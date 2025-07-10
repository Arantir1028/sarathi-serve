import copy
import time
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from typing import List, Tuple

import zmq

from sarathi.config import SystemConfig
from sarathi.core.datatypes.request_output import RequestOutput
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SamplerOutputs, SequenceMetadata, Sequence
from sarathi.core.datatypes.step_inputs import StepInputs
from sarathi.engine.base_llm_engine import BaseLLMEngine
from sarathi.engine.ray_utils import RayWorker, initialize_cluster, ray
from sarathi.core.datatypes.comm_info import CommInfo
from sarathi.logger import init_logger
from sarathi.utils import get_ip, unset_cuda_visible_devices
from sarathi.utils.threading_utils import exit_on_error, synchronized

logger = init_logger(__name__)

SCHEDULER_LOOP_DELAY = 0.01

@dataclass
class ScheduleStageOutputs:
    ignored_seqs: List[Sequence]
    seq_metadata_list: List[SequenceMetadata]
    scheduler_outputs: SchedulerOutputs
    start_time: float

class PipelineConcurrentLLMEngine(BaseLLMEngine):
    """单GPU多stage流水线并行的LLM Engine。"""
    def __init__(self, config: SystemConfig) -> None:
        super().__init__(config)
        self.has_started_execution_loops = False
        self.scheduler_output_queue = Queue()
        self.output_queue = Queue()
        self.schedule_event = Event()
        self.microbatch_watch_event = Event()
        self.schedule_thread = Thread(target=self._schedule_loop, daemon=True)
        self.microbatch_watch_thread = Thread(
            target=self._microbatch_watch_loop, daemon=True
        )
        self.output_thread = Thread(target=self._output_loop, daemon=True)
        self.scheduler_timer_thread = Thread(
            target=self._scheduler_timer_loop, daemon=True
        )
        self.pending_step_outputs: List[Tuple[SchedulerOutputs, SamplerOutputs]] = []

    def _validate_parallel_config(self) -> None:
        assert self.config.parallel_config.tensor_parallel_size == 1
        assert self.config.parallel_config.pipeline_parallel_size > 1

    def _get_worker_impl(self):
        from sarathi.worker.pipeline_concurrent_worker import PipelineConcurrentWorker
        return PipelineConcurrentWorker

    def _init_workers_ray(self, **ray_remote_kwargs):
        # 所有worker都在同一块GPU上
        num_stages = self.config.parallel_config.pipeline_parallel_size
        
        # 强制所有stage都在GPU 0上
        self.config.replica_config.resource_mapping = [(None, 0) for _ in range(num_stages)]
        resource_mapping = self.config.replica_config.get_resource_mapping(num_stages)
        
        logger.info(f"Pipeline concurrent mode: Starting {num_stages} workers on single GPU with resource mapping: {resource_mapping}")
        
        self.workers = []
        unset_cuda_visible_devices()
        driver_ip = get_ip()
        
        for rank in range(num_stages):
            worker_class = ray.remote(num_cpus=1, **ray_remote_kwargs)(RayWorker)
            worker_class = worker_class.options(max_concurrency=1)
            worker = worker_class.remote(self.config.model_config.trust_remote_code)
            self.workers.append(worker)
            
        self.comm_info = CommInfo(driver_ip)
        
        # Initialize torch distributed process group for the workers.
        config = copy.deepcopy(self.config)
        config.metrics_config = self.metrics_store.get_config_for_worker()
        
        worker_impl = self._get_worker_impl()
        
        for rank, worker in enumerate(self.workers):
            # 所有worker都使用local_rank=0，因为都在同一块GPU上
            promise = worker.init_worker.remote(
                lambda rank=rank, local_rank=0: worker_impl(
                    config,
                    local_rank,
                    rank,
                    self.comm_info,
                )
            )
            ray.get(promise)
            
        self._run_workers("init_model", get_all_outputs=True)
        
        # 在pipeline_concurrent模式下，使用动态资源分配
        self._init_concurrent_resources()
    
    def _init_concurrent_resources(self) -> None:
        """在pipeline_concurrent模式下初始化动态资源分配"""
        # 首先获取总的可用GPU blocks
        total_gpu_blocks = self._run_workers(
            "profile_num_available_blocks",
            get_all_outputs=True,
            block_size=self.config.cache_config.block_size,
            gpu_memory_utilization=self.config.worker_config.gpu_memory_utilization,
        )
        total_gpu_blocks = total_gpu_blocks[0]  # 使用第一个worker的结果
        
        # 获取算力资源信息
        total_compute_units = self._run_workers(
            "profile_compute_resources",
            get_all_outputs=True,
            compute_utilization=self.config.worker_config.compute_utilization,
        )
        total_compute_units = total_compute_units[0]  # 使用第一个worker的结果
        
        # 使用资源分配策略
        from sarathi.config.pipeline_resource_strategy import get_resource_strategy
        
        strategy_name = self.config.worker_config.resource_allocation_strategy
        strategy_kwargs = {}
        
        # 设置策略特定参数
        if strategy_name == "layer_based":
            strategy_kwargs.update({
                "cache_weight": self.config.worker_config.cache_weight,
                "compute_weight": self.config.worker_config.compute_weight,
            })
        elif strategy_name == "performance_based":
            if self.config.worker_config.compute_allocation_weights:
                strategy_kwargs["performance_weights"] = self.config.worker_config.compute_allocation_weights
        
        strategy = get_resource_strategy(strategy_name, **strategy_kwargs)
        allocation_result = strategy.allocate_resources(
            total_gpu_blocks,
            total_compute_units,
            self.config.model_config, 
            self.config.parallel_config
        )
        
        logger.info(f"Pipeline concurrent resource allocation ({allocation_result.strategy_name}):")
        logger.info(f"Cache blocks: {allocation_result.cache_allocation.stage_values}")
        logger.info(f"Cache ratios: {allocation_result.cache_allocation.stage_allocations}")
        logger.info(f"Compute units: {allocation_result.compute_allocation.stage_values}")
        logger.info(f"Compute ratios: {allocation_result.compute_allocation.stage_allocations}")
        
        # 为每个worker设置对应的资源配置
        for i, worker in enumerate(self.workers):
            # 设置缓存配置
            worker_cache_config = copy.deepcopy(self.config.cache_config)
            worker_cache_config.num_gpu_blocks = allocation_result.cache_allocation.stage_values[i]
            
            # 设置算力配置
            worker_compute_config = {
                "compute_units": allocation_result.compute_allocation.stage_values[i],
                "compute_ratio": allocation_result.compute_allocation.stage_allocations[i],
                "enable_dynamic_allocation": self.config.worker_config.enable_dynamic_compute_allocation,
            }
            
            # 直接调用worker的init_resources方法并等待完成
            ray.get(worker.execute_method.remote("init_resources", 
                                               cache_config=worker_cache_config,
                                               compute_config=worker_compute_config))
        
        # 等待所有worker完成初始化
        self._run_workers("wait_till_ready", get_all_outputs=True)
    
    def _init_cache(self) -> None:
        """在pipeline_concurrent模式下，缓存初始化已经在_init_concurrent_resources中完成"""
        # 在pipeline_concurrent模式下，缓存初始化已经在_init_concurrent_resources中完成
        # 这里不需要做任何事情，避免重复初始化
        logger.info("Pipeline concurrent mode: cache initialization already completed in _init_concurrent_resources")
        
        # 但是我们需要确保cache_config.num_gpu_blocks有值，因为scheduler需要它
        # 这里设置一个默认值，实际的分配在_init_concurrent_resources中完成
        if self.config.cache_config.num_gpu_blocks is None:
            # 临时设置一个值，避免scheduler初始化失败
            # 实际的分配会在_init_concurrent_resources中重新设置
            self.config.cache_config.num_gpu_blocks = 1000  # 临时值

    def _init_zmq_sockets(self):
        super()._init_zmq_sockets()
        self.microbatch_socket = self.zmq_context.socket(zmq.PULL)
        self.microbatch_socket.bind(f"tcp://*:{self.comm_info.microbatch_socket_port}")

    def start_execution_loops(self) -> None:
        self.has_started_execution_loops = True
        self.schedule_event.set()
        self.schedule_thread.start()
        self.output_thread.start()
        self.scheduler_timer_thread.start()
        self.microbatch_watch_thread.start()

    @exit_on_error
    def _scheduler_timer_loop(self) -> None:
        while True:
            time.sleep(SCHEDULER_LOOP_DELAY)
            self.schedule_event.set()

    @synchronized
    def _append_pending_step_output(
        self, scheduler_outputs: SchedulerOutputs, sampler_outputs: SamplerOutputs
    ) -> None:
        self.pending_step_outputs.append((scheduler_outputs, sampler_outputs))

    @synchronized
    def _get_pending_step_outputs(
        self,
    ) -> List[Tuple[SchedulerOutputs, SamplerOutputs]]:
        pending_step_outputs = self.pending_step_outputs
        self.pending_step_outputs = []
        return pending_step_outputs

    @exit_on_error
    def _schedule_loop(self) -> None:
        while True:
            self.schedule_event.wait()
            self.schedule_event.clear()
            start_time = time.perf_counter()
            scheduler_outputs = self.scheduler.schedule()
            if scheduler_outputs.has_no_output():
                continue
            ignored_seqs, seq_metadata_list = self.seq_manager.on_schedule(
                scheduler_outputs
            )
            self.scheduler_output_queue.put(
                ScheduleStageOutputs(
                    ignored_seqs,
                    seq_metadata_list,
                    scheduler_outputs,
                    start_time,
                )
            )
            end_time = time.perf_counter()
            if not scheduler_outputs.is_empty():
                self.microbatch_watch_event.set()
                self.enqueue_socket.send_pyobj(
                    StepInputs(
                        scheduler_outputs,
                        new_seqs=self._get_new_seqs(),
                        pending_step_outputs=self._get_pending_step_outputs(),
                    )
                )
            self.metrics_store.on_schedule(seq_metadata_list, start_time, end_time)

    @exit_on_error
    def _microbatch_watch_loop(self) -> None:
        while True:
            self.microbatch_watch_event.wait()
            self.microbatch_socket.recv_pyobj()
            self.schedule_event.set()

    @exit_on_error
    def _output_loop(self) -> None:
        while True:
            scheduler_stage_output = self.scheduler_output_queue.get()
            sampler_outputs = self.output_socket.recv_pyobj()
            self._append_pending_step_output(
                scheduler_stage_output.scheduler_outputs, sampler_outputs
            )
            all_request_outputs = self._on_step_completed(
                scheduler_stage_output.scheduler_outputs,
                scheduler_stage_output.ignored_seqs,
                scheduler_stage_output.seq_metadata_list,
                sampler_outputs,
                scheduler_stage_output.start_time,
            )
            self.schedule_event.set()
            self.output_queue.put(all_request_outputs)

    def step(self, block: bool = True) -> List[RequestOutput]:
        if not self.has_started_execution_loops:
            self.start_execution_loops()
        if block:
            return self.output_queue.get()
        try:
            return self.output_queue.get(block=False)
        except Empty:
            return [] 
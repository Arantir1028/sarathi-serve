import time
from typing import Tuple, Optional

import torch
import torch.distributed
import zmq

from sarathi.config import CacheConfig
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SamplerOutputs
from sarathi.logger import init_logger
from sarathi.utils.threading_utils import exit_on_error, synchronized
from sarathi.worker.base_worker import BaseWorker

logger = init_logger(__name__)

class PipelineConcurrentWorker(BaseWorker):
    """单GPU多stage流水线的Worker，每个stage都在同一块GPU上。"""
    def _verify_parallel_config(self) -> None:
        assert self.config.parallel_config.tensor_parallel_size == 1
        assert self.config.parallel_config.pipeline_parallel_size > 1

    def init_model(self):
        import torch
        import os
        
        # 强制所有worker都用同一块GPU
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        
        # 设置环境变量避免NCCL问题
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        os.environ["KINETO_LOG_LEVEL"] = "5"
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        os.environ["PIPELINE_CONCURRENT_MODE"] = "1"
        
        # 手动设置parallel_state模块的全局变量，避免依赖分布式环境
        from sarathi.model_executor.parallel_utils.parallel_state import (
            set_tensor_model_parallel_rank,
            set_pipeline_model_parallel_rank,
            set_tensor_model_parallel_world_size,
            set_pipeline_model_parallel_world_size,
        )
        
        # 设置tensor parallel相关
        set_tensor_model_parallel_rank(0)
        set_tensor_model_parallel_world_size(self.config.parallel_config.tensor_parallel_size)
        
        # 记录原始pipeline配置
        original_pp_size = self.config.parallel_config.pipeline_parallel_size
        original_pp_rank = self.rank
        
        # 临时patch pipeline config，保证权重加载完整
        self.config.parallel_config.pipeline_parallel_size = 1
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)
        
        # 初始化模型并加载完整权重
        from sarathi.model_executor import set_random_seed
        from sarathi.model_executor.model_runner import ModelRunner
        set_random_seed(self.config.model_config.seed)
        self.model_runner = ModelRunner(
            self.config,
            self.device,
            self.rank,
        )
        
        # 恢复原始pipeline配置
        self.config.parallel_config.pipeline_parallel_size = original_pp_size
        set_pipeline_model_parallel_world_size(original_pp_size)
        set_pipeline_model_parallel_rank(original_pp_rank)

        # 重新设置本地rank属性
        self.tensor_model_parallel_rank = 0
        self.pipeline_model_parallel_rank = self.rank
        self.is_tensor_parallel_rank_zero = True
        self.is_first_pipeline_stage = self.pipeline_model_parallel_rank == 0
        self.is_last_pipeline_stage = (
            self.pipeline_model_parallel_rank
            == self.config.parallel_config.pipeline_parallel_size - 1
        )
        
        # 伪造分布式组（保持原有逻辑）
        pipeline_ranks = list(range(self.config.parallel_config.pipeline_parallel_size))
        import sarathi.model_executor.parallel_utils.parallel_state as parallel_state
        parallel_state._PIPELINE_MODEL_PARALLEL_GROUP = type('FakeGroup', (), {
            'size': lambda *args: self.config.parallel_config.pipeline_parallel_size,
            'rank': lambda *args: self.pipeline_model_parallel_rank,
        })()
        parallel_state._PIPELINE_GLOBAL_RANKS = pipeline_ranks
        parallel_state._TENSOR_MODEL_PARALLEL_GROUP = type('FakeGroup', (), {
            'size': lambda *args: self.config.parallel_config.tensor_parallel_size,
            'rank': lambda *args: self.tensor_model_parallel_rank,
        })()
        parallel_state._DATA_PARALLEL_GROUP = type('FakeGroup', (), {
            'size': lambda *args: 1,
            'rank': lambda *args: 0,
        })()
        parallel_state._DATA_PARALLEL_GLOBAL_RANKS = [0]
        parallel_state._MODEL_PARALLEL_GROUP = type('FakeGroup', (), {
            'size': lambda *args: self.config.parallel_config.world_size,
            'rank': lambda *args: self.rank,
        })()
        parallel_state._EMBEDDING_GROUP = type('FakeGroup', (), {
            'size': lambda *args: 1,
            'rank': lambda *args: 0,
        })()
        parallel_state._EMBEDDING_GLOBAL_RANKS = [0]
        parallel_state._POSITION_EMBEDDING_GROUP = type('FakeGroup', (), {
            'size': lambda *args: 1,
            'rank': lambda *args: 0,
        })()
        parallel_state._POSITION_EMBEDDING_GLOBAL_RANKS = [0]
        
        # sampler/lm_head初始化逻辑保持不变
        if self.is_last_pipeline_stage:
            if self.model_runner.sampler is None:
                from sarathi.model_executor.layers.sampler import Sampler
                if hasattr(self.model_runner.model, 'lm_head') and self.model_runner.model.lm_head is not None:
                    from sarathi.model_executor.weight_utils import load_padded_tensor_parallel_vocab, hf_model_weights_iterator
                    from sarathi.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_rank
                    tp_rank = get_tensor_model_parallel_rank()
                    for name, loaded_weight in hf_model_weights_iterator(
                        self.config.model_config.model,
                        self.config.model_config.download_dir,
                        self.config.model_config.load_format,
                        self.config.model_config.revision,
                    ):
                        if "lm_head" in name:
                            param = self.model_runner.model.state_dict()[name]
                            load_padded_tensor_parallel_vocab(param, loaded_weight, tp_rank)
                            break
                    self.model_runner.sampler = Sampler(
                        self.model_runner.model.lm_head.weight, 
                        self.model_runner.model.config.vocab_size
                    )
                else:
                    from sarathi.model_executor.parallel_utils.tensor_parallel import ColumnParallelLinear
                    vocab_size = self.model_runner.model.config.vocab_size
                    vocab_size = ((vocab_size + 63) // 64) * 64
                    self.model_runner.model.lm_head = ColumnParallelLinear(
                        self.model_runner.model.config.hidden_size,
                        vocab_size,
                        bias=False,
                        gather_output=False,
                        perform_initialization=False,
                    )
                    self.model_runner.sampler = Sampler(
                        self.model_runner.model.lm_head.weight, 
                        self.model_runner.model.config.vocab_size
                    )
        else:
            self.model_runner.sampler = None
    
    @torch.inference_mode()
    @synchronized
    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        """向后兼容的缓存初始化方法"""
        self.init_resources(cache_config=cache_config)
    
    @torch.inference_mode()
    @synchronized
    def init_resources(self, cache_config: CacheConfig, compute_config: Optional[dict] = None) -> None:
        """初始化缓存和算力资源"""
        torch.cuda.set_device(self.device)

        self.config.cache_config = cache_config

        if cache_config.num_gpu_blocks is None:
            raise ValueError("num_gpu_blocks must be set before calling init_resources")
        
        self.model_runner.init_kv_cache(cache_config.num_gpu_blocks)

        # 设置算力资源配置
        if compute_config:
            self.compute_units = compute_config.get("compute_units", 1)
            self.compute_ratio = compute_config.get("compute_ratio", 1.0)
            self.enable_dynamic_allocation = compute_config.get("enable_dynamic_allocation", True)
        else:
            self.compute_units = 1
            self.compute_ratio = 1.0
            self.enable_dynamic_allocation = False

        # 初始化seq_manager
        from sarathi.core.sequence_manager.worker_sequence_manager import WorkerSequenceManager
        self.seq_manager = WorkerSequenceManager(
            self.config,
        )

        self.execution_thread.start()
    
    @synchronized
    def profile_compute_resources(self, compute_utilization: float) -> int:
        """分析可用的算力资源"""
        # 获取GPU的SM数量作为算力单位
        if torch.cuda.is_available():
            device_properties = torch.cuda.get_device_properties(self.device)
            total_sms = device_properties.multi_processor_count
            available_sms = int(total_sms * compute_utilization)
            return available_sms
        else:
            return 1

    def _init_zmq_sockets(self):
        super()._init_zmq_sockets()
        self.microbatch_socket = self.zmq_context.socket(zmq.PUSH)
        self.microbatch_socket.connect(
            f"tcp://{self.comm_info.engine_ip_address}:{self.comm_info.microbatch_socket_port}"
        )

    def on_step_completed(
        self, scheduler_outputs: SchedulerOutputs, sampler_outputs: SamplerOutputs
    ) -> None:
        # 在流水线并行下，每个stage通常没有独立的sampler output
        pass

    @synchronized
    def on_sampling_completed(
        self, scheduler_outputs: SchedulerOutputs, sampler_outputs: SamplerOutputs
    ) -> None:
        self.seq_manager.on_step_completed(scheduler_outputs, sampler_outputs)

    @exit_on_error
    def _execution_loop(self) -> None:
        torch.cuda.set_device(self.device)
        self.worker_ready_event.set()
        while True:
            step_inputs = self.enqueue_socket.recv_pyobj()
            for new_seq in step_inputs.new_seqs:
                self.seq_manager.add_seq(new_seq)
            for pending_step_output in getattr(step_inputs, 'pending_step_outputs', []):
                self.seq_manager.on_step_completed(
                    pending_step_output[0], pending_step_output[1]
                )
            output = self.execute_model(step_inputs.scheduler_outputs)
            if not self.is_tensor_parallel_rank_zero:
                continue
            if self.is_last_pipeline_stage:
                self.output_socket.send_pyobj(output)
            elif self.is_first_pipeline_stage:
                self.microbatch_socket.send_pyobj(None)

    @synchronized
    def get_model_parallel_ranks(self) -> Tuple[int, int]:
        return self.tensor_model_parallel_rank, self.pipeline_model_parallel_rank 

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_outputs: SchedulerOutputs,
    ) -> Optional[SamplerOutputs]:
        """在pipeline_concurrent模式下执行模型"""
        torch.cuda.synchronize()
        batch_stage_start_time = time.monotonic()

        _, seq_metadata_list = self.seq_manager.on_schedule(scheduler_outputs)

        # 在pipeline_concurrent模式下，模型的前向传播会通过send/recv函数在内存中传递数据
        # 只有最后一个stage才需要返回采样输出
        if self.is_last_pipeline_stage:
            # 确保sampler存在且seq_metadata_list不为空
            if self.model_runner.sampler is None:
                # 再次尝试修复sampler
                from sarathi.model_executor.layers.sampler import Sampler
                if hasattr(self.model_runner.model, 'lm_head') and self.model_runner.model.lm_head is not None:
                    self.model_runner.sampler = Sampler(
                        self.model_runner.model.lm_head.weight, 
                        self.model_runner.model.config.vocab_size
                    )
                else:
                    # 创建空的SamplerOutputs作为fallback
                    from sarathi.core.datatypes.sequence import SamplerOutput
                    output = []
                    for seq_metadata in seq_metadata_list:
                        sampler_output = SamplerOutput(
                            seq_id=seq_metadata.seq.seq_id,
                            output_token=0  # 占位符token
                        )
                        output.append(sampler_output)
                    return output
            
            # 正常执行模型推理和采样
            output = self.model_runner.run(seq_metadata_list)
            
            # 确保output是SamplerOutputs类型
            if not isinstance(output, list):
                # 创建空的SamplerOutputs作为fallback
                from sarathi.core.datatypes.sequence import SamplerOutput
                output = []
                for seq_metadata in seq_metadata_list:
                    sampler_output = SamplerOutput(
                        seq_id=seq_metadata.seq.seq_id,
                        output_token=0  # 占位符token
                    )
                    output.append(sampler_output)
        else:
            # 非最后一个stage只执行前向传播，不进行采样
            # 临时禁用sampler，让模型只执行前向传播
            original_sampler = self.model_runner.sampler
            self.model_runner.sampler = None
            
            try:
                # 这里会返回torch.Tensor，但我们不需要它
                self.model_runner.run(seq_metadata_list)
                output = None
            finally:
                # 恢复sampler
                self.model_runner.sampler = original_sampler

        # 在pipeline_concurrent模式下，我们不需要调用on_step_completed
        # 因为每个stage只处理部分计算，最终的序列管理由engine处理

        torch.cuda.synchronize()

        batch_stage_end_time = time.monotonic()

        self.metrics_store.on_batch_stage_end(
            seq_metadata_list,
            scheduler_outputs,
            self.tensor_model_parallel_rank,
            self.pipeline_model_parallel_rank,
            batch_stage_start_time,
            batch_stage_end_time,
        )

        return output 
# edit

import time
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from typing import List, Tuple

import zmq
from sarathi.config import SystemConfig
from sarathi.core.datatypes.request_output import RequestOutput
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SamplerOutputs, SequenceMetadata
from sarathi.core.datatypes.step_inputs import StepInputs
from sarathi.engine.base_llm_engine import BaseLLMEngine
from sarathi.logger import init_logger
from sarathi.utils.threading_utils import exit_on_error, synchronized

logger = init_logger(__name__)

SCHEDULER_LOOP_DELAY = 0.01

@dataclass
class ScheduleStageOutputs:
    ignored_seqs: List[SequenceMetadata]
    seq_metadata_list: List[SequenceMetadata]
    scheduler_outputs: SchedulerOutputs
    start_time: float

class PipelineConcurrentLLMEngine(BaseLLMEngine):
    """LLM引擎支持单GPU内多Stream管道并发"""
    
    def __init__(self, config: SystemConfig) -> None:
        super().__init__(config)
        
        # 并发控制队列
        self.has_started_execution_loops = False
        self.scheduler_output_queue = Queue()
        self.output_queue = Queue()
        self.schedule_event = Event()
        
        # 每个管道有自己的完成事件
        self.pipeline_events = [Event() for _ in range(config.parallel_config.pipeline_parallel_size)]
        
        # 线程池
        self.schedule_thread = Thread(target=self._schedule_loop, daemon=True)
        self.output_thread = Thread(target=self._output_loop, daemon=True)
        
        self.pending_step_outputs: List[Tuple[SchedulerOutputs, SamplerOutputs]] = []

    def _validate_parallel_config(self) -> None:
        assert self.config.parallel_config.pipeline_parallel_size >= 2
        assert self.config.parallel_config.tensor_parallel_size == 1  # 单GPU

    def start_execution_loops(self) -> None:
        self.has_started_execution_loops = True
        self.schedule_event.set()
        self.schedule_thread.start()
        self.output_thread.start()

    @exit_on_error
    def _schedule_loop(self) -> None:
        while True:
            self.schedule_event.wait()
            self.schedule_event.clear()

            start_time = time.perf_counter()
            scheduler_outputs = self.scheduler.schedule()

            if scheduler_outputs.has_no_output():
                continue

            ignored_seqs, seq_metadata_list = self.seq_manager.on_schedule(scheduler_outputs)

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
                # 通知所有管道开始处理
                for event in self.pipeline_events:
                    event.set()
                
                self.enqueue_socket.send_pyobj(
                    StepInputs(
                        scheduler_outputs,
                        new_seqs=self._get_new_seqs(),
                        pending_step_outputs=self._get_pending_step_outputs(),
                    )
                )

            self.metrics_store.on_schedule(seq_metadata_list, start_time, end_time)

    @exit_on_error
    def _output_loop(self) -> None:
        while True:
            scheduler_stage_output = self.scheduler_output_queue.get()
            
            # 等待所有管道完成
            sampler_outputs = []
            for _ in range(self.config.parallel_config.pipeline_parallel_size):
                output = self.output_socket.recv_pyobj()
                sampler_outputs.append(output)
            
            # 合并结果 (简单策略: 取第一个有效输出)
            final_output = next(out for out in sampler_outputs if out is not None)
            
            self._append_pending_step_output(
                scheduler_stage_output.scheduler_outputs, 
                final_output
            )

            all_request_outputs = self._on_step_completed(
                scheduler_stage_output.scheduler_outputs,
                scheduler_stage_output.ignored_seqs,
                scheduler_stage_output.seq_metadata_list,
                final_output,
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
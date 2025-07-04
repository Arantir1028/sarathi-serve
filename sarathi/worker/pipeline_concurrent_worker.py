#edit

import torch
import zmq
from typing import Optional

from sarathi.config import SystemConfig
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import SamplerOutputs
from sarathi.logger import init_logger
from sarathi.utils.threading_utils import exit_on_error, synchronized
from sarathi.worker.base_worker import BaseWorker

logger = init_logger(__name__)

class PipelineConcurrentWorker(BaseWorker):
    """单GPU内多Stream管道并发的Worker实现
    
    关键特性：
    1. 在单个GPU上创建多个CUDA Stream实现伪管道并行
    2. 自动轮询调度不同Stream执行计算任务
    3. 保持与原始PipelineParallelWorker相同的接口
    """
    
    def _verify_parallel_config(self) -> None:
        """验证并行配置必须满足：
        1. 管道并行度>=2（但实际在单GPU运行）
        2. 张量并行度=1（单GPU）
        """
        assert self.config.parallel_config.pipeline_parallel_size >= 2
        assert self.config.parallel_config.tensor_parallel_size == 1

    def __init__(
        self,
        config: SystemConfig,
        local_rank: int,
        rank: int,
        comm_info,
    ) -> None:

        super().__init__(config, local_rank, rank, comm_info)
        
        # 初始化多Stream
        self.streams = [
            torch.cuda.Stream(device=self.device) 
            for _ in range(config.parallel_config.pipeline_parallel_size)
        ]
        self.current_stream_idx = 0

    def _init_zmq_sockets(self):
        """初始化通信端口（硬编码方式）"""
        self.zmq_context = zmq.Context()
        
        # 输入端口（接收引擎调度指令）
        self.enqueue_socket = self.zmq_context.socket(zmq.SUB)
        self.enqueue_socket.connect("tcp://localhost:5555")  # 硬编码端口
        self.enqueue_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # 输出端口（返回计算结果）
        self.output_socket = self.zmq_context.socket(zmq.PUSH)
        self.output_socket.connect("tcp://localhost:5556")
        
        # 微批次控制端口（协调Stream间同步）
        self.microbatch_socket = self.zmq_context.socket(zmq.PUSH)
        self.microbatch_socket.connect("tcp://localhost:5557")



    @synchronized
    def execute_model(
        self,
        scheduler_outputs: SchedulerOutputs,
    ) -> Optional[SamplerOutputs]:
        """在指定Stream上执行模型计算"""
        # 轮询选择Stream
        stream = self.streams[self.current_stream_idx]
        self.current_stream_idx = (self.current_stream_idx + 1) % len(self.streams)
        
        with torch.cuda.stream(stream):
            # 准备序列数据
            _, seq_metadata_list = self.seq_manager.on_schedule(scheduler_outputs)
            
            # 执行模型计算
            sampler_outputs = self.model_runner.run(seq_metadata_list)
            
            # 只有最后一个"管道阶段"返回结果
            if self.is_last_pipeline_stage:
                return sampler_outputs
            return None

    @exit_on_error
    def _execution_loop(self) -> None:
        """主执行循环"""
        torch.cuda.set_device(self.device)
        self.worker_ready_event.set()

        while True:
            # 1. 接收输入数据
            step_inputs = self.enqueue_socket.recv_pyobj()

            # 2. 处理新序列
            for new_seq in step_inputs.new_seqs:
                self.seq_manager.add_seq(new_seq)

            # 3. 处理待更新状态
            for pending_step_output in step_inputs.pending_step_outputs:
                self.seq_manager.on_step_completed(*pending_step_output)

            # 4. 执行计算
            output = self.execute_model(step_inputs.scheduler_outputs)

            # 5. 处理输出
            if not self.is_tensor_parallel_rank_zero:
                continue  # 仅rank0需要返回结果

            if output is not None:  # 末阶段
                self.output_socket.send_pyobj(output)
            elif self.is_first_pipeline_stage:  # 首阶段
                self.microbatch_socket.send_pyobj(None)  # 触发下一阶段

    def on_step_completed(
        self, 
        scheduler_outputs: SchedulerOutputs, 
        sampler_outputs: SamplerOutputs
    ) -> None:
        """空实现（状态更新由主循环统一处理）"""
        pass
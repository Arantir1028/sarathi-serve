import torch
import os

from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.cuda_timer import CudaTimer
from sarathi.model_executor.parallel_utils.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_pipeline_model_parallel_rank,
)

# 在pipeline_concurrent模式下用于存储中间结果的全局变量
_pipeline_concurrent_buffer = {}

def send(hidden_states: torch.tensor):
    """Send hidden states to the next pipeline stage."""
    # Bypass the function if we are using only 1 stage.
    if get_pipeline_model_parallel_group().size() == 1:
        return hidden_states
    
    # 在pipeline_concurrent模式下，使用内存共享
    if os.environ.get('PIPELINE_CONCURRENT_MODE') == '1':
        current_rank = get_pipeline_model_parallel_rank()
        next_rank = get_pipeline_model_parallel_next_rank()
        
        # 将数据存储到全局缓冲区中
        global _pipeline_concurrent_buffer
        _pipeline_concurrent_buffer[f"stage_{next_rank}"] = hidden_states.clone()
        
        return hidden_states

    with CudaTimer(OperationMetrics.NCCL_SEND):
        # Send the tensor.
        torch.distributed.send(
            tensor=hidden_states,
            dst=get_pipeline_model_parallel_next_rank(),
            group=get_pipeline_model_parallel_group(),
        )


def recv(hidden_states: torch.tensor):
    """Receive hidden states from the previous pipeline stage."""
    # Bypass the function if we are using only 1 stage.
    if get_pipeline_model_parallel_group().size() == 1:
        return hidden_states
    
    # 在pipeline_concurrent模式下，使用内存共享
    if os.environ.get('PIPELINE_CONCURRENT_MODE') == '1':
        current_rank = get_pipeline_model_parallel_rank()
        prev_rank = get_pipeline_model_parallel_prev_rank()
        
        # 从全局缓冲区中获取数据
        global _pipeline_concurrent_buffer
        buffer_key = f"stage_{current_rank}"
        
        if buffer_key in _pipeline_concurrent_buffer:
            # 将数据复制到目标张量中
            hidden_states.copy_(_pipeline_concurrent_buffer[buffer_key])
            # 清理缓冲区
            del _pipeline_concurrent_buffer[buffer_key]
        else:
            # 如果没有数据，使用零张量（这种情况不应该发生）
            hidden_states.zero_()
        
        return hidden_states

    # Receive the tensor.
    with CudaTimer(OperationMetrics.NCCL_RECV):
        torch.distributed.recv(
            tensor=hidden_states,
            src=get_pipeline_model_parallel_prev_rank(),
            group=get_pipeline_model_parallel_group(),
        )

    return hidden_states

import time
from typing import Tuple, Optional

import torch
import torch.distributed
import zmq
import threading
import queue

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

    def _split_layers(self):
        # 适配Mistral模型，假设self.model_runner.model.model.layers为nn.ModuleList
        all_layers = list(self.model_runner.model.model.layers)
        num_stages = self.config.parallel_config.pipeline_parallel_size
        layers_per_stage = len(all_layers) // num_stages
        stage_layers = []
        for i in range(num_stages):
            start = i * layers_per_stage
            end = (i + 1) * layers_per_stage if i < num_stages - 1 else len(all_layers)
            stage_layers.append(all_layers[start:end])
        return stage_layers

    def pipeline_inference(self, input_tensor, positions, attention_backend_wrapper):
        num_stages = self.config.parallel_config.pipeline_parallel_size
        device = self.device
        stage_layers = self._split_layers()
        streams = [torch.cuda.Stream() for _ in range(num_stages)]
        queues = [queue.Queue() for _ in range(num_stages + 1)]
        threads = []

        def stage_worker(stage_id, input_queue, output_queue, layers, stream):
            torch.cuda.set_device(device)
            while True:
                x = input_queue.get()
                if x is None:
                    output_queue.put(None)
                    break
                with torch.cuda.stream(stream):
                    for idx, layer in enumerate(layers):
                        global_layer_id = stage_id * len(layers) + idx
                        # 只保留最小必要的shape/dtype修正
                        if isinstance(x, torch.Tensor):
                            if x.dim() == 1:
                                x = x.unsqueeze(0)
                            if x.dtype != torch.float16:
                                x = x.to(dtype=torch.float16)
                        x = layer(positions, x, global_layer_id, attention_backend_wrapper)
                output_queue.put(x)

        for i in range(num_stages):
            t = threading.Thread(target=stage_worker, args=(i, queues[i], queues[i+1], stage_layers[i], streams[i]))
            t.start()
            threads.append(t)

        if isinstance(input_tensor, torch.Tensor):
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)
            if input_tensor.dtype != torch.float16:
                input_tensor = input_tensor.to(dtype=torch.float16)
        queues[0].put(input_tensor)
        queues[0].put(None)

        result = None
        while True:
            out = queues[-1].get()
            if out is None:
                break
            result = out

        for t in threads:
            t.join()
        return result

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_outputs: SchedulerOutputs,
    ) -> Optional[SamplerOutputs]:
        torch.cuda.synchronize()
        batch_stage_start_time = time.monotonic()
        for seq_sched_metadata in scheduler_outputs.scheduled_seq_metadata_list:
            seq_id = seq_sched_metadata.seq_id
            seq = self.seq_manager.seq_map[seq_id]
            if seq.is_waiting() or seq.is_paused():
                self.seq_manager._resume_seq(seq_id)
            if not hasattr(self.seq_manager.block_manager, 'block_tables') or seq_id not in self.seq_manager.block_manager.block_tables:
                self.seq_manager.block_manager.allocate(seq)
        seq_metadata_list = []
        from sarathi.core.datatypes.sequence import SequenceMetadata
        for seq_sched_metadata in scheduler_outputs.scheduled_seq_metadata_list:
            seq = self.seq_manager.seq_map[seq_sched_metadata.seq_id]
            block_table = self.seq_manager._get_block_table(seq)
            seq_metadata_list.append(
                SequenceMetadata(
                    seq,
                    block_table,
                    seq_sched_metadata.num_prompt_tokens,
                )
            )
        outputs = []
        if not self.is_last_pipeline_stage:
            return None
        outputs = []
        for seq_metadata in seq_metadata_list:
            input_tokens, input_positions = self.model_runner._prepare_inputs([seq_metadata])
            self.model_runner.attention_backend_wrapper.begin_forward([seq_metadata])
            if hasattr(self.model_runner.model.model, 'embed_tokens') and input_tokens.dtype in (torch.int32, torch.int64):
                if input_tokens.dim() == 2 and input_tokens.shape[0] == 1:
                    input_tokens = input_tokens.squeeze(0)
                hidden_states = self.model_runner.model.model.embed_tokens(input_tokens)
                if hidden_states.dim() == 3 and hidden_states.shape[0] == 1:
                    hidden_states = hidden_states.squeeze(0)
            else:
                hidden_states = input_tokens
            hidden_states = self.pipeline_inference(hidden_states, input_positions, self.model_runner.attention_backend_wrapper)
            self.model_runner.attention_backend_wrapper.end_forward()
            if self.model_runner.sampler is not None:
                sampler_output = self.model_runner.sampler(hidden_states, [seq_metadata])
                if isinstance(sampler_output, list):
                    outputs.extend(sampler_output)
                elif sampler_output is not None:
                    outputs.append(sampler_output)
            else:
                from sarathi.core.datatypes.sequence import SamplerOutput
                outputs.append(SamplerOutput(seq_id=seq_metadata.seq.seq_id, output_token=0))
        return outputs

    @synchronized
    def get_model_parallel_ranks(self) -> Tuple[int, int]:
        return self.tensor_model_parallel_rank, self.pipeline_model_parallel_rank 

    def _execution_loop(self) -> None:
        import torch
        torch.cuda.set_device(self.device)
        self.worker_ready_event.set()
        while True:
            step_inputs = self.enqueue_socket.recv_pyobj()
            for new_seq in step_inputs.new_seqs:
                self.seq_manager.add_seq(new_seq)
            output = self.execute_model(step_inputs.scheduler_outputs)
            # 只让最后一个 stage 发送 output，其余 stage 不发送
            if self.is_last_pipeline_stage:
                self.output_socket.send_pyobj(output)
from sarathi.config import SystemConfig
from sarathi.engine.base_llm_engine import BaseLLMEngine
from sarathi.engine.pipeline_parallel_llm_engine import PipelineParallelLLMEngine
from sarathi.engine.pipeline_concurrent_engine import PipelineConcurrentLLMEngine


class LLMEngine:

    @classmethod
    def from_system_config(cls, config: SystemConfig):
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        if (
            config.parallel_config.tensor_parallel_size == 1 and
            config.parallel_config.pipeline_parallel_size == 1
        ):
            # 当tensor_parallel_size=1且pipeline_parallel_size=1时，使用BaseLLMEngine
            engine = BaseLLMEngine(config)
        elif (
            hasattr(config.worker_config, 'force_concurrent') and 
            config.worker_config.force_concurrent and
            config.parallel_config.pipeline_parallel_size > 1 and
            config.parallel_config.tensor_parallel_size == 1
        ):
            # 当force_concurrent=True且满足pipeline_concurrent条件时，使用PipelineConcurrentLLMEngine
            engine = PipelineConcurrentLLMEngine(config)
        elif config.parallel_config.pipeline_parallel_size > 1:
            # 其他pipeline并行情况使用PipelineParallelLLMEngine
            engine = PipelineParallelLLMEngine(config)
        else:
            # 默认使用BaseLLMEngine
            engine = BaseLLMEngine(config)

        return engine

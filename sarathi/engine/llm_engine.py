#edit

from sarathi.config import SystemConfig
from sarathi.engine.base_llm_engine import BaseLLMEngine
from sarathi.engine.pipeline_parallel_llm_engine import PipelineParallelLLMEngine
from sarathi.engine.pipeline_concurrent_llm_engine import PipelineConcurrentLLMEngine


class LLMEngine:

    @classmethod
    def from_system_config(cls, config: SystemConfig) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        # if config.parallel_config.pipeline_parallel_size > 1:
        if config.parallel_config.pipeline_parallel_size > 2:
            engine = PipelineParallelLLMEngine(config)
        elif config.parallel_config.pipeline_parallel_size == 2:
            engine = PipelineConcurrentLLMEngine(config) # 按说可以考虑直接大于2，但GPU估计跑不动
        else:
            engine = BaseLLMEngine(config)

        return engine

"""
Pipeline Cache Allocation Strategies

This module defines different strategies for allocating GPU cache blocks
among pipeline stages in pipeline_concurrent mode.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

from sarathi.config import ModelConfig, ParallelConfig


@dataclass
class CacheAllocationResult:
    """Result of cache allocation strategy."""
    stage_blocks: List[int]
    allocation_ratios: List[float]
    strategy_name: str


class CacheAllocationStrategy(ABC):
    """Abstract base class for cache allocation strategies."""
    
    @abstractmethod
    def allocate(self, total_blocks: int, model_config: ModelConfig, 
                parallel_config: ParallelConfig) -> CacheAllocationResult:
        """Allocate cache blocks among pipeline stages."""
        pass


class LayerBasedStrategy(CacheAllocationStrategy):
    """Allocate cache based on the number of layers in each stage."""
    
    def allocate(self, total_blocks: int, model_config: ModelConfig, 
                parallel_config: ParallelConfig) -> CacheAllocationResult:
        pipeline_size = parallel_config.pipeline_parallel_size
        total_layers = model_config.get_total_num_layers()
        layers_per_stage = total_layers // pipeline_size
        
        # Calculate allocation ratios based on layer count
        stage_allocations = []
        for stage in range(pipeline_size):
            start_layer = stage * layers_per_stage
            end_layer = (stage + 1) * layers_per_stage if stage < pipeline_size - 1 else total_layers
            stage_layers = end_layer - start_layer
            
            # Each stage gets at least 10% of total blocks
            min_allocation = 0.1
            layer_weight = stage_layers / total_layers
            allocation = max(min_allocation, layer_weight)
            stage_allocations.append(allocation)
        
        # Normalize allocations
        total_allocation = sum(stage_allocations)
        stage_allocations = [alloc / total_allocation for alloc in stage_allocations]
        
        # Calculate actual block counts
        stage_blocks = []
        remaining_blocks = total_blocks
        
        for i, allocation in enumerate(stage_allocations):
            if i == len(stage_allocations) - 1:
                blocks = remaining_blocks
            else:
                blocks = int(total_blocks * allocation)
                remaining_blocks -= blocks
            stage_blocks.append(blocks)
        
        return CacheAllocationResult(
            stage_blocks=stage_blocks,
            allocation_ratios=stage_allocations,
            strategy_name="layer_based"
        )


class EqualStrategy(CacheAllocationStrategy):
    """Equal allocation among all stages."""
    
    def allocate(self, total_blocks: int, model_config: ModelConfig, 
                parallel_config: ParallelConfig) -> CacheAllocationResult:
        pipeline_size = parallel_config.pipeline_parallel_size
        blocks_per_stage = total_blocks // pipeline_size
        
        stage_blocks = [blocks_per_stage] * pipeline_size
        # Give remaining blocks to the last stage
        stage_blocks[-1] += total_blocks % pipeline_size
        
        allocation_ratios = [1.0 / pipeline_size] * pipeline_size
        
        return CacheAllocationResult(
            stage_blocks=stage_blocks,
            allocation_ratios=allocation_ratios,
            strategy_name="equal"
        )


class WeightedStrategy(CacheAllocationStrategy):
    """Weighted allocation based on custom weights."""
    
    def __init__(self, weights: List[float]):
        self.weights = weights
    
    def allocate(self, total_blocks: int, model_config: ModelConfig, 
                parallel_config: ParallelConfig) -> CacheAllocationResult:
        pipeline_size = parallel_config.pipeline_parallel_size
        
        if len(self.weights) != pipeline_size:
            raise ValueError(f"Expected {pipeline_size} weights, got {len(self.weights)}")
        
        # Normalize weights
        total_weight = sum(self.weights)
        normalized_weights = [w / total_weight for w in self.weights]
        
        # Calculate block counts
        stage_blocks = []
        remaining_blocks = total_blocks
        
        for i, weight in enumerate(normalized_weights):
            if i == len(normalized_weights) - 1:
                blocks = remaining_blocks
            else:
                blocks = int(total_blocks * weight)
                remaining_blocks -= blocks
            stage_blocks.append(blocks)
        
        return CacheAllocationResult(
            stage_blocks=stage_blocks,
            allocation_ratios=normalized_weights,
            strategy_name="weighted"
        )


class AdaptiveStrategy(CacheAllocationStrategy):
    """Adaptive allocation based on runtime performance metrics."""
    
    def __init__(self, initial_strategy: CacheAllocationStrategy = None):
        self.initial_strategy = initial_strategy or LayerBasedStrategy()
        self.performance_history: Dict[int, List[float]] = {}
    
    def allocate(self, total_blocks: int, model_config: ModelConfig, 
                parallel_config: ParallelConfig) -> CacheAllocationResult:
        # For now, use the initial strategy
        # In the future, this could analyze performance history and adjust allocation
        return self.initial_strategy.allocate(total_blocks, model_config, parallel_config)
    
    def update_performance(self, stage_id: int, performance_metric: float):
        """Update performance metrics for a stage."""
        if stage_id not in self.performance_history:
            self.performance_history[stage_id] = []
        self.performance_history[stage_id].append(performance_metric)


# Strategy registry
STRATEGY_REGISTRY = {
    "layer_based": LayerBasedStrategy,
    "equal": EqualStrategy,
    "adaptive": AdaptiveStrategy,
}


def get_strategy(strategy_name: str, **kwargs) -> CacheAllocationStrategy:
    """Get a cache allocation strategy by name."""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    return strategy_class(**kwargs) 
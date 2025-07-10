"""
Pipeline Resource Allocation Strategies

This module defines strategies for allocating both cache blocks and compute resources
among pipeline stages in pipeline_concurrent mode.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from sarathi.config import ModelConfig, ParallelConfig


class ResourceType(Enum):
    """Resource types that can be allocated."""
    CACHE = "cache"
    COMPUTE = "compute"
    MEMORY = "memory"


@dataclass
class ResourceAllocation:
    """Allocation for a specific resource type."""
    resource_type: ResourceType
    stage_allocations: List[float]  # Allocation ratios for each stage
    stage_values: List[Any]  # Actual values for each stage (e.g., block counts)


@dataclass
class PipelineResourceResult:
    """Result of pipeline resource allocation."""
    cache_allocation: ResourceAllocation
    compute_allocation: ResourceAllocation
    memory_allocation: Optional[ResourceAllocation] = None
    strategy_name: str = ""


class ResourceAllocationStrategy(ABC):
    """Abstract base class for resource allocation strategies."""
    
    @abstractmethod
    def allocate_resources(
        self, 
        total_cache_blocks: int,
        total_compute_units: int,
        model_config: ModelConfig, 
        parallel_config: ParallelConfig,
        **kwargs
    ) -> PipelineResourceResult:
        """Allocate both cache and compute resources among pipeline stages."""
        pass


class LayerBasedResourceStrategy(ResourceAllocationStrategy):
    """Allocate resources based on the number of layers in each stage."""
    
    def __init__(self, cache_weight: float = 0.6, compute_weight: float = 0.4):
        """
        Args:
            cache_weight: Weight for cache allocation (0.0 to 1.0)
            compute_weight: Weight for compute allocation (0.0 to 1.0)
        """
        self.cache_weight = cache_weight
        self.compute_weight = compute_weight
    
    def allocate_resources(
        self, 
        total_cache_blocks: int,
        total_compute_units: int,
        model_config: ModelConfig, 
        parallel_config: ParallelConfig,
        **kwargs
    ) -> PipelineResourceResult:
        pipeline_size = parallel_config.pipeline_parallel_size
        total_layers = model_config.get_total_num_layers()
        layers_per_stage = total_layers // pipeline_size
        
        # Calculate layer-based allocation ratios
        stage_allocations = []
        for stage in range(pipeline_size):
            start_layer = stage * layers_per_stage
            end_layer = (stage + 1) * layers_per_stage if stage < pipeline_size - 1 else total_layers
            stage_layers = end_layer - start_layer
            
            # Each stage gets at least 10% of resources
            min_allocation = 0.1
            layer_weight = stage_layers / total_layers
            allocation = max(min_allocation, layer_weight)
            stage_allocations.append(allocation)
        
        # Normalize allocations
        total_allocation = sum(stage_allocations)
        stage_allocations = [alloc / total_allocation for alloc in stage_allocations]
        
        # Allocate cache blocks
        cache_blocks = self._allocate_blocks(total_cache_blocks, stage_allocations)
        cache_allocation = ResourceAllocation(
            resource_type=ResourceType.CACHE,
            stage_allocations=stage_allocations,
            stage_values=cache_blocks
        )
        
        # Allocate compute units (could be CUDA cores, SMs, etc.)
        compute_units = self._allocate_blocks(total_compute_units, stage_allocations)
        compute_allocation = ResourceAllocation(
            resource_type=ResourceType.COMPUTE,
            stage_allocations=stage_allocations,
            stage_values=compute_units
        )
        
        return PipelineResourceResult(
            cache_allocation=cache_allocation,
            compute_allocation=compute_allocation,
            strategy_name="layer_based"
        )
    
    def _allocate_blocks(self, total_blocks: int, ratios: List[float]) -> List[int]:
        """Allocate blocks based on ratios."""
        stage_blocks = []
        remaining_blocks = total_blocks
        
        for i, ratio in enumerate(ratios):
            if i == len(ratios) - 1:
                blocks = remaining_blocks
            else:
                blocks = int(total_blocks * ratio)
                remaining_blocks -= blocks
            stage_blocks.append(blocks)
        
        return stage_blocks


class PerformanceBasedStrategy(ResourceAllocationStrategy):
    """Allocate resources based on performance characteristics of each stage."""
    
    def __init__(self, performance_weights: Optional[List[float]] = None):
        """
        Args:
            performance_weights: Custom performance weights for each stage
        """
        self.performance_weights = performance_weights
    
    def allocate_resources(
        self, 
        total_cache_blocks: int,
        total_compute_units: int,
        model_config: ModelConfig, 
        parallel_config: ParallelConfig,
        **kwargs
    ) -> PipelineResourceResult:
        pipeline_size = parallel_config.pipeline_parallel_size
        
        if self.performance_weights:
            # Use custom performance weights
            if len(self.performance_weights) != pipeline_size:
                raise ValueError(f"Expected {pipeline_size} weights, got {len(self.performance_weights)}")
            stage_allocations = self.performance_weights
        else:
            # Estimate performance based on model characteristics
            stage_allocations = self._estimate_performance_weights(model_config, parallel_config)
        
        # Normalize allocations
        total_allocation = sum(stage_allocations)
        stage_allocations = [alloc / total_allocation for alloc in stage_allocations]
        
        # Allocate resources
        cache_blocks = self._allocate_blocks(total_cache_blocks, stage_allocations)
        compute_units = self._allocate_blocks(total_compute_units, stage_allocations)
        
        return PipelineResourceResult(
            cache_allocation=ResourceAllocation(
                resource_type=ResourceType.CACHE,
                stage_allocations=stage_allocations,
                stage_values=cache_blocks
            ),
            compute_allocation=ResourceAllocation(
                resource_type=ResourceType.COMPUTE,
                stage_allocations=stage_allocations,
                stage_values=compute_units
            ),
            strategy_name="performance_based"
        )
    
    def _estimate_performance_weights(self, model_config: ModelConfig, parallel_config: ParallelConfig) -> List[float]:
        """Estimate performance weights based on model characteristics."""
        pipeline_size = parallel_config.pipeline_parallel_size
        total_layers = model_config.get_total_num_layers()
        
        # Simple estimation: later stages might need more resources due to accumulated activations
        weights = []
        for stage in range(pipeline_size):
            # Give more weight to later stages
            weight = 1.0 + (stage * 0.2)  # 20% increase per stage
            weights.append(weight)
        
        return weights
    
    def _allocate_blocks(self, total_blocks: int, ratios: List[float]) -> List[int]:
        """Allocate blocks based on ratios."""
        stage_blocks = []
        remaining_blocks = total_blocks
        
        for i, ratio in enumerate(ratios):
            if i == len(ratios) - 1:
                blocks = remaining_blocks
            else:
                blocks = int(total_blocks * ratio)
                remaining_blocks -= blocks
            stage_blocks.append(blocks)
        
        return stage_blocks


class AdaptiveResourceStrategy(ResourceAllocationStrategy):
    """Adaptive resource allocation based on runtime performance metrics."""
    
    def __init__(self, initial_strategy: ResourceAllocationStrategy = None):
        self.initial_strategy = initial_strategy or LayerBasedResourceStrategy()
        self.performance_history: Dict[int, List[float]] = {}
        self.resource_history: Dict[int, List[ResourceAllocation]] = {}
    
    def allocate_resources(
        self, 
        total_cache_blocks: int,
        total_compute_units: int,
        model_config: ModelConfig, 
        parallel_config: ParallelConfig,
        **kwargs
    ) -> PipelineResourceResult:
        # For now, use the initial strategy
        # In the future, this could analyze performance history and adjust allocation
        return self.initial_strategy.allocate_resources(
            total_cache_blocks, total_compute_units, model_config, parallel_config, **kwargs
        )
    
    def update_performance(self, stage_id: int, performance_metrics: Dict[str, float]):
        """Update performance metrics for a stage."""
        if stage_id not in self.performance_history:
            self.performance_history[stage_id] = []
        
        # Store performance metrics
        self.performance_history[stage_id].append(performance_metrics)
    
    def update_resource_usage(self, stage_id: int, resource_usage: ResourceAllocation):
        """Update resource usage for a stage."""
        if stage_id not in self.resource_history:
            self.resource_history[stage_id] = []
        self.resource_history[stage_id].append(resource_usage)


class EqualResourceStrategy(ResourceAllocationStrategy):
    """Equal allocation of all resources among stages."""
    
    def allocate_resources(
        self, 
        total_cache_blocks: int,
        total_compute_units: int,
        model_config: ModelConfig, 
        parallel_config: ParallelConfig,
        **kwargs
    ) -> PipelineResourceResult:
        pipeline_size = parallel_config.pipeline_parallel_size
        
        # Equal allocation ratios
        equal_ratio = 1.0 / pipeline_size
        stage_allocations = [equal_ratio] * pipeline_size
        
        # Allocate cache blocks
        cache_blocks = self._allocate_blocks(total_cache_blocks, stage_allocations)
        
        # Allocate compute units
        compute_units = self._allocate_blocks(total_compute_units, stage_allocations)
        
        return PipelineResourceResult(
            cache_allocation=ResourceAllocation(
                resource_type=ResourceType.CACHE,
                stage_allocations=stage_allocations,
                stage_values=cache_blocks
            ),
            compute_allocation=ResourceAllocation(
                resource_type=ResourceType.COMPUTE,
                stage_allocations=stage_allocations,
                stage_values=compute_units
            ),
            strategy_name="equal"
        )
    
    def _allocate_blocks(self, total_blocks: int, ratios: List[float]) -> List[int]:
        """Allocate blocks based on ratios."""
        stage_blocks = []
        remaining_blocks = total_blocks
        
        for i, ratio in enumerate(ratios):
            if i == len(ratios) - 1:
                blocks = remaining_blocks
            else:
                blocks = int(total_blocks * ratio)
                remaining_blocks -= blocks
            stage_blocks.append(blocks)
        
        return stage_blocks


# Strategy registry
RESOURCE_STRATEGY_REGISTRY = {
    "layer_based": LayerBasedResourceStrategy,
    "performance_based": PerformanceBasedStrategy,
    "adaptive": AdaptiveResourceStrategy,
    "equal": EqualResourceStrategy,
}


def get_resource_strategy(strategy_name: str, **kwargs) -> ResourceAllocationStrategy:
    """Get a resource allocation strategy by name."""
    if strategy_name not in RESOURCE_STRATEGY_REGISTRY:
        raise ValueError(f"Unknown resource strategy: {strategy_name}")
    
    strategy_class = RESOURCE_STRATEGY_REGISTRY[strategy_name]
    return strategy_class(**kwargs) 
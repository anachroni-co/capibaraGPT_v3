from dataclasses import dataclass
from typing import Tuple


@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory and compute optimizations.

    This configuration groups options related to memory savings, mixed precision
    and parallelism used across the model.
    """

    # Gradient checkpointing and rematerialization
    use_gradient_checkpointing: bool = True
    checkpoint_policy: str = "block_depth"
    checkpoint_blocks: int = 2
    remat_layer_interval: int = 2
    preserve_rng_state: bool = True

    # Memory thresholds and monitoring
    memory_pressure_threshold: float = 0.85
    enable_memory_monitoring: bool = True
    cleanup_interval: float = 300.0
    max_memory_usage_gb: float = 32.0
    force_device_gc: bool = True

    # Precision and dtype
    use_mixed_precision: bool = True
    compute_dtype: str = "bfloat16"
    param_dtype: str = "float32"
    output_dtype: str = "float32"

    # Parallelism and sharding
    use_model_parallelism: bool = True
    model_parallel_submesh: Tuple[int, int] = (2, 2)
    data_parallel_submesh: Tuple[int, int] = (2, 1)
    shard_strategy: str = "axis_0"

    # Dynamic batch size
    dynamic_batch_size: bool = True
    min_batch_size: int = 4
    max_batch_size: int = 32
    batch_growth_factor: float = 1.5
"""
PyTorch FSDP (Fully Sharded Data Parallel) Wrapper

Alternative to DeepSpeed for distributed training with:
- Full parameter sharding
- Mixed precision training
- Gradient checkpointing
- Optimized for PyTorch 2.x
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, Type, Set
from enum import Enum
from functools import partial

logger = logging.getLogger(__name__)

# Lazy imports
torch = None
FSDP = None
ShardingStrategy_torch = None
MixedPrecision = None
CPUOffload = None
BackwardPrefetch = None
FullStateDictConfig = None
StateDictType = None


def _ensure_torch():
    global torch, FSDP, ShardingStrategy_torch, MixedPrecision, CPUOffload
    global BackwardPrefetch, FullStateDictConfig, StateDictType

    if torch is None:
        import torch as _torch
        from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP
        from torch.distributed.fsdp import (
            ShardingStrategy as _ShardingStrategy,
            MixedPrecision as _MixedPrecision,
            CPUOffload as _CPUOffload,
            BackwardPrefetch as _BackwardPrefetch,
        )
        from torch.distributed.fsdp.api import (
            FullStateDictConfig as _FullStateDictConfig,
            StateDictType as _StateDictType,
        )

        torch = _torch
        FSDP = _FSDP
        ShardingStrategy_torch = _ShardingStrategy
        MixedPrecision = _MixedPrecision
        CPUOffload = _CPUOffload
        BackwardPrefetch = _BackwardPrefetch
        FullStateDictConfig = _FullStateDictConfig
        StateDictType = _StateDictType


class ShardingStrategy(Enum):
    """FSDP sharding strategies."""
    FULL_SHARD = "full_shard"           # ZeRO-3 equivalent
    SHARD_GRAD_OP = "shard_grad_op"     # ZeRO-2 equivalent
    NO_SHARD = "no_shard"               # DDP equivalent
    HYBRID_SHARD = "hybrid_shard"       # Shard within node, replicate across


class BackwardPrefetchPolicy(Enum):
    """Backward prefetch policies."""
    BACKWARD_PRE = "backward_pre"
    BACKWARD_POST = "backward_post"
    NONE = "none"


@dataclass
class FSDPConfig:
    """Configuration for FSDP wrapper."""

    # Sharding strategy
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD

    # Mixed precision
    use_mixed_precision: bool = True
    param_dtype: str = "bfloat16"
    reduce_dtype: str = "float32"
    buffer_dtype: str = "bfloat16"

    # CPU offloading
    cpu_offload: bool = False

    # Backward prefetch
    backward_prefetch: BackwardPrefetchPolicy = BackwardPrefetchPolicy.BACKWARD_PRE

    # Gradient checkpointing
    activation_checkpointing: bool = True
    checkpoint_every_n_layers: int = 1

    # Memory optimization
    limit_all_gathers: bool = True
    forward_prefetch: bool = True

    # State dict configuration
    use_orig_params: bool = True  # Needed for torch.compile

    # Auto wrapping
    auto_wrap: bool = True
    min_num_params: int = 100_000_000  # 100M params minimum for wrapping

    # Sync module states
    sync_module_states: bool = True

    def get_torch_sharding_strategy(self):
        """Get PyTorch ShardingStrategy enum."""
        _ensure_torch()
        mapping = {
            ShardingStrategy.FULL_SHARD: ShardingStrategy_torch.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP: ShardingStrategy_torch.SHARD_GRAD_OP,
            ShardingStrategy.NO_SHARD: ShardingStrategy_torch.NO_SHARD,
            ShardingStrategy.HYBRID_SHARD: ShardingStrategy_torch.HYBRID_SHARD,
        }
        return mapping[self.sharding_strategy]

    def get_mixed_precision(self):
        """Get MixedPrecision configuration."""
        if not self.use_mixed_precision:
            return None

        _ensure_torch()

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        return MixedPrecision(
            param_dtype=dtype_map[self.param_dtype],
            reduce_dtype=dtype_map[self.reduce_dtype],
            buffer_dtype=dtype_map[self.buffer_dtype],
        )

    def get_cpu_offload(self):
        """Get CPUOffload configuration."""
        if not self.cpu_offload:
            return None

        _ensure_torch()
        return CPUOffload(offload_params=True)

    def get_backward_prefetch(self):
        """Get BackwardPrefetch configuration."""
        _ensure_torch()
        mapping = {
            BackwardPrefetchPolicy.BACKWARD_PRE: BackwardPrefetch.BACKWARD_PRE,
            BackwardPrefetchPolicy.BACKWARD_POST: BackwardPrefetch.BACKWARD_POST,
            BackwardPrefetchPolicy.NONE: None,
        }
        return mapping[self.backward_prefetch]


class FSDPWrapper:
    """Wrapper for applying FSDP to models."""

    def __init__(self, config: FSDPConfig):
        self.config = config
        _ensure_torch()

    def wrap_model(
        self,
        model: "torch.nn.Module",
        auto_wrap_policy: Optional[Callable] = None,
        transformer_layer_cls: Optional[Set[Type]] = None,
    ) -> "FSDP":
        """
        Wrap a model with FSDP.

        Args:
            model: PyTorch model to wrap
            auto_wrap_policy: Custom auto wrap policy
            transformer_layer_cls: Set of transformer layer classes for auto-wrapping

        Returns:
            FSDP-wrapped model
        """
        cfg = self.config

        # Determine auto wrap policy
        if auto_wrap_policy is None and cfg.auto_wrap:
            if transformer_layer_cls:
                from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
                auto_wrap_policy = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=transformer_layer_cls,
                )
            else:
                from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
                auto_wrap_policy = partial(
                    size_based_auto_wrap_policy,
                    min_num_params=cfg.min_num_params,
                )

        # Apply activation checkpointing before FSDP wrapping
        if cfg.activation_checkpointing:
            self._apply_activation_checkpointing(model, transformer_layer_cls)

        # Wrap with FSDP
        wrapped_model = FSDP(
            model,
            sharding_strategy=cfg.get_torch_sharding_strategy(),
            mixed_precision=cfg.get_mixed_precision(),
            cpu_offload=cfg.get_cpu_offload(),
            backward_prefetch=cfg.get_backward_prefetch(),
            auto_wrap_policy=auto_wrap_policy,
            limit_all_gathers=cfg.limit_all_gathers,
            forward_prefetch=cfg.forward_prefetch,
            use_orig_params=cfg.use_orig_params,
            sync_module_states=cfg.sync_module_states,
        )

        logger.info(f"Model wrapped with FSDP: strategy={cfg.sharding_strategy.value}, "
                   f"mixed_precision={cfg.use_mixed_precision}, cpu_offload={cfg.cpu_offload}")

        return wrapped_model

    def _apply_activation_checkpointing(
        self,
        model: "torch.nn.Module",
        layer_classes: Optional[Set[Type]] = None,
    ):
        """Apply activation checkpointing to model layers."""
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointImpl,
            apply_activation_checkpointing,
        )

        if layer_classes:
            check_fn = lambda submodule: isinstance(submodule, tuple(layer_classes))
        else:
            check_fn = lambda submodule: hasattr(submodule, 'forward') and \
                sum(p.numel() for p in submodule.parameters()) > 1_000_000

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=check_fn,
        )

        logger.info("Activation checkpointing applied")

    def save_checkpoint(
        self,
        model: "FSDP",
        optimizer: "torch.optim.Optimizer",
        path: str,
        rank: int = 0,
        full_state_dict: bool = True,
    ) -> None:
        """
        Save FSDP checkpoint.

        Args:
            model: FSDP-wrapped model
            optimizer: Optimizer
            path: Checkpoint path
            rank: Current process rank
            full_state_dict: Whether to gather full state dict on rank 0
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType

        if full_state_dict:
            # Gather full state dict on rank 0
            full_state_dict_config = FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True,
            )

            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                full_state_dict_config,
            ):
                model_state = model.state_dict()
                optim_state = FSDP.optim_state_dict(model, optimizer)

            if rank == 0:
                checkpoint = {
                    "model": model_state,
                    "optimizer": optim_state,
                }
                torch.save(checkpoint, path)
                logger.info(f"Full checkpoint saved to: {path}")
        else:
            # Save sharded checkpoint (each rank saves its shard)
            sharded_path = f"{path}.rank{rank}"

            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                model_state = model.state_dict()
                optim_state = FSDP.optim_state_dict(model, optimizer)

            checkpoint = {
                "model": model_state,
                "optimizer": optim_state,
            }
            torch.save(checkpoint, sharded_path)
            logger.info(f"Sharded checkpoint saved to: {sharded_path}")

    def load_checkpoint(
        self,
        model: "FSDP",
        optimizer: "torch.optim.Optimizer",
        path: str,
        full_state_dict: bool = True,
    ) -> None:
        """
        Load FSDP checkpoint.

        Args:
            model: FSDP-wrapped model
            optimizer: Optimizer
            path: Checkpoint path
            full_state_dict: Whether loading full state dict
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType

        if full_state_dict:
            full_state_dict_config = FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True,
            )

            checkpoint = torch.load(path, map_location="cpu")

            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                full_state_dict_config,
            ):
                model.load_state_dict(checkpoint["model"])
                FSDP.optim_state_dict_to_load(
                    model, optimizer, checkpoint["optimizer"]
                )

            logger.info(f"Full checkpoint loaded from: {path}")
        else:
            # Load sharded checkpoint
            checkpoint = torch.load(path, map_location="cpu")

            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                model.load_state_dict(checkpoint["model"])
                FSDP.optim_state_dict_to_load(
                    model, optimizer, checkpoint["optimizer"]
                )

            logger.info(f"Sharded checkpoint loaded from: {path}")


def create_fsdp_config(
    model_size: str = "7B",
    num_gpus: int = 8,
    gpu_memory_gb: int = 80,
    use_cpu_offload: bool = False,
) -> FSDPConfig:
    """
    Create optimized FSDP configuration.

    Args:
        model_size: Model size (e.g., "3B", "7B", "13B", "30B", "70B")
        num_gpus: Total number of GPUs
        gpu_memory_gb: GPU memory in GB
        use_cpu_offload: Enable CPU offloading

    Returns:
        Optimized FSDPConfig
    """
    model_params = {
        "3B": 3_000_000_000,
        "7B": 7_000_000_000,
        "13B": 13_000_000_000,
        "30B": 30_000_000_000,
        "70B": 70_000_000_000,
    }

    params = model_params.get(model_size, 7_000_000_000)
    params_per_gpu = params / num_gpus
    bytes_per_param = 2  # BF16
    memory_per_gpu_gb = (params_per_gpu * bytes_per_param * 4) / (1024**3)

    # Choose sharding strategy based on memory requirements
    if memory_per_gpu_gb > gpu_memory_gb * 0.7:
        strategy = ShardingStrategy.FULL_SHARD
    elif memory_per_gpu_gb > gpu_memory_gb * 0.4:
        strategy = ShardingStrategy.SHARD_GRAD_OP
    else:
        strategy = ShardingStrategy.HYBRID_SHARD

    logger.info(f"Creating FSDP config for {model_size}: strategy={strategy.value}, "
                f"cpu_offload={use_cpu_offload}")

    return FSDPConfig(
        sharding_strategy=strategy,
        use_mixed_precision=True,
        param_dtype="bfloat16",
        reduce_dtype="float32",
        buffer_dtype="bfloat16",
        cpu_offload=use_cpu_offload,
        activation_checkpointing=True,
        limit_all_gathers=True,
        forward_prefetch=True,
        use_orig_params=True,
    )

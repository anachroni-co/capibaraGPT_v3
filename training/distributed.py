"""Capibara Slim — gradient checkpointing + FSDP (T6.1, T6.2).

Gradient checkpointing:
    Wraps each TransformerBlock / MambaBlock in torch.utils.checkpoint
    to recompute activations on the backward pass instead of storing them.
    Memory saving: ~60 % at ~33 % extra compute cost.
    Enables 3B on 1× A100 40 GB and 7B on 2× A100.

FSDP (Fully Sharded Data Parallel):
    Wraps the model for multi-GPU training using PyTorch FSDP with
    ZeRO-3 style sharding (parameters + gradients + optimizer states).
    Enables 7B on 4–8× A100 without model parallelism.

Usage:
    # Single-node gradient checkpointing
    model = apply_gradient_checkpointing(model)

    # Multi-GPU FSDP (run with torchrun)
    model = wrap_fsdp(model, cfg)
    trainer = SlimTrainer(model, train_cfg, loader)
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.checkpoint import checkpoint as _ckpt
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        BackwardPrefetch,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    import functools
    _FSDP_AVAILABLE = True
except ImportError:
    _FSDP_AVAILABLE = False


def apply_gradient_checkpointing(model: "nn.Module") -> "nn.Module":
    """Wrap each transformer/mamba block to recompute activations on backward.

    Reduces peak activation memory by ~60 % at the cost of ~33 % extra
    forward passes during backward.  Call before wrapping with FSDP.
    """
    if not _TORCH:
        raise ImportError("PyTorch required for gradient checkpointing.")

    from models.architecture import TransformerBlock, MambaBlock

    wrapped = 0
    for module in model.modules():
        if isinstance(module, (TransformerBlock, MambaBlock)):
            _patch_block(module)
            wrapped += 1

    logger.info("Gradient checkpointing enabled on %d blocks", wrapped)
    return model


def _patch_block(block: "nn.Module") -> None:
    """Monkey-patch block.forward to run through torch.utils.checkpoint."""
    original_forward = block.forward

    def checkpointed_forward(*args, **kwargs):
        # checkpoint requires non-keyword tensor inputs; re-wrap kwargs
        def _fn(*a):
            return original_forward(*a, **kwargs)
        return _ckpt(_fn, *args, use_reentrant=False)

    block.forward = checkpointed_forward


def wrap_fsdp(
    model: "nn.Module",
    slim_config=None,
    dtype: str = "bf16",
    sharding: str = "full",
) -> "nn.Module":
    """Wrap model in PyTorch FSDP for multi-GPU ZeRO-3 training.

    Requires torchrun / torch.distributed to be initialised before calling.

    Args:
        model:       SlimModel (or any nn.Module)
        slim_config: SlimConfig instance — used to set auto-wrap policy
        dtype:       mixed-precision dtype ("bf16", "fp16", "fp32")
        sharding:    "full" (ZeRO-3) | "shard_grad_op" (ZeRO-2) | "no_shard"

    Returns:
        FSDP-wrapped model, ready for distributed training.
    """
    if not _TORCH:
        raise ImportError("PyTorch required for FSDP.")
    if not _FSDP_AVAILABLE:
        raise ImportError(
            "FSDP not available — upgrade to PyTorch ≥ 1.12: pip install torch>=1.12"
        )

    _dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    param_dtype = _dtype_map.get(dtype, torch.bfloat16)

    mp_policy = MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
        buffer_dtype=param_dtype,
    )

    _sharding_map = {
        "full":          ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard":      ShardingStrategy.NO_SHARD,
    }
    sharding_strategy = _sharding_map.get(sharding, ShardingStrategy.FULL_SHARD)

    # Auto-wrap each TransformerBlock / MambaBlock as its own FSDP unit
    from models.architecture import TransformerBlock, MambaBlock
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock, MambaBlock},
    )

    wrapped = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=sharding_strategy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
    )

    logger.info(
        "FSDP enabled | sharding=%s dtype=%s",
        sharding_strategy.name, dtype,
    )
    return wrapped


def setup_distributed() -> tuple[int, int]:
    """Initialise torch.distributed (call at start of torchrun worker).

    Returns: (rank, world_size)
    """
    if not _TORCH:
        raise ImportError("PyTorch required.")
    import os
    import torch.distributed as dist

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    logger.info("Distributed init: rank=%d world_size=%d backend=%s", rank, world_size, backend)
    return rank, world_size

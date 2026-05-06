"""Layers Module - CapibaraGPT v3.

Surviving layers after BACKLOG-017 cleanup. Imports are explicit; no
silent-fallback is permitted (CONTRIBUTING.md §1).

Available:
- BaseLayer / LayerConfig (layers.base)
- SelfAttention / SelfAttentionConfig / TpuAttentionCache (layers.self_attention)
- jax_compat (layers.jax_compat) - JAX/Flax availability + shims
- attention_utils (split_heads, merge_heads)
"""
from .base import BaseLayer, LayerConfig
from .self_attention import SelfAttention, SelfAttentionConfig, TpuAttentionCache
from .jax_compat import jax, jnp, nn, JAX_AVAILABLE
from .attention_utils import split_heads, merge_heads

__all__ = [
    "BaseLayer",
    "LayerConfig",
    "SelfAttention",
    "SelfAttentionConfig",
    "TpuAttentionCache",
    "JAX_AVAILABLE",
    "split_heads",
    "merge_heads",
    "jax",
    "jnp",
    "nn",
]

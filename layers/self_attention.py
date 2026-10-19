"""
Self-attention layers for CapibaraGPT v3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Any

from layers.jax_compat import jnp, nn, JAX_AVAILABLE
from layers.base import LayerConfig

logger = logging.getLogger(__name__)


@dataclass
class SelfAttentionConfig(LayerConfig):
    """Configuration for SelfAttention."""
    num_heads: int = 8
    dropout_rate: float = 0.1


class SelfAttention(nn.Module):
    """Minimal multi-head self-attention layer."""
    config: SelfAttentionConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None,
                 training: bool = False) -> jnp.ndarray:
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.config.num_heads,
            qkv_features=x.shape[-1],
            dropout_rate=self.config.dropout_rate,
        )
        return attn(x, x, mask=mask, deterministic=not training)


class TpuAttentionCache:
    """Simple attention cache container (CPU/JAX friendly)."""

    def __init__(self):
        self.keys = None
        self.values = None

    def update(self, keys: Any, values: Any) -> None:
        self.keys = keys
        self.values = values

    def reset(self) -> None:
        self.keys = None
        self.values = None


def main() -> bool:
    logger.info("Module self_attention.py starting")
    if not JAX_AVAILABLE:
        logger.warning("JAX/Flax not available; SelfAttention will not be usable.")
    return True


if __name__ == "__main__":
    main()

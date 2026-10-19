"""
Hybrid SSM + Attention layer stack for CapibaraGPT v3.

Implements the TRUE hybrid Transformer-Mamba architecture where SSM (Mamba/S4)
layers and self-attention layers are interleaved throughout the model depth,
rather than routing an entire forward pass to one architecture or the other.

Layer assignment is controlled by `ssm_layers` and `attention_layers` index
lists (from HybridModelConfig). The default interleaved pattern alternates
SSM on even indices and attention on odd indices.

Each layer follows the pre-norm residual pattern:
    x = x + Block(LayerNorm(x))
    x = x + FFN(LayerNorm(x))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from layers.base import LayerConfig
from layers.self_attention import SelfAttention, SelfAttentionConfig
from capibara.ssm.ssm_tpu import SSMBlock

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class HybridLayerStackConfig:
    """
    Configuration for HybridLayerStack.

    Attributes:
        num_layers: Total number of model layers.
        hidden_size: Residual stream width.
        num_heads: Number of attention heads (attention layers only).
        d_state: SSM state dimension (SSM layers only).
        dropout_rate: Dropout applied inside attention.
        ffn_mult: FFN hidden size multiplier (hidden_size * ffn_mult).
        ssm_layers: Layer indices that use SSM. If None, defaults to even
            indices [0, 2, 4, ...].
        attention_layers: Layer indices that use attention. If None, defaults
            to odd indices [1, 3, 5, ...].
    """
    num_layers: int = 12
    hidden_size: int = 768
    num_heads: int = 12
    d_state: int = 64
    dropout_rate: float = 0.1
    ffn_mult: int = 4
    ssm_layers: Optional[List[int]] = None
    attention_layers: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if self.ssm_layers is None:
            self.ssm_layers = list(range(0, self.num_layers, 2))
        if self.attention_layers is None:
            self.attention_layers = list(range(1, self.num_layers, 2))

        ssm_set = set(self.ssm_layers)
        attn_set = set(self.attention_layers)
        overlap = ssm_set & attn_set
        if overlap:
            raise ValueError(
                f"Layer indices appear in both ssm_layers and attention_layers: {overlap}"
            )
        assigned = ssm_set | attn_set
        missing = set(range(self.num_layers)) - assigned
        if missing:
            logger.warning(
                "Layers %s have no assigned block type — they will be identity.", missing
            )


# ── Single hybrid layer ───────────────────────────────────────────────────────

class _SSMLayer(nn.Module):
    """Pre-norm SSM layer with residual connection and FFN."""
    hidden_size: int
    d_state: int
    ffn_mult: int
    layer_idx: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # ── sequence mixing (SSM) ──────────────────────────────────────────
        residual = x
        x = nn.LayerNorm(name="ssm_norm")(x)
        x = SSMBlock(hidden_size=self.hidden_size, state_dim=self.d_state)(x)
        x = residual + x

        # ── channel mixing (FFN) ──────────────────────────────────────────
        residual = x
        x = nn.LayerNorm(name="ffn_norm")(x)
        x = nn.Dense(self.hidden_size * self.ffn_mult, name="ffn_up")(x)
        x = jax.nn.gelu(x)
        x = nn.Dense(self.hidden_size, name="ffn_down")(x)
        x = residual + x
        return x


class _AttentionLayer(nn.Module):
    """Pre-norm attention layer with residual connection and FFN."""
    hidden_size: int
    num_heads: int
    dropout_rate: float
    ffn_mult: int
    layer_idx: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        attn_cfg = SelfAttentionConfig(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )

        # ── sequence mixing (attention) ────────────────────────────────────
        residual = x
        x = nn.LayerNorm(name="attn_norm")(x)
        x = SelfAttention(config=attn_cfg)(x, training=training)
        x = residual + x

        # ── channel mixing (FFN) ──────────────────────────────────────────
        residual = x
        x = nn.LayerNorm(name="ffn_norm")(x)
        x = nn.Dense(self.hidden_size * self.ffn_mult, name="ffn_up")(x)
        x = jax.nn.gelu(x)
        x = nn.Dense(self.hidden_size, name="ffn_down")(x)
        x = residual + x
        return x


# ── Full hybrid layer stack ───────────────────────────────────────────────────

class HybridLayerStack(nn.Module):
    """
    Interleaved Transformer-Mamba layer stack.

    Each layer in the stack is either:
    - SSM layer (Mamba-style O(n) state-space block), or
    - Attention layer (standard O(n²) multi-head self-attention).

    The assignment is determined by `ssm_layers` and `attention_layers`
    index sets. Layers not present in either set pass through unchanged.

    Args:
        config: HybridLayerStackConfig controlling layer count and types.

    Example usage::

        cfg = HybridLayerStackConfig(num_layers=12, hidden_size=768)
        model = HybridLayerStack(config=cfg)
        params = model.init(jax.random.PRNGKey(0), x)
        y = model.apply(params, x)
    """

    config: HybridLayerStackConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        ssm_set = set(self.config.ssm_layers)
        attn_set = set(self.config.attention_layers)

        for i in range(self.config.num_layers):
            if i in ssm_set:
                x = _SSMLayer(
                    hidden_size=self.config.hidden_size,
                    d_state=self.config.d_state,
                    ffn_mult=self.config.ffn_mult,
                    layer_idx=i,
                    name=f"ssm_layer_{i}",
                )(x, training=training)

            elif i in attn_set:
                x = _AttentionLayer(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_heads,
                    dropout_rate=self.config.dropout_rate,
                    ffn_mult=self.config.ffn_mult,
                    layer_idx=i,
                    name=f"attn_layer_{i}",
                )(x, training=training)
            # else: identity (no-op for unassigned indices)

        return x


# ── Factory ───────────────────────────────────────────────────────────────────

def build_hybrid_stack(
    num_layers: int = 12,
    hidden_size: int = 768,
    num_heads: int = 12,
    d_state: int = 64,
    ssm_layers: Optional[List[int]] = None,
    attention_layers: Optional[List[int]] = None,
    dropout_rate: float = 0.1,
    ffn_mult: int = 4,
) -> HybridLayerStack:
    """
    Convenience factory for building a hybrid SSM+Attention stack.

    With default arguments the stack alternates SSM on even-index layers and
    attention on odd-index layers — the canonical hybrid interleaving used in
    models like Jamba.
    """
    cfg = HybridLayerStackConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        d_state=d_state,
        dropout_rate=dropout_rate,
        ffn_mult=ffn_mult,
        ssm_layers=ssm_layers,
        attention_layers=attention_layers,
    )
    return HybridLayerStack(config=cfg)


def main() -> bool:
    logger.info("Module ssm_hybrid_layers.py starting")
    return True


if __name__ == "__main__":
    main()

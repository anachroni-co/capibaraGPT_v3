"""DeepDialog — CPU/JAX compatible version (CapibaraGPT v3)."""

from __future__ import annotations

from typing import Optional, Callable
from dataclasses import dataclass
import logging
import numpy as np

# Optional JAX/Flax
try:
    from flax import linen as nn
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except Exception:
    nn = None  # type: ignore
    jnp = np  # type: ignore
    JAX_AVAILABLE = False

logger = logging.getLogger(__name__)


def _gelu_np(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# =====================================================================
# CONFIG
# =====================================================================

@dataclass
class DeepDialogConfig:
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 8
    dropout_rate: float = 0.1
    activation_fn: Callable = jax.nn.gelu if JAX_AVAILABLE else _gelu_np
    context_dim: Optional[int] = None
    residual_scale: float = 1.0


# =====================================================================
# JAX/Flax MODEL
# =====================================================================

if JAX_AVAILABLE:
    class DeepDialog(nn.Module):
        config: DeepDialogConfig

        def setup(self):
            self.input_proj = nn.Dense(self.config.hidden_size)

            if self.config.context_dim:
                self.context_proj = nn.Dense(self.config.hidden_size)
                self.cross_layers = [
                    CrossAttentionLayer(self.config)
                    for _ in range(self.config.num_layers)
                ]

            self.layers = [
                TransformerLayer(self.config)
                for _ in range(self.config.num_layers)
            ]

            self.norm = nn.LayerNorm()

        def __call__(
            self,
            x: jnp.ndarray,
            context: Optional[jnp.ndarray] = None,
            deterministic: bool = True
        ) -> jnp.ndarray:
            x = self.input_proj(x)

            if context is not None and self.config.context_dim:
                context = self.context_proj(context)
                for layer in self.cross_layers:
                    x = layer(x, context, deterministic)

            for layer in self.layers:
                x = layer(x, deterministic)

            return self.norm(x)


    class CrossAttentionLayer(nn.Module):
        config: DeepDialogConfig

        @nn.compact
        def __call__(self, x, context, deterministic):
            h = nn.LayerNorm()(x)
            c = nn.LayerNorm()(context)

            attn = nn.MultiHeadDotProductAttention(
                num_heads=self.config.num_heads,
                dropout_rate=self.config.dropout_rate,
            )(h, c, deterministic=deterministic)

            x = x + self.config.residual_scale * attn

            y = nn.Dense(self.config.hidden_size * 4)(x)
            y = self.config.activation_fn(y)
            y = nn.Dropout(self.config.dropout_rate)(y, deterministic=deterministic)
            y = nn.Dense(self.config.hidden_size)(y)

            return x + self.config.residual_scale * y


    class TransformerLayer(nn.Module):
        config: DeepDialogConfig

        @nn.compact
        def __call__(self, x, deterministic):
            h = nn.LayerNorm()(x)

            attn = nn.MultiHeadDotProductAttention(
                num_heads=self.config.num_heads,
                dropout_rate=self.config.dropout_rate,
            )(h, h, deterministic=deterministic)

            x = x + self.config.residual_scale * attn

            y = nn.LayerNorm()(x)
            y = nn.Dense(self.config.hidden_size * 4)(y)
            y = self.config.activation_fn(y)
            y = nn.Dropout(self.config.dropout_rate)(y, deterministic=deterministic)
            y = nn.Dense(self.config.hidden_size)(y)

            return x + self.config.residual_scale * y

else:
    # =================================================================
    # CPU fallback implementation (NumPy)
    # =================================================================

    class DeepDialog:
        def __init__(self, config: Optional[DeepDialogConfig] = None):
            self.config = config or DeepDialogConfig()
            self._initialized = False
            self._rng = np.random.default_rng(0)

        def _init_weights(self, input_dim: int, context_dim: Optional[int]) -> None:
            h = self.config.hidden_size
            self.W_in = self._rng.standard_normal((input_dim, h)).astype(np.float32) * 0.02
            self.b_in = np.zeros((h,), dtype=np.float32)

            if context_dim:
                self.W_ctx = self._rng.standard_normal((context_dim, h)).astype(np.float32) * 0.02
                self.b_ctx = np.zeros((h,), dtype=np.float32)
            else:
                self.W_ctx = None
                self.b_ctx = None

            self.layers = []
            for _ in range(self.config.num_layers):
                W1 = self._rng.standard_normal((h, h * 4)).astype(np.float32) * 0.02
                b1 = np.zeros((h * 4,), dtype=np.float32)
                W2 = self._rng.standard_normal((h * 4, h)).astype(np.float32) * 0.02
                b2 = np.zeros((h,), dtype=np.float32)
                self.layers.append((W1, b1, W2, b2))

            self._initialized = True

        def __call__(self, x, context: Optional[np.ndarray] = None, deterministic: bool = True):
            x = np.asarray(x, dtype=np.float32)
            if x.ndim < 2:
                raise ValueError("DeepDialog expects at least 2D input (batch, seq, hidden)")

            if not self._initialized:
                context_dim = context.shape[-1] if context is not None else None
                self._init_weights(x.shape[-1], context_dim)

            # Input projection
            h = x @ self.W_in + self.b_in

            # Context projection (simple additive)
            if context is not None and self.W_ctx is not None:
                ctx = np.asarray(context, dtype=np.float32)
                ctx_proj = ctx @ self.W_ctx + self.b_ctx
                ctx_mean = ctx_proj.mean(axis=-2, keepdims=True)
                h = h + ctx_mean

            # Simple MLP stack
            for W1, b1, W2, b2 in self.layers:
                y = _gelu_np(h @ W1 + b1)
                h = h + (y @ W2 + b2) * self.config.residual_scale

            return h


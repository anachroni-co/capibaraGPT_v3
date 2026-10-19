"""
Distributed Attention Layer for CapibaraGPT v3.

Implements multi-head attention with optional cross-attention for
multi-expert distributed processing in the PASIVE framework.
"""

import logging
import numpy as np

from layers.attention_utils import split_heads, merge_heads

from layers.jax_compat import nn, jnp, JAX_AVAILABLE as FLAX_AVAILABLE

logger = logging.getLogger(__name__)


class DistributedAttention:
    """Distributed attention mechanism for multi-expert processing.

    Supports self-attention and cross-attention (when ``context`` is provided).
    Falls back to a pure-NumPy scaled dot-product path when JAX/Flax is
    unavailable so the layer remains usable on CPU-only environments.
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.hidden_size = self.config.get('hidden_size', 768)
        self.num_heads = self.config.get('num_heads', 12)
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        # Lazily initialised NumPy projection matrices (for non-JAX path)
        self._np_projections = None

        logger.info(
            f"DistributedAttention initialized with {self.num_heads} heads, "
            f"head_dim={self.head_dim}"
        )

    # ------------------------------------------------------------------
    # NumPy fallback helpers
    # ------------------------------------------------------------------

    def _init_np_projections(self, dim_in):
        """Xavier-uniform init for Q/K/V/O projections (NumPy path)."""
        rng = np.random.RandomState(42)
        limit = np.sqrt(6.0 / (dim_in + self.hidden_size))
        def _rand():
            return rng.uniform(-limit, limit, (dim_in, self.hidden_size)).astype(np.float32)
        self._np_projections = {
            'q': _rand(), 'k': _rand(), 'v': _rand(), 'o': _rand()
        }

    def _np_attention(self, inputs, context):
        """Pure NumPy scaled dot-product multi-head attention."""
        x = np.asarray(inputs, dtype=np.float32)
        c = np.asarray(context, dtype=np.float32) if context is not None else x

        if self._np_projections is None:
            self._init_np_projections(x.shape[-1])

        # Project Q, K, V
        Q = x @ self._np_projections['q']
        K = c @ self._np_projections['k']
        V = c @ self._np_projections['v']

        Q = split_heads(Q, self.num_heads)
        K = split_heads(K, self.num_heads)
        V = split_heads(V, self.num_heads)

        # Scaled dot-product attention
        scores = np.matmul(Q, np.swapaxes(K, -1, -2)) * self.scale
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-8)

        attn_out = np.matmul(attn_weights, V)
        attn_out = merge_heads(attn_out, self.hidden_size)

        # Output projection
        output = attn_out @ self._np_projections['o']
        return output

    # ------------------------------------------------------------------
    # JAX / Flax path
    # ------------------------------------------------------------------

    def _jax_attention(self, inputs, context, training):
        """JAX multi-head attention via manual projections."""
        x = jnp.asarray(inputs)
        c = jnp.asarray(context) if context is not None else x

        dim_in = x.shape[-1]

        # Simple deterministic projections (no trainable params outside Flax)
        rng = np.random.RandomState(42)
        limit = float(np.sqrt(6.0 / (dim_in + self.hidden_size)))
        def _param(shape):
            return jnp.array(rng.uniform(-limit, limit, shape).astype(np.float32))

        if self._np_projections is None:
            self._np_projections = {
                'q': _param((dim_in, self.hidden_size)),
                'k': _param((dim_in, self.hidden_size)),
                'v': _param((dim_in, self.hidden_size)),
                'o': _param((self.hidden_size, self.hidden_size)),
            }

        Q = jnp.dot(x, self._np_projections['q'])
        K = jnp.dot(c, self._np_projections['k'])
        V = jnp.dot(c, self._np_projections['v'])

        Q = split_heads(Q, self.num_heads)
        K = split_heads(K, self.num_heads)
        V = split_heads(V, self.num_heads)

        scores = jnp.matmul(Q, jnp.swapaxes(K, -1, -2)) * self.scale
        attn_weights = nn.softmax(scores, axis=-1)
        attn_out = jnp.matmul(attn_weights, V)

        attn_out = merge_heads(attn_out, self.hidden_size)

        output = jnp.dot(attn_out, self._np_projections['o'])
        return output

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, inputs, context=None, training=False):
        """Apply distributed multi-head attention.

        Args:
            inputs: Query tensor of shape (..., seq_len, hidden_size).
            context: Optional key/value tensor for cross-attention.
                     If None, self-attention is performed.
            training: Whether we are in training mode.

        Returns:
            Output tensor of same shape as ``inputs``.
        """
        if FLAX_AVAILABLE:
            return self._jax_attention(inputs, context, training)

        return self._np_attention(inputs, context)


def main():
    logger.info("DistributedAttention module loaded successfully")
    return True


if __name__ == "__main__":
    main()

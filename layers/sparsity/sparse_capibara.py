"""
Sparse Capibara layer implementation.

This module provides sparse attention and computation layers for the Capibara model.
Optimized with JIT compilation and cached masks for performance.
"""

import functools
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from layers.jax_compat import jax, jnp, nn, JAX_AVAILABLE


# JIT-compiled attention computation (extracted for optimization)
if JAX_AVAILABLE:
    @functools.partial(jax.jit, static_argnums=(4, 5))
    def _sparse_attention_kernel(
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        mask: Optional[jnp.ndarray],
        head_dim: int,
        apply_sparsity: bool,
        sparsity_k: int = 1
    ) -> jnp.ndarray:
        """
        JIT-compiled sparse attention kernel.

        Extracted from SparseCapibara.__call__ for optimal compilation.
        """
        # Compute attention scores
        scale = 1.0 / jnp.sqrt(head_dim)
        scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale

        # Apply mask if provided
        if mask is not None:
            scores = jnp.where(mask, scores, -jnp.inf)

        # Apply sparsity pattern (top-k attention) - only during training
        if apply_sparsity and sparsity_k > 0:
            k_sparse = max(1, sparsity_k)
            # Get top-k values and indices
            top_k_values, _ = jax.lax.top_k(scores, k_sparse)
            # Use the k-th largest value as threshold (vectorized)
            threshold = top_k_values[..., -1:]  # [..., 1] - minimum of top-k
            # Create mask directly from threshold comparison (no index arrays needed)
            sparse_mask = scores >= threshold
            scores = jnp.where(sparse_mask, scores, -jnp.inf)

        # Compute attention weights
        attn_weights = jax.nn.softmax(scores, axis=-1)

        # Apply attention to values
        return jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)


if JAX_AVAILABLE:
    class SparseCapibara(nn.Module):
        """
        Sparse Capibara layer with optimized attention and computation.

        This layer implements sparse attention mechanisms and efficient computation
        patterns for the Capibara model architecture.

        Uses @nn.compact for better JIT compilation and extracted attention
        kernel for optimal performance.
        """

        features: int = 512
        num_heads: int = 8
        sparsity_ratio: float = 0.9
        dtype: Any = jnp.float32

        @nn.compact
        def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, training: bool = False) -> jnp.ndarray:
            """
            Apply sparse capibara layer computation.

            Args:
                x: Input tensor of shape (batch_size, seq_len, features)
                mask: Optional attention mask
                training: Whether in training mode

            Returns:
                Output tensor of same shape as input
            """
            batch_size, seq_len, features = x.shape
            head_dim = self.features // self.num_heads

            # Apply layer normalization
            x_norm = nn.LayerNorm(dtype=self.dtype)(x)

            # Compute Q, K, V with fused Dense layer (3x features, then split)
            # This is more efficient than 3 separate Dense calls
            qkv = nn.Dense(3 * self.features, dtype=self.dtype,
                          kernel_init=nn.initializers.xavier_uniform(),
                          name='qkv_proj')(x_norm)
            q, k, v = jnp.split(qkv, 3, axis=-1)

            # Reshape for multi-head attention
            q = q.reshape(batch_size, seq_len, self.num_heads, head_dim)
            k = k.reshape(batch_size, seq_len, self.num_heads, head_dim)
            v = v.reshape(batch_size, seq_len, self.num_heads, head_dim)

            # Transpose for attention computation: (batch, heads, seq_len, head_dim)
            q = jnp.transpose(q, (0, 2, 1, 3))
            k = jnp.transpose(k, (0, 2, 1, 3))
            v = jnp.transpose(v, (0, 2, 1, 3))

            # Prepare mask for attention kernel
            attn_mask = None
            if mask is not None:
                attn_mask = mask.reshape(batch_size, 1, 1, seq_len)

            # Compute sparsity parameters
            apply_sparsity = training and self.sparsity_ratio > 0
            sparsity_k = max(1, int(seq_len * (1 - self.sparsity_ratio))) if apply_sparsity else 0

            # Call JIT-compiled attention kernel
            out = _sparse_attention_kernel(
                q, k, v, attn_mask, head_dim, apply_sparsity, sparsity_k
            )

            # Transpose back and reshape
            out = jnp.transpose(out, (0, 2, 1, 3))  # (batch, seq_len, heads, head_dim)
            out = out.reshape(batch_size, seq_len, features)

            # Apply output projection
            out = nn.Dense(self.features, dtype=self.dtype,
                          kernel_init=nn.initializers.xavier_uniform())(out)

            # Residual connection
            return x + out
else:
    # Fallback implementation when JAX is not available
    class SparseCapibara:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("SparseCapibara requires JAX/Flax to be installed")

def main():
    """Main function for this module."""
    logger.info("SparseCapibara module initialized")
    return True

if __name__ == "__main__":
    main()

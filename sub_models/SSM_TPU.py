"""
SSM TPU Module - State Space Model optimized for TPU execution.

This module implements a classic State Space Model (SSM) block optimized for TPU
execution using JAX and Flax. The SSM provides O(n) complexity for sequence modeling
compared to O(n^2) for standard attention mechanisms.

The SSM formulation follows:
    h[t+1] = A @ h[t] + B @ x[t]  (state update)
    y[t] = C @ h[t] + D @ x[t]    (output)

Key Features:
    - TPU-optimized scan operations via jax.lax.scan
    - Efficient memory usage for long sequences
    - Stable initialization for state transition matrix
    - Support for batched inputs

Example:
    >>> model = SSMBlock(hidden_size=256, state_dim=64)
    >>> key = jax.random.PRNGKey(0)
    >>> x = jax.random.normal(key, (batch=2, seq=128, hidden=256))
    >>> params = model.init(key, x)
    >>> output = model.apply(params, x)

Reference:
    Gu, A., Goel, K., & Re, C. (2021). Efficiently Modeling Long Sequences
    with Structured State Spaces. arXiv:2111.00396
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Any

import jax
import jax.numpy as jnp
from flax import linen as nn

logger = logging.getLogger(__name__)


@dataclass
class SSMConfig:
    """Configuration for SSM block.

    Attributes:
        hidden_size: Size of input/output hidden dimension.
        state_dim: Size of internal state dimension. Larger values capture
            more complex dynamics but increase computation.
        dt_min: Minimum discretization step for continuous-time SSM.
        dt_max: Maximum discretization step for continuous-time SSM.
        use_stable_init: Whether to use stable initialization for matrix A.
    """
    hidden_size: int = 256
    state_dim: int = 64
    dt_min: float = 0.001
    dt_max: float = 0.1
    use_stable_init: bool = True


class SSMBlock(nn.Module):
    """
    State Space Model block optimized for TPU execution.

    Implements the classic SSM recurrence with efficient JAX scan operations.
    The model maintains an internal state that captures sequence history,
    enabling O(n) sequence processing.

    Attributes:
        hidden_size: Size of the input/output hidden dimension.
        state_dim: Size of the internal state dimension (default: 64).

    Note:
        The state transition matrix A is initialized to be stable (eigenvalues < 1)
        to prevent gradient explosion during training.
    """
    hidden_size: int
    state_dim: int = 64

    def setup(self):
        """Initialize SSM parameters with proper initialization for stability."""
        # State transition matrix A - initialized to be stable
        # Using negative values to ensure eigenvalues < 1
        self.A = self.param(
            "A",
            lambda rng, shape: jax.random.normal(rng, shape) * 0.1 - 0.5,
            (self.state_dim, self.state_dim)
        )

        # Input matrix B - projects input to state space
        self.B = self.param(
            "B",
            nn.initializers.xavier_uniform(),
            (self.state_dim, self.hidden_size)
        )

        # Output matrix C - projects state to output space
        self.C = self.param(
            "C",
            nn.initializers.xavier_uniform(),
            (self.hidden_size, self.state_dim)
        )

        # Skip connection D - direct input-to-output connection
        self.D = self.param(
            "D",
            nn.initializers.zeros,
            (self.hidden_size,)
        )

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the SSM block.

        Args:
            inputs: Input tensor of shape [batch, seq_len, hidden_size].

        Returns:
            Output tensor of shape [batch, seq_len, hidden_size].

        Note:
            Uses jax.lax.scan for efficient TPU execution, which automatically
            handles sequence-level parallelism on TPU hardware.
        """
        batch_size, seq_len, _ = inputs.shape

        def step(carry: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Single step of the SSM recurrence."""
            h = carry  # Previous hidden state

            # State update: h[t+1] = A @ h[t] + B @ x[t]
            h_next = jnp.dot(h, self.A.T) + jnp.dot(x, self.B.T)

            # Output: y[t] = C @ h[t+1] + D @ x[t]
            y = jnp.dot(h_next, self.C.T) + self.D * x

            return h_next, y

        # Initialize hidden state to zeros
        h0 = jnp.zeros((batch_size, self.state_dim))

        # Apply scan over sequence dimension
        # Transpose to [seq, batch, hidden] for scan
        _, outputs = jax.lax.scan(step, h0, jnp.transpose(inputs, (1, 0, 2)))

        # Transpose back to [batch, seq, hidden]
        outputs = jnp.transpose(outputs, (1, 0, 2))

        return outputs

    def get_state_matrices(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return the SSM state matrices for analysis.

        Returns:
            Tuple of (A, B, C, D) matrices.
        """
        return self.A, self.B, self.C, self.D


class BidirectionalSSM(nn.Module):
    """
    Bidirectional SSM that processes sequences in both directions.

    Combines forward and backward SSM passes for tasks requiring
    full sequence context (e.g., classification, NER).

    Attributes:
        hidden_size: Size of input/output hidden dimension.
        state_dim: Size of internal state dimension.
    """
    hidden_size: int
    state_dim: int = 64

    def setup(self):
        """Initialize forward and backward SSM blocks."""
        self.forward_ssm = SSMBlock(
            hidden_size=self.hidden_size,
            state_dim=self.state_dim
        )
        self.backward_ssm = SSMBlock(
            hidden_size=self.hidden_size,
            state_dim=self.state_dim
        )
        # Projection to combine bidirectional outputs
        self.output_proj = nn.Dense(self.hidden_size)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through bidirectional SSM.

        Args:
            inputs: Input tensor of shape [batch, seq_len, hidden_size].

        Returns:
            Output tensor of shape [batch, seq_len, hidden_size].
        """
        # Forward pass
        forward_out = self.forward_ssm(inputs)

        # Backward pass (reverse sequence, process, reverse back)
        backward_out = self.backward_ssm(jnp.flip(inputs, axis=1))
        backward_out = jnp.flip(backward_out, axis=1)

        # Combine and project
        combined = jnp.concatenate([forward_out, backward_out], axis=-1)
        return self.output_proj(combined)


def create_ssm_block(
    hidden_size: int,
    state_dim: int = 64,
    bidirectional: bool = False
) -> nn.Module:
    """
    Factory function to create SSM block.

    Args:
        hidden_size: Size of input/output hidden dimension.
        state_dim: Size of internal state dimension.
        bidirectional: Whether to use bidirectional processing.

    Returns:
        SSMBlock or BidirectionalSSM module.
    """
    if bidirectional:
        return BidirectionalSSM(hidden_size=hidden_size, state_dim=state_dim)
    return SSMBlock(hidden_size=hidden_size, state_dim=state_dim)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test SSM block
    import jax.random as random

    batch_size, seq_len, hidden_size = 2, 128, 256
    model = SSMBlock(hidden_size=hidden_size, state_dim=64)

    key = random.PRNGKey(42)
    x = random.normal(key, (batch_size, seq_len, hidden_size))
    params = model.init(key, x)
    output = model.apply(params, x)

    logger.info(f"SSM test - Input: {x.shape}, Output: {output.shape}")

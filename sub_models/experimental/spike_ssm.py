"""
Spiking SSM Module - Bio-inspired State Space Model with surrogate gradients.

This module implements a bio-inspired spiking neural network version of SSM
that uses surrogate gradients for backpropagation through spike discontinuities.
The model provides sparsity and energy efficiency benefits similar to biological
neural networks.

Key Features:
    - Surrogate gradient learning for spike discontinuities
    - Leaky integrate-and-fire neuron dynamics
    - Adaptive thresholds based on recent activity
    - Multiple decay timescales for rich temporal dynamics
    - Lateral inhibition for competitive learning

Applications:
    - Energy-efficient inference on neuromorphic hardware
    - Sparse activations for reduced computation
    - Temporal pattern recognition

Reference:
    Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning
    in spiking neural networks. IEEE Signal Processing Magazine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

logger = logging.getLogger(__name__)


class SpikeSSM(nn.Module):
    """
    Bio-inspired Spiking State Space Model with surrogate gradient learning.

    This model combines the memory efficiency of SSMs with the sparsity benefits
    of spiking neural networks. It uses a surrogate gradient approach to handle
    the non-differentiable spike function during backpropagation.

    Attributes:
        hidden_size: Size of the input/output hidden dimension.
        state_dim: Size of the internal state dimension (default: 64).
        threshold: Spike threshold for neuron activation (default: 0.5).
        surrogate_scale: Scale factor for surrogate gradient (default: 10.0).

    Note:
        The surrogate gradient uses a sigmoid derivative approximation,
        which provides smooth gradients through the hard threshold function.
    """
    hidden_size: int
    state_dim: int = 64
    threshold: float = 0.5
    surrogate_scale: float = 10.0

    def setup(self):
        """Initialize SpikeSSM parameters."""
        # Input transformation
        self.linear_in = nn.Dense(
            self.state_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros
        )

        # Output transformation
        self.linear_out = nn.Dense(
            self.hidden_size,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros
        )

        # Learnable decay factor for membrane potential
        self.decay = self.param(
            "decay",
            lambda rng, shape: jnp.ones(shape) * 0.9,
            (self.state_dim,)
        )

    def surrogate_spike(self, membrane_potential: jnp.ndarray) -> jnp.ndarray:
        """
        Surrogate gradient spike function.

        Forward pass uses hard threshold (spike or no spike).
        Backward pass uses smooth sigmoid derivative for gradient flow.

        Args:
            membrane_potential: Membrane potential values.

        Returns:
            Binary spike outputs with surrogate gradients attached.
        """
        # Forward pass: hard threshold
        spikes = jnp.where(membrane_potential > self.threshold, 1.0, 0.0)

        # Surrogate gradient: sigmoid derivative approximation
        sigmoid_val = jax.nn.sigmoid(
            self.surrogate_scale * (membrane_potential - self.threshold)
        )
        surrogate_grad = self.surrogate_scale * sigmoid_val * (1 - sigmoid_val)

        # Straight-through estimator with surrogate gradient
        return spikes + jax.lax.stop_gradient(spikes - surrogate_grad) + surrogate_grad - jax.lax.stop_gradient(surrogate_grad)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the SpikeSSM.

        Args:
            inputs: Input tensor of shape [batch, seq_len, hidden_size].

        Returns:
            Output tensor of shape [batch, seq_len, hidden_size].
        """
        batch_size, seq_len, _ = inputs.shape

        # Project the whole sequence OUTSIDE the scan: calling Flax
        # submodules inside a raw jax.lax.scan body leaks tracers. The
        # per-step recurrence below is pure elementwise math.
        driven = self.linear_in(inputs)  # [batch, seq_len, state_dim]
        decay = self.decay
        threshold = self.threshold

        def step(carry: jnp.ndarray, u_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Single step of the spiking SSM (parameter-free math only)."""
            membrane_potential = carry * decay + u_t

            # Generate spikes using surrogate gradient
            spikes = self.surrogate_spike(membrane_potential)

            # Reset membrane potential where spikes occurred
            membrane_potential = membrane_potential - spikes * threshold

            return membrane_potential, spikes

        # Initialize membrane potential
        membrane_0 = jnp.zeros((batch_size, self.state_dim))

        # Apply scan over sequence dimension (spikes only)
        _, spikes_seq = jax.lax.scan(step, membrane_0, jnp.transpose(driven, (1, 0, 2)))

        # Transpose back to [batch, seq, state_dim]
        spikes_seq = jnp.transpose(spikes_seq, (1, 0, 2))

        # Output projection outside the scan (same math, no tracer leak)
        return self.linear_out(spikes_seq)


class AdaptiveSpikeSSM(nn.Module):
    """
    Advanced SpikeSSM with adaptive thresholds and multiple timescales.

    This variant includes several biologically-inspired mechanisms:
    - Adaptive spike thresholds that increase with activity
    - Multiple decay timescales for multi-scale temporal integration
    - Lateral inhibition for competition between neurons

    Attributes:
        hidden_size: Size of input/output hidden dimension.
        state_dim: Size of internal state dimension.
        base_threshold: Base spike threshold (default: 0.5).
        surrogate_scale: Scale for surrogate gradient (default: 10.0).
        num_timescales: Number of different decay timescales (default: 3).

    Note:
        Multiple timescales allow the model to capture both fast and slow
        temporal dynamics simultaneously, similar to biological neurons.
    """
    hidden_size: int
    state_dim: int = 64
    base_threshold: float = 0.5
    surrogate_scale: float = 10.0
    num_timescales: int = 3

    def setup(self):
        """Initialize adaptive SpikeSSM parameters."""
        self.linear_in = nn.Dense(self.state_dim)
        self.linear_out = nn.Dense(self.hidden_size)

        # Adaptive threshold parameters
        self.threshold_adaptation = nn.Dense(self.state_dim, use_bias=False)

        # Multiple decay timescales (from fast to slow)
        self.decays = self.param(
            "decays",
            lambda rng, shape: jnp.linspace(0.7, 0.95, self.num_timescales).reshape(-1, 1),
            (self.num_timescales, 1)
        )

        # Lateral inhibition weights
        self.lateral_weights = self.param(
            "lateral_weights",
            lambda rng, shape: jax.random.normal(rng, shape) * 0.1,
            (self.state_dim, self.state_dim)
        )

    def adaptive_threshold(
        self,
        membrane_potential: jnp.ndarray,
        spike_history: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute adaptive thresholds based on recent activity.

        Higher recent activity leads to higher thresholds, implementing
        spike-frequency adaptation.

        Args:
            membrane_potential: Current membrane potential.
            spike_history: Exponential moving average of recent spikes.

        Returns:
            Adapted threshold values for each neuron.
        """
        adaptation = self.threshold_adaptation(spike_history)
        return self.base_threshold + 0.1 * jax.nn.tanh(adaptation)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass with adaptive mechanisms.

        Args:
            inputs: Input tensor of shape [batch, seq_len, hidden_size].

        Returns:
            Output tensor of shape [batch, seq_len, hidden_size].
        """
        batch_size, seq_len, _ = inputs.shape

        def step(carry, x):
            membrane_potentials, spike_history = carry

            # Multi-timescale integration
            new_potentials = []
            for i, decay in enumerate(self.decays):
                mp = membrane_potentials[i] * decay + self.linear_in(x) / self.num_timescales
                new_potentials.append(mp)

            # Combine multi-timescale potentials
            combined_potential = jnp.mean(jnp.stack(new_potentials), axis=0)

            # Apply lateral inhibition
            inhibited_potential = combined_potential - 0.1 * jnp.dot(
                spike_history, self.lateral_weights
            )

            # Adaptive threshold
            threshold = self.adaptive_threshold(inhibited_potential, spike_history)

            # Generate spikes
            spikes = jnp.where(inhibited_potential > threshold, 1.0, 0.0)

            # Reset potentials where spikes occurred
            reset_potentials = []
            for mp in new_potentials:
                reset_potentials.append(mp - spikes * threshold)

            # Update spike history (exponential moving average)
            new_spike_history = 0.9 * spike_history + 0.1 * spikes

            # Output
            y = self.linear_out(spikes)

            return (jnp.stack(reset_potentials), new_spike_history), y

        # Initialize states
        membrane_0 = jnp.zeros((self.num_timescales, batch_size, self.state_dim))
        spike_history_0 = jnp.zeros((batch_size, self.state_dim))

        _, outputs = jax.lax.scan(
            step,
            (membrane_0, spike_history_0),
            jnp.transpose(inputs, (1, 0, 2))
        )

        return jnp.transpose(outputs, (1, 0, 2))


def create_spike_ssm(
    hidden_size: int,
    state_dim: int = 64,
    adaptive: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create Spiking SSM.

    Args:
        hidden_size: Size of input/output hidden dimension.
        state_dim: Size of internal state dimension.
        adaptive: Whether to use adaptive variant with multiple timescales.
        **kwargs: Additional arguments passed to the model.

    Returns:
        SpikeSSM or AdaptiveSpikeSSM module.
    """
    if adaptive:
        return AdaptiveSpikeSSM(
            hidden_size=hidden_size,
            state_dim=state_dim,
            **kwargs
        )
    return SpikeSSM(
        hidden_size=hidden_size,
        state_dim=state_dim,
        **kwargs
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import jax.random as random

    batch_size, seq_len, hidden_size = 2, 64, 128

    # Test basic SpikeSSM
    model = SpikeSSM(hidden_size=hidden_size, state_dim=64)
    key = random.PRNGKey(42)
    x = random.normal(key, (batch_size, seq_len, hidden_size))

    params = model.init(key, x)
    output = model.apply(params, x)
    logger.info(f"SpikeSSM - Input: {x.shape}, Output: {output.shape}")

    # Test adaptive version
    adaptive_model = AdaptiveSpikeSSM(hidden_size=hidden_size, state_dim=64)
    adaptive_params = adaptive_model.init(key, x)
    adaptive_output = adaptive_model.apply(adaptive_params, x)
    logger.info(f"AdaptiveSpikeSSM - Output: {adaptive_output.shape}")

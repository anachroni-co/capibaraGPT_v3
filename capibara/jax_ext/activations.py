"""
Activation functions for JAX-based CapibaraGPT.
"""

import numpy as np
from typing import Any, Callable, Optional

# Try to import JAX, fallback to numpy
try:
    import jax.numpy as jnp
    from jax import jit as jax_jit
    HAS_JAX = True
    jit = jax_jit
except ImportError:
    import numpy as jnp
    HAS_JAX = False

    def jit(fn):
        """Mock jit decorator."""
        return fn

# Activation functions
def gelu(x: Any) -> Any:
    """Gaussian Error Linear Unit activation."""
    if HAS_JAX:
        return x * 0.5 * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))
    else:
        return x * 0.5 * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))

def relu(x: Any) -> Any:
    """Rectified Linear Unit activation."""
    return jnp.maximum(0, x)

def leaky_relu(x: Any, alpha: float = 0.01) -> Any:
    """Leaky ReLU activation."""
    return jnp.where(x >= 0, x, alpha * x)

def swish(x: Any) -> Any:
    """Swish activation (SiLU)."""
    return x * sigmoid(x)

def sigmoid(x: Any) -> Any:
    """Sigmoid activation."""
    return 1.0 / (1.0 + jnp.exp(-x))

def tanh(x: Any) -> Any:
    """Hyperbolic tangent activation."""
    return jnp.tanh(x)

def softmax(x: Any, axis: int = -1) -> Any:
    """Softmax activation."""
    x_max = jnp.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    exp_x = jnp.exp(x_shifted)
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)

def log_softmax(x: Any, axis: int = -1) -> Any:
    """Log softmax activation."""
    x_max = jnp.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    return x_shifted - jnp.log(jnp.sum(jnp.exp(x_shifted), axis=axis, keepdims=True))

def glu(x: Any, axis: int = -1) -> Any:
    """Gated Linear Unit activation."""
    a, b = jnp.split(x, 2, axis=axis)
    return a * sigmoid(b)

def mish(x: Any) -> Any:
    """Mish activation."""
    return x * tanh(softplus(x))

def softplus(x: Any) -> Any:
    """Softplus activation."""
    return jnp.log(1.0 + jnp.exp(x))

def elu(x: Any, alpha: float = 1.0) -> Any:
    """Exponential Linear Unit activation."""
    return jnp.where(x >= 0, x, alpha * (jnp.exp(x) - 1))

def selu(x: Any) -> Any:
    """Scaled Exponential Linear Unit activation."""
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * elu(x, alpha)

def prelu(x: Any, alpha: Any) -> Any:
    """Parametric ReLU activation."""
    return jnp.where(x >= 0, x, alpha * x)

# Advanced activations
def gated_gelu(x: Any) -> Any:
    """Gated GELU activation."""
    a, b = jnp.split(x, 2, axis=-1)
    return gelu(a) * sigmoid(b)

def squared_relu(x: Any) -> Any:
    """Squared ReLU activation."""
    return jnp.square(relu(x))

def hard_sigmoid(x: Any) -> Any:
    """Hard sigmoid activation."""
    return jnp.clip((x + 3) / 6, 0, 1)

def hard_swish(x: Any) -> Any:
    """Hard swish activation."""
    return x * hard_sigmoid(x)

# Ultra activations for CapibaraGPT
def ultra_gelu(x: Any, alpha: float = 1.702) -> Any:
    """Ultra GELU with learnable parameter."""
    return x * 0.5 * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + alpha * x**3)))

def adaptive_activation(x: Any, weights: Any) -> Any:
    """Adaptive activation that combines multiple activations."""
    activations = jnp.stack([
        relu(x),
        gelu(x),
        swish(x),
        tanh(x)
    ], axis=-1)

    # Apply weights
    weighted = activations * weights
    return jnp.sum(weighted, axis=-1)

def neuromorphic_activation(x: Any, threshold: float = 0.5) -> Any:
    """Neuromorphic spiking activation."""
    spikes = jnp.where(x > threshold, 1.0, 0.0)
    return spikes * (x - threshold)

# Activation registry
ACTIVATIONS = {
    'relu': relu,
    'gelu': gelu,
    'leaky_relu': leaky_relu,
    'swish': swish,
    'silu': swish,  # Alias
    'sigmoid': sigmoid,
    'tanh': tanh,
    'softmax': softmax,
    'log_softmax': log_softmax,
    'glu': glu,
    'mish': mish,
    'softplus': softplus,
    'elu': elu,
    'selu': selu,
    'prelu': prelu,
    'gated_gelu': gated_gelu,
    'squared_relu': squared_relu,
    'hard_sigmoid': hard_sigmoid,
    'hard_swish': hard_swish,
    'ultra_gelu': ultra_gelu,
    'adaptive': adaptive_activation,
    'neuromorphic': neuromorphic_activation
}

def get_activation(name: str) -> Callable:
    """Get activation function by name."""
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation: {name}. Available: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]

def apply_activation(x: Any, activation: str, **kwargs) -> Any:
    """Apply activation function by name."""
    fn = get_activation(activation)
    return fn(x, **kwargs)

# JIT compiled versions
if HAS_JAX:
    relu = jit(relu)
    gelu = jit(gelu)
    swish = jit(swish)
    sigmoid = jit(sigmoid)
    tanh = jit(tanh)
    softmax = jit(softmax)
    log_softmax = jit(log_softmax)

__all__ = [
    'relu', 'gelu', 'leaky_relu', 'swish', 'sigmoid', 'tanh',
    'softmax', 'log_softmax', 'glu', 'mish', 'softplus', 'elu',
    'selu', 'prelu', 'gated_gelu', 'squared_relu', 'hard_sigmoid',
    'hard_swish', 'ultra_gelu', 'adaptive_activation',
    'neuromorphic_activation', 'get_activation', 'apply_activation',
    'ACTIVATIONS'
]

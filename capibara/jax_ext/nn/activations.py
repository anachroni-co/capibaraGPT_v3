"""
Advanced Activation Functions for CapibaraGPT
============================================

State-of-the-art activation functions including modern variants
used in the latest transformer architectures.
"""

try:
    import jax.numpy as jnp
    from jax import lax
    
    def softmax(x, axis=-1):
        """Softmax activation function with numerical stability."""
        x_max = lax.stop_gradient(jnp.max(x, axis=axis, keepdims=True))
        exp_x = jnp.exp(x - x_max)
        return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)
    
    def sigmoid(x):
        """Sigmoid activation function."""
        return 1.0 / (1.0 + jnp.exp(-jnp.clip(x, -500, 500)))
    
    def relu(x):
        """ReLU activation function."""
        return jnp.maximum(0, x)
    
    def gelu(x):
        """GELU activation function (Gaussian error Linear Units)."""
        return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3))))
    
    def swish(x):
        """Swish activation function (x * sigmoid(x))."""
        return x * sigmoid(x)
    
    def silu(x):
        """SiLU activation function (same as Swish)."""
        return swish(x)
    
    def tanh(x):
        """Tanh activation function."""
        return jnp.tanh(x)
    
    def log_softmax(x, axis=-1):
        """Log-softmax function with numerical stability."""
        x_max = lax.stop_gradient(jnp.max(x, axis=axis, keepdims=True))
        shifted = x - x_max
        return shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=axis, keepdims=True))
    
    # NEW ADVANCED ACTIVATIONS
    
    def glu(x, axis=-1):
        """Gated Linear Unit."""
        a, b = jnp.split(x, 2, axis=axis)
        return a * sigmoid(b)
    
    def swiglu(x, axis=-1):
        """SwiGLU activation (used in PaLM, LLaMA)."""
        a, b = jnp.split(x, 2, axis=axis)
        return swish(a) * b
    
    def geglu(x, axis=-1):
        """GeGLU activation (GELU variant)."""
        a, b = jnp.split(x, 2, axis=axis)
        return gelu(a) * b
    
    def reglu(x, axis=-1):
        """ReGLU activation (ReLU variant)."""
        a, b = jnp.split(x, 2, axis=axis)
        return relu(a) * b
    
    def mish(x):
        """Mish activation function."""
        return x * tanh(jnp.log(1 + jnp.exp(x)))
    
    def leaky_relu(x, negative_slope=0.01):
        """Leaky ReLU activation."""
        return jnp.where(x >= 0, x, negative_slope * x)
    
    def elu(x, alpha=1.0):
        """Exponential Linear Unit."""
        return jnp.where(x >= 0, x, alpha * (jnp.exp(x) - 1))
    
    def selu(x):
        """Scaled Exponential Linear Unit."""
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * elu(x, alpha)
    
    def hard_sigmoid(x):
        """Hard sigmoid activation (computationally efficient)."""
        return jnp.clip((x + 1) / 2, 0, 1)
    
    def hard_swish(x):
        """Hard swish activation (MobileNet-V3)."""
        return x * hard_sigmoid(x)
    
    def hard_tanh(x):
        """Hard tanh activation."""
        return jnp.clip(x, -1, 1)

except ImportError:
    # Fallback using numpy
    import numpy as np
    
    def softmax(x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def relu(x):
        return np.maximum(0, x)
    
    def gelu(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    
    def swish(x):
        return x * sigmoid(x)
    
    def silu(x):
        return swish(x)
    
    def tanh(x):
        return np.tanh(x)
    
    def log_softmax(x, axis=-1):
        return x - np.log(np.sum(np.exp(x - np.max(x, axis=axis, keepdims=True)), axis=axis, keepdims=True))
    
    def glu(x, axis=-1):
        a, b = np.split(x, 2, axis=axis)
        return a * sigmoid(b)
    
    def swiglu(x, axis=-1):
        a, b = np.split(x, 2, axis=axis)
        return swish(a) * b
    
    def geglu(x, axis=-1):
        a, b = np.split(x, 2, axis=axis)
        return gelu(a) * b
    
    def reglu(x, axis=-1):
        a, b = np.split(x, 2, axis=axis)
        return relu(a) * b
    
    def mish(x):
        return x * tanh(np.log(1 + np.exp(np.clip(x, -20, 20))))
    
    def leaky_relu(x, negative_slope=0.01):
        return np.where(x >= 0, x, negative_slope * x)
    
    def elu(x, alpha=1.0):
        return np.where(x >= 0, x, alpha * (np.exp(np.clip(x, -500, 500)) - 1))
    
    def selu(x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * elu(x, alpha)
    
    def hard_sigmoid(x):
        return np.clip((x + 1) / 2, 0, 1)
    
    def hard_swish(x):
        return x * hard_sigmoid(x)
    
    def hard_tanh(x):
        return np.clip(x, -1, 1)

# Export all activation functions
__all__ = [
    'softmax', 'log_softmax', 'sigmoid', 'relu', 'gelu', 'swish', 'silu', 'tanh',
    'glu', 'swiglu', 'geglu', 'reglu', 'mish', 'leaky_relu', 'elu', 'selu',
    'hard_sigmoid', 'hard_swish', 'hard_tanh'
]
"""
Advanced Weight Initializers for Neural Networks
===============================================

State-of-the-art weight initialization schemes for optimal training.
"""

import math

# Pre-computed constants (avoid recomputing on every call)
_SQRT_2 = math.sqrt(2.0)
_SQRT_3 = math.sqrt(3.0)
_SQRT_6 = math.sqrt(6.0)

try:
    import jax.numpy as jnp
    from jax import random
    
    def xavier_uniform(key, shape, gain=1.0):
        """Xavier/Glorot uniform initialization."""
        fan_in, fan_out = _calculate_fan_in_fan_out(shape)
        bound = gain * _SQRT_6 / math.sqrt(fan_in + fan_out)
        return random.uniform(key, shape, minval=-bound, maxval=bound)

    def xavier_normal(key, shape, gain=1.0):
        """Xavier/Glorot normal initialization."""
        fan_in, fan_out = _calculate_fan_in_fan_out(shape)
        std = gain * _SQRT_2 / math.sqrt(fan_in + fan_out)
        return random.normal(key, shape) * std

    def kaiming_uniform(key, shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        """Kaiming/He uniform initialization."""
        fan = _calculate_correct_fan(shape, mode)
        gain = _calculate_gain(nonlinearity, a)
        bound = gain * _SQRT_3 / math.sqrt(fan)
        return random.uniform(key, shape, minval=-bound, maxval=bound)

    def kaiming_normal(key, shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        """Kaiming/He normal initialization."""
        fan = _calculate_correct_fan(shape, mode)
        gain = _calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        return random.normal(key, shape) * std
    
    def lecun_uniform(key, shape):
        """LeCun uniform initialization."""
        fan_in = _calculate_fan_in_fan_out(shape)[0]
        bound = _SQRT_3 / math.sqrt(fan_in)
        return random.uniform(key, shape, minval=-bound, maxval=bound)

    def lecun_normal(key, shape):
        """LeCun normal initialization."""
        fan_in = _calculate_fan_in_fan_out(shape)[0]
        std = 1.0 / math.sqrt(fan_in)
        return random.normal(key, shape) * std
    
    def truncated_normal(key, shape, stddev=1.0, lower=-2.0, upper=2.0):
        """Truncated normal initialization."""
        # simple approximation using clipping
        values = random.normal(key, shape) * stddev
        return jnp.clip(values, lower * stddev, upper * stddev)
    
    def orthogonal(key, shape, gain=1.0):
        """Orthogonal initialization."""
        if len(shape) < 2:
            raise ValueError("Orthogonal initialization requires at least 2D tensor")
        
        rows, cols = shape[0], jnp.prod(jnp.array(shape[1:]))
        flattened_shape = (rows, cols) if rows >= cols else (cols, rows)
        
        # Generate random matrix and perform QR decomposition
        a = random.normal(key, flattened_shape)
        q, r = jnp.linalg.qr(a)
        
        # Make Q uniform
        d = jnp.diag(r)
        q *= jnp.sign(d)
        
        if rows < cols:
            q = q.T
            
        q = q.reshape(shape)
        return gain * q
    
    def sparse(key, shape, sparsity=0.1, std=0.01):
        """Sparse initialization."""
        tensor = jnp.zeros(shape)
        num_nonzero = int(sparsity * jnp.prod(jnp.array(shape)))
        
        if len(shape) == 2:
            rows, cols = shape
            indices = random.choice(key, rows * cols, shape=(num_nonzero,), replace=False)
            row_indices = indices // cols
            col_indices = indices % cols
            values = random.normal(random.split(key)[1], (num_nonzero,)) * std
            tensor = tensor.at[row_indices, col_indices].set(values)
        
        return tensor
    
    # Transformer-specific initializers
    
    def transformer_init(key, shape, d_model, layer_type='linear'):
        """Transformer initialization (used in T5, GPT-3)."""
        if layer_type == 'embedding':
            std = 1.0
        elif layer_type == 'linear':
            std = (d_model ** -0.5)
        elif layer_type == 'output':
            std = (2 * d_model) ** -0.5
        else:
            std = 1.0
            
        return random.normal(key, shape) * std
    
    def gpt_init(key, shape, n_layer, layer_idx=0):
        """GPT-style initialization with layer-dependent scaling."""
        std = 0.02
        if len(shape) == 2 and shape[0] > shape[1]:  # Output projection
            std = 0.02 / math.sqrt(2 * n_layer)
        return random.normal(key, shape) * std
    
    def llama_init(key, shape, dim):
        """LLaMA initialization scheme."""
        if len(shape) == 2:  # Linear layer
            bound = 1 / math.sqrt(shape[0])
            return random.uniform(key, shape, minval=-bound, maxval=bound)
        else:
            return xavier_uniform(key, shape)
    
    # Utility functions
    
    def _calculate_fan_in_fan_out(shape):
        """Calculate fan_in and fan_out for a tensor shape."""
        if len(shape) < 2:
            fan_in = fan_out = shape[0] if len(shape) == 1 else 1
        elif len(shape) == 2:
            fan_in, fan_out = shape
        else:
            # For conv layers: fan_in/out = kernel_size * channels
            receptive_field_size = jnp.prod(jnp.array(shape[2:]))
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        
        return fan_in, fan_out
    
    def _calculate_correct_fan(shape, mode):
        """Calculate the correct fan value based on mode."""
        fan_in, fan_out = _calculate_fan_in_fan_out(shape)
        if mode == 'fan_in':
            return fan_in
        elif mode == 'fan_out':
            return fan_out
        else:  # fan_avg
            return (fan_in + fan_out) / 2.0
    
    def _calculate_gain(nonlinearity, param=None):
        """Calculate gain for different nonlinearities."""
        gains = {
            'linear': 1,
            'conv1d': 1,
            'conv2d': 1,
            'conv3d': 1,
            'conv_transpose1d': 1,
            'conv_transpose2d': 1,
            'conv_transpose3d': 1,
            'sigmoid': 1,
            'tanh': 5.0 / 3,
            'relu': _SQRT_2,
            'leaky_relu': math.sqrt(2.0 / (1 + (param or 0.01) ** 2)),  # param-dependent
            'selu': 3.0 / 4,
            'silu': 1.054,
            'gelu': 1.7159,
            'swish': 1.054,
        }
        return gains.get(nonlinearity, 1)

except ImportError:
    # Fallback using numpy
    import numpy as np
    
    def xavier_uniform(key, shape, gain=1.0):
        np.random.seed(key)
        fan_in, fan_out = _calculate_fan_in_fan_out(shape)
        bound = gain * _SQRT_6 / math.sqrt(fan_in + fan_out)
        return np.random.uniform(-bound, bound, shape)

    def xavier_normal(key, shape, gain=1.0):
        np.random.seed(key)
        fan_in, fan_out = _calculate_fan_in_fan_out(shape)
        std = gain * _SQRT_2 / math.sqrt(fan_in + fan_out)
        return np.random.normal(0, std, shape)
    
    def kaiming_uniform(key, shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        np.random.seed(key)
        fan = _calculate_correct_fan(shape, mode)
        gain = _calculate_gain(nonlinearity, a)
        bound = gain * math.sqrt(3.0 / fan)
        return np.random.uniform(-bound, bound, shape)
    
    def kaiming_normal(key, shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        np.random.seed(key)
        fan = _calculate_correct_fan(shape, mode)
        gain = _calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        return np.random.normal(0, std, shape)
    
    def lecun_uniform(key, shape):
        np.random.seed(key)
        fan_in = _calculate_fan_in_fan_out(shape)[0]
        bound = math.sqrt(3.0 / fan_in)
        return np.random.uniform(-bound, bound, shape)
    
    def lecun_normal(key, shape):
        np.random.seed(key)
        fan_in = _calculate_fan_in_fan_out(shape)[0]
        std = math.sqrt(1.0 / fan_in)
        return np.random.normal(0, std, shape)
    
    def truncated_normal(key, shape, stddev=1.0, lower=-2.0, upper=2.0):
        np.random.seed(key)
        values = np.random.normal(0, stddev, shape)
        return np.clip(values, lower * stddev, upper * stddev)
    
    def orthogonal(key, shape, gain=1.0):
        np.random.seed(key)
        if len(shape) < 2:
            raise ValueError("Orthogonal initialization requires at least 2D tensor")
        
        rows, cols = shape[0], np.prod(shape[1:])
        flattened_shape = (rows, cols) if rows >= cols else (cols, rows)
        
        a = np.random.normal(0, 1, flattened_shape)
        q, r = np.linalg.qr(a)
        d = np.diag(r)
        q *= np.sign(d)
        
        if rows < cols:
            q = q.T
            
        q = q.reshape(shape)
        return gain * q
    
    def sparse(key, shape, sparsity=0.1, std=0.01):
        np.random.seed(key)
        tensor = np.zeros(shape)
        num_nonzero = int(sparsity * np.prod(shape))
        
        if len(shape) == 2:
            rows, cols = shape
            indices = np.random.choice(rows * cols, num_nonzero, replace=False)
            row_indices = indices // cols
            col_indices = indices % cols
            values = np.random.normal(0, std, num_nonzero)
            tensor[row_indices, col_indices] = values
        
        return tensor
    
    def transformer_init(key, shape, d_model, layer_type='linear'):
        np.random.seed(key)
        if layer_type == 'embedding':
            std = 1.0
        elif layer_type == 'linear':
            std = (d_model ** -0.5)
        elif layer_type == 'output':
            std = (2 * d_model) ** -0.5
        else:
            std = 1.0
        return np.random.normal(0, std, shape)
    
    def gpt_init(key, shape, n_layer, layer_idx=0):
        np.random.seed(key)
        std = 0.02
        if len(shape) == 2 and shape[0] > shape[1]:
            std = 0.02 / math.sqrt(2 * n_layer)
        return np.random.normal(0, std, shape)
    
    def llama_init(key, shape, dim):
        np.random.seed(key)
        if len(shape) == 2:
            bound = 1 / math.sqrt(shape[0])
            return np.random.uniform(-bound, bound, shape)
        else:
            return xavier_uniform(key, shape)
    
    def _calculate_fan_in_fan_out(shape):
        if len(shape) < 2:
            fan_in = fan_out = shape[0] if len(shape) == 1 else 1
        elif len(shape) == 2:
            fan_in, fan_out = shape
        else:
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        return fan_in, fan_out
    
    def _calculate_correct_fan(shape, mode):
        fan_in, fan_out = _calculate_fan_in_fan_out(shape)
        if mode == 'fan_in':
            return fan_in
        elif mode == 'fan_out':
            return fan_out
        else:
            return (fan_in + fan_out) / 2.0
    
    def _calculate_gain(nonlinearity, param=None):
        gains = {
            'linear': 1, 'sigmoid': 1, 'tanh': 5.0/3, 'relu': math.sqrt(2.0),
            'leaky_relu': math.sqrt(2.0 / (1 + (param or 0.01) ** 2)),
            'selu': 3.0/4, 'silu': 1.054, 'gelu': 1.7159, 'swish': 1.054,
        }
        return gains.get(nonlinearity, 1)

__all__ = [
    'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
    'lecun_uniform', 'lecun_normal', 'truncated_normal', 'orthogonal', 'sparse',
    'transformer_init', 'gpt_init', 'llama_init'
]
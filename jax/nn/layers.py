"""
nn layers module.

# This module provides functionality for layers.
"""

import logging

try:
    import jax.numpy as jnp
    from jax import random, lax
    from . import initializers
except Exception:
    pass


    
    class Dense:
        """Dense/Linear layer with optional bias."""
        
        def __init__(self, features, use_bias=True, kernel_init='kaiming_normal', bias_init='zeros'):
            self.features = features
            self.use_bias = use_bias
            self.kernel_init = kernel_init
            self.bias_init = bias_init
            
        def init_params(self, key, input_shape):
            """Initialize parameters."""
            in_features = input_shape[-1]
            
            # Initialize kernel
            if self.kernel_init == 'kaiming_normal':
                kernel = initializers.kaiming_normal(key, (in_features, self.features), nonlinearity='relu')
            elif self.kernel_init == 'xavier_normal':
                kernel = initializers.xavier_normal(key, (in_features, self.features))
            elif self.kernel_init == 'gpt_init':
                kernel = initializers.gpt_init(key, (in_features, self.features), n_layer=12)
            else:
                kernel = initializers.kaiming_normal(key, (in_features, self.features))
            
            params = {'kernel': kernel}
            
            # Initialize bias
            if self.use_bias:
                if self.bias_init == 'zeros':
                    bias = jnp.zeros((self.features,))
                else:
                    bias = jnp.zeros((self.features,))
                params['bias'] = bias
                
            return params
        
        def __call__(self, x, params):
            """Forward pass."""
            y = jnp.dot(x, params['kernel'])
            if self.use_bias and 'bias' in params:
                y = y + params['bias']
            return y
    
    class Embedding:
        """Embedding layer for discrete inputs."""
        
        def __init__(self, num_embeddings, features, embedding_init='normal'):
            self.num_embeddings = num_embeddings
            self.features = features
            self.embedding_init = embedding_init
            
        def init_params(self, key):
            """Initialize embedding table."""
            if self.embedding_init == 'normal':
                embedding = random.normal(key, (self.num_embeddings, self.features)) * 0.02
            elif self.embedding_init == 'uniform':
                bound = 1.0 / jnp.sqrt(self.features)
                embedding = random.uniform(key, (self.num_embeddings, self.features), 
                                         minval=-bound, maxval=bound)
            else:
                embedding = random.normal(key, (self.num_embeddings, self.features)) * 0.02
                
            return {'embedding': embedding}
        
        def __call__(self, indices, params):
            """Forward pass - lookup embeddings."""
            return params['embedding'][indices]
        
        def attend(self, query, params):
            """Compute attention weights over embeddings (for output layer)."""
            return jnp.dot(query, params['embedding'].T)
    
    class PositionalEncoding:
        """Sinusoidal positional encoding (Transformer)."""
        
        def __init__(self, d_model, max_len=5000):
            self.d_model = d_model
            self.max_len = max_len
            
        def __call__(self, positions):
            """Generate positional encodings."""
            d_model = self.d_model
            
            # Create position indices
            pos = positions.reshape(-1, 1)  # (seq_len, 1)
            
            # Create dimension indices
            div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
            
            # Compute positional encodings
            pe = jnp.zeros((positions.shape[0], d_model))
            pe = pe.at[:, 0::2].set(jnp.sin(pos * div_term))
            pe = pe.at[:, 1::2].set(jnp.cos(pos * div_term))
            
            return pe
    
    class RotaryPositionalEmbedding:
        """Rotary Position Embedding (RoPE) - used in RoFormer, GPT-Neo-X."""
        
        def __init__(self, dim, max_seq_len=2048, base=10000):
            self.dim = dim
            self.max_seq_len = max_seq_len
            self.base = base
            
        def __call__(self, seq_len):
            """Generate rotary embeddings."""
            # Create position indices
            positions = jnp.arange(seq_len)
            
            # Create frequency bands
            inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2) / self.dim))
            
            # Compute sinusoidal values
            sinusoid_inp = jnp.outer(positions, inv_freq)
            
            # Create cos and without tensors
            cos_cached = jnp.cos(sinusoid_inp)
            sin_cached = jnp.sin(sinusoid_inp)
            
            return cos_cached, sin_cached
        
        def apply_rotary_emb(self, x, cos, sin):
            """Apply rotary embedding to input tensor."""
            def rotate_half(x):
                x1, x2 = jnp.split(x, 2, axis=-1)
                return jnp.concatenate([-x2, x1], axis=-1)
            
            return x * cos + rotate_half(x) * sin
    
    class FeedForward:
        """Position-wise feed-forward network (FFN)."""
        
        def __init__(self, d_model, d_ff, activation='relu', dropout_rate=0.1):
            self.d_model = d_model
            self.d_ff = d_ff
            self.activation = activation
            self.dropout_rate = dropout_rate
            
        def init_params(self, key):
            """Initialize FFN parameters."""
            k1, k2 = random.split(key)
            
            # First linear layer
            w1 = initializers.kaiming_normal(k1, (self.d_model, self.d_ff))
            b1 = jnp.zeros((self.d_ff,))
            
            # Second linear layer  
            w2 = initializers.kaiming_normal(k2, (self.d_ff, self.d_model))
            b2 = jnp.zeros((self.d_model,))
            
            return {
                'w1': w1, 'b1': b1,
                'w2': w2, 'b2': b2
            }
        
        def __call__(self, x, params, training=True, key=None):
            """Forward pass through FFN."""
            # First linear transformation
            hidden = jnp.dot(x, params['w1']) + params['b1']
            
            # Apply activation
            if self.activation == 'relu':
                from .activations import relu
                hidden = relu(hidden)
            elif self.activation == 'gelu':
                from .activations import gelu  
                hidden = gelu(hidden)
            elif self.activation == 'swish':
                from .activations import swish
                hidden = swish(hidden)
            
            # Apply dropout
            if training and self.dropout_rate > 0 and key is not None:
                hidden = dropout(hidden, self.dropout_rate, key)
            
            # Second linear transformation
            output = jnp.dot(hidden, params['w2']) + params['b2']
            
            return output
    
    class SwiGLU:
        """SwiGLU FFN (used in PaLM, LLaMA)."""
        
        def __init__(self, d_model, d_ff=None, bias=False):
            self.d_model = d_model
            self.d_ff = d_ff or int(2.67 * d_model)  # LLaMA scaling
            self.bias = bias
            
        def init_params(self, key):
            """Initialize SwiGLU parameters."""
            k1, k2, k3 = random.split(key, 3)
            
            # Gate and up projections (both to d_ff)
            w_gate = initializers.kaiming_normal(k1, (self.d_model, self.d_ff))
            w_up = initializers.kaiming_normal(k2, (self.d_model, self.d_ff))
            
            # Down projection
            w_down = initializers.kaiming_normal(k3, (self.d_ff, self.d_model))
            
            params = {
                'w_gate': w_gate,
                'w_up': w_up, 
                'w_down': w_down
            }
            
            if self.bias:
                params.update({
                    'b_gate': jnp.zeros((self.d_ff,)),
                    'b_up': jnp.zeros((self.d_ff,)),
                    'b_down': jnp.zeros((self.d_model,))
                })
                
            return params
        
        def __call__(self, x, params):
            """Forward pass through SwiGLU."""
            from .activations import swish
            
            # Gate and up projections
            gate = jnp.dot(x, params['w_gate'])
            up = jnp.dot(x, params['w_up'])
            
            if self.bias:
                gate = gate + params['b_gate']
                up = up + params['b_up']
            
            # Apply SwiGLU: swish(gate) * up
            hidden = swish(gate) * up
            
            # Down projection
            output = jnp.dot(hidden, params['w_down'])
            if self.bias:
                output = output + params['b_down']
                
            return output
    
    class GLU:
        """Gated Linear Unit and variants."""
        
        def __init__(self, d_model, d_ff, variant='glu', bias=True):
            self.d_model = d_model
            self.d_ff = d_ff
            self.variant = variant  # 'glu', 'geglu', 'swiglu', 'reglu'
            self.bias = bias
            
        def init_params(self, key):
            """Initialize GLU parameters."""
            k1, k2 = random.split(key)
            
            # Input projection (to 2 * d_ff for gating)
            w1 = initializers.kaiming_normal(k1, (self.d_model, 2 * self.d_ff))
            
            # Output projection  
            w2 = initializers.kaiming_normal(k2, (self.d_ff, self.d_model))
            
            params = {'w1': w1, 'w2': w2}
            
            if self.bias:
                params.update({
                    'b1': jnp.zeros((2 * self.d_ff,)),
                    'b2': jnp.zeros((self.d_model,))
                })
                
            return params
        
        def __call__(self, x, params):
            """Forward pass through GLU variant."""
            # Input projection
            hidden = jnp.dot(x, params['w1'])
            if self.bias:
                hidden = hidden + params['b1']
            
            # Apply GLU variant
            if self.variant == 'glu':
                from .activations import sigmoid
                a, b = jnp.split(hidden, 2, axis=-1)
                gated = a * sigmoid(b)
            elif self.variant == 'geglu':
                from .activations import gelu
                a, b = jnp.split(hidden, 2, axis=-1)
                gated = gelu(a) * b
            elif self.variant == 'swiglu':
                from .activations import swish
                a, b = jnp.split(hidden, 2, axis=-1)
                gated = swish(a) * b
            elif self.variant == 'reglu':
                from .activations import relu
                a, b = jnp.split(hidden, 2, axis=-1)
                gated = relu(a) * b
            else:
                raise ValueError(f"Unknown GLU variant: {self.variant}")
            
            # Output projection
            output = jnp.dot(gated, params['w2'])
            if self.bias:
                output = output + params['b2']
                
            return output
    
    # Utility functions
    
    def dropout(x, rate, key, training=True):
        """Apply dropout."""
        if not training or rate == 0:
            return x
        
        keep_prob = 1.0 - rate
        mask = random.bernoulli(key, keep_prob, x.shape)
        return jnp.where(mask, x / keep_prob, 0.0)
    
    def linear(x, weight, bias=None):
        """Functional linear transformation."""
        y = jnp.dot(x, weight)
        if bias is not None:
            y = y + bias
        return y
    
    def embedding_lookup(indices, embedding_table):
        """Functional embedding lookup."""
        return embedding_table[indices]

logger = logging.getLogger(__name__)

def main():
    # Main function for this module.
    logger.info("Module layers.py starting")
    return True

if __name__ == "__main__":
    try:
        main()
    except Exception: 
        pass

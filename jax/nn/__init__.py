"""
CapibaraGPT Native Flax Implementation
"""

import logging
from typing import Any

# Use internal capibara.jax.numpy to avoid direct JAX dependency
try:
    from .. import numpy as jnp
except ImportError:
    # Fallback if circular import
    try:
        import jax.numpy as jnp
    except ImportError:
        import numpy as jnp

logger = logging.getLogger(__name__)

# Try to import from Flax first
try:
    from flax import linen as nn
    Module = nn.Module
    logger.info("Using Flax linen Module")
except ImportError:
    try:
        # Fallback to standard Flax
        from flax import nn as flax_nn
        Module = flax_nn.Module
        logger.info("Using standard Flax Module")
    except ImportError:
        # Create a basic fallback Module class
        logger.warning("Flax not available, using fallback Module")
        
        class Module:
            """Fallback Module class when Flax is not available"""
            def __init__(self, *args, **kwargs):
                pass
            
            def __call__(self, *args, **kwargs):
                return args[0] if args else None
        
        # Create a namespace for compatibility
        class _FallbackNN:
            Module = Module
            
        nn = _FallbackNN()


class FeedForwardNetwork(nn.Module):
    """Feed-forward network with GELU activation."""
    
    intermediate_size: int
    hidden_size: int
    dropout_rate: float = 0.1
    
    def setup(self):
        self.dense_1 = nn.Dense(features=self.intermediate_size, name='dense_1')
        self.dense_2 = nn.Dense(features=self.hidden_size, name='dense_2')
        self.dropout = nn.Dropout(rate=self.dropout_rate)
    
    def __call__(self, hidden_states, training=True):
        hidden_states = self.dense_1(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=not training)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states


class TransformerLayer(nn.Module):
    """Transformer layer with multi-head attention and FFN."""
    
    hidden_size: int
    num_heads: int
    intermediate_size: int
    dropout_rate: float = 0.1
    layer_norm_eps: float = 1e-12
    
    def setup(self):
        # Multi-head attention
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            dropout_rate=self.dropout_rate,
            name='attention'
        )
        
        # Layer norms
        self.attention_norm = nn.LayerNorm(name='attention_norm')
        self.ffn_norm = nn.LayerNorm(name='ffn_norm')
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(
            intermediate_size=self.intermediate_size,
            hidden_size=self.hidden_size,
            dropout_rate=self.dropout_rate,
            name='ffn'
        )
    
    def __call__(self, hidden_states, attention_mask=None, training=True):
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        
        attention_output = self.attention(
            hidden_states,
            mask=attention_mask,
            deterministic=not training
        )
        
        hidden_states = residual + attention_output
        
        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        ffn_output = self.ffn(hidden_states, training=training)
        hidden_states = residual + ffn_output
        
        return hidden_states


# CapibaraGPT 300M Model Definition
class CapibaraGPT300M(nn.Module):
    """300M Parameter CapibaraGPT Model - Real Architecture."""
    
    vocab_size: int = 50000
    hidden_size: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    intermediate_size: int = 4096
    max_position_embeddings: int = 2048
    dropout_rate: float = 0.1
    layer_norm_eps: float = 1e-12
    
    def setup(self):
        # Embedding layers
        self.token_embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            name='token_embedding'
        )
        self.position_embedding = nn.Embed(
            num_embeddings=self.max_position_embeddings,
            features=self.hidden_size,
            name='position_embedding'
        )
        
        # Transformer layers
        self.layers = [
            TransformerLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                dropout_rate=self.dropout_rate,
                layer_norm_eps=self.layer_norm_eps,
                name=f'layer_{i}'
            ) for i in range(self.num_layers)
        ]
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(name='final_layer_norm')
        
        # Language modeling head
        self.lm_head = nn.Dense(
            features=self.vocab_size,
            use_bias=False,
            name='lm_head'
        )
        
    def __call__(self, input_ids, attention_mask=None, training=True):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = jnp.arange(seq_len)[None, :]
        position_embeds = self.position_embedding(position_ids)
        
        hidden_states = token_embeds + position_embeds
        
        if training:
            hidden_states = nn.Dropout(rate=self.dropout_rate)(
                hidden_states, deterministic=not training
            )
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, 
                attention_mask=attention_mask,
                training=training
            )
        
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        return logits


# Activation functions
def softmax(x, axis=-1):
    """Compute softmax along specified axis."""
    return jnp.exp(x) / jnp.sum(jnp.exp(x), axis=axis, keepdims=True)

def relu(x):
    """ReLU activation function."""
    return jnp.maximum(0, x)

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + jnp.exp(-x))

def gelu(x):
    """GELU activation function."""
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))

# Export all classes and functions
__all__ = [
    'Module', 'CapibaraGPT300M', 'TransformerLayer', 'FeedForwardNetwork',
    'softmax', 'relu', 'sigmoid', 'gelu'
]
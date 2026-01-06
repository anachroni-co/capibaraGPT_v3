"""Ultra-Optimized DeepDialog Model with Advanced Features"""

import os
import sys
from capibara.jax import jax
from capibara.jax import numpy as jnp
from capibara.jax import random  # Import random from capibara.jax
# Gets the current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to obtain project root -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Add project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from capibara.jax import jax
from capibara.jax.numpy import jnp
from flax import linen as nn
from typing import Tuple, Optional, Dict, Any, Callable, List
import logging
from pydantic import BaseModel, Field, validator
from functools import partial
from enum import Enum

logger = logging.getLogger(__name__)

class ActivationType(str, Enum):
    """Supported activation functions"""
    GELU = "gelu"
    RELU = "relu"
    SWISH = "swish"
    SILU = "silu"
    TANH = "tanh"

class DeepDialogConfig(BaseModel):
    """Enhanced configuration with validation and optimization parameters"""
    hidden_size: int = Field(default=768, gt=0, description="Hidden dimension size")
    num_layers: int = Field(default=12, gt=0, description="Number of transformer layers")
    num_heads: int = Field(default=8, gt=0, description="Number of attention heads")
    key_size: int = Field(default=64, gt=0, description="Attention key dimension")
    dropout_rate: float = Field(default=0.1, ge=0, lt=1, description="Dropout rate")
    activation: ActivationType = Field(default=ActivationType.GELU, description="Activation function")
    context_dim: Optional[int] = Field(default=None, description="Context dimension")
    max_seq_len: int = Field(default=512, gt=0, description="Maximum sequence length")
    use_memory_efficient: bool = Field(default=True, description="Enable memory optimizations")
    gradient_checkpointing: bool = Field(default=False, description="Enable gradient checkpointing")
    residual_scale: float = Field(default=1.0, gt=0, description="Residual connection scaling")
    meta_la_config: Optional[Dict[str, Any]] = Field(default=None, description="Meta-Learning Adapter configuration")
    vq_config: Optional[Dict[str, Any]] = Field(default=None, description="Vector Quantization configuration")

    @property
    def activation_fn(self) -> Callable:
        """Get activation function with optimized implementations"""
        activations = {
            ActivationType.GELU: jax.nn.gelu,
            ActivationType.RELU: jax.nn.relu,
            ActivationType.SWISH: jax.nn.swish,
            ActivationType.SILU: jax.nn.silu,
            ActivationType.TANH: jnp.tanh
        }
        return activations[self.activation]

class DeepDialog(nn.Module):
    """Advanced Dialogue Model with Cross-Attention and Optimizations"""
    
    config: DeepDialogConfig

    def setup(self):
        """Initialize model with enhanced components"""
        try:
            # Input projections with kernel scaling
            kernel_init = nn.initializers.variance_scaling(
                1.0, 'fan_in', 'truncated_normal')
            
            self.input_proj = nn.Dense(
                self.config.hidden_size,
                kernel_init=kernel_init,
                dtype=jnp.float32
            )
            
            # Context processing
            if self.config.context_dim:
                self.context_proj = nn.Dense(
                    self.config.hidden_size,
                    kernel_init=kernel_init,
                    dtype=jnp.float32
                )
                self.cross_attn_layers = [
                    self._create_cross_attn_layer(i)
                    for i in range(self.config.num_layers)
                ]
            
            # Main transformer layers
            self.layers = [
                self._create_transformer_layer(i)
                for i in range(self.config.num_layers)
            ]

            self.output_norm = nn.LayerNorm(epsilon=1e-6, dtype=jnp.float32)
            
            # Reduce logging verbosity during training
            if not hasattr(DeepDialogModel, '_logged_init'):
                logger.info("Model initialized with %d layers", self.config.num_layers)
                DeepDialogModel._logged_init = True
            
        except Exception as e:
            logger.error("Initialization failed: %s", str(e), exc_info=True)
            raise

    def _create_cross_attn_layer(self, layer_idx: int) -> nn.Module:
        """Creates optimized cross-attention layer"""
        return CrossAttentionLayer(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            key_size=self.config.key_size,
            dropout_rate=self.config.dropout_rate,
            activation_fn=self.config.activation_fn,
            use_checkpoint=self.config.gradient_checkpointing,
            name=f"cross_attn_{layer_idx}"
        )

    def _create_transformer_layer(self, layer_idx: int) -> nn.Module:
        """Creates optimized transformer layer"""
        return TransformerLayer(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_heads,
            key_size=self.config.key_size,
            dropout_rate=self.config.dropout_rate,
            activation_fn=self.config.activation_fn,
            residual_scale=self.config.residual_scale,
            use_checkpoint=self.config.gradient_checkpointing,
            name=f"transformer_{layer_idx}"
        )

    def __call__(
        self,
        x: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        training: bool = False,
        rng: Optional[random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Enhanced forward pass with:
        - Better error handling
        - Memory optimizations
        - RNG support
        """
        try:
            # Input validation - convert all JAX array types
            x = jnp.asarray(x)
                
            if x.ndim not in [2, 3]:
                raise ValueError(
                    f"Input must be 2D (batch, features) or 3D (batch, seq, features), got {x.shape}"
                )
            
            # Project inputs
            x = self.input_proj(x)
            
            # Process context if provided
            if context is not None and self.config.context_dim:
                context = jnp.asarray(context)
                    
                context_proj = self.context_proj(context)
                
                # Apply cross-attention
                for layer in self.cross_attn_layers:
                    x = layer(x, context_proj, training=training, rng=rng)
            
            # Process through main layers
            for layer in self.layers:
                x = layer(x, training=training, rng=rng)
            
            return self.output_norm(x)

        except Exception as e:
            logger.error("Forward pass failed: %s", str(e), exc_info=True)
            raise

class CrossAttentionLayer(nn.Module):
    """Optimized Cross-Attention Layer with Memory Efficiency"""
    hidden_size: int
    num_heads: int
    key_size: int
    dropout_rate: float
    activation_fn: Callable
    use_checkpoint: bool = False
    residual_scale: float = 1.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        context: jnp.ndarray,
        training: bool,
        rng: Optional[random.PRNGKey] = None
    ) -> jnp.ndarray:
        def _forward(x, context):
            # Layer normalization
            x_norm = nn.LayerNorm(epsilon=1e-6, dtype=jnp.float32)(x)
            context_norm = nn.LayerNorm(epsilon=1e-6, dtype=jnp.float32)(context)
            
            # Cross-attention
            attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                dtype=jnp.float32,
                param_dtype=jnp.float32
            )(x_norm, context_norm, deterministic=not training)
            
            # Scaled residual connection
            x = x + self.residual_scale * nn.Dropout(
                self.dropout_rate,
                deterministic=not training
            )(attn)
            
            # Feed-forward with activation
            x = nn.Sequential([
                nn.Dense(self.hidden_size * 4, dtype=jnp.float32),
                self.activation_fn,
                nn.Dropout(self.dropout_rate, deterministic=not training),
                nn.Dense(self.hidden_size, dtype=jnp.float32)
            ])(x)
            
            return x
        
        if self.use_checkpoint:
            return jax.checkpoint(_forward)(x, context)
        return _forward(x, context)

class TransformerLayer(nn.Module):
    """Enhanced Transformer Layer with Advanced Features"""
    hidden_size: int
    num_heads: int
    key_size: int
    dropout_rate: float
    activation_fn: Callable
    residual_scale: float = 1.0
    use_checkpoint: bool = False

    def setup(self):
        self.self_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )
        # FFN layers (manual implementation to handle deterministic)
        self.ffn_dense1 = nn.Dense(self.hidden_size * 4, dtype=jnp.float32)
        self.ffn_dropout = nn.Dropout(self.dropout_rate)
        self.ffn_dense2 = nn.Dense(self.hidden_size, dtype=jnp.float32)
        self.norm1 = nn.LayerNorm(epsilon=1e-6, dtype=jnp.float32)
        self.norm2 = nn.LayerNorm(epsilon=1e-6, dtype=jnp.float32)

    def __call__(
        self,
        x: jnp.ndarray,
        training: bool,
        rng: Optional[random.PRNGKey] = None
    ) -> jnp.ndarray:
        def _forward(x):
            # Self-attention
            attn_out = self.self_attn(
                inputs_q=x,
                inputs_kv=x,
                deterministic=not training
            )
            x = self.norm1(x + self.residual_scale * attn_out)
            
            # Feed-forward with manual dropout handling
            ffn_x = self.ffn_dense1(x)
            ffn_x = self.activation_fn(ffn_x)
            ffn_x = self.ffn_dropout(ffn_x, deterministic=not training)
            ffn_out = self.ffn_dense2(ffn_x)
            return self.norm2(x + self.residual_scale * ffn_out)
        
        # Temporarily disable checkpointing to avoid tracer leaks
        # if self.use_checkpoint:
        #     return jax.checkpoint(_forward)(x)
        return _forward(x)

# Example Usage with Enhanced Features
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        logger.info("Starting Enhanced DeepDialog Test")
        
        # Configuration with advanced options
        config = DeepDialogConfig(
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            key_size=64,
            dropout_rate=0.1,
            activation=ActivationType.SWISH,
            context_dim=128,
            max_seq_len=128,
            gradient_checkpointing=True,
            residual_scale=0.9
        )

        # Initialize model with key
        key = random.PRNGKey(42)
        model = DeepDialog(config)
        
        # Generate test data
        batch_size = 2
        seq_len = 16
        input_dim = 128
        context_dim = config.context_dim
        
        inputs = random.normal(key, (batch_size, seq_len, input_dim))
        context = random.normal(key, (batch_size, seq_len, context_dim))
        
        # Initialize parameters with separate RNG
        params_key, dropout_key = random.split(key)
        params = model.init(params_key, inputs, context)
        
        # Forward pass with dropout RNG
        output = model.apply(
            params,
            inputs,
            context,
            training=True,
            rngs={'dropout': dropout_key}
        )
        
        logger.info("Test completed successfully. Output shape: %s", str(output.shape))

    except Exception as e:
        logger.error("Test failed: %s", str(e), exc_info=True)
        raise

# Create alias for backwards compatibility
DeepDialogModel = DeepDialog

# Export both names
__all__ = ['DeepDialog', 'DeepDialogModel', 'DeepDialogConfig']
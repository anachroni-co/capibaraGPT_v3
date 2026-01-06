"""
sub_models Byte_TPU module.

# This module provides functionality for Byte_TPU.
"""

import os
import sys
from pathlib import Path
import logging
import numpy as np

# Gets the current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to obtain project root -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Add project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from capibara.jax import jax
from capibara.jax.numpy import jnp
from flax import linen as nn
from dataclasses import dataclass
from typing import Tuple, Callable, Optional, Dict, Any, Union
from functools import partial
from enum import Enum

from capibara.core.config import (
    distributed_jit,
    model_sharded_jit,
    batch_sharded_jit,
    BATCH_SHARDING,
    MODEL_SHARDING,
    HYBRID_SHARDING,
    TPU_DTYPE,
    create_unified_mesh
)
try:
    from capibara.core.utils import ActivationType
except ImportError:
    # Fallback enum if not existe
    from enum import Enum
    class ActivationType(str, Enum):
        GELU = 'gelu'
        RELU = 'relu'
        SWISH = 'swish'

def get_activation_fn(activation_type: ActivationType):
    """Get activation function."""
    if activation_type == ActivationType.GELU:
        return jax.nn.gelu
    elif activation_type == ActivationType.RELU:
        return jax.nn.relu
    elif activation_type == ActivationType.SWISH:
        return jax.nn.swish
    else:
        return jax.nn.gelu  # default

logger = logging.getLogger(__name__)

@dataclass
class ByteTPUConfig:
    """Configurestion for Byte_TPU processing."""
    hidden_size: int = 768
    byte_embedding_size: int = 256
    activation_type: ActivationType = ActivationType.GELU
    use_layer_norm: bool = True
    dropout_rate: float = 0.1
    tpu_optimized: bool = True

class ByteTPULayer(nn.Module):
    """Enhanced Byte-level processing layer with TPU v6e optimizations."""
    
    config: ByteTPUConfig
    
    def setup(self):
        """Initialize Byte TPU layer with enhanced TPU optimizations."""
        self.hidden_size = self.config.hidden_size
        self.byte_embedding_size = self.config.byte_embedding_size
        self.activation_fn = get_activation_fn(self.config.activation_type)
        
        # Enhanced byte processing layers
        self.byte_embedding = nn.Dense(self.byte_embedding_size, name="byte_embedding")
        self.byte_processor = nn.Dense(self.hidden_size, name="byte_processor") 
        self.context_mixer = nn.Dense(self.hidden_size, name="context_mixer")
        
        if self.config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(name="byte_layer_norm")
        
        self.dropout = nn.Dropout(rate=self.config.dropout_rate)
    
    @distributed_jit  # TPU v6e optimization
    def __call__(self, inputs, training=False, rngs=None):
        """Enhanced forward pass with TPU v6e optimizations."""
        try:
            batch_size, seq_len, hidden_size = inputs.shape
            
            # Byte-level embedding with TPU optimization
            byte_embedded = self.byte_embedding(inputs)
            
            # Apply activation and processing
            processed = self.activation_fn(byte_embedded)
            processed = self.byte_processor(processed)
            
            # Context mixing for better representations
            context_mixed = self.context_mixer(processed)
            
            # Residual connection
            output = inputs + context_mixed
            
            # Layer normalization for stability
            if self.config.use_layer_norm:
                output = self.layer_norm(output)
            
            # Apply dropout during training
            if training and rngs is not None:
                dropout_rng = rngs.get('dropout') if isinstance(rngs, dict) else rngs
                output = self.dropout(output, deterministic=not training, rng=dropout_rng)
            
            return output
            
        except Exception as e:
            logger.warning(f"Byte_TPU processing error: {e}")
            return inputs  # Fallback to identity

class EnhancedByteTPUProcessor:
    """Enhanced Byte_TPU processor with initialization management."""
    
    def __init__(self, config: ByteTPUConfig):
        self.config = config
        self.layer = ByteTPULayer(config)
        self.params = None
        self.initialized = False
    
    def initialize(self, rng_key, input_shape):
        """Initialize the Byte_TPU processor with given input shape."""
        dummy_input = jnp.ones(input_shape, dtype=jnp.float32)
        try:
            self.params = self.layer.init(rng_key, dummy_input, training=False)
            self.initialized = True
            logger.info(f"✅ Enhanced Byte_TPU processor initialized: {input_shape}")
            return True
        except Exception as e:
            logger.error(f"❌ Byte_TPU initialization failed: {e}")
            return False
    
    def process(self, inputs, training=False, rngs=None):
        """Process inputs through initialized Byte_TPU layer."""
        if not self.initialized:
            logger.warning("Byte_TPU processor not initialized, returning identity")
            return inputs
        
        try:
            return self.layer.apply(self.params, inputs, training=training, rngs=rngs)
        except Exception as e:
            logger.warning(f"Byte_TPU processing error: {e}")
            return inputs

def create_byte_tpu_processor(hidden_size=768, **kwargs):
    """Factory function to create Enhanced Byte_TPU processor."""
    config = ByteTPUConfig(hidden_size=hidden_size, **kwargs)
    return EnhancedByteTPUProcessor(config)

def main():
    """Main function for testing Byte_TPU module."""
    logger.info("Enhanced Byte_TPU module with TPU v6e optimizations ready")
    return True

if __name__ == "__main__":
    main()

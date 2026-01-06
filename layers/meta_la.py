"""
layers meta_la module.

# This module provides functionality for meta_la.
"""

import os
import sys

import logging
# Gets the current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to obtain project root -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Add project root to sys.path
if project_root not in sys.path:
    # Fixed: Using proper imports instead of sys.path manipulation
    pass

from capibara.jax import jax
from flax import linen as nn
from dataclasses import dataclass
from capibara.jax import numpy as jnp
from typing import Tuple, Optional, Dict, Any

from capibara.interfaces.ilayer import ILayer
from capibara.layers.base import BaseLayer, LayerConfig

logger = logging.getLogger(__name__)

@dataclass
class MetaLAConfig(LayerConfig):
    """Configuration for Meta-Learning Attention (Meta-LA) layer"""
    hidden_dim: int = 768  # Use main's larger default
    hidden_size: int = 768  # Compatibility alias
    num_heads: int = 12  # Use main's larger default
    num_attention_heads: int = 12  # Compatibility alias
    dropout_rate: float = 0.1
    meta_learning_rate: float = 0.001
    adaptation_steps: int = 5  # From main branch

class MetaLA(BaseLayer):
    """
    Meta-Learning Attention (Meta-LA) Layer
    
    Advanced attention mechanism that adapts based on meta-learning
    principles for improved context understanding.
    """
    config: MetaLAConfig
    
    def setup(self):
        # Use the larger of the two head counts for better performance
        num_heads = max(getattr(self.config, 'num_heads', 8), 
                       getattr(self.config, 'num_attention_heads', 12))
        hidden_dim = max(getattr(self.config, 'hidden_dim', 512),
                        getattr(self.config, 'hidden_size', 768))
        
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=num_heads,
            qkv_features=hidden_dim,
            dropout_rate=self.config.dropout_rate
        )
        
        # Combine both approaches: meta_adapter from PR + meta_weights from main
        self.meta_adapter = nn.Dense(hidden_dim)
        self.meta_weights = self.param(
            'meta_weights',
            nn.initializers.normal(0.02),
            (hidden_dim, hidden_dim)
        )
        self.adaptation_layer = nn.Dense(hidden_dim)
        self.layer_norm = nn.LayerNorm()
        
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, 
                 training: bool = False) -> jnp.ndarray:
        """Forward pass with enhanced meta-learning adaptation"""
        batch_size, seq_len, hidden_size = x.shape
        
        # Apply meta-adaptation (from PR branch)
        adapted_x = self.meta_adapter(x)
        
        # Apply attention with adaptation
        attended = self.attention(adapted_x, adapted_x, mask=mask, 
                                deterministic=not training)
        
        # Meta-learning adaptation (from main branch)
        meta_context = jnp.dot(x, self.meta_weights)
        adapted_features = self.adaptation_layer(meta_context)
        
        # Combine all features: attention + PR meta-adaptation + main meta-features
        combined = attended + adapted_features
        output = self.layer_norm(x + combined)
        
        return output
    
    def fast_adapt(self, support_x, support_y, query_x):
        """Fast adaptation for few-shot learning."""
        # Simplified meta-learning adaptation
        adapted_params = self.adapt_parameters(support_x, support_y)
        return self.apply_adapted_params(query_x, adapted_params)
    
    def adapt_parameters(self, support_x, support_y):
        """Adapt parameters based on support set."""
        # Placeholder for meta-learning parameter adaptation
        return self.meta_weights
    
    def apply_adapted_params(self, query_x, adapted_params):
        """Apply adapted parameters to query set."""
        adapted_features = jnp.dot(query_x, adapted_params)
        return adapted_features

# Alias for backward compatibility
class MetaLearningAttention(MetaLA):
    """Backward compatibility alias."""
    pass

def main():
    # Main function for this module.
    logger.info("Module meta_la.py starting")
    # Test MetaLA configuration with updated parameter names
    config = MetaLAConfig(hidden_dim=768, num_heads=12)
    logger.info(f"✅ MetaLA config created: {config}")
    return True

if __name__ == "__main__":
    main()

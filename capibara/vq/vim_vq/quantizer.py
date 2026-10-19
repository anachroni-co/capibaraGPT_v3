"""
ViM-VQ: vector Quantization with Vision in Mind
implementation optimizada for CapibaraGPT with soporte tpu.
"""

import jax
import logging
import jax.numpy as jnp
from flax import linen as nn
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from functools import partial

logger = logging.getLogger(__name__)

@dataclass
class ViMVQConfig:
    """setup for ViM-VQ Quantizer"""
    num_embeddings: int = 8192
    embedding_dim: int = 512
    commitment_cost: float = 0.25
    decay: float = 0.99
    use_vision_bias: bool = True
    temperature: float = 0.5
    
class ViMVQQuantizer(nn.Module):
    """Cuantizador ViM-VQ optimizado for aplicaciones de visión"""
    
    config: ViMVQConfig
    
    def setup(self):
        self.embeddings = nn.Embed(
            self.config.num_embeddings,
            self.config.embedding_dim
        )
        
    def __call__(self, x: jnp.ndarray, training: bool = True) -> Dict[str, jnp.ndarray]:
        """Forward pass del cuantizador"""
        # Flatten input
        input_shape = x.shape
        flat_input = x.reshape(-1, x.shape[-1])
        
        # Compute distances to embeddings
        distances = jnp.sum((flat_input[..., None, :] - self.embeddings.embedding) ** 2, axis=-1)
        
        # Find closest embeddings
        encoding_indices = jnp.argmin(distances, axis=-1)
        
        # Get quantized values
        quantized = self.embeddings(encoding_indices)
        quantized = quantized.reshape(input_shape)
        
        # Compute losses
        commitment_loss = jnp.mean((jax.lax.stop_gradient(quantized) - x) ** 2)
        vq_loss = jnp.mean((quantized - jax.lax.stop_gradient(x)) ** 2)
        
        # Straight-through estimator
        quantized = x + jax.lax.stop_gradient(quantized - x)
        
        return {
            'quantized': quantized,
            'indices': encoding_indices,
            'commitment_loss': commitment_loss,
            'vq_loss': vq_loss,
            'perplexity': jnp.exp(-jnp.sum(jnp.mean(distances, axis=0) * jnp.log(jnp.mean(distances, axis=0) + 1e-10)))
        }
        
    def quantize_model(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """Cuantiza un model complete"""
        quantized_params = {}
        total_params_original = 0
        total_params_quantized = 0
        
        for layer_name, params in model_params.items():
            if isinstance(params, jnp.ndarray) and params.ndim >= 2:
                # Apply quantization
                result = self(params, training=False)
                quantized_params[layer_name] = result['quantized']
                
                total_params_original += params.size
                total_params_quantized += result['indices'].size * jnp.log2(self.config.num_embeddings) / 8
            else:
                quantized_params[layer_name] = params
                total_params_original += params.size if hasattr(params, 'size') else 0
                total_params_quantized += params.size if hasattr(params, 'size') else 0
        
        compression = total_params_original / max(total_params_quantized, 1)
        
        return {
            'quantized_params': quantized_params,
            'compression_ratio': compression,
            'original_size': total_params_original,
            'quantized_size': total_params_quantized
        }

def create_vim_vq_quantizer(config: Optional[ViMVQConfig] = None) -> ViMVQQuantizer:
    """Factory function for create un cuantizador ViM-VQ"""
    if config is None:
        config = ViMVQConfig()
    return ViMVQQuantizer(config=config)

# Configuraciones predefinidas
VIM_VQ_SMALL = ViMVQConfig(
    num_embeddings=1024,
    embedding_dim=256,
    commitment_cost=0.25,
    decay=0.99,
    use_vision_bias=True,
    temperature=0.5
)

VIM_VQ_BASE = ViMVQConfig(
    num_embeddings=8192,
    embedding_dim=512,
    commitment_cost=0.25,
    decay=0.99,
    use_vision_bias=True,
    temperature=0.5
)

VIM_VQ_LARGE = ViMVQConfig(
    num_embeddings=16384,
    embedding_dim=1024,
    commitment_cost=0.25,
    decay=0.99,
    use_vision_bias=True,
    temperature=0.5
)

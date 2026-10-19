"""
VQbit Layer implementation for CapibaraGPT.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

# Try to import JAX
try:
    import jax.numpy as jnp
    from jax import random, jit, vmap, lax
    HAS_JAX = True
except ImportError:
    import numpy as jnp
    HAS_JAX = False
    lax = None

    def random(*args, **kwargs):
        return None

    def jit(fn):
        return fn

    def vmap(fn):
        return fn

    def stop_gradient(x):
        return x

class VQCompressionMode(Enum):
    """vector quantization compression modes."""
    STANDARD = "standard"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"
    RESIDUAL = "residual"
    PRODUCT = "product"

@dataclass
class VQbitConfig:
    """Configuration for VQbit layer."""
    
    # Basic VQ parameters
    codebook_size: int = 512
    embedding_dim: int = 64
    commitment_cost: float = 0.25
    decay: float = 0.99
    
    # Compression settings
    compression_mode: VQCompressionMode = VQCompressionMode.STANDARD
    compression_ratio: float = 0.1
    
    # Adaptive features
    enable_adaptive_codebook: bool = True
    adaptation_rate: float = 0.01
    
    # Multi-level quantization
    num_levels: int = 1
    level_weights: List[float] = None
    
    # Advanced features
    enable_ema_update: bool = True
    enable_straight_through: bool = True
    enable_commitment_loss: bool = True
    
    def __post_init__(self):
        if self.level_weights is None:
            self.level_weights = [1.0] * self.num_levels

class VQbitLayer:
    """vector Quantization with bit-level compression layer."""
    
    def __init__(self, config: VQbitConfig):
        self.config = config
        self.codebook = None
        self.codebook_usage = None
        self.initialized = False
        
    def _initialize_codebook(self, input_shape: Tuple[int, ...], key: Any = None):
        """Initialize the codebook."""
        if HAS_JAX and key is not None:
            # JAX initialization
            self.codebook = random.normal(
                key, 
                (self.config.codebook_size, self.config.embedding_dim)
            ) * 0.1
        else:
            # Numpy fallback
            np.random.seed(42)
            self.codebook = np.random.normal(
                0, 0.1, 
                (self.config.codebook_size, self.config.embedding_dim)
            ).astype(np.float32)
        
        # Initialize usage tracking
        self.codebook_usage = jnp.ones(self.config.codebook_size)
        self.initialized = True
    
    def quantize(self, inputs: Any, key: Any = None) -> Tuple[Any, Dict[str, Any]]:
        """Quantize input vectors."""
        if not self.initialized:
            self._initialize_codebook(inputs.shape, key)
        
        # Flatten inputs for quantization
        input_shape = inputs.shape
        flat_inputs = inputs.reshape(-1, self.config.embedding_dim)
        
        # Compute distances to codebook
        distances = self._compute_distances(flat_inputs, self.codebook)
        
        # Find closest codebook entries
        encoding_indices = jnp.argmin(distances, axis=1)
        
        # Get quantized vectors
        quantized = self.codebook[encoding_indices]
        
        # Reshape back
        quantized = quantized.reshape(input_shape)
        
        # Compute losses
        stop_grad = lax.stop_gradient if HAS_JAX and lax is not None else stop_gradient
        commitment_loss = jnp.mean((inputs - stop_grad(quantized)) ** 2)
        codebook_loss = jnp.mean((stop_grad(inputs) - quantized) ** 2)
        
        # Straight-through estimator
        if self.config.enable_straight_through and HAS_JAX and lax is not None:
            quantized = inputs + stop_grad(quantized - inputs)
        
        # Update codebook usage
        if self.config.enable_ema_update:
            self._update_codebook_ema(flat_inputs, encoding_indices)
        
        info = {
            'encoding_indices': encoding_indices.reshape(input_shape[:-1]),
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'perplexity': self._compute_perplexity(encoding_indices),
            'codebook_usage': self.codebook_usage
        }
        
        return quantized, info
    
    def _compute_distances(self, inputs: Any, codebook: Any) -> Any:
        """Compute distances between inputs and codebook."""
        # L2 distance
        input_norm = jnp.sum(inputs**2, axis=1, keepdims=True)
        codebook_norm = jnp.sum(codebook**2, axis=1)
        dot_product = jnp.dot(inputs, codebook.T)
        
        distances = input_norm + codebook_norm - 2 * dot_product
        return distances
    
    def _update_codebook_ema(self, inputs: Any, indices: Any):
        """Update codebook using exponential moving average."""
        # This would typically be done with proper JAX state management
        # For now, we'll skip the current update in this mock implementation
        pass
    
    def _compute_perplexity(self, indices: Any) -> float:
        """Compute perplexity of the encoding distribution."""
        # Count usage of each codebook entry
        if HAS_JAX:
            counts = jnp.bincount(indices, length=self.config.codebook_size)
        else:
            counts = jnp.bincount(indices, minlength=self.config.codebook_size)
        probs = counts / jnp.sum(counts)
        
        # Avoid log(0)
        probs = jnp.where(probs > 0, probs, 1e-10)
        
        # Compute perplexity
        entropy = -jnp.sum(probs * jnp.log(probs))
        perplexity = jnp.exp(entropy)
        
        return float(perplexity)
    
    def compress(self, inputs: Any, target_ratio: Optional[float] = None) -> Tuple[Any, Dict[str, Any]]:
        """Compress inputs using VQ compression."""
        if target_ratio is None:
            target_ratio = self.config.compression_ratio
        
        # Apply quantization
        quantized, info = self.quantize(inputs)
        
        # Apply compression based on mode
        if self.config.compression_mode == VQCompressionMode.ADAPTIVE:
            compressed = self._adaptive_compress(quantized, target_ratio)
        elif self.config.compression_mode == VQCompressionMode.HIERARCHICAL:
            compressed = self._hierarchical_compress(quantized, target_ratio)
        elif self.config.compression_mode == VQCompressionMode.RESIDUAL:
            compressed = self._residual_compress(inputs, quantized, target_ratio)
        else:
            compressed = quantized
        
        compression_info = {
            'compression_ratio': self._compute_compression_ratio(inputs, compressed),
            'original_size': inputs.size,
            'compressed_size': compressed.size if hasattr(compressed, 'size') else len(compressed)
        }
        
        info.update(compression_info)
        return compressed, info
    
    def _adaptive_compress(self, quantized: Any, target_ratio: float) -> Any:
        """Adaptive compression based on importance."""
        # simple importance-based compression (mock implementation)
        importance = jnp.abs(quantized)
        threshold = jnp.percentile(importance, (1 - target_ratio) * 100)
        compressed = jnp.where(importance > threshold, quantized, 0)
        return compressed
    
    def _hierarchical_compress(self, quantized: Any, target_ratio: float) -> Any:
        """Hierarchical compression with multiple levels."""
        # Multi-level quantization using JAX fori_loop for JIT compatibility
        level_ratio = target_ratio ** (1 / self.config.num_levels)

        if HAS_JAX and lax is not None:
            # Use lax.fori_loop for JIT compilation
            def body_fn(_, compressed):
                return self._adaptive_compress(compressed, level_ratio)
            return lax.fori_loop(0, self.config.num_levels, body_fn, quantized)
        else:
            # Fallback for non-JAX
            compressed = quantized
            for _ in range(self.config.num_levels):
                compressed = self._adaptive_compress(compressed, level_ratio)
            return compressed
    
    def _residual_compress(self, original: Any, quantized: Any, target_ratio: float) -> Any:
        """Residual compression of quantization error."""
        residual = original - quantized
        compressed_residual = self._adaptive_compress(residual, target_ratio)
        return quantized + compressed_residual
    
    def _compute_compression_ratio(self, original: Any, compressed: Any) -> float:
        """Compute current compression ratio achieved."""
        if hasattr(compressed, 'size') and hasattr(original, 'size'):
            return compressed.size / original.size
        return 1.0
    
    def decompress(self, compressed: Any, original_shape: Optional[Tuple[int, ...]] = None) -> Any:
        """Decompress VQ-compressed data."""
        # For this mock implementation, just return the compressed data
        # In a real implementation, this would reconstruct from indices
        if original_shape is not None and hasattr(compressed, 'reshape'):
            return compressed.reshape(original_shape)
        return compressed
    
    def get_codebook_stats(self) -> Dict[str, Any]:
        """Get statistics about the codebook."""
        if not self.initialized:
            return {}
        
        return {
            'codebook_size': self.config.codebook_size,
            'embedding_dim': self.config.embedding_dim,
            'usage_mean': float(jnp.mean(self.codebook_usage)),
            'usage_std': float(jnp.std(self.codebook_usage)),
            'usage_min': float(jnp.min(self.codebook_usage)),
            'usage_max': float(jnp.max(self.codebook_usage)),
            'active_entries': int(jnp.sum(self.codebook_usage > 0))
        }

# Factory functions
def create_vqbit_layer(
    codebook_size: int = 512,
    embedding_dim: int = 64,
    compression_mode: str = "standard",
    **kwargs
) -> VQbitLayer:
    """Create a VQbit layer with specified configuration."""
    
    config = VQbitConfig(
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        compression_mode=VQCompressionMode(compression_mode),
        **kwargs
    )
    
    return VQbitLayer(config)

def create_ultra_vqbit_layer(**kwargs) -> VQbitLayer:
    """Create an ultra-compressed VQbit layer."""
    
    ultra_config = VQbitConfig(
        codebook_size=1024,
        embedding_dim=128,
        compression_mode=VQCompressionMode.HIERARCHICAL,
        compression_ratio=0.05,
        enable_adaptive_codebook=True,
        num_levels=3,
        **kwargs
    )
    
    return VQbitLayer(ultra_config)

# Utility functions
if HAS_JAX:
    quantize_jit = jit(lambda layer, inputs: layer.quantize(inputs))
    compress_jit = jit(lambda layer, inputs: layer.compress(inputs))
else:
    quantize_jit = lambda layer, inputs: layer.quantize(inputs)
    compress_jit = lambda layer, inputs: layer.compress(inputs)

__all__ = [
    'VQbitLayer',
    'VQbitConfig', 
    'VQCompressionMode',
    'create_vqbit_layer',
    'create_ultra_vqbit_layer',
    'quantize_jit',
    'compress_jit'
]

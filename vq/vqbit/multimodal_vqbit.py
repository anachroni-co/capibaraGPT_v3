"""
Multimodal VQbit Module for Capibara-6

Advanced multimodal vector quantization supporting:
- Cross-modal quantization (text, image, audio)
- Shared codebook learning across modalities
- Modal-specific adaptations
- Fusion strategies for multimodal representations
- Cache-aware processing for adaptive systems
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math

# JAX/Flax imports (preferred for TPU)
try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    from flax import struct
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    nn = None

# PyTorch fallback
try:
    import torch
    import torch.nn as torch_nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    torch_nn = None
    F = None

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MultimodalVQConfig:
    """Configurestion for Multimodal VQbit."""
    # Core VQ parameters
    codebook_size: int = 64
    embedding_dim: int = 768
    commitment_weight: float = 0.25
    
    # Multimodal settings
    modalities: List[str] = None  # ['text', 'image', 'audio']
    shared_codebook: bool = True
    modal_adaptation: bool = True
    fusion_strategy: str = "concat"  # concat, add, attention
    
    # Cache settings for adaptive compatibility
    adaptive_compatible: bool = False
    cache_size_ratio: float = 0.4
    tile_size: int = 256
    enable_prefetch: bool = True
    
    # EMA settings
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_epsilon: float = 1e-5
    
    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ['text', 'image', 'audio']


if JAX_AVAILABLE:
    class VQbitModule(nn.Module):
        """
        Multimodal VQbit Module - JAX/Flax implementation.
        
        Features:
        - Multimodal vector quantization
        - Shared or separate codebooks per modality
        - Modal adaptation layers
        - Fusion strategies for cross-modal representations
        - Cache-aware processing for adaptive systems
        """
        
        config: MultimodalVQConfig
        
        def setup(self):
            """Initialize the multimodal VQbit module."""
            num_modalities = len(self.config.modalities)
            
            # Modal adaptation layers
            if self.config.modal_adaptation:
                self.modal_adapters = {}
                for modality in self.config.modalities:
                    self.modal_adapters[modality] = nn.Dense(
                        self.config.embedding_dim,
                        name=f"adapter_{modality}"
                    )
            
            # Codebook initialization
            if self.config.shared_codebook:
                # Single shared codebook
                self.codebook = self.param(
                    'shared_codebook',
                    nn.initializers.uniform(scale=1.0),
                    (self.config.codebook_size, self.config.embedding_dim)
                )
            else:
                # Separate codebooks per modality
                self.codebooks = {}
                for modality in self.config.modalities:
                    self.codebooks[modality] = self.param(
                        f'codebook_{modality}',
                        nn.initializers.uniform(scale=1.0),
                        (self.config.codebook_size, self.config.embedding_dim)
                    )
            
            # Fusion layer
            if self.config.fusion_strategy == "attention":
                self.fusion_attention = nn.MultiHeadDotProductAttention(
                    num_heads=8,
                    qkv_features=self.config.embedding_dim
                )
            elif self.config.fusion_strategy == "concat":
                self.fusion_projection = nn.Dense(self.config.embedding_dim)
            
            # EMA variables if enabled
            if self.config.use_ema:
                if self.config.shared_codebook:
                    self.ema_cluster_size = self.variable(
                        'ema_stats', 'cluster_size',
                        lambda: jnp.zeros(self.config.codebook_size)
                    )
                    self.ema_embedding_avg = self.variable(
                        'ema_stats', 'embedding_avg',
                        lambda: jnp.zeros((self.config.codebook_size, self.config.embedding_dim))
                    )
                else:
                    self.ema_stats = {}
                    for modality in self.config.modalities:
                        self.ema_stats[modality] = {
                            'cluster_size': self.variable(
                                'ema_stats', f'cluster_size_{modality}',
                                lambda: jnp.zeros(self.config.codebook_size)
                            ),
                            'embedding_avg': self.variable(
                                'ema_stats', f'embedding_avg_{modality}',
                                lambda: jnp.zeros((self.config.codebook_size, self.config.embedding_dim))
                            )
                        }
        
        def __call__(self, 
                    inputs: Union[jnp.ndarray, Dict[str, jnp.ndarray]], 
                    modality: Optional[str] = None,
                    training: bool = True,
                    **kwargs) -> Dict[str, Any]:
            """
            Forward pass of multimodal VQbit module.
            
            Args:
                inputs: Input embeddings (single array or dict of modality arrays)
                modality: Modality name if single input
                training: Whether in training mode
                **kwargs: Additional arguments (for adaptive compatibility)
                
            Returns:
                Dictionary with quantized outputs, indices, and metrics
            """
            
            if isinstance(inputs, dict):
                # Multiple modalities
                return self._process_multimodal(inputs, training)
            else:
                # Single modality
                if modality is None:
                    modality = self.config.modalities[0]  # Default to first modality
                return self._process_single_modality(inputs, modality, training)
        
        def _process_single_modality(self, 
                                   inputs: jnp.ndarray, 
                                   modality: str, 
                                   training: bool) -> Dict[str, Any]:
            """Process single modality input."""
            
            # Modal adaptation
            if self.config.modal_adaptation and modality in self.modal_adapters:
                adapted_inputs = self.modal_adapters[modality](inputs)
            else:
                adapted_inputs = inputs
            
            # Get appropriate codebook
            if self.config.shared_codebook:
                codebook = self.codebook
            else:
                codebook = self.codebooks[modality]
            
            # Quantize
            input_shape = adapted_inputs.shape
            flat_inputs = adapted_inputs.reshape(-1, self.config.embedding_dim)
            
            # Compute distances and quantize
            distances = jnp.linalg.norm(
                flat_inputs[:, None, :] - codebook[None, :, :],
                axis=2
            )
            
            indices = jnp.argmin(distances, axis=1)
            quantized = codebook[indices]
            
            # Compute losses
            commitment_loss = jnp.mean((jax.lax.stop_gradient(quantized) - flat_inputs) ** 2)
            codebook_loss = jnp.mean((quantized - jax.lax.stop_gradient(flat_inputs)) ** 2)
            
            # Straight-through estimator
            quantized = flat_inputs + jax.lax.stop_gradient(quantized - flat_inputs)
            
            # Reshape back
            quantized = quantized.reshape(input_shape)
            indices = indices.reshape(input_shape[:-1])
            
            # Compute metrics
            metrics = self._compute_metrics(flat_inputs, quantized.reshape(-1, self.config.embedding_dim), 
                                          indices.reshape(-1), commitment_loss, codebook_loss, modality)
            
            return {
                'quantized': quantized,
                'indices': indices,
                'metrics': metrics,
                'modality': modality,
                'cache_info': self._get_cache_info() if self.config.adaptive_compatible else {}
            }
        
        def _process_multimodal(self, 
                              inputs: Dict[str, jnp.ndarray], 
                              training: bool) -> Dict[str, Any]:
            """Process multiple modalities and fuse results."""
            
            modal_results = {}
            all_quantized = []
            all_metrics = {}
            
            # Process each modality
            for modality, modal_input in inputs.items():
                if modality in self.config.modalities:
                    result = self._process_single_modality(modal_input, modality, training)
                    modal_results[modality] = result
                    all_quantized.append(result['quantized'])
                    all_metrics[f"{modality}_metrics"] = result['metrics']
            
            # Fuse modalities
            if len(all_quantized) > 1:
                fused_quantized = self._fuse_modalities(all_quantized)
            else:
                fused_quantized = all_quantized[0] if all_quantized else jnp.zeros((1, self.config.embedding_dim))
            
            # Compute global metrics
            global_metrics = self._compute_global_metrics(modal_results)
            all_metrics['global'] = global_metrics
            
            return {
                'quantized': fused_quantized,
                'modal_results': modal_results,
                'metrics': all_metrics,
                'cache_info': self._get_cache_info() if self.config.adaptive_compatible else {}
            }
        
        def _fuse_modalities(self, quantized_list: List[jnp.ndarray]) -> jnp.ndarray:
            """Fuse quantized representations from multiple modalities."""
            
            if self.config.fusion_strategy == "concat":
                # Concatenate and project
                concatenated = jnp.concatenate(quantized_list, axis=-1)
                if hasattr(self, 'fusion_projection'):
                    return self.fusion_projection(concatenated)
                else:
                    # Simple average if no projection
                    return jnp.mean(jnp.stack(quantized_list), axis=0)
                    
            elif self.config.fusion_strategy == "add":
                # Element-wise addition
                return jnp.sum(jnp.stack(quantized_list), axis=0)
                
            elif self.config.fusion_strategy == "attention":
                # Attention-based fusion
                stacked = jnp.stack(quantized_list, axis=1)  # [batch, num_modalities, dim]
                if hasattr(self, 'fusion_attention'):
                    fused = self.fusion_attention(stacked, stacked)
                    return jnp.mean(fused, axis=1)  # Average over modalities
                else:
                    return jnp.mean(stacked, axis=1)
            
            else:
                # Default: average
                return jnp.mean(jnp.stack(quantized_list), axis=0)
        
        def _compute_metrics(self, inputs: jnp.ndarray, quantized: jnp.ndarray,
                           indices: jnp.ndarray, commitment_loss: float,
                           codebook_loss: float, modality: str) -> Dict[str, Any]:
            """Compute quantization metrics for a modality."""
            
            # Basic metrics
            mse = jnp.mean((inputs - quantized) ** 2)
            
            # Codebook usage
            unique_indices = len(jnp.unique(indices))
            codebook_usage = unique_indices / self.config.codebook_size
            
            # Perplexity
            avg_probs = jnp.mean(jax.nn.one_hot(indices, self.config.codebook_size), axis=0)
            perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))
            
            # Compression ratio
            original_bits = inputs.size * 32  # float32
            quantized_bits = indices.size * math.ceil(math.log2(self.config.codebook_size))
            compression_ratio = original_bits / quantized_bits
            
            return {
                'modality': modality,
                'commitment_loss': commitment_loss,
                'codebook_loss': codebook_loss,
                'mse': mse,
                'codebook_usage': codebook_usage,
                'perplexity': perplexity,
                'compression_ratio': compression_ratio,
                'unique_codes': unique_indices
            }
        
        def _compute_global_metrics(self, modal_results: Dict[str, Any]) -> Dict[str, Any]:
            """Compute global metrics across all modalities."""
            
            if not modal_results:
                return {'total_modalities': 0}
            
            # Average metrics across modalities
            total_commitment = 0.0
            total_codebook = 0.0
            total_mse = 0.0
            total_usage = 0.0
            total_perplexity = 0.0
            
            for result in modal_results.values():
                metrics = result['metrics']
                total_commitment += metrics['commitment_loss']
                total_codebook += metrics['codebook_loss']
                total_mse += metrics['mse']
                total_usage += metrics['codebook_usage']
                total_perplexity += metrics['perplexity']
            
            num_modalities = len(modal_results)
            
            return {
                'total_modalities': num_modalities,
                'avg_commitment_loss': total_commitment / num_modalities,
                'avg_codebook_loss': total_codebook / num_modalities,
                'avg_mse': total_mse / num_modalities,
                'avg_codebook_usage': total_usage / num_modalities,
                'avg_perplexity': total_perplexity / num_modalities,
                'fusion_strategy': self.config.fusion_strategy
            }
        
        def _get_cache_info(self) -> Dict[str, Any]:
            """Get cache information for adaptive compatibility."""
            return {
                'cache_size_ratio': self.config.cache_size_ratio,
                'tile_size': self.config.tile_size,
                'prefetch_enabled': self.config.enable_prefetch,
                'adaptive_compatible': self.config.adaptive_compatible
            }

elif TORCH_AVAILABLE:
    class VQbitModule(torch_nn.Module):
        """
        Multimodal VQbit Module - PyTorch implementation.
        """
        
        def __init__(self, config: Optional[MultimodalVQConfig] = None, **kwargs):
            super().__init__()
            if config is None:
                # Extract config parameters from kwargs
                config_dict = {}
                for key in ['codebook_size', 'embedding_dim', 'commitment_weight', 
                           'modalities', 'shared_codebook', 'modal_adaptation',
                           'fusion_strategy', 'adaptive_compatible', 'cache_size_ratio',
                           'tile_size', 'enable_prefetch']:
                    if key in kwargs:
                        config_dict[key] = kwargs.pop(key)
                config = MultimodalVQConfig(**config_dict)
            self.config = config
            
            # Modal adaptation layers
            if self.config.modal_adaptation:
                self.modal_adapters = torch_nn.ModuleDict()
                for modality in self.config.modalities:
                    self.modal_adapters[modality] = torch_nn.Linear(
                        self.config.embedding_dim, self.config.embedding_dim
                    )
            
            # Codebook initialization
            if self.config.shared_codebook:
                self.codebook = torch_nn.Parameter(
                    torch.randn(self.config.codebook_size, self.config.embedding_dim)
                )
            else:
                self.codebooks = torch_nn.ParameterDict()
                for modality in self.config.modalities:
                    self.codebooks[modality] = torch_nn.Parameter(
                        torch.randn(self.config.codebook_size, self.config.embedding_dim)
                    )
            
            # Fusion layer
            if self.config.fusion_strategy == "attention":
                self.fusion_attention = torch_nn.MultiheadAttention(
                    self.config.embedding_dim, num_heads=8, batch_first=True
                )
            elif self.config.fusion_strategy == "concat":
                fusion_input_dim = len(self.config.modalities) * self.config.embedding_dim
                self.fusion_projection = torch_nn.Linear(fusion_input_dim, self.config.embedding_dim)
            
            # EMA buffers
            if self.config.use_ema:
                if self.config.shared_codebook:
                    self.register_buffer('ema_cluster_size', torch.zeros(self.config.codebook_size))
                    self.register_buffer('ema_embedding_avg', torch.zeros(self.config.codebook_size, self.config.embedding_dim))
                else:
                    for modality in self.config.modalities:
                        self.register_buffer(f'ema_cluster_size_{modality}', torch.zeros(self.config.codebook_size))
                        self.register_buffer(f'ema_embedding_avg_{modality}', torch.zeros(self.config.codebook_size, self.config.embedding_dim))
        
        def forward(self, 
                   inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                   modality: Optional[str] = None,
                   training: bool = True,
                   **kwargs) -> Dict[str, Any]:
            """Forward pass of multimodal VQbit module."""
            
            if isinstance(inputs, dict):
                return self._process_multimodal(inputs, training)
            else:
                if modality is None:
                    modality = self.config.modalities[0]
                return self._process_single_modality(inputs, modality, training)
        
        def _process_single_modality(self, 
                                   inputs: torch.Tensor, 
                                   modality: str, 
                                   training: bool) -> Dict[str, Any]:
            """Process single modality input."""
            
            # Modal adaptation
            if self.config.modal_adaptation and modality in self.modal_adapters:
                adapted_inputs = self.modal_adapters[modality](inputs)
            else:
                adapted_inputs = inputs
            
            # Get appropriate codebook
            if self.config.shared_codebook:
                codebook = self.codebook
            else:
                codebook = self.codebooks[modality]
            
            # Quantize
            input_shape = adapted_inputs.shape
            flat_inputs = adapted_inputs.view(-1, self.config.embedding_dim)
            
            # Compute distances and quantize
            distances = torch.norm(
                flat_inputs.unsqueeze(1) - codebook.unsqueeze(0),
                dim=2
            )
            
            indices = torch.argmin(distances, dim=1)
            quantized = codebook[indices]
            
            # Compute losses
            commitment_loss = F.mse_loss(quantized.detach(), flat_inputs)
            codebook_loss = F.mse_loss(quantized, flat_inputs.detach())
            
            # Straight-through estimator
            quantized = flat_inputs + (quantized - flat_inputs).detach()
            
            # Reshape back
            quantized = quantized.view(input_shape)
            indices = indices.view(input_shape[:-1])
            
            # Compute metrics
            metrics = self._compute_metrics(flat_inputs, quantized.view(-1, self.config.embedding_dim), 
                                          indices.view(-1), commitment_loss, codebook_loss, modality)
            
            return {
                'quantized': quantized,
                'indices': indices,
                'metrics': metrics,
                'modality': modality,
                'cache_info': self._get_cache_info() if self.config.adaptive_compatible else {}
            }
        
        def _process_multimodal(self, 
                              inputs: Dict[str, torch.Tensor], 
                              training: bool) -> Dict[str, Any]:
            """Process multiple modalities and fuse results."""
            
            modal_results = {}
            all_quantized = []
            all_metrics = {}
            
            # Process each modality
            for modality, modal_input in inputs.items():
                if modality in self.config.modalities:
                    result = self._process_single_modality(modal_input, modality, training)
                    modal_results[modality] = result
                    all_quantized.append(result['quantized'])
                    all_metrics[f"{modality}_metrics"] = result['metrics']
            
            # Fuse modalities
            if len(all_quantized) > 1:
                fused_quantized = self._fuse_modalities(all_quantized)
            else:
                fused_quantized = all_quantized[0] if all_quantized else torch.zeros(1, self.config.embedding_dim)
            
            # Compute global metrics
            global_metrics = self._compute_global_metrics(modal_results)
            all_metrics['global'] = global_metrics
            
            return {
                'quantized': fused_quantized,
                'modal_results': modal_results,
                'metrics': all_metrics,
                'cache_info': self._get_cache_info() if self.config.adaptive_compatible else {}
            }
        
        def _fuse_modalities(self, quantized_list: List[torch.Tensor]) -> torch.Tensor:
            """Fuse quantized representations from multiple modalities."""
            
            if self.config.fusion_strategy == "concat":
                # Concatenate and project
                concatenated = torch.cat(quantized_list, dim=-1)
                if hasattr(self, 'fusion_projection'):
                    return self.fusion_projection(concatenated)
                else:
                    return torch.mean(torch.stack(quantized_list), dim=0)
                    
            elif self.config.fusion_strategy == "add":
                return torch.sum(torch.stack(quantized_list), dim=0)
                
            elif self.config.fusion_strategy == "attention":
                stacked = torch.stack(quantized_list, dim=1)  # [batch, num_modalities, dim]
                if hasattr(self, 'fusion_attention'):
                    fused, _ = self.fusion_attention(stacked, stacked, stacked)
                    return torch.mean(fused, dim=1)
                else:
                    return torch.mean(stacked, dim=1)
            
            else:
                return torch.mean(torch.stack(quantized_list), dim=0)
        
        def _compute_metrics(self, inputs: torch.Tensor, quantized: torch.Tensor,
                           indices: torch.Tensor, commitment_loss: torch.Tensor,
                           codebook_loss: torch.Tensor, modality: str) -> Dict[str, Any]:
            """Compute quantization metrics for a modality."""
            
            # Basic metrics
            mse = F.mse_loss(inputs, quantized)
            
            # Codebook usage
            unique_indices = len(torch.unique(indices))
            codebook_usage = unique_indices / self.config.codebook_size
            
            # Perplexity
            one_hot = F.one_hot(indices, self.config.codebook_size).float()
            avg_probs = torch.mean(one_hot, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            
            return {
                'modality': modality,
                'commitment_loss': commitment_loss.item(),
                'codebook_loss': codebook_loss.item(),
                'mse': mse.item(),
                'codebook_usage': codebook_usage.item(),
                'perplexity': perplexity.item(),
                'unique_codes': unique_indices
            }
        
        def _compute_global_metrics(self, modal_results: Dict[str, Any]) -> Dict[str, Any]:
            """Compute global metrics across all modalities."""
            
            if not modal_results:
                return {'total_modalities': 0}
            
            total_commitment = 0.0
            total_codebook = 0.0
            total_mse = 0.0
            total_usage = 0.0
            total_perplexity = 0.0
            
            for result in modal_results.values():
                metrics = result['metrics']
                total_commitment += metrics['commitment_loss']
                total_codebook += metrics['codebook_loss']
                total_mse += metrics['mse']
                total_usage += metrics['codebook_usage']
                total_perplexity += metrics['perplexity']
            
            num_modalities = len(modal_results)
            
            return {
                'total_modalities': num_modalities,
                'avg_commitment_loss': total_commitment / num_modalities,
                'avg_codebook_loss': total_codebook / num_modalities,
                'avg_mse': total_mse / num_modalities,
                'avg_codebook_usage': total_usage / num_modalities,
                'avg_perplexity': total_perplexity / num_modalities,
                'fusion_strategy': self.config.fusion_strategy
            }
        
        def _get_cache_info(self) -> Dict[str, Any]:
            """Get cache information for adaptive compatibility."""
            return {
                'cache_size_ratio': self.config.cache_size_ratio,
                'tile_size': self.config.tile_size,
                'prefetch_enabled': self.config.enable_prefetch,
                'adaptive_compatible': self.config.adaptive_compatible
            }

else:
    # Fallback implementation using NumPy
    class VQbitModule:
        """
        Multimodal VQbit Module - NumPy fallback implementation.
        """
        
        def __init__(self, config: Optional[MultimodalVQConfig] = None, **kwargs):
            if config is None:
                config_dict = {}
                for key in ['codebook_size', 'embedding_dim', 'commitment_weight', 
                           'modalities', 'shared_codebook', 'modal_adaptation',
                           'fusion_strategy', 'adaptive_compatible', 'cache_size_ratio',
                           'tile_size', 'enable_prefetch']:
                    if key in kwargs:
                        config_dict[key] = kwargs.pop(key)
                config = MultimodalVQConfig(**config_dict)
            self.config = config
            
            # Initialize codebooks
            if self.config.shared_codebook:
                self.codebook = np.random.randn(self.config.codebook_size, self.config.embedding_dim).astype(np.float32)
            else:
                self.codebooks = {}
                for modality in self.config.modalities:
                    self.codebooks[modality] = np.random.randn(
                        self.config.codebook_size, self.config.embedding_dim
                    ).astype(np.float32)
        
        def __call__(self, 
                    inputs: Union[np.ndarray, Dict[str, np.ndarray]], 
                    modality: Optional[str] = None,
                    training: bool = True,
                    **kwargs) -> Dict[str, Any]:
            """Forward pass (simplified NumPy implementation)."""
            
            if isinstance(inputs, dict):
                # Simple multimodal processing
                results = {}
                for mod, inp in inputs.items():
                    if mod in self.config.modalities:
                        results[mod] = self._process_single(inp, mod)
                
                # Simple fusion (average)
                if results:
                    all_quantized = [r['quantized'] for r in results.values()]
                    fused = np.mean(np.stack(all_quantized), axis=0)
                else:
                    fused = np.zeros((1, self.config.embedding_dim))
                
                return {
                    'quantized': fused,
                    'modal_results': results,
                    'metrics': {'total_modalities': len(results)},
                    'cache_info': self._get_cache_info() if self.config.adaptive_compatible else {}
                }
            else:
                if modality is None:
                    modality = self.config.modalities[0]
                return self._process_single(inputs, modality)
        
        def _process_single(self, inputs: np.ndarray, modality: str) -> Dict[str, Any]:
            """Process single modality (simplified)."""
            
            # Get codebook
            if self.config.shared_codebook:
                codebook = self.codebook
            else:
                codebook = self.codebooks.get(modality, self.codebook if hasattr(self, 'codebook') else np.random.randn(self.config.codebook_size, self.config.embedding_dim))
            
            # Simple quantization
            input_shape = inputs.shape
            flat_inputs = inputs.reshape(-1, self.config.embedding_dim)
            
            distances = np.linalg.norm(
                flat_inputs[:, None, :] - codebook[None, :, :],
                axis=2
            )
            
            indices = np.argmin(distances, axis=1)
            quantized = codebook[indices].reshape(input_shape)
            indices = indices.reshape(input_shape[:-1])
            
            # Basic metrics
            unique_codes = len(np.unique(indices))
            
            return {
                'quantized': quantized,
                'indices': indices,
                'metrics': {
                    'modality': modality,
                    'unique_codes': unique_codes,
                    'codebook_usage': unique_codes / self.config.codebook_size
                },
                'cache_info': self._get_cache_info() if self.config.adaptive_compatible else {}
            }
        
        def _get_cache_info(self) -> Dict[str, Any]:
            """Get cache information for adaptive compatibility."""
            return {
                'cache_size_ratio': self.config.cache_size_ratio,
                'tile_size': self.config.tile_size,
                'prefetch_enabled': self.config.enable_prefetch,
                'adaptive_compatible': self.config.adaptive_compatible
            }


# Factory functions
def create_multimodal_vqbit(modalities: List[str] = None,
                           codebook_size: int = 64,
                           embedding_dim: int = 768,
                           shared_codebook: bool = True,
                           fusion_strategy: str = "concat",
                           **kwargs) -> VQbitModule:
    """
    Create a multimodal VQbit module.
    
    Args:
        modalities: List of supported modalities
        codebook_size: Number of codebook entries
        embedding_dim: Dimension of embeddings
        shared_codebook: Whether to use shared codebook across modalities
        fusion_strategy: Strategy for fusing modalities
        **kwargs: Additional configuration
        
    Returns:
        VQbitModule instance
    """
    if modalities is None:
        modalities = ['text', 'image', 'audio']
    
    config = MultimodalVQConfig(
        modalities=modalities,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        shared_codebook=shared_codebook,
        fusion_strategy=fusion_strategy,
        **kwargs
    )
    
    return VQbitModule(config)


logger.info(f"Multimodal VQbit initialized - JAX: {JAX_AVAILABLE}, PyTorch: {TORCH_AVAILABLE}")

def main():
    # Main function for this module.
    logger.info("Module multimodal_vqbit.py starting")
    return True

if __name__ == "__main__":
    main()

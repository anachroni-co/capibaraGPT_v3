"""
VQbit Layer Implementation for Capibara-6

Core Vector Quantization layer with support for:
- JAX/Flax implementation for TPU optimization
- PyTorch fallback for CPU/GPU
- Adaptive codebook management
- Performance monitoring
- Integration with MoE and Adaptive systems
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
class VQbitConfig:
    """Configurestion for VQbit Layer."""
    codebook_size: int = 64
    embedding_dim: int = 768
    commitment_weight: float = 0.25
    use_tpu_optimizations: bool = True
    diversity_regularization: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_epsilon: float = 1e-5
    use_product_quantization: bool = False
    num_subspaces: int = 8


if JAX_AVAILABLE:
    class VQbitLayer(nn.Module):
        """
        VQbit Layer - JAX/Flax implementation for TPU optimization.
        
        Features:
        - Vector quantization with learnable codebook
        - EMA updates for stable training
        - Commitment loss and diversity regularization
        - TPU-optimized operations
        - Product quantization support
        """
        
        config: VQbitConfig
        
        def setup(self):
            """Initialize the VQbit layer."""
            # Initialize codebook
            if self.config.use_product_quantization:
                # Product quantization: split embedding into subspaces
                subspace_dim = self.config.embedding_dim // self.config.num_subspaces
                codebook_shape = (self.config.num_subspaces, self.config.codebook_size, subspace_dim)
            else:
                codebook_shape = (self.config.codebook_size, self.config.embedding_dim)
                
            self.codebook = self.param(
                'codebook',
                nn.initializers.uniform(scale=1.0),
                codebook_shape
            )
            
            if self.config.use_ema:
                # EMA variables for stable training
                self.ema_cluster_size = self.variable(
                    'ema_stats', 'cluster_size',
                    lambda: jnp.zeros(self.config.codebook_size)
                )
                self.ema_embedding_avg = self.variable(
                    'ema_stats', 'embedding_avg',
                    lambda: jnp.zeros_like(self.codebook)
                )
        
        def __call__(self, inputs: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
            """
            Forward pass of VQbit layer.
            
            Args:
                inputs: Input embeddings [batch, ..., embedding_dim]
                training: Whether in training mode
                
            Returns:
                quantized: Quantized embeddings
                indices: Codebook indices
                metrics: Quantization metrics
            """
            
            input_shape = inputs.shape
            flat_inputs = inputs.reshape(-1, self.config.embedding_dim)
            
            if self.config.use_product_quantization:
                quantized, indices, metrics = self._product_quantize(flat_inputs, training)
            else:
                quantized, indices, metrics = self._standard_quantize(flat_inputs, training)
            
            # Reshape back to original shape
            quantized = quantized.reshape(input_shape)
            indices = indices.reshape(input_shape[:-1])
            
            return quantized, indices, metrics
        
        def _standard_quantize(self, flat_inputs: jnp.ndarray, training: bool) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
            """Standard vector quantization."""
            # Compute distances to codebook
            distances = jnp.linalg.norm(
                flat_inputs[:, None, :] - self.codebook[None, :, :], 
                axis=2
            )
            
            # Find nearest codebook entries
            indices = jnp.argmin(distances, axis=1)
            quantized = self.codebook[indices]
            
            # Compute losses
            commitment_loss = jnp.mean((jax.lax.stop_gradient(quantized) - flat_inputs) ** 2)
            codebook_loss = jnp.mean((quantized - jax.lax.stop_gradient(flat_inputs)) ** 2)
            
            # Straight-through estimator
            quantized = flat_inputs + jax.lax.stop_gradient(quantized - flat_inputs)
            
            # Update EMA statistics if training
            if training and self.config.use_ema:
                self._update_ema_stats(flat_inputs, indices)
            
            # Compute metrics
            metrics = self._compute_metrics(flat_inputs, quantized, indices, commitment_loss, codebook_loss)
            
            return quantized, indices, metrics
        
        def _product_quantize(self, flat_inputs: jnp.ndarray, training: bool) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
            """Product quantization for memory efficiency."""
            batch_size = flat_inputs.shape[0]
            subspace_dim = self.config.embedding_dim // self.config.num_subspaces
            
            # Reshape for product quantization
            inputs_reshaped = flat_inputs.reshape(batch_size, self.config.num_subspaces, subspace_dim)
            
            quantized_subspaces = []
            indices_subspaces = []
            total_commitment_loss = 0.0
            total_codebook_loss = 0.0
            
            for i in range(self.config.num_subspaces):
                subspace_input = inputs_reshaped[:, i, :]
                subspace_codebook = self.codebook[i]
                
                # Quantize subspace
                distances = jnp.linalg.norm(
                    subspace_input[:, None, :] - subspace_codebook[None, :, :],
                    axis=2
                )
                subspace_indices = jnp.argmin(distances, axis=1)
                subspace_quantized = subspace_codebook[subspace_indices]
                
                # Compute subspace losses
                commitment_loss = jnp.mean((jax.lax.stop_gradient(subspace_quantized) - subspace_input) ** 2)
                codebook_loss = jnp.mean((subspace_quantized - jax.lax.stop_gradient(subspace_input)) ** 2)
                
                total_commitment_loss += commitment_loss
                total_codebook_loss += codebook_loss
                
                # Straight-through estimator
                subspace_quantized = subspace_input + jax.lax.stop_gradient(subspace_quantized - subspace_input)
                
                quantized_subspaces.append(subspace_quantized)
                indices_subspaces.append(subspace_indices)
            
            # Combine subspaces
            quantized = jnp.concatenate(quantized_subspaces, axis=1)
            indices = jnp.stack(indices_subspaces, axis=1)
            
            # Average losses
            commitment_loss = total_commitment_loss / self.config.num_subspaces
            codebook_loss = total_codebook_loss / self.config.num_subspaces
            
            # Compute metrics
            metrics = self._compute_metrics(flat_inputs, quantized, indices, commitment_loss, codebook_loss)
            
            return quantized, indices, metrics
        
        def _update_ema_stats(self, flat_inputs: jnp.ndarray, indices: jnp.ndarray):
            """Update EMA statistics for stable training."""
            # Update cluster sizes
            one_hot = jax.nn.one_hot(indices, self.config.codebook_size)
            cluster_size = jnp.sum(one_hot, axis=0)
            
            self.ema_cluster_size.value = (
                self.config.ema_decay * self.ema_cluster_size.value +
                (1 - self.config.ema_decay) * cluster_size
            )
            
            # Update embedding averages
            embedding_sum = jnp.dot(one_hot.T, flat_inputs)
            self.ema_embedding_avg.value = (
                self.config.ema_decay * self.ema_embedding_avg.value +
                (1 - self.config.ema_decay) * embedding_sum
            )
        
        def _compute_metrics(self, inputs: jnp.ndarray, quantized: jnp.ndarray, 
                           indices: jnp.ndarray, commitment_loss: float, 
                           codebook_loss: float) -> Dict[str, Any]:
            """Compute quantization metrics."""
            
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
                'commitment_loss': commitment_loss,
                'codebook_loss': codebook_loss,
                'mse': mse,
                'codebook_usage': codebook_usage,
                'perplexity': perplexity,
                'compression_ratio': compression_ratio,
                'unique_codes': unique_indices
            }
        
        @property
        def codebook_size(self) -> int:
            """Get codebook size."""
            return self.config.codebook_size

elif TORCH_AVAILABLE:
    class VQbitLayer(torch_nn.Module):
        """
        VQbit Layer - PyTorch implementation for CPU/GPU fallback.
        """
        
        def __init__(self, config: Optional[VQbitConfig] = None, **kwargs):
            super().__init__()
            if config is None:
                config = VQbitConfig(**kwargs)
            self.config = config
            
            # Initialize codebook
            if self.config.use_product_quantization:
                subspace_dim = self.config.embedding_dim // self.config.num_subspaces
                codebook_shape = (self.config.num_subspaces, self.config.codebook_size, subspace_dim)
            else:
                codebook_shape = (self.config.codebook_size, self.config.embedding_dim)
                
            self.codebook = torch_nn.Parameter(torch.randn(codebook_shape))
            
            if self.config.use_ema:
                self.register_buffer('ema_cluster_size', torch.zeros(self.config.codebook_size))
                self.register_buffer('ema_embedding_avg', torch.zeros_like(self.codebook))
        
        def forward(self, inputs: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
            """Forward pass of VQbit layer."""
            input_shape = inputs.shape
            flat_inputs = inputs.view(-1, self.config.embedding_dim)
            
            if self.config.use_product_quantization:
                quantized, indices, metrics = self._product_quantize(flat_inputs, training)
            else:
                quantized, indices, metrics = self._standard_quantize(flat_inputs, training)
            
            # Reshape back to original shape
            quantized = quantized.view(input_shape)
            indices = indices.view(input_shape[:-1])
            
            return quantized, indices, metrics
        
        def _standard_quantize(self, flat_inputs: torch.Tensor, training: bool) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
            """Standard vector quantization."""
            # Compute distances to codebook
            distances = torch.norm(
                flat_inputs.unsqueeze(1) - self.codebook.unsqueeze(0),
                dim=2
            )
            
            # Find nearest codebook entries
            indices = torch.argmin(distances, dim=1)
            quantized = self.codebook[indices]
            
            # Compute losses
            commitment_loss = F.mse_loss(quantized.detach(), flat_inputs)
            codebook_loss = F.mse_loss(quantized, flat_inputs.detach())
            
            # Straight-through estimator
            quantized = flat_inputs + (quantized - flat_inputs).detach()
            
            # Update EMA statistics if training
            if training and self.config.use_ema:
                self._update_ema_stats(flat_inputs, indices)
            
            # Compute metrics
            metrics = self._compute_metrics(flat_inputs, quantized, indices, commitment_loss, codebook_loss)
            
            return quantized, indices, metrics
        
        def _product_quantize(self, flat_inputs: torch.Tensor, training: bool) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
            """Product quantization for memory efficiency."""
            batch_size = flat_inputs.shape[0]
            subspace_dim = self.config.embedding_dim // self.config.num_subspaces
            
            # Reshape for product quantization
            inputs_reshaped = flat_inputs.view(batch_size, self.config.num_subspaces, subspace_dim)
            
            quantized_subspaces = []
            indices_subspaces = []
            total_commitment_loss = 0.0
            total_codebook_loss = 0.0
            
            for i in range(self.config.num_subspaces):
                subspace_input = inputs_reshaped[:, i, :]
                subspace_codebook = self.codebook[i]
                
                # Quantize subspace
                distances = torch.norm(
                    subspace_input.unsqueeze(1) - subspace_codebook.unsqueeze(0),
                    dim=2
                )
                subspace_indices = torch.argmin(distances, dim=1)
                subspace_quantized = subspace_codebook[subspace_indices]
                
                # Compute subspace losses
                commitment_loss = F.mse_loss(subspace_quantized.detach(), subspace_input)
                codebook_loss = F.mse_loss(subspace_quantized, subspace_input.detach())
                
                total_commitment_loss += commitment_loss
                total_codebook_loss += codebook_loss
                
                # Straight-through estimator
                subspace_quantized = subspace_input + (subspace_quantized - subspace_input).detach()
                
                quantized_subspaces.append(subspace_quantized)
                indices_subspaces.append(subspace_indices)
            
            # Combine subspaces
            quantized = torch.cat(quantized_subspaces, dim=1)
            indices = torch.stack(indices_subspaces, dim=1)
            
            # Average losses
            commitment_loss = total_commitment_loss / self.config.num_subspaces
            codebook_loss = total_codebook_loss / self.config.num_subspaces
            
            # Compute metrics
            metrics = self._compute_metrics(flat_inputs, quantized, indices, commitment_loss, codebook_loss)
            
            return quantized, indices, metrics
        
        def _update_ema_stats(self, flat_inputs: torch.Tensor, indices: torch.Tensor):
            """Update EMA statistics for stable training."""
            with torch.no_grad():
                # Update cluster sizes
                one_hot = F.one_hot(indices, self.config.codebook_size).float()
                cluster_size = torch.sum(one_hot, dim=0)
                
                self.ema_cluster_size.mul_(self.config.ema_decay).add_(
                    cluster_size, alpha=1 - self.config.ema_decay
                )
                
                # Update embedding averages
                embedding_sum = torch.matmul(one_hot.t(), flat_inputs)
                self.ema_embedding_avg.mul_(self.config.ema_decay).add_(
                    embedding_sum, alpha=1 - self.config.ema_decay
                )
        
        def _compute_metrics(self, inputs: torch.Tensor, quantized: torch.Tensor,
                           indices: torch.Tensor, commitment_loss: float,
                           codebook_loss: float) -> Dict[str, Any]:
            """Compute quantization metrics."""
            
            # Basic metrics
            mse = F.mse_loss(inputs, quantized)
            
            # Codebook usage
            unique_indices = len(torch.unique(indices))
            codebook_usage = unique_indices / self.config.codebook_size
            
            # Perplexity
            one_hot = F.one_hot(indices, self.config.codebook_size).float()
            avg_probs = torch.mean(one_hot, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            
            # Compression ratio
            original_bits = inputs.numel() * 32  # float32
            quantized_bits = indices.numel() * math.ceil(math.log2(self.config.codebook_size))
            compression_ratio = original_bits / quantized_bits
            
            return {
                'commitment_loss': commitment_loss.item() if hasattr(commitment_loss, 'item') else float(commitment_loss),
                'codebook_loss': codebook_loss.item() if hasattr(codebook_loss, 'item') else float(codebook_loss),
                'mse': mse.item(),
                'codebook_usage': codebook_usage.item(),
                'perplexity': perplexity.item(),
                'compression_ratio': compression_ratio,
                'unique_codes': unique_indices
            }
        
        @property
        def codebook_size(self) -> int:
            """Get codebook size."""
            return self.config.codebook_size

else:
    # Fallback implementation using NumPy
    class VQbitLayer:
        """
        VQbit Layer - NumPy fallback implementation.
        """
        
        def __init__(self, config: Optional[VQbitConfig] = None, **kwargs):
            if config is None:
                config = VQbitConfig(**kwargs)
            self.config = config
            
            # Initialize codebook
            if self.config.use_product_quantization:
                subspace_dim = self.config.embedding_dim // self.config.num_subspaces
                codebook_shape = (self.config.num_subspaces, self.config.codebook_size, subspace_dim)
            else:
                codebook_shape = (self.config.codebook_size, self.config.embedding_dim)
                
            self.codebook = np.random.randn(*codebook_shape).astype(np.float32)
            
            if self.config.use_ema:
                self.ema_cluster_size = np.zeros(self.config.codebook_size)
                self.ema_embedding_avg = np.zeros_like(self.codebook)
        
        def __call__(self, inputs: np.ndarray, training: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
            """Forward pass of VQbit layer."""
            input_shape = inputs.shape
            flat_inputs = inputs.reshape(-1, self.config.embedding_dim)
            
            # Standard quantization (simplified for NumPy)
            distances = np.linalg.norm(
                flat_inputs[:, None, :] - self.codebook[None, :, :],
                axis=2
            )
            
            indices = np.argmin(distances, axis=1)
            quantized = self.codebook[indices]
            
            # Compute basic metrics
            mse = np.mean((inputs.reshape(-1, self.config.embedding_dim) - quantized) ** 2)
            unique_indices = len(np.unique(indices))
            codebook_usage = unique_indices / self.config.codebook_size
            
            metrics = {
                'commitment_loss': 0.0,
                'codebook_loss': 0.0,
                'mse': float(mse),
                'codebook_usage': float(codebook_usage),
                'perplexity': float(unique_indices),
                'compression_ratio': 32.0 / math.ceil(math.log2(self.config.codebook_size)),
                'unique_codes': unique_indices
            }
            
            # Reshape back
            quantized = quantized.reshape(input_shape)
            indices = indices.reshape(input_shape[:-1])
            
            return quantized, indices, metrics
        
        @property
        def codebook_size(self) -> int:
            """Get codebook size."""
            return self.config.codebook_size


# Factory function for easy creation
def create_vqbit_layer(codebook_size: int = 64,
                      embedding_dim: int = 768,
                      use_tpu_optimizations: bool = True,
                      commitment_weight: float = 0.25,
                      diversity_regularization: bool = True,
                      **kwargs) -> VQbitLayer:
    """
    Create a VQbit layer with the specified configuration.
    
    Args:
        codebook_size: Number of codebook entries
        embedding_dim: Dimension of embeddings
        use_tpu_optimizations: Enable TPU optimizations
        commitment_weight: Weight for commitment loss
        diversity_regularization: Enable diversity regularization
        **kwargs: Additional configuration options
        
    Returns:
        VQbitLayer instance
    """
    config = VQbitConfig(
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        use_tpu_optimizations=use_tpu_optimizations,
        commitment_weight=commitment_weight,
        diversity_regularization=diversity_regularization,
        **kwargs
    )
    
    return VQbitLayer(config)


# Alias for compatibility
VQBitLayer = VQbitLayer

logger.info(f"VQbit Layer initialized - JAX: {JAX_AVAILABLE}, PyTorch: {TORCH_AVAILABLE}")

def main():
    # Main function for this module.
    logger.info("Module vqbit_layer.py starting")
    return True

if __name__ == "__main__":
    main()

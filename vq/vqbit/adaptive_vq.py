"""
Adaptive VQ Module for Capibara-6

Advanced adaptive vector quantization providing:
- Dynamic codebook adaptation
- Performance-based optimization
- Resource-aware scaling
- Integration with MoE and routing systems
- Real-time adaptation to data distribution changes
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import time
import math
from collections import deque, defaultdict

# JAX/Flax imports
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
class AdaptiveVQConfig:
    """Configurestion for Adaptive VQ."""
    # Core VQ parameters
    initial_codebook_size: int = 64
    embedding_dim: int = 768
    min_codebook_size: int = 16
    max_codebook_size: int = 256
    
    # Adaptation parameters
    adaptation_rate: float = 0.01
    growth_threshold: float = 0.95  # Codebook usage threshold for growth
    shrink_threshold: float = 0.3   # Codebook usage threshold for shrinking
    adaptation_interval: int = 100   # Steps between adaptations
    
    # Performance monitoring
    performance_window: int = 50
    target_perplexity: float = 32.0
    target_compression_ratio: float = 8.0
    
    # Resource constraints
    memory_limit_mb: float = 1000.0
    latency_target_ms: float = 10.0
    
    # Advanced features
    use_hierarchical_quantization: bool = False
    use_learned_codebook_init: bool = True
    enable_dynamic_embedding_dim: bool = False
    
    # Integration settings
    moe_compatible: bool = True
    routing_compatible: bool = True


class AdaptationHistory:
    """Track adaptation history and decisions."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.adaptations = deque(maxlen=max_history)
        self.performance_history = deque(maxlen=max_history)
    
    def record_adaptation(self, 
                         old_config: Dict[str, Any], 
                         new_config: Dict[str, Any], 
                         reason: str,
                         performance_before: float,
                         performance_after: float):
        """Record an adaptation event."""
        
        adaptation_event = {
            'timestamp': time.time(),
            'old_config': old_config.copy(),
            'new_config': new_config.copy(),
            'reason': reason,
            'performance_before': performance_before,
            'performance_after': performance_after,
            'improvement': performance_after - performance_before
        }
        
        self.adaptations.append(adaptation_event)
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        
        if not self.adaptations:
            return {'total_adaptations': 0}
        
        total_adaptations = len(self.adaptations)
        successful_adaptations = sum(1 for a in self.adaptations if a['improvement'] > 0)
        
        reasons = defaultdict(int)
        for adaptation in self.adaptations:
            reasons[adaptation['reason']] += 1
        
        return {
            'total_adaptations': total_adaptations,
            'successful_adaptations': successful_adaptations,
            'success_rate': successful_adaptations / total_adaptations,
            'adaptation_reasons': dict(reasons),
            'recent_adaptations': list(self.adaptations)[-10:]  # Last 10
        }


if JAX_AVAILABLE:
    class AdaptiveVectorQuantizer(nn.Module):
        """
        Adaptive Vector Quantizer - JAX/Flax implementation.
        
        Features:
        - Dynamic codebook size adaptation
        - Performance-based optimization
        - Resource-aware scaling
        - Integration with MoE systems
        """
        
        config: AdaptiveVQConfig
        
        def setup(self):
            """Initialize the adaptive VQ."""
            
            # Dynamic codebook - starts with initial size
            self.current_codebook_size = self.variable(
                'adaptive_state', 'codebook_size',
                lambda: jnp.array(self.config.initial_codebook_size, dtype=jnp.int32)
            )
            
            # Main codebook (max size, unused entries are masked)
            self.codebook = self.param(
                'codebook',
                nn.initializers.uniform(scale=1.0),
                (self.config.max_codebook_size, self.config.embedding_dim)
            )
            
            # Usage statistics for adaptation
            self.usage_stats = self.variable(
                'adaptive_state', 'usage_stats',
                lambda: jnp.zeros(self.config.max_codebook_size)
            )
            
            self.adaptation_step = self.variable(
                'adaptive_state', 'adaptation_step',
                lambda: jnp.array(0, dtype=jnp.int32)
            )
            
            # Performance tracking
            self.performance_history = self.variable(
                'adaptive_state', 'performance_history',
                lambda: jnp.zeros(self.config.performance_window)
            )
            
            # Hierarchical quantization if enabled
            if self.config.use_hierarchical_quantization:
                self.hierarchical_levels = 3
                self.level_codebooks = {}
                for level in range(self.hierarchical_levels):
                    size = self.config.initial_codebook_size // (2 ** level)
                    self.level_codebooks[level] = self.param(
                        f'codebook_level_{level}',
                        nn.initializers.uniform(scale=1.0),
                        (size, self.config.embedding_dim)
                    )
        
        def __call__(self, 
                    inputs: jnp.ndarray, 
                    training: bool = True,
                    force_adaptation: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
            """
            Adaptive quantization forward pass.
            
            Args:
                inputs: Input embeddings
                training: Whether in training mode
                force_adaptation: Force adaptation check
                
            Returns:
                quantized: Quantized embeddings
                indices: Codebook indices
                metrics: Adaptation and performance metrics
            """
            
            input_shape = inputs.shape
            flat_inputs = inputs.reshape(-1, self.config.embedding_dim)
            
            # Get current active codebook size
            active_size = self.current_codebook_size.value
            active_codebook = self.codebook[:active_size]
            
            # Perform quantization
            if self.config.use_hierarchical_quantization:
                quantized, indices, base_metrics = self._hierarchical_quantize(flat_inputs, training)
            else:
                quantized, indices, base_metrics = self._standard_adaptive_quantize(
                    flat_inputs, active_codebook, training
                )
            
            # Update adaptation step
            self.adaptation_step.value += 1
            
            # Check for adaptation
            adaptation_info = {}
            if training and (force_adaptation or self.adaptation_step.value % self.config.adaptation_interval == 0):
                adaptation_info = self._check_and_adapt(base_metrics)
            
            # Reshape outputs
            quantized = quantized.reshape(input_shape)
            indices = indices.reshape(input_shape[:-1])
            
            # Combine metrics
            metrics = {
                **base_metrics,
                'adaptation_info': adaptation_info,
                'current_codebook_size': int(active_size),
                'adaptation_step': int(self.adaptation_step.value),
                'adaptive_features': {
                    'hierarchical': self.config.use_hierarchical_quantization,
                    'dynamic_size': True,
                    'performance_tracking': True
                }
            }
            
            return quantized, indices, metrics
        
        def _standard_adaptive_quantize(self, 
                                      flat_inputs: jnp.ndarray, 
                                      active_codebook: jnp.ndarray,
                                      training: bool) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
            """Standard adaptive quantization."""
            
            # Compute distances to active codebook
            distances = jnp.linalg.norm(
                flat_inputs[:, None, :] - active_codebook[None, :, :],
                axis=2
            )
            
            # Find nearest entries
            indices = jnp.argmin(distances, axis=1)
            quantized = active_codebook[indices]
            
            # Update usage statistics
            if training:
                one_hot = jax.nn.one_hot(indices, active_codebook.shape[0])
                usage_update = jnp.sum(one_hot, axis=0)
                
                # Update usage stats (only for active portion)
                current_usage = self.usage_stats.value
                current_usage = current_usage.at[:active_codebook.shape[0]].add(usage_update)
                self.usage_stats.value = current_usage
            
            # Compute losses
            commitment_loss = jnp.mean((jax.lax.stop_gradient(quantized) - flat_inputs) ** 2)
            codebook_loss = jnp.mean((quantized - jax.lax.stop_gradient(flat_inputs)) ** 2)
            
            # Straight-through estimator
            quantized = flat_inputs + jax.lax.stop_gradient(quantized - flat_inputs)
            
            # Compute metrics
            metrics = self._compute_adaptive_metrics(flat_inputs, quantized, indices, 
                                                   commitment_loss, codebook_loss, active_codebook.shape[0])
            
            return quantized, indices, metrics
        
        def _hierarchical_quantize(self, 
                                 flat_inputs: jnp.ndarray, 
                                 training: bool) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
            """Hierarchical quantization with multiple levels."""
            
            # Start with coarsest level
            current_inputs = flat_inputs
            total_indices = []
            total_loss = 0.0
            
            for level in range(self.hierarchical_levels):
                level_codebook = self.level_codebooks[level]
                
                # Quantize at current level
                distances = jnp.linalg.norm(
                    current_inputs[:, None, :] - level_codebook[None, :, :],
                    axis=2
                )
                
                level_indices = jnp.argmin(distances, axis=1)
                level_quantized = level_codebook[level_indices]
                
                # Compute residual for next level
                residual = current_inputs - level_quantized
                current_inputs = residual
                
                total_indices.append(level_indices)
                total_loss += jnp.mean(residual ** 2)
            
            # Reconstruct from all levels
            reconstructed = jnp.zeros_like(flat_inputs)
            for level, indices in enumerate(total_indices):
                level_codebook = self.level_codebooks[level]
                reconstructed += level_codebook[indices]
            
            # Use finest level indices as primary
            primary_indices = total_indices[-1]
            
            metrics = {
                'hierarchical_loss': total_loss,
                'num_levels': self.hierarchical_levels,
                'level_indices': total_indices
            }
            
            return reconstructed, primary_indices, metrics
        
        def _compute_adaptive_metrics(self, 
                                    inputs: jnp.ndarray, 
                                    quantized: jnp.ndarray,
                                    indices: jnp.ndarray, 
                                    commitment_loss: float,
                                    codebook_loss: float,
                                    active_codebook_size: int) -> Dict[str, Any]:
            """Compute metrics with adaptive information."""
            
            # Standard metrics
            mse = jnp.mean((inputs - quantized) ** 2)
            
            # Codebook usage (for active portion)
            unique_indices = len(jnp.unique(indices))
            codebook_usage = unique_indices / active_codebook_size
            
            # Perplexity
            avg_probs = jnp.mean(jax.nn.one_hot(indices, active_codebook_size), axis=0)
            perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))
            
            # Adaptive-specific metrics
            usage_distribution = self.usage_stats.value[:active_codebook_size]
            usage_entropy = -jnp.sum(
                (usage_distribution / jnp.sum(usage_distribution + 1e-10)) * 
                jnp.log(usage_distribution / jnp.sum(usage_distribution + 1e-10) + 1e-10)
            )
            
            return {
                'commitment_loss': commitment_loss,
                'codebook_loss': codebook_loss,
                'mse': mse,
                'codebook_usage': codebook_usage,
                'perplexity': perplexity,
                'unique_codes': unique_indices,
                'usage_entropy': usage_entropy,
                'active_codebook_size': active_codebook_size,
                'adaptation_potential': self._compute_adaptation_potential(codebook_usage, perplexity)
            }
        
        def _compute_adaptation_potential(self, usage: float, perplexity: float) -> float:
            """Compute potential for adaptation based on current metrics."""
            
            # High usage suggests need for larger codebook
            usage_pressure = max(0, usage - self.config.growth_threshold)
            
            # Low perplexity suggests underutilized codebook
            target_perplexity = min(self.config.target_perplexity, self.current_codebook_size.value * 0.8)
            perplexity_deficit = max(0, target_perplexity - perplexity)
            
            # Combine signals
            adaptation_potential = usage_pressure * 2.0 + perplexity_deficit * 0.1
            
            return float(adaptation_potential)
        
        def _check_and_adapt(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
            """Check if adaptation is needed and perform it."""
            
            current_size = int(self.current_codebook_size.value)
            usage = metrics['codebook_usage']
            perplexity = metrics['perplexity']
            
            adaptation_info = {
                'adapted': False,
                'reason': None,
                'old_size': current_size,
                'new_size': current_size
            }
            
            # Check for growth (high usage)
            if (usage > self.config.growth_threshold and 
                current_size < self.config.max_codebook_size):
                
                new_size = min(self.config.max_codebook_size, current_size * 2)
                self.current_codebook_size.value = new_size
                
                adaptation_info.update({
                    'adapted': True,
                    'reason': f'high_usage_{usage:.2f}',
                    'new_size': new_size
                })
                
                logger.info(f"Adapted codebook size: {current_size} -> {new_size} (high usage: {usage:.2f})")
            
            # Check for shrinking (low usage)
            elif (usage < self.config.shrink_threshold and 
                  current_size > self.config.min_codebook_size):
                
                new_size = max(self.config.min_codebook_size, current_size // 2)
                self.current_codebook_size.value = new_size
                
                adaptation_info.update({
                    'adapted': True,
                    'reason': f'low_usage_{usage:.2f}',
                    'new_size': new_size
                })
                
                logger.info(f"Adapted codebook size: {current_size} -> {new_size} (low usage: {usage:.2f})")
            
            return adaptation_info

elif TORCH_AVAILABLE:
    class AdaptiveVectorQuantizer(torch_nn.Module):
        """
        Adaptive Vector Quantizer - PyTorch implementation.
        """
        
        def __init__(self, config: Optional[AdaptiveVQConfig] = None):
            super().__init__()
            if config is None:
                config = AdaptiveVQConfig()
            self.config = config
            
            # Dynamic codebook
            self.register_buffer('current_codebook_size', 
                               torch.tensor(self.config.initial_codebook_size, dtype=torch.int32))
            
            # Main codebook (max size)
            self.codebook = torch_nn.Parameter(
                torch.randn(self.config.max_codebook_size, self.config.embedding_dim)
            )
            
            # Usage statistics
            self.register_buffer('usage_stats', torch.zeros(self.config.max_codebook_size))
            self.register_buffer('adaptation_step', torch.tensor(0, dtype=torch.int32))
            
            # Adaptation history
            self.adaptation_history = AdaptationHistory()
        
        def forward(self, 
                   inputs: torch.Tensor, 
                   training: bool = True,
                   force_adaptation: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
            """Adaptive quantization forward pass."""
            
            input_shape = inputs.shape
            flat_inputs = inputs.view(-1, self.config.embedding_dim)
            
            # Get active codebook
            active_size = int(self.current_codebook_size.item())
            active_codebook = self.codebook[:active_size]
            
            # Perform quantization
            quantized, indices, metrics = self._adaptive_quantize(flat_inputs, active_codebook, training)
            
            # Update adaptation step
            self.adaptation_step += 1
            
            # Check for adaptation
            adaptation_info = {}
            if training and (force_adaptation or self.adaptation_step % self.config.adaptation_interval == 0):
                adaptation_info = self._check_and_adapt(metrics)
            
            # Reshape outputs
            quantized = quantized.view(input_shape)
            indices = indices.view(input_shape[:-1])
            
            # Update metrics
            metrics.update({
                'adaptation_info': adaptation_info,
                'current_codebook_size': active_size,
                'adaptation_step': int(self.adaptation_step.item())
            })
            
            return quantized, indices, metrics
        
        def _adaptive_quantize(self, 
                             flat_inputs: torch.Tensor, 
                             active_codebook: torch.Tensor,
                             training: bool) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
            """Perform adaptive quantization."""
            
            # Compute distances
            distances = torch.norm(
                flat_inputs.unsqueeze(1) - active_codebook.unsqueeze(0),
                dim=2
            )
            
            # Find nearest entries
            indices = torch.argmin(distances, dim=1)
            quantized = active_codebook[indices]
            
            # Update usage statistics
            if training:
                with torch.no_grad():
                    one_hot = F.one_hot(indices, active_codebook.shape[0]).float()
                    usage_update = torch.sum(one_hot, dim=0)
                    self.usage_stats[:active_codebook.shape[0]] += usage_update
            
            # Compute losses
            commitment_loss = F.mse_loss(quantized.detach(), flat_inputs)
            codebook_loss = F.mse_loss(quantized, flat_inputs.detach())
            
            # Straight-through estimator
            quantized = flat_inputs + (quantized - flat_inputs).detach()
            
            # Compute metrics
            metrics = self._compute_adaptive_metrics(flat_inputs, quantized, indices, 
                                                   commitment_loss, codebook_loss, active_codebook.shape[0])
            
            return quantized, indices, metrics
        
        def _compute_adaptive_metrics(self, 
                                    inputs: torch.Tensor, 
                                    quantized: torch.Tensor,
                                    indices: torch.Tensor, 
                                    commitment_loss: torch.Tensor,
                                    codebook_loss: torch.Tensor,
                                    active_codebook_size: int) -> Dict[str, Any]:
            """Compute adaptive metrics."""
            
            # Standard metrics
            mse = F.mse_loss(inputs, quantized)
            unique_indices = len(torch.unique(indices))
            codebook_usage = unique_indices / active_codebook_size
            
            # Perplexity
            one_hot = F.one_hot(indices, active_codebook_size).float()
            avg_probs = torch.mean(one_hot, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            
            # Usage entropy
            usage_dist = self.usage_stats[:active_codebook_size]
            normalized_usage = usage_dist / (torch.sum(usage_dist) + 1e-10)
            usage_entropy = -torch.sum(normalized_usage * torch.log(normalized_usage + 1e-10))
            
            return {
                'commitment_loss': commitment_loss.item(),
                'codebook_loss': codebook_loss.item(),
                'mse': mse.item(),
                'codebook_usage': codebook_usage.item(),
                'perplexity': perplexity.item(),
                'unique_codes': unique_indices,
                'usage_entropy': usage_entropy.item(),
                'active_codebook_size': active_codebook_size
            }
        
        def _check_and_adapt(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
            """Check and perform adaptation."""
            
            current_size = int(self.current_codebook_size.item())
            usage = metrics['codebook_usage']
            
            adaptation_info = {
                'adapted': False,
                'reason': None,
                'old_size': current_size,
                'new_size': current_size
            }
            
            # Growth check
            if (usage > self.config.growth_threshold and 
                current_size < self.config.max_codebook_size):
                
                new_size = min(self.config.max_codebook_size, current_size * 2)
                self.current_codebook_size.fill_(new_size)
                
                adaptation_info.update({
                    'adapted': True,
                    'reason': f'high_usage_{usage:.2f}',
                    'new_size': new_size
                })
            
            # Shrink check
            elif (usage < self.config.shrink_threshold and 
                  current_size > self.config.min_codebook_size):
                
                new_size = max(self.config.min_codebook_size, current_size // 2)
                self.current_codebook_size.fill_(new_size)
                
                adaptation_info.update({
                    'adapted': True,
                    'reason': f'low_usage_{usage:.2f}',
                    'new_size': new_size
                })
            
            return adaptation_info

else:
    # NumPy fallback
    class AdaptiveVectorQuantizer:
        """Adaptive Vector Quantizer - NumPy fallback."""
        
        def __init__(self, config: Optional[AdaptiveVQConfig] = None):
            if config is None:
                config = AdaptiveVQConfig()
            self.config = config
            
            self.current_codebook_size = self.config.initial_codebook_size
            self.codebook = np.random.randn(self.config.max_codebook_size, self.config.embedding_dim).astype(np.float32)
            self.usage_stats = np.zeros(self.config.max_codebook_size)
            self.adaptation_step = 0
        
        def __call__(self, inputs: np.ndarray, training: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
            """Simple adaptive quantization."""
            
            input_shape = inputs.shape
            flat_inputs = inputs.reshape(-1, self.config.embedding_dim)
            
            # Use active portion of codebook
            active_codebook = self.codebook[:self.current_codebook_size]
            
            # Simple quantization
            distances = np.linalg.norm(
                flat_inputs[:, None, :] - active_codebook[None, :, :],
                axis=2
            )
            
            indices = np.argmin(distances, axis=1)
            quantized = active_codebook[indices]
            
            # Basic metrics
            unique_codes = len(np.unique(indices))
            usage = unique_codes / self.current_codebook_size
            
            metrics = {
                'mse': float(np.mean((flat_inputs - quantized) ** 2)),
                'codebook_usage': usage,
                'unique_codes': unique_codes,
                'current_codebook_size': self.current_codebook_size,
                'backend': 'numpy_fallback'
            }
            
            # Simple adaptation
            self.adaptation_step += 1
            if self.adaptation_step % self.config.adaptation_interval == 0:
                if usage > self.config.growth_threshold and self.current_codebook_size < self.config.max_codebook_size:
                    self.current_codebook_size = min(self.config.max_codebook_size, self.current_codebook_size * 2)
                elif usage < self.config.shrink_threshold and self.current_codebook_size > self.config.min_codebook_size:
                    self.current_codebook_size = max(self.config.min_codebook_size, self.current_codebook_size // 2)
            
            # Reshape back
            quantized = quantized.reshape(input_shape)
            indices = indices.reshape(input_shape[:-1])
            
            return quantized, indices, metrics


# Factory functions
def create_adaptive_vq(initial_codebook_size: int = 64,
                      embedding_dim: int = 768,
                      adaptation_rate: float = 0.01,
                      use_hierarchical: bool = False,
                      **kwargs) -> Union[AdaptiveVectorQuantizer, AdaptiveVectorQuantizer]:
    """
    Create an adaptive VQ with specified configuration.
    
    Args:
        initial_codebook_size: Starting codebook size
        embedding_dim: Embedding dimension
        adaptation_rate: Rate of adaptation
        use_hierarchical: Enable hierarchical quantization
        **kwargs: Additional configuration
        
    Returns:
        AdaptiveVectorQuantizer instance
    """
    
    config = AdaptiveVQConfig(
        initial_codebook_size=initial_codebook_size,
        embedding_dim=embedding_dim,
        adaptation_rate=adaptation_rate,
        use_hierarchical_quantization=use_hierarchical,
        **kwargs
    )
    
    return AdaptiveVectorQuantizer(config)


def create_performance_adaptive_vq(target_performance: float = 0.9,
                                  memory_limit_mb: float = 500.0,
                                  latency_target_ms: float = 5.0) -> Union[AdaptiveVectorQuantizer, AdaptiveVectorQuantizer]:
    """
    Create performance-optimized adaptive VQ.
    
    Args:
        target_performance: Target performance score
        memory_limit_mb: Memory limit in MB
        latency_target_ms: Target latency in milliseconds
        
    Returns:
        Performance-optimized AdaptiveVectorQuantizer
    """
    
    # Estimate optimal parameters based on constraints
    if memory_limit_mb < 100:
        initial_size = 16
        max_size = 32
    elif memory_limit_mb < 500:
        initial_size = 32
        max_size = 64
    else:
        initial_size = 64
        max_size = 128
    
    config = AdaptiveVQConfig(
        initial_codebook_size=initial_size,
        max_codebook_size=max_size,
        memory_limit_mb=memory_limit_mb,
        latency_target_ms=latency_target_ms,
        growth_threshold=0.9 if target_performance > 0.8 else 0.8,
        adaptation_interval=50 if target_performance > 0.9 else 100
    )
    
    return AdaptiveVectorQuantizer(config)


# Alias for compatibility
AdaptiveVQ = AdaptiveVectorQuantizer

logger.info(f"Adaptive VQ initialized - JAX: {JAX_AVAILABLE}, PyTorch: {TORCH_AVAILABLE}")

def main():
    # Main function for this module.
    logger.info("Module adaptive_vq.py starting")
    return True

if __name__ == "__main__":
    main()

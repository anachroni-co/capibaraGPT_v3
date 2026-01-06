"""
Enhanced VQ System v2.0 for CapibaraGPT
Advanced production-ready implementation with cutting-edge features
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Any, Union, Callable
from functools import partial
import json
import time

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct

logger = logging.getLogger(__name__)

# ============================================================================
# Enhanced Configuration System
# ============================================================================

@dataclass
class EnhancedVQConfig:
    """Advanced VQ configuration with new features."""
    
    # Core VQ parameters
    num_embeddings: int = 8192
    embedding_dim: int = 1024
    commitment_cost: float = 0.25
    
    # EMA updates
    use_ema: bool = True
    ema_decay: float = 0.99
    ema_epsilon: float = 1e-5
    
    # Advanced quantization features
    use_cosine_sim: bool = False
    use_l2_normalize: bool = False
    dead_code_threshold: int = 100
    
    # Gumbel softmax
    use_gumbel: bool = False
    temperature: float = 1.0
    temperature_decay: float = 0.9999
    min_temperature: float = 0.5
    
    # Multi-stage VQ
    num_stages: int = 1
    share_codebook: bool = False
    
    # NEW: Product Quantization
    use_product_quantization: bool = False
    num_subspaces: int = 8
    subspace_dim: Optional[int] = None  # Will be computed as embedding_dim // num_subspaces
    
    # NEW: Learnable codebook initialization
    learnable_init: bool = True
    init_method: str = "xavier"  # "xavier", "kaiming", "orthogonal", "normal"
    
    # NEW: Adaptive quantization
    use_adaptive_commitment: bool = False
    commitment_schedule: str = "constant"  # "constant", "cosine", "linear"
    min_commitment: float = 0.1
    max_commitment: float = 1.0
    
    # NEW: Entropy regularization
    use_entropy_loss: bool = False
    entropy_weight: float = 0.01
    
    # NEW: Diversity loss
    use_diversity_loss: bool = False
    diversity_weight: float = 0.1
    
    # NEW: Vector compression
    use_compression: bool = False
    compression_ratio: float = 0.5
    
    # Performance and monitoring
    dtype: jnp.dtype = jnp.float32
    use_jit: bool = True
    enable_monitoring: bool = True
    log_frequency: int = 100
    
    # NEW: Experiment tracking
    experiment_name: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validates and compute derived parameters."""
        if self.subspace_dim is None and self.use_product_quantization:
            self.subspace_dim = self.embedding_dim // self.num_subspaces
            if self.embedding_dim % self.num_subspaces != 0:
                logger.warning(f"embedding_dim ({self.embedding_dim}) not divisible by num_subspaces ({self.num_subspaces})")

# ============================================================================
# Enhanced Vector Quantizer with New Features
# ============================================================================

class EnhancedVectorQuantizer(nn.Module):
    """Production-ready VQ with advanced features."""
    
    config: EnhancedVQConfig
    
    def setup(self):
        """Initialize enhanced VQ components."""
        # Main embedding table
        self._setup_embeddings()
        
        # EMA state if enabled
        if self.config.use_ema:
            self._setup_ema_state()
        
        # Usage and stats tracking
        self._setup_tracking()
        
        # Temperature for Gumbel softmax
        if self.config.use_gumbel:
            self._setup_gumbel()
        
        # NEW: Product quantization
        if self.config.use_product_quantization:
            self._setup_product_quantization()
        
        # NEW: Compression layers
        if self.config.use_compression:
            self._setup_compression()
    
    def _setup_embeddings(self):
        """Setup main embedding table with learnable initialization."""
        if self.config.learnable_init:
            init_fn = self._get_initializer()
        else:
            init_fn = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0)
        
        self.embeddings = nn.Embed(
            num_embeddings=self.config.num_embeddings,
            features=self.config.embedding_dim,
            dtype=self.config.dtype,
            param_dtype=self.config.dtype,
            embedding_init=init_fn
        )
    
    def _get_initializer(self):
        """Get the specified initializer."""
        if self.config.init_method == "xavier":
            return nn.initializers.xavier_normal()
        elif self.config.init_method == "kaiming":
            return nn.initializers.kaiming_normal()
        elif self.config.init_method == "orthogonal":
            return nn.initializers.orthogonal()
        elif self.config.init_method == "normal":
            return nn.initializers.normal(stddev=0.02)
        else:
            return nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0)
    
    def _setup_ema_state(self):
        """Setup EMA state variables."""
        self.ema_cluster_size = self.variable(
            'ema', 'cluster_size',
            lambda: jnp.zeros(self.config.num_embeddings, dtype=self.config.dtype)
        )
        self.ema_embeddings = self.variable(
            'ema', 'embeddings',
            lambda: self.embeddings.embedding.value.copy()
        )
    
    def _setup_tracking(self):
        """Setup usage and statistics tracking."""
        self.usage_count = self.variable(
            'stats', 'usage_count',
            lambda: jnp.zeros(self.config.num_embeddings, dtype=jnp.int32)
        )
        
        # NEW: Enhanced tracking
        if self.config.enable_monitoring:
            self.step_count = self.variable(
                'stats', 'step_count',
                lambda: jnp.array(0, dtype=jnp.int32)
            )
            self.perplexity_history = self.variable(
                'stats', 'perplexity_history',
                lambda: jnp.zeros(1000, dtype=self.config.dtype)  # Rolling window
            )
    
    def _setup_gumbel(self):
        """Setup Gumbel softmax temperature."""
        self.temperature = self.variable(
            'stats', 'temperature',
            lambda: jnp.array(self.config.temperature, dtype=self.config.dtype)
        )
    
    def _setup_product_quantization(self):
        """Setup product quantization components."""
        if self.config.subspace_dim is None:
            raise ValueError("subspace_dim must be specified for product quantization")
        
        # Separate codebooks for each subspace
        self.pq_codebooks = []
        for i in range(self.config.num_subspaces):
            codebook = nn.Embed(
                num_embeddings=self.config.num_embeddings,
                features=self.config.subspace_dim,
                dtype=self.config.dtype,
                embedding_init=self._get_initializer()
            )
            self.pq_codebooks.append(codebook)
    
    def _setup_compression(self):
        """Setup compression layers."""
        compressed_dim = int(self.config.embedding_dim * self.config.compression_ratio)
        
        self.compression_encoder = nn.Dense(
            features=compressed_dim,
            dtype=self.config.dtype,
            name="compression_encoder"
        )
        
        self.compression_decoder = nn.Dense(
            features=self.config.embedding_dim,
            dtype=self.config.dtype,
            name="compression_decoder"
        )
    
    def quantize(self, z: jnp.ndarray, training: bool = False) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Enhanced quantization with new features."""
        batch_shape = z.shape[:-1]
        z_flat = z.reshape(-1, self.config.embedding_dim)
        
        # NEW: Apply compression if enabled
        if self.config.use_compression:
            z_compressed = self.compression_encoder(z_flat)
            z_flat = self.compression_decoder(z_compressed)
        
        # Product quantization path
        if self.config.use_product_quantization:
            return self._product_quantize(z, z_flat, batch_shape, training)
        
        # Standard quantization path
        return self._standard_quantize(z, z_flat, batch_shape, training)
    
    def _standard_quantize(self, z: jnp.ndarray, z_flat: jnp.ndarray, 
                          batch_shape: Tuple, training: bool) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Standard quantization implementation."""
        # Get embeddings
        embeddings = self.embeddings.embedding.value
        if self.config.use_ema and hasattr(self, 'ema_embeddings'):
            if self.ema_embeddings.value is not None:
                embeddings = self.ema_embeddings.value
        
        # Normalize if configured
        if self.config.use_l2_normalize:
            z_flat = z_flat / (jnp.linalg.norm(z_flat, axis=-1, keepdims=True) + 1e-8)
            embeddings = embeddings / (jnp.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8)
        
        # Compute distances
        distances = self._compute_distances(z_flat, embeddings)
        
        # Quantization (Gumbel or hard)
        if self.config.use_gumbel and training:
            quantized_flat, encoding_indices = self._gumbel_softmax_quantize(distances, embeddings, z_flat)
        else:
            encoding_indices = jnp.argmin(distances, axis=1)
            quantized_flat = embeddings[encoding_indices]
        
        # Reshape to original shape
        quantized = quantized_flat.reshape(z.shape)
        encoding_indices = encoding_indices.reshape(batch_shape)
        
        # Compute losses
        losses = self._compute_losses(z, quantized, z_flat, encoding_indices, embeddings, distances)
        
        # Straight-through estimator
        quantized = z + jax.lax.stop_gradient(quantized - z)
        
        # Updates during training
        if training:
            self._perform_updates(z_flat, encoding_indices)
        
        # Compute metrics
        metrics = self._compute_enhanced_metrics(z, quantized, encoding_indices, distances, losses)
        
        return quantized, metrics
    
    def _product_quantize(self, z: jnp.ndarray, z_flat: jnp.ndarray,
                         batch_shape: Tuple, training: bool) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Product quantization implementation."""
        # Split into subspaces
        subspaces = jnp.split(z_flat, self.config.num_subspaces, axis=-1)
        
        quantized_subspaces = []
        all_indices = []
        total_loss = 0.0
        
        for i, subspace in enumerate(subspaces):
            # Get subspace codebook
            codebook = self.pq_codebooks[i].embedding.value
            
            # Quantize subspace
            distances = self._compute_distances(subspace, codebook)
            indices = jnp.argmin(distances, axis=1)
            quantized_subspace = codebook[indices]
            
            # Compute subspace loss
            commitment_loss = jnp.mean((subspace - jax.lax.stop_gradient(quantized_subspace))**2)
            embedding_loss = jnp.mean((jax.lax.stop_gradient(subspace) - quantized_subspace)**2)
            subspace_loss = embedding_loss + self.config.commitment_cost * commitment_loss
            
            total_loss += subspace_loss
            quantized_subspaces.append(quantized_subspace)
            all_indices.append(indices)
        
        # Concatenate quantized subspaces
        quantized_flat = jnp.concatenate(quantized_subspaces, axis=-1)
        quantized = quantized_flat.reshape(z.shape)
        
        # Combine indices (for metrics)
        combined_indices = jnp.stack(all_indices, axis=-1)
        
        # Straight-through estimator
        quantized = z + jax.lax.stop_gradient(quantized - z)
        
        # Basic metrics for PQ
        mse = jnp.mean((z - quantized)**2)
        metrics = {
            'loss': total_loss,
            'mse': float(mse),
            'pq_loss': float(total_loss),
            'num_subspaces': self.config.num_subspaces,
            'mode': 'product_quantization'
        }
        
        return quantized, metrics
    
    def _compute_distances(self, z_flat: jnp.ndarray, embeddings: jnp.ndarray) -> jnp.ndarray:
        """Compute distances between inputs and embeddings."""
        if self.config.use_cosine_sim:
            # Cosine similarity (higher is better, so negate)
            distances = -jnp.matmul(z_flat, embeddings.T)
        else:
            # L2 distance
            distances = (
                jnp.sum(z_flat**2, axis=1, keepdims=True) +
                jnp.sum(embeddings**2, axis=1) -
                2 * jnp.matmul(z_flat, embeddings.T)
            )
        return distances
    
    def _compute_losses(self, z: jnp.ndarray, quantized: jnp.ndarray,
                       z_flat: jnp.ndarray, encoding_indices: jnp.ndarray,
                       embeddings: jnp.ndarray, distances: jnp.ndarray) -> Dict[str, float]:
        """Compute all loss components."""
        losses = {}
        
        # Standard VQ losses
        if self.config.use_ema:
            commitment_loss = jnp.mean((z - jax.lax.stop_gradient(quantized))**2)
            losses['commitment'] = float(commitment_loss)
            total_loss = self._get_adaptive_commitment() * commitment_loss
        else:
            commitment_loss = jnp.mean((z - jax.lax.stop_gradient(quantized))**2)
            embedding_loss = jnp.mean((jax.lax.stop_gradient(z) - quantized)**2)
            losses['commitment'] = float(commitment_loss)
            losses['embedding'] = float(embedding_loss)
            total_loss = embedding_loss + self._get_adaptive_commitment() * commitment_loss
        
        # NEW: Entropy regularization
        if self.config.use_entropy_loss:
            entropy_loss = self._compute_entropy_loss(encoding_indices)
            losses['entropy'] = float(entropy_loss)
            total_loss += self.config.entropy_weight * entropy_loss
        
        # NEW: Diversity loss
        if self.config.use_diversity_loss:
            diversity_loss = self._compute_diversity_loss(embeddings, encoding_indices)
            losses['diversity'] = float(diversity_loss)
            total_loss += self.config.diversity_weight * diversity_loss
        
        losses['total'] = float(total_loss)
        return losses
    
    def _get_adaptive_commitment(self) -> float:
        """Get adaptive commitment cost."""
        if not self.config.use_adaptive_commitment:
            return self.config.commitment_cost
        
        if not hasattr(self, 'step_count'):
            return self.config.commitment_cost
        
        step = float(self.step_count.value)
        
        if self.config.commitment_schedule == "cosine":
            # Cosine annealing
            progress = step / 10000.0  # Assume 10k steps for full cycle
            factor = 0.5 * (1 + jnp.cos(jnp.pi * progress))
            commitment = self.config.min_commitment + (self.config.max_commitment - self.config.min_commitment) * factor
        elif self.config.commitment_schedule == "linear":
            # Linear decay
            progress = jnp.minimum(step / 10000.0, 1.0)
            commitment = self.config.max_commitment - progress * (self.config.max_commitment - self.config.min_commitment)
        else:
            commitment = self.config.commitment_cost
        
        return float(commitment)
    
    def _compute_entropy_loss(self, encoding_indices: jnp.ndarray) -> jnp.ndarray:
        """Compute entropy regularization loss."""
        # Compute empirical distribution
        indices_flat = encoding_indices.reshape(-1)
        probs = jnp.bincount(indices_flat, length=self.config.num_embeddings) / len(indices_flat)
        
        # Compute entropy
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-10))
        
        # We want high entropy, so minimize negative entropy
        return -entropy
    
    def _compute_diversity_loss(self, embeddings: jnp.ndarray, encoding_indices: jnp.ndarray) -> jnp.ndarray:
        """Compute diversity loss to encourage usage of all codes."""
        # Get unique indices and their counts
        indices_flat = encoding_indices.reshape(-1)
        unique_indices, counts = jnp.unique(indices_flat, return_counts=True, size=self.config.num_embeddings, fill_value=0)
        
        # Compute coefficient of variation (std/mean) of usage
        mean_usage = jnp.mean(counts)
        std_usage = jnp.std(counts)
        diversity_loss = std_usage / (mean_usage + 1e-8)
        
        return diversity_loss
    
    def _gumbel_softmax_quantize(self, distances: jnp.ndarray, embeddings: jnp.ndarray,
                                z_flat: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Enhanced Gumbel softmax quantization."""
        temp = self.temperature.value
        
        # Convert distances to logits
        logits = -distances / temp
        
        # Add Gumbel noise
        gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(
            self.make_rng('gumbel'),
            logits.shape,
            minval=1e-20
        )))
        
        # Gumbel softmax
        y_soft = jax.nn.softmax((logits + gumbel_noise) / temp, axis=-1)
        
        # Soft quantization
        quantized = jnp.matmul(y_soft, embeddings)
        
        # Hard indices for metrics
        encoding_indices = jnp.argmax(y_soft, axis=-1)
        
        # Update temperature
        new_temp = jnp.maximum(
            temp * self.config.temperature_decay,
            self.config.min_temperature
        )
        self.temperature.value = new_temp
        
        return quantized, encoding_indices
    
    def _perform_updates(self, z_flat: jnp.ndarray, encoding_indices: jnp.ndarray):
        """Perform all training updates."""
        # EMA updates
        if self.config.use_ema:
            self._update_ema(z_flat, encoding_indices)
        
        # Usage tracking
        self._update_usage(encoding_indices)
        
        # Step counting
        if self.config.enable_monitoring and hasattr(self, 'step_count'):
            self.step_count.value = self.step_count.value + 1
    
    def _update_ema(self, z_flat: jnp.ndarray, encoding_indices: jnp.ndarray):
        """Update EMA statistics."""
        # One-hot encoding
        encodings = jax.nn.one_hot(
            encoding_indices,
            self.config.num_embeddings,
            dtype=self.config.dtype
        )
        
        # Update cluster sizes
        new_cluster_size = jnp.sum(encodings, axis=0)
        updated_cluster_size = (
            self.ema_cluster_size.value * self.config.ema_decay +
            new_cluster_size * (1 - self.config.ema_decay)
        )
        
        # Update embeddings
        dw = jnp.matmul(encodings.T, z_flat)
        updated_embeddings = (
            self.ema_embeddings.value * self.config.ema_decay +
            dw * (1 - self.config.ema_decay)
        )
        
        # Laplace smoothing
        n = jnp.sum(updated_cluster_size)
        updated_cluster_size = (
            (updated_cluster_size + self.config.ema_epsilon) /
            (n + self.config.num_embeddings * self.config.ema_epsilon) * n
        )
        
        # Normalize embeddings
        normalized_embeddings = updated_embeddings / (updated_cluster_size[:, None] + 1e-8)
        
        # Update variables
        self.ema_cluster_size.value = updated_cluster_size
        self.ema_embeddings.value = normalized_embeddings
        self.embeddings.embedding.value = normalized_embeddings
    
    def _update_usage(self, encoding_indices: jnp.ndarray):
        """Update usage statistics and handle dead codes."""
        # Count usage
        usage = jnp.bincount(
            encoding_indices.reshape(-1),
            minlength=self.config.num_embeddings,
            length=self.config.num_embeddings
        )
        self.usage_count.value = self.usage_count.value + usage
        
        # Check for dead code reinitialization
        total_usage = jnp.sum(self.usage_count.value)
        if total_usage > 0 and total_usage % (self.config.dead_code_threshold * self.config.num_embeddings) == 0:
            self._reinit_dead_codes()
    
    def _reinit_dead_codes(self):
        """Reinitialize dead codes with improved strategy."""
        dead_codes = self.usage_count.value == 0
        num_dead = jnp.sum(dead_codes)
        
        if num_dead > 0:
            logger.info(f"Reinitializing {num_dead} dead codes")
            
            live_indices = jnp.where(~dead_codes)[0]
            if len(live_indices) > 0:
                rng = self.make_rng('reinit')
                
                # NEW: Improved reinitialization strategy
                # Sample from high-usage codes with added noise
                usage_probs = self.usage_count.value[live_indices] / jnp.sum(self.usage_count.value[live_indices])
                selected = jax.random.choice(rng, live_indices, shape=(num_dead,), p=usage_probs)
                
                # Add more noise for better diversity
                noise_scale = 0.1  # Increased from 0.01
                noise = jax.random.normal(rng, (num_dead, self.config.embedding_dim)) * noise_scale
                
                new_embeddings = self.embeddings.embedding.value[selected] + noise
                
                # Update dead codes
                dead_indices = jnp.where(dead_codes)[0]
                self.embeddings.embedding.value = self.embeddings.embedding.value.at[dead_indices].set(new_embeddings)
                
                if self.config.use_ema:
                    self.ema_embeddings.value = self.ema_embeddings.value.at[dead_indices].set(new_embeddings)
            
            # Reset usage counts
            self.usage_count.value = jnp.zeros_like(self.usage_count.value)
    
    def _compute_enhanced_metrics(self, z: jnp.ndarray, quantized: jnp.ndarray,
                                 encoding_indices: jnp.ndarray, distances: jnp.ndarray,
                                 losses: Dict[str, float]) -> Dict[str, Any]:
        """Compute enhanced metrics with new features."""
        # Basic metrics
        mse = jnp.mean((z - quantized)**2)
        
        # Perplexity
        encodings = jax.nn.one_hot(encoding_indices.reshape(-1), self.config.num_embeddings)
        avg_probs = jnp.mean(encodings, axis=0)
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))
        
        # Usage statistics
        unique_codes = jnp.unique(encoding_indices).shape[0]
        usage_rate = unique_codes / self.config.num_embeddings
        
        # Distance statistics
        min_distances = jnp.min(distances, axis=1)
        
        # NEW: Enhanced metrics
        metrics = {
            'mse': float(mse),
            'perplexity': float(perplexity),
            'usage_rate': float(usage_rate),
            'unique_codes': int(unique_codes),
            'mean_distance': float(jnp.mean(min_distances)),
            'std_distance': float(jnp.std(min_distances)),
            'temperature': float(self.temperature.value) if self.config.use_gumbel else 1.0,
            'commitment_cost': self._get_adaptive_commitment(),
            **losses  # Include all loss components
        }
        
        # Add step count if monitoring enabled
        if self.config.enable_monitoring and hasattr(self, 'step_count'):
            metrics['step'] = int(self.step_count.value)
            
            # Update perplexity history
            step = self.step_count.value % 1000
            self.perplexity_history.value = self.perplexity_history.value.at[step].set(perplexity)
            
            # Rolling average perplexity
            metrics['avg_perplexity'] = float(jnp.mean(self.perplexity_history.value))
        
        return metrics
    
    def __call__(self, inputs: jnp.ndarray, training: bool = False) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Main forward pass."""
        return self.quantize(inputs, training)

# ============================================================================
# Enhanced Factory Functions
# ============================================================================

def create_enhanced_vq_layer(
    num_embeddings: int = 8192,
    embedding_dim: int = 1024,
    use_ema: bool = True,
    use_product_quantization: bool = False,
    num_subspaces: int = 8,
    use_adaptive_commitment: bool = False,
    use_entropy_loss: bool = False,
    use_diversity_loss: bool = False,
    experiment_name: Optional[str] = None,
    **kwargs
) -> EnhancedVectorQuantizer:
    """Creates enhanced VQ layer with new features."""
    config = EnhancedVQConfig(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        use_ema=use_ema,
        use_product_quantization=use_product_quantization,
        num_subspaces=num_subspaces,
        use_adaptive_commitment=use_adaptive_commitment,
        use_entropy_loss=use_entropy_loss,
        use_diversity_loss=use_diversity_loss,
        experiment_name=experiment_name,
        **kwargs
    )
    return EnhancedVectorQuantizer(config)

# ============================================================================
# Benchmark and Comparison Tools
# ============================================================================

def benchmark_vq_variants():
    """Benchmark different VQ configurations."""
    import time
    
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (4, 32, 256))
    
    configs = [
        ("Standard VQ", {"use_ema": True}),
        ("Product VQ", {"use_product_quantization": True, "num_subspaces": 4}),
        ("Adaptive VQ", {"use_adaptive_commitment": True, "use_entropy_loss": True}),
        ("Diversity VQ", {"use_diversity_loss": True, "use_entropy_loss": True}),
        ("Gumbel VQ", {"use_gumbel": True, "temperature": 1.0}),
    ]
    
    results = {}
    
    for name, config_kwargs in configs:
        print(f"\n🔬 Benchmarking {name}...")
        
        try:
            vq = create_enhanced_vq_layer(
                num_embeddings=512,
                embedding_dim=256,
                **config_kwargs
            )
            
            params = vq.init(key, x)
            
            # Warmup
            _ = vq.apply(params, x, training=True)
            
            # Benchmark
            start_time = time.time()
            for _ in range(10):
                quantized, metrics = vq.apply(params, x, training=True)
            elapsed = (time.time() - start_time) / 10
            
            results[name] = {
                'time_ms': elapsed * 1000,
                'perplexity': metrics['perplexity'],
                'usage_rate': metrics['usage_rate'],
                'mse': metrics['mse']
            }
            
            print(f"   Time: {elapsed*1000:.2f}ms")
            print(f"   Perplexity: {metrics['perplexity']:.2f}")
            print(f"   Usage: {metrics['usage_rate']:.2%}")
            print(f"   MSE: {metrics['mse']:.6f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[name] = {"error": str(e)}
    
    return results

# ============================================================================
# Usage Example with Enhanced Features
# ============================================================================

def enhanced_example_usage():
    """Demonstrate enhanced VQ system features."""
    print("🚀 Enhanced VQ System v2.0 Demo")
    print("=" * 60)
    
    key = jax.random.PRNGKey(42)
    
    # 1. Enhanced Standard VQ with new features
    print("\n1️⃣ Enhanced VQ with Adaptive Commitment:")
    vq = create_enhanced_vq_layer(
        num_embeddings=1024,
        embedding_dim=256,
        use_adaptive_commitment=True,
        use_entropy_loss=True,
        use_diversity_loss=True,
        experiment_name="enhanced_demo"
    )
    
    x = jax.random.normal(key, (4, 32, 256))
    params = vq.init(key, x)
    
    quantized, metrics = vq.apply(params, x, training=True)
    print(f"   Enhanced metrics: {len(metrics)} tracked")
    print(f"   Adaptive commitment: {metrics['commitment_cost']:.3f}")
    print(f"   Total loss: {metrics['total']:.6f}")
    
    # 2. Product Quantization
    print("\n2️⃣ Product Quantization:")
    pq_vq = create_enhanced_vq_layer(
        num_embeddings=256,
        embedding_dim=256,
        use_product_quantization=True,
        num_subspaces=8,
        experiment_name="pq_demo"
    )
    
    params_pq = pq_vq.init(key, x)
    quantized_pq, metrics_pq = pq_vq.apply(params_pq, x, training=True)
    print(f"   PQ subspaces: {metrics_pq['num_subspaces']}")
    print(f"   PQ loss: {metrics_pq['pq_loss']:.6f}")
    
    # 3. Run benchmarks
    print("\n3️⃣ Benchmarking VQ variants...")
    benchmark_results = benchmark_vq_variants()
    
    print("\n✅ Enhanced VQ Demo completed!")
    return benchmark_results

if __name__ == "__main__":
    enhanced_example_usage()
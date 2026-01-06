"""
Dynamic Mixture of Experts (MoE) System for Capibara-6

Optimized for TPU v6e-64 with advanced routing and load balancing.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import time

# JAX imports with fallbacks
try:
    # Try to import real JAX first
    import jax.numpy as jnp
    from jax import nn, random
    from jax.sharding import PartitionSpec as P
except ImportError:
    # Fall back to capibara.jax
    from capibara.jax import numpy as jnp
    from capibara.jax import nn, random
    from capibara.jax.sharding import PartitionSpec as P

logger = logging.getLogger(__name__)


@dataclass
class MoEConfig:
    """Configuration for Dynamic MoE system."""
    num_experts: int = 32
    num_active_experts: int = 4
    hidden_size: int = 768
    expert_hidden_size: int = 3072
    use_specialization: bool = True
    tpu_optimized: bool = True
    load_balance_weight: float = 0.01
    routing_temperature: float = 1.0
    expert_dropout: float = 0.1
    capacity_factor: float = 1.25
    use_auxiliary_loss: bool = True
    expert_types: List[str] = None
    
    def __post_init__(self):
        if self.expert_types is None:
            self.expert_types = [
                "general", "reasoning", "creative", "analytical",
                "mathematical", "linguistic", "factual", "conversational",
                "technical", "scientific", "artistic", "logical",
                "memory", "planning", "synthesis", "critique",
                "translation", "summarization", "classification", "generation",
                "question_answering", "code", "multimodal", "temporal",
                "spatial", "causal", "ethical", "emotional",
                "strategic", "tactical", "optimization", "debugging"
            ]


def create_moe_config(**kwargs) -> MoEConfig:
    """Factory function to create MoE configuration."""
    return MoEConfig(**kwargs)


class ExpertLayer:
    """Individual expert in the MoE system."""
    
    def __init__(self, config: MoEConfig, expert_id: int, expert_type: str):
        self.config = config
        self.expert_id = expert_id
        self.expert_type = expert_type
        
        # Expert-specific formeters
        self.w1 = jnp.ones((config.hidden_size, config.expert_hidden_size)) * 0.02
        self.w2 = jnp.ones((config.expert_hidden_size, config.hidden_size)) * 0.02
        self.bias1 = jnp.zeros(config.expert_hidden_size)
        self.bias2 = jnp.zeros(config.hidden_size)
        
        # Specialization parameters
        if config.use_specialization:
            self.specialization_weight = self._get_specialization_weight(expert_type)
        else:
            self.specialization_weight = 1.0
            
        # Performance tracking
        self.usage_count = 0
        self.total_processing_time = 0.0
        self.last_used = time.time()
        
    def _get_specialization_weight(self, expert_type: str) -> float:
        """Get specialization weight based on expert type."""
        specialization_map = {
            "reasoning": 1.2, "mathematical": 1.3, "logical": 1.2,
            "creative": 1.1, "artistic": 1.1, "linguistic": 1.1,
            "technical": 1.2, "scientific": 1.2, "code": 1.3,
            "analytical": 1.2, "factual": 1.0, "conversational": 0.9,
            "general": 1.0
        }
        return specialization_map.get(expert_type, 1.0)
        
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass through expert."""
        start_time = time.time()
        
        # Apply expert transformation
        # x: [batch_size, seq_len, hidden_size]
        hidden = jnp.dot(x, self.w1) + self.bias1
        
        # Apply activation (SwiGLU-style)
        hidden = nn.silu(hidden)  # SiLU activation
        
        # Apply dropout if training
        if training and self.config.expert_dropout > 0:
            dropout_mask = random.bernoulli(
                key=random.PRNGKey(int(time.time() * 1000) % 2**32),
                p=1.0 - self.config.expert_dropout,
                shape=hidden.shape
            )
            hidden = hidden * dropout_mask / (1.0 - self.config.expert_dropout)
            
        # Output projection
        output = jnp.dot(hidden, self.w2) + self.bias2
        
        # Apply specialization weight
        output = output * self.specialization_weight
        
        # Update usage statistics
        self.usage_count += 1
        self.total_processing_time += time.time() - start_time
        self.last_used = time.time()
        
        return output
        
    def get_stats(self) -> Dict[str, Any]:
        """Get expert statistics."""
        avg_time = self.total_processing_time / max(self.usage_count, 1)
        return {
            "expert_id": self.expert_id,
            "expert_type": self.expert_type,
            "usage_count": self.usage_count,
            "avg_processing_time": avg_time,
            "specialization_weight": self.specialization_weight,
            "last_used": self.last_used
        }


class DynamicRouter:
    """Dynamic routing system for MoE."""
    
    def __init__(self, config: MoEConfig):
        self.config = config
        
        # Router parameters
        self.router_weights = jnp.ones((config.hidden_size, config.num_experts)) * 0.02
        self.router_bias = jnp.zeros(config.num_experts)
        
        # Routing history for adaptation
        self.routing_history = []
        self.expert_performance = jnp.ones(config.num_experts)
        
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Route inputs to experts.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            training: Whether in training mode
            
        Returns:
            routing_weights: [batch_size, seq_len, num_experts]
            expert_indices: [batch_size, seq_len, num_active_experts]
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute routing logits
        # Average pool over sequence dimension for routing decision
        pooled_x = jnp.mean(x, axis=1)  # [batch_size, hidden_size]
        
        # Router forward pass
        router_logits = jnp.dot(pooled_x, self.router_weights) + self.router_bias
        
        # Apply temperature scaling
        router_logits = router_logits / self.config.routing_temperature
        
        # Apply expert performance weighting
        router_logits = router_logits + jnp.log(self.expert_performance + 1e-8)
        
        # Compute routing probabilities
        routing_probs = nn.softmax(router_logits, axis=-1)
        
        # Select top-k experts
        top_k_indices = jnp.argsort(routing_probs, axis=-1)[:, -self.config.num_active_experts:]
        
        # Get routing weights for selected experts
        routing_weights = jnp.zeros_like(routing_probs)
        
        # Handle JAX vs numpy compatibility for .at[] operations
        try:
            # JAX version with .at[] syntax
            for i in range(batch_size):
                for j in range(self.config.num_active_experts):
                    expert_idx = top_k_indices[i, j]
                    routing_weights = routing_weights.at[i, expert_idx].set(routing_probs[i, expert_idx])
        except AttributeError:
            # Numpy fallback without .at[] syntax
            routing_weights = routing_weights.copy()  # Make it writable
            for i in range(batch_size):
                for j in range(self.config.num_active_experts):
                    expert_idx = top_k_indices[i, j]
                    routing_weights[i, expert_idx] = routing_probs[i, expert_idx]
        
        # Normalize routing weights
        routing_weights = routing_weights / (jnp.sum(routing_weights, axis=-1, keepdims=True) + 1e-8)
        
        # Expand to sequence dimension
        routing_weights = jnp.expand_dims(routing_weights, axis=1)  # [batch_size, 1, num_experts]
        routing_weights = jnp.broadcast_to(routing_weights, (batch_size, seq_len, self.config.num_experts))
        
        return routing_weights, top_k_indices
        
    def update_expert_performance(self, expert_loads: jnp.ndarray, expert_losses: jnp.ndarray):
        """Update expert performance based on usage and losses."""
        # Exponential moving average of performance
        alpha = 0.1
        performance_update = 1.0 / (expert_losses + 1e-8)  # Lower loss = higher performance
        self.expert_performance = (1 - alpha) * self.expert_performance + alpha * performance_update
        
        # Store routing history
        self.routing_history.append({
            "expert_loads": expert_loads,
            "expert_losses": expert_losses,
            "timestamp": time.time()
        })
        
        # Keep only recent history
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]


class DynamicMoELayer:
    """Dynamic MoE layer with load balancing and specialization."""
    
    def __init__(self, config: MoEConfig, layer_id: int):
        self.config = config
        self.layer_id = layer_id
        
        # Create experts
        self.experts = []
        for i in range(config.num_experts):
            expert_type = config.expert_types[i % len(config.expert_types)]
            expert = ExpertLayer(config, i, expert_type)
            self.experts.append(expert)
            
        # Create router
        self.router = DynamicRouter(config)
        
        # Layer normalization
        self.layer_norm = lambda x: x  # Simplified for now
        
        # Metrics tracking
        self.total_tokens_processed = 0
        self.routing_entropy_history = []
        self.load_balance_history = []
        
    def __call__(self, x: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None, 
                 training: bool = False) -> Dict[str, Any]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            training: Whether in training mode
            
        Returns:
            Dictionary containing output and metrics
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Apply layer normalization
        normalized_x = self.layer_norm(x)
        
        # Route to experts
        routing_weights, expert_indices = self.router(normalized_x, training)
        
        # Process through experts
        expert_outputs = []
        expert_loads = jnp.zeros(self.config.num_experts)
        
        for i, expert in enumerate(self.experts):
            # Check if expert is selected
            expert_mask = routing_weights[:, :, i] > 1e-6
            
            if jnp.any(expert_mask):
                # Process tokens assigned to this expert
                expert_input = jnp.where(
                    jnp.expand_dims(expert_mask, axis=-1),
                    normalized_x,
                    jnp.zeros_like(normalized_x)
                )
                
                expert_output = expert(expert_input, training)
                expert_outputs.append(expert_output)
                
                # Track expert load
                expert_load = jnp.sum(expert_mask)
                try:
                    # JAX version with .at[] syntax
                    expert_loads = expert_loads.at[i].set(expert_load)
                except AttributeError:
                    # Numpy fallback without .at[] syntax
                    expert_loads = expert_loads.copy()
                    expert_loads[i] = expert_load
            else:
                expert_outputs.append(jnp.zeros_like(normalized_x))
        
        # Combine expert outputs
        combined_output = jnp.zeros_like(x)
        for i, expert_output in enumerate(expert_outputs):
            weight = jnp.expand_dims(routing_weights[:, :, i], axis=-1)
            combined_output += weight * expert_output
            
        # Add residual connection
        output = x + combined_output
        
        # Compute metrics
        moe_metrics = self._compute_metrics(routing_weights, expert_loads, training)
        
        # Note: History tracking removed from compiled function to avoid concretization errors
        # Metrics are still computed and returned for external tracking
        
        return {
            "output": output,
            "expert_loads": expert_loads,
            "routing_weights": routing_weights,
            "moe_metrics": moe_metrics,
            "expert_stats": [expert.get_stats() for expert in self.experts]
        }
        
    def _compute_metrics(self, routing_weights: jnp.ndarray, expert_loads: jnp.ndarray, 
                        training: bool) -> Dict[str, Any]:
        """Compute MoE-specific metrics."""
        
        # Routing entropy (higher = more diverse routing)
        routing_probs = jnp.mean(routing_weights, axis=(0, 1))  # Average over batch and sequence
        routing_entropy = -jnp.sum(routing_probs * jnp.log(routing_probs + 1e-8))
        
        # Load balance loss (lower = better balance)
        total_load = jnp.sum(expert_loads)
        expected_load = total_load / self.config.num_experts
        load_balance_loss = jnp.sum(jnp.square(expert_loads - expected_load)) / self.config.num_experts
        
        # Active experts per token
        active_experts_per_token = jnp.sum(routing_weights > 1e-6, axis=-1)
        avg_active_experts = jnp.mean(active_experts_per_token)
        
        # Expert utilization efficiency
        non_zero_loads = jnp.sum(expert_loads > 0)
        utilization_efficiency = non_zero_loads / self.config.num_experts
        
        # Routing concentration (lower = more distributed)
        max_routing_weight = jnp.max(routing_probs)
        routing_concentration = max_routing_weight
        
        return {
            "routing_entropy": routing_entropy,
            "load_balance_loss": load_balance_loss,
            "active_experts_per_token": avg_active_experts,
            "utilization_efficiency": utilization_efficiency,
            "routing_concentration": routing_concentration,
            "total_tokens_processed": self.total_tokens_processed
        }
        
    def get_layer_stats(self) -> Dict[str, Any]:
        """Get comprehensive layer statistics."""
        expert_stats = [expert.get_stats() for expert in self.experts]
        
        # Aggregate expert statistics
        total_usage = sum(stats["usage_count"] for stats in expert_stats)
        avg_processing_time = sum(stats["avg_processing_time"] for stats in expert_stats) / len(expert_stats)
        
        # Most and least used experts
        most_used = max(expert_stats, key=lambda x: x["usage_count"])
        least_used = min(expert_stats, key=lambda x: x["usage_count"])
        
        return {
            "layer_id": self.layer_id,
            "total_tokens_processed": self.total_tokens_processed,
            "total_expert_usage": total_usage,
            "avg_expert_processing_time": avg_processing_time,
            "most_used_expert": most_used,
            "least_used_expert": least_used,
            "avg_routing_entropy": sum(self.routing_entropy_history) / max(len(self.routing_entropy_history), 1),
            "avg_load_balance_loss": sum(self.load_balance_history) / max(len(self.load_balance_history), 1),
            "expert_stats": expert_stats
        }
        
    def optimize_routing(self):
        """Optimize routing based on performance history."""
        if len(self.routing_entropy_history) < 10:
            return
            
        # Analyze routing patterns
        recent_entropy = self.routing_entropy_history[-10:]
        recent_balance = self.load_balance_history[-10:]
        
        avg_entropy = sum(recent_entropy) / len(recent_entropy)
        avg_balance = sum(recent_balance) / len(recent_balance)
        
        # Adjust routing temperature based on performance
        if avg_entropy < 2.0:  # Too concentrated
            self.router.config.routing_temperature *= 1.1
        elif avg_entropy > 4.0:  # Too dispersed
            self.router.config.routing_temperature *= 0.9
            
        # Update expert performance
        expert_loads = jnp.array([expert.usage_count for expert in self.experts])
        expert_times = jnp.array([expert.total_processing_time / max(expert.usage_count, 1) 
                                 for expert in self.experts])
        
        # Lower processing time = better performance
        expert_losses = expert_times / (jnp.mean(expert_times) + 1e-8)
        self.router.update_expert_performance(expert_loads, expert_losses)
        
        logger.info(f"MoE Layer {self.layer_id} optimization: "
                   f"entropy={avg_entropy:.2f}, balance={avg_balance:.2f}, "
                   f"temperature={self.router.config.routing_temperature:.2f}")


class MoEManager:
    """Manager for multiple MoE layers."""
    
    def __init__(self, config: MoEConfig):
        self.config = config
        self.moe_layers = {}
        self.global_stats = {
            "total_forward_passes": 0,
            "total_processing_time": 0.0,
            "optimization_runs": 0
        }
        
    def create_layer(self, layer_id: int) -> DynamicMoELayer:
        """Create a new MoE layer."""
        layer = DynamicMoELayer(self.config, layer_id)
        self.moe_layers[layer_id] = layer
        return layer
        
    def get_layer(self, layer_id: int) -> Optional[DynamicMoELayer]:
        """Get existing MoE layer."""
        return self.moe_layers.get(layer_id)
        
    def optimize_all_layers(self):
        """Optimize all MoE layers."""
        for layer in self.moe_layers.values():
            layer.optimize_routing()
        self.global_stats["optimization_runs"] += 1
        
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global MoE statistics."""
        layer_stats = {f"layer_{layer_id}": layer.get_layer_stats() 
                      for layer_id, layer in self.moe_layers.items()}
        
        # Aggregate statistics
        total_tokens = sum(layer.total_tokens_processed for layer in self.moe_layers.values())
        total_experts = len(self.moe_layers) * self.config.num_experts
        
        return {
            "global_stats": self.global_stats,
            "total_tokens_processed": total_tokens,
            "total_moe_layers": len(self.moe_layers),
            "total_experts": total_experts,
            "config": self.config,
            "layer_stats": layer_stats
        }
        
    def cleanup_unused_experts(self, min_usage_threshold: int = 10):
        """Clean up experts that are rarely used."""
        for layer in self.moe_layers.values():
            for expert in layer.experts:
                if expert.usage_count < min_usage_threshold:
                    # Reset expert parameters to encourage exploration
                    expert.w1 = jnp.ones_like(expert.w1) * 0.02
                    expert.w2 = jnp.ones_like(expert.w2) * 0.02
                    expert.usage_count = 0
                    expert.total_processing_time = 0.0
                    
        logger.info(f"Cleaned up unused experts (threshold: {min_usage_threshold})")


# Factory functions
def create_dynamic_moe_layer(config: MoEConfig, layer_id: int) -> DynamicMoELayer:
    """Factory function to create a dynamic MoE layer."""
    return DynamicMoELayer(config, layer_id)


def create_moe_manager(config: MoEConfig) -> MoEManager:
    """Factory function to create a MoE manager."""
    return MoEManager(config)
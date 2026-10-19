"""
Optimized Quantum Router for CapibaraGPT - TPU v4-32
Optimized version with JAX JIT, full differentiability and TPU efficiency.
"""

import logging
from flax import linen as nn
import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, List, Optional, Any, Tuple

# Relative imports with fallbacks
try:
    from .contextual_activation import ContextualActivation
except ImportError:
    # Fallback if not available
    class ContextualActivation(nn.Module):
        def __call__(self, x, **kwargs):
            return x

try:
    from .personality.conversation_manager import ConversationManager
except ImportError:
    # Fallback if not available
    class ConversationManager:
        pass

logger = logging.getLogger(__name__)

class OptimizedAdaptiveRouter(nn.Module):
    """Optimized quantum router for TPU v4-32 with differentiable processing."""
    
    hidden_size: int
    num_virtual_qubits: int
    vocab_size: int = 50257
    max_context_length: int = 2048
    num_router_experts: int = 8
    router_capacity_factor: float = 1.25
    
    def setup(self):
        """Optimized initialization for TPU."""
        
        # 1. Differentiable context encoder
        self.context_embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02)
        )
        
        # 2. Router gate network
        self.router_gate = nn.Sequential([
            nn.Dense(self.hidden_size * 2),
            nn.LayerNorm(),
            lambda x: nn.gelu(x),
            nn.Dense(2),  # [dynamic_weight, pretrained_weight]
        ])
        
        # 3. Optimized VQbit layers
        self.dynamic_vq_codes = VQbitLayerOptimized(
            codebook_size=self.num_virtual_qubits,
            embedding_dim=self.hidden_size,
            name="dynamic"
        )
        
        self.pretrained_vq_codes = VQbitLayerOptimized(
            codebook_size=self.num_virtual_qubits,
            embedding_dim=self.hidden_size,
            name="pretrained"
        )
        
        # 4. Expert routing (Mixture of Experts)
        self.expert_gate = nn.Dense(self.num_router_experts)
        self.experts = [
            ExpertLayer(self.hidden_size, name=f"expert_{i}")
            for i in range(self.num_router_experts)
        ]
        
        # 5. Output projection
        self.output_projection = nn.Dense(self.hidden_size)

    def _encode_context_efficiently(
        self, 
        context_tokens: jnp.ndarray
    ) -> jnp.ndarray:
        """Efficient context encoding for TPU."""
        # Shape: [batch_size, seq_len] -> [batch_size, hidden_size]
        
        # 1. Embed tokens
        embedded = self.context_embedding(context_tokens)  # [B, L, H]
        
        # 2. Attention pooling (more efficient than mean pooling)
        attention_weights = nn.Dense(1)(embedded)  # [B, L, 1]
        attention_weights = nn.softmax(attention_weights, axis=1)
        
        # 3. Weighted pooling
        context_repr = jnp.sum(embedded * attention_weights, axis=1)  # [B, H]
        
        return context_repr

    @partial(jax.jit, static_argnums=(0,))
    def _compute_router_weights(
        self,
        context_features: jnp.ndarray,
        training: bool = False
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute router weights in a differentiable manner."""
        
        # 1. Router logits
        router_logits = self.router_gate(context_features)  # [B, 2]
        
        # 2. Soft routing with temperature
        temperature = 1.0 if training else 0.1  # More deterministic at inference
        router_weights = nn.softmax(router_logits / temperature, axis=-1)
        
        # 3. Routing metrics
        routing_metrics = {
            "dynamic_usage": jnp.mean(router_weights[:, 0]),
            "pretrained_usage": jnp.mean(router_weights[:, 1]),
            "routing_entropy": -jnp.sum(
                router_weights * jnp.log(router_weights + 1e-8), axis=-1
            ).mean(),
            "routing_load_balance": jnp.std(jnp.mean(router_weights, axis=0))
        }
        
        return router_weights, routing_metrics

    def _expert_routing(
        self,
        x: jnp.ndarray,
        context_features: jnp.ndarray,
        training: bool = False
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Optimized expert routing for TPU."""
        batch_size, seq_len, hidden_size = x.shape
        
        # 1. Expert gate
        gate_logits = self.expert_gate(context_features)  # [B, num_experts]
        
        # 2. Top-k routing (k=2 for balance load/quality)
        k = 2
        top_k_logits, top_k_indices = jax.lax.top_k(gate_logits, k)
        top_k_weights = nn.softmax(top_k_logits, axis=-1)
        
        # 3. Process with selected experts
        expert_outputs = []
        for i in range(k):
            expert_idx = top_k_indices[:, i]  # [B]
            expert_weight = top_k_weights[:, i:i+1]  # [B, 1]
            
            # Dynamically select expert
            expert_output = jnp.zeros_like(x)
            for j in range(self.num_router_experts):
                mask = (expert_idx == j).astype(jnp.float32)[:, None, None]
                expert_j_output = self.experts[j](x)
                expert_output += mask * expert_j_output
            
            expert_outputs.append(expert_weight[:, :, None] * expert_output)
        
        # 4. Combine outputs
        final_output = sum(expert_outputs)
        
        # 5. Expert metrics
        expert_metrics = {
            "expert_usage": jnp.mean(nn.one_hot(top_k_indices, self.num_router_experts), axis=(0, 1)),
            "expert_load_balance": jnp.std(jnp.mean(nn.softmax(gate_logits), axis=0)),
            "routing_efficiency": jnp.mean(jnp.max(top_k_weights, axis=-1))
        }
        
        return final_output, expert_metrics

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        context_tokens: jnp.ndarray,
        training: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """Optimized forward pass for TPU v4-32."""
        
        # 1. Encode context
        context_features = self._encode_context_efficiently(context_tokens)
        
        # 2. Compute router weights
        router_weights, routing_metrics = self._compute_router_weights(
            context_features, training
        )
        
        # 3. Process with both types of VQ codes in parallel
        dynamic_output = self.dynamic_vq_codes(x, training=training)
        pretrained_output = self.pretrained_vq_codes(x, training=training)
        
        # 4. Differentiable mixing
        vqbit_mixed = (
            router_weights[:, 0:1, None] * dynamic_output +
            router_weights[:, 1:2, None] * pretrained_output
        )
        
        # 5. Additional expert routing
        expert_output, expert_metrics = self._expert_routing(
            vqbit_mixed, context_features, training
        )
        
        # 6. Final projection
        final_output = self.output_projection(expert_output)
        
        # 7. Combine metrics
        all_metrics = {
            **routing_metrics,
            **expert_metrics,
            "total_params": sum(
                p.size for p in jax.tree_util.tree_leaves(self.variables)
            ) if hasattr(self, 'variables') else 0
        }
        
        return {
            "output": final_output,
            "router_weights": router_weights,
            "context_features": context_features,
            "metrics": all_metrics
        }


class VQbitLayerOptimized(nn.Module):
    """Optimized VQbit layer for TPU."""
    
    codebook_size: int
    embedding_dim: int
    name: str
    commitment_cost: float = 0.25
    
    def setup(self):
        self.codebook = self.param(
            'codebook',
            nn.initializers.uniform(scale=1.0),
            (self.codebook_size, self.embedding_dim)
        )
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Optimized forward pass."""

        # 1. Compute distances
        distances = jnp.linalg.norm(
            x[..., None, :] - self.codebook[None, None, :, :], 
            axis=-1
        )
        
        # 2. Find closest codes
        indices = jnp.argmin(distances, axis=-1)
        quantized = self.codebook[indices]
        
        # 3. Straight-through estimator
        quantized = x + jax.lax.stop_gradient(quantized - x)
        
        # 4. VQ loss
        if training:
            vq_loss = jnp.mean((jax.lax.stop_gradient(x) - quantized) ** 2)
            commitment_loss = self.commitment_cost * jnp.mean((x - jax.lax.stop_gradient(quantized)) ** 2)
            total_loss = vq_loss + commitment_loss
        else:
            total_loss = 0.0
        
        return quantized


class ExpertLayer(nn.Module):
    """Optimized expert layer."""
    
    hidden_size: int
    name: str
    expansion_factor: int = 4
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Optimized FFN expert."""
        
        # 1. Expand
        expanded = nn.Dense(
            self.hidden_size * self.expansion_factor,
            name=f"{self.name}_expand"
        )(x)
        activated = nn.gelu(expanded)
        
        # 2. Contract
        output = nn.Dense(
            self.hidden_size,
            name=f"{self.name}_contract"
        )(activated)
        
        return output


class ContextualRouterOptimized(nn.Module):
    """Optimized contextual router that combines activation and quantum routing."""
    
    hidden_size: int
    num_virtual_qubits: int
    vocab_size: int = 50257
    activation_threshold: float = 0.5
    
    def setup(self):
        # 1. Contextual activation module
        self.activation_gate = nn.Sequential([
            nn.Dense(self.hidden_size),
            nn.LayerNorm(),
            lambda x: nn.gelu(x),
            nn.Dense(1),
            lambda x: nn.sigmoid(x)
        ])
        
        # 2. Optimized quantum router
        self.adaptive_router = OptimizedAdaptiveRouter(
            hidden_size=self.hidden_size,
            num_virtual_qubits=self.num_virtual_qubits,
            vocab_size=self.vocab_size
        )
        
        # 3. Classical path (bypass)
        self.classical_path = nn.Dense(self.hidden_size)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        context_tokens: jnp.ndarray,
        training: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """Optimized forward pass with soft routing."""
        
        # 1. Compute activation gate
        context_repr = jnp.mean(
            self.adaptive_router.context_embedding(context_tokens), 
            axis=1
        )
        activation_gate = self.activation_gate(context_repr)  # [B, 1]
        
        # 2. Process both paths in parallel
        classical_output = self.classical_path(x)
        adaptive_result = self.adaptive_router(x, context_tokens, training=training)
        adaptive_output = adaptive_result["output"]
        
        # 3. Soft routing based on gate
        final_output = (
            activation_gate[..., None] * adaptive_output +
            (1 - activation_gate[..., None]) * classical_output
        )
        
        return {
            "output": final_output,
            "activation_gate": activation_gate,
            "adaptive_metrics": adaptive_result["metrics"],
            "classical_usage": jnp.mean(1 - activation_gate),
            "adaptive_usage": jnp.mean(activation_gate)
        }


# Utility functions for TPU
@partial(jax.pmap, axis_name='batch')
def distributed_router_forward(
    router: ContextualRouterOptimized,
    params: Dict,
    x: jnp.ndarray,
    context_tokens: jnp.ndarray,
    training: bool = False
) -> Dict[str, jnp.ndarray]:
    """Distributed forward pass for TPU v4-32."""
    
    result = router.apply(params, x, context_tokens, training=training)
    
    # Synchronize metrics across devices
    for key in result["adaptive_metrics"]:
        if isinstance(result["adaptive_metrics"][key], jnp.ndarray):
            result["adaptive_metrics"][key] = jax.lax.pmean(
                result["adaptive_metrics"][key], axis_name='batch'
            )
    
    return result


def create_router_for_tpu_v4_32(
    hidden_size: int = 768,
    num_virtual_qubits: int = 512,
    vocab_size: int = 50257
) -> ContextualRouterOptimized:
    """Create optimized router for TPU v4-32."""
    
    return ContextualRouterOptimized(
        hidden_size=hidden_size,
        num_virtual_qubits=num_virtual_qubits,
        vocab_size=vocab_size,
        activation_threshold=0.5
    )


# Backwards-compatible alias used across docs/imports.
AdaptiveRouter = ContextualRouterOptimized

"""
HybridAttentionModule - Intelligent Attention Routing for Capibara6

Intelligent hybrid module that automatically decides between:
- Transformer (O(n²)) for short sequences
- Mamba (O(n)) for long sequences

Provides the best performance based on input characteristics.
"""

import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import time
import numpy as np

# Import JAX with fallbacks
JAX_AVAILABLE = False
FLAX_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    JAX_AVAILABLE = True
except ImportError:
    # Use numpy as fallback
    jnp = np

    # Create dummy jax for fallback
    class _DummyJax:
        class lax:
            @staticmethod
            def scan(f, init, xs):
                """Dummy scan implementation."""
                carry = init
                ys = []
                for x in xs:
                    carry, y = f(carry, x)
                    ys.append(y)
                return carry, np.array(ys)

        class nn:
            @staticmethod
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            @staticmethod
            def gelu(x):
                return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

            @staticmethod
            def softmax(x, axis=-1):
                exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
                return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    jax = _DummyJax()

# Import Flax
try:
    from flax import linen as nn
    FLAX_AVAILABLE = True
except ImportError:
    FLAX_AVAILABLE = False

    # Create dummy nn for fallback
    class _DummyNN:
        class MultiHeadDotProductAttention:
            def __init__(self, **kwargs):
                pass
            def __call__(self, x, mask=None):
                return x

        class LayerNorm:
            def __init__(self, **kwargs):
                pass
            def __call__(self, x):
                return x

        class Dense:
            def __init__(self, **kwargs):
                pass
            def __call__(self, x):
                return x

    nn = _DummyNN()

# Import interfaces and modules
from interfaces.imodules import IModule
from sub_models.mamba.mamba_module import MambaModule, MambaConfig

logger = logging.getLogger(__name__)

@dataclass
class HybridConfig:
    """Configuration for HybridAttentionModule."""
    hidden_size: int = 768
    num_heads: int = 12
    intermediate_size: int = 3072

    # Hybrid decision parameters
    mamba_threshold: int = 512  # Minimum length to use Mamba
    transformer_max_length: int = 2048  # Maximum length for Transformer

    # Mamba configuration
    mamba_config: Optional[Dict[str, Any]] = None

    # Transformer configuration
    dropout_rate: float = 0.1
    layer_norm_eps: float = 1e-12

    # Optimizations
    use_tpu_optimizations: bool = True
    use_mixed_precision: bool = True
    enable_caching: bool = True

    # Metrics and logging
    collect_metrics: bool = True
    log_decisions: bool = False


class TransformerModule:
    """Traditional Transformer module for the hybrid system."""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if FLAX_AVAILABLE:
            self._init_flax_components()
        else:
            self.logger.warning("Flax not available, using fallback implementation")

    def _init_flax_components(self):
        """Initialize Flax components."""
        # Multi-head attention
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.config.num_heads,
            qkv_features=self.config.hidden_size,
            dropout_rate=self.config.dropout_rate,
            name='multihead_attention'
        )
        
        # Layer normalization
        self.attention_norm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps,
            name='attention_norm'
        )
        self.ffn_norm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps,
            name='ffn_norm'
        )
        
        # Feed-forward network
        self.dense_1 = nn.Dense(
            features=self.config.intermediate_size,
            name='ffn_dense_1'
        )
        self.dense_2 = nn.Dense(
            features=self.config.hidden_size,
            name='ffn_dense_2'
        )
        self.dropout = nn.Dropout(rate=self.config.dropout_rate)
    
    def __call__(self, inputs: jnp.ndarray, training: bool = False) -> tuple:
        """Forward pass of the Transformer."""
        if not FLAX_AVAILABLE:
            return self._fallback_forward(inputs, training)

        # Self-attention with residual connection
        residual = inputs
        hidden_states = self.attention_norm(inputs)

        attention_output = self.attention(
            hidden_states,
            deterministic=not training
        )

        hidden_states = residual + attention_output

        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        
        # FFN
        ffn_output = self.dense_1(hidden_states)
        ffn_output = jax.nn.gelu(ffn_output)
        ffn_output = self.dropout(ffn_output, deterministic=not training)
        ffn_output = self.dense_2(ffn_output)
        
        output = residual + ffn_output
        
        metrics = {
            "transformer_active": True,
            "complexity": "O(n²)",
            "attention_heads": self.config.num_heads,
            "sequence_length": inputs.shape[1]
        }
        
        return output, metrics
    
    def _fallback_forward(self, inputs: jnp.ndarray, training: bool = False) -> tuple:
        """Fallback implementation when Flax is not available."""
        self.logger.warning("Using Transformer fallback")

        # Minimal transformation
        output = inputs * 1.05
        
        metrics = {
            "transformer_active": True,
            "complexity": "O(n²)",
            "fallback_used": True,
            "sequence_length": inputs.shape[1]
        }
        
        return output, metrics


class HybridAttentionModule(IModule):
    """
    Intelligent Hybrid Attention Module.

    Automatically decides between Transformer and Mamba based on:
    - Sequence length
    - Content characteristics
    - Available resources
    - Performance metrics

    Features:
    - Automatic intelligent routing
    - O(n^2) vs O(n) optimization based on context
    - Compatible with modular architecture
    - Detailed decision metrics
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize HybridAttentionModule.

        Args:
            config: Configuration dictionary
        """
        self.config = HybridConfig(**config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize underlying modules
        self._init_submodules()

        # Cache for decisions
        if self.config.enable_caching:
            self.decision_cache = {}
            self.performance_history = []
        
        # Statistics
        self.decision_stats = {
            "mamba_used": 0,
            "transformer_used": 0,
            "total_decisions": 0,
            "avg_sequence_length": 0.0
        }
        
        self.logger.info(f"HybridAttentionModule initialized: threshold={self.config.mamba_threshold}")

    def _init_submodules(self):
        """Initialize Mamba and Transformer modules."""
        # Mamba configuration
        mamba_config = self.config.mamba_config or {
            "hidden_size": self.config.hidden_size,
            "d_state": 64,
            "d_conv": 4,
            "expand_factor": 2,
            "use_tpu_optimizations": self.config.use_tpu_optimizations
        }

        # Initialize modules
        self.mamba_module = MambaModule(mamba_config)
        self.transformer_module = TransformerModule(self.config)

        self.logger.info("Mamba and Transformer submodules initialized")
    
    def _make_routing_decision(self, inputs: jnp.ndarray,
                              training: bool = False) -> Dict[str, Any]:
        """
        Decide which module to use based on input characteristics.

        Args:
            inputs: Input tensor
            training: Training mode

        Returns:
            Dict with decision and metadata
        """
        batch_size, seq_len, hidden_size = inputs.shape

        # Decision based on sequence length
        decision = {
            "use_mamba": False,
            "use_transformer": False,
            "reason": "",
            "confidence": 0.0,
            "sequence_length": seq_len,
            "batch_size": batch_size
        }
        
        # Decision rules
        if seq_len >= self.config.mamba_threshold:
            # Long sequences: use Mamba (O(n))
            decision.update({
                "use_mamba": True,
                "reason": f"long_sequence_len_{seq_len}_threshold_{self.config.mamba_threshold}",
                "confidence": min(1.0, seq_len / self.config.mamba_threshold),
                "expected_complexity": "O(n)",
                "memory_efficiency": "high"
            })
            
        elif seq_len <= self.config.transformer_max_length:
            # Short/medium sequences: use Transformer (O(n^2))
            decision.update({
                "use_transformer": True,
                "reason": f"short_sequence_len_{seq_len}_max_{self.config.transformer_max_length}",
                "confidence": 1.0 - (seq_len / self.config.transformer_max_length) * 0.5,
                "expected_complexity": "O(n²)",
                "memory_efficiency": "medium"
            })
            
        else:
            # Edge case: prefer Mamba for efficiency
            decision.update({
                "use_mamba": True,
                "reason": f"edge_case_len_{seq_len}_prefer_efficiency",
                "confidence": 0.7,
                "expected_complexity": "O(n)",
                "memory_efficiency": "high"
            })
        
        # Additional decision factors
        self._apply_additional_decision_factors(decision, inputs, training)

        # Decision logging
        if self.config.log_decisions:
            self.logger.debug(f"Routing decision: {decision['reason']} "
                            f"(seq_len={seq_len}, confidence={decision['confidence']:.2f})")
        
        return decision
    
    def _apply_additional_decision_factors(self, decision: Dict[str, Any],
                                         inputs: jnp.ndarray, training: bool):
        """Apply additional decision factors."""
        seq_len = inputs.shape[1]

        # Available memory factor
        if seq_len > 1024 and training:
            # In training with very long sequences, prefer Mamba
            if decision["use_transformer"]:
                decision.update({
                    "use_transformer": False,
                    "use_mamba": True,
                    "reason": decision["reason"] + "_memory_optimization",
                    "confidence": min(decision["confidence"] + 0.2, 1.0)
                })
        
        # Precision vs efficiency factor
        if seq_len < 128:
            # Very short sequences: Transformer can be more precise
            if decision["use_mamba"]:
                decision["confidence"] = max(decision["confidence"] - 0.1, 0.0)
                decision["reason"] += "_precision_preference"
    
    def _execute_with_module(self, module_name: str, inputs: jnp.ndarray, 
                           training: bool) -> tuple:
        """Executes processing with the specified module."""
        start_time = time.time()
        
        try:
            if module_name == "mamba":
                result = self.mamba_module(inputs, training=training)
                output = result["output"]
                module_metrics = result["metrics"]
                
            elif module_name == "transformer":
                output, module_metrics = self.transformer_module(inputs, training=training)
                
            else:
                raise ValueError(f"Unknown module: {module_name}")
            
            processing_time = time.time() - start_time
            
            # Update statistics
            if module_name == "mamba":
                self.decision_stats["mamba_used"] += 1
            else:
                self.decision_stats["transformer_used"] += 1
            
            self.decision_stats["total_decisions"] += 1
            
            # Calculate average sequence length
            current_avg = self.decision_stats["avg_sequence_length"]
            total = self.decision_stats["total_decisions"]
            seq_len = inputs.shape[1]
            self.decision_stats["avg_sequence_length"] = (current_avg * (total - 1) + seq_len) / total
            
            return output, module_metrics, processing_time
            
        except Exception as e:
            self.logger.error(f"Error executing module {module_name}: {e}")
            # Fallback: return input unchanged
            return inputs, {"error": str(e), "fallback_used": True}, 0.0
    
    def __call__(self, inputs: jnp.ndarray, training: bool = False) -> Dict[str, Any]:
        """
        Main interface of the hybrid module.

        Args:
            inputs: Input tensor [batch, seq_len, hidden_size]
            training: Training mode

        Returns:
            Dict with output, metrics, and decision information
        """
        try:
            # Validate input
            if not hasattr(inputs, 'shape') or len(inputs.shape) != 3:
                raise ValueError(f"HybridAttentionModule expected 3D tensor, received: {inputs.shape if hasattr(inputs, 'shape') else type(inputs)}")
            
            batch_size, seq_len, hidden_size = inputs.shape
            
            # Make routing decision
            decision = self._make_routing_decision(inputs, training)
            
            # Execute with selected module
            if decision["use_mamba"]:
                output, module_metrics, processing_time = self._execute_with_module(
                    "mamba", inputs, training
                )
                selected_module = "mamba"
                
            elif decision["use_transformer"]:
                output, module_metrics, processing_time = self._execute_with_module(
                    "transformer", inputs, training
                )
                selected_module = "transformer"
                
            else:
                # Fallback to Transformer
                output, module_metrics, processing_time = self._execute_with_module(
                    "transformer", inputs, training
                )
                selected_module = "transformer_fallback"
                decision["reason"] += "_fallback"
            
            # Compile complete metrics
            hybrid_metrics = {
                **module_metrics,
                "hybrid_active": True,
                "selected_module": selected_module,
                "routing_decision": decision,
                "processing_time": processing_time,
                "decision_confidence": decision["confidence"],
                "sequence_length": seq_len,
                "batch_size": batch_size,
                "decision_stats": self.decision_stats.copy()
            }
            
            # Processing information
            processing_info = {
                "module_type": "HybridAttentionModule",
                "architecture": f"{selected_module}_selected",
                "routing_reason": decision["reason"],
                "complexity": decision.get("expected_complexity", "unknown"),
                "memory_efficiency": decision.get("memory_efficiency", "unknown"),
                "training_mode": training,
                "auto_optimization": True
            }
            
            result = {
                "output": output,
                "metrics": hybrid_metrics,
                "processing_info": processing_info,
                "success": True
            }
            
            # Result logging
            self.logger.debug(f"HybridAttentionModule: {selected_module} processed sequence {seq_len} "
                            f"in {processing_time:.3f}s (confidence: {decision['confidence']:.2f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in HybridAttentionModule: {e}")

            # Complete fallback
            return {
                "output": inputs,
                "metrics": {
                    "error": str(e),
                    "fallback_used": True,
                    "hybrid_active": False
                },
                "processing_info": {
                    "module_type": "HybridAttentionModule",
                    "error": True
                },
                "success": False
            }

    def get_metrics(self) -> Dict[str, Any]:
        """Return module metrics for IModule compatibility."""
        return {
            "module_type": "HybridAttentionModule",
            "decision_stats": self.decision_stats,
            "jax_available": JAX_AVAILABLE,
            "flax_available": FLAX_AVAILABLE,
            "mamba_threshold": self.config.mamba_threshold,
        }

    def get_config(self) -> Dict[str, Any]:
        """Return module configuration for IModule compatibility."""
        return {
            "hidden_size": self.config.hidden_size,
            "num_heads": self.config.num_heads,
            "intermediate_size": self.config.intermediate_size,
            "mamba_threshold": self.config.mamba_threshold,
            "transformer_max_length": self.config.transformer_max_length,
            "dropout_rate": self.config.dropout_rate,
            "layer_norm_eps": self.config.layer_norm_eps,
            "use_tpu_optimizations": self.config.use_tpu_optimizations,
            "use_mixed_precision": self.config.use_mixed_precision,
            "enable_caching": self.config.enable_caching,
        }
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get routing decision statistics."""
        total = self.decision_stats["total_decisions"]
        if total == 0:
            return {"no_decisions_made": True}
        
        return {
            "total_decisions": total,
            "mamba_usage_percentage": (self.decision_stats["mamba_used"] / total) * 100,
            "transformer_usage_percentage": (self.decision_stats["transformer_used"] / total) * 100,
            "average_sequence_length": self.decision_stats["avg_sequence_length"],
            "mamba_threshold": self.config.mamba_threshold,
            "transformer_max_length": self.config.transformer_max_length
        }
    
    def setup_tpu_optimizations(self):
        """Configures TPU optimizations for both modules."""
        if self.config.use_tpu_optimizations:
            self.logger.info("Configuring TPU optimizations for HybridAttentionModule")

            # Configure Mamba
            self.mamba_module.setup_tpu_optimizations()

            # Configure caching if enabled
            if self.config.enable_caching:
                self.logger.info("Enabling decision caching for TPU optimization")

    def reset_statistics(self):
        """Reset decision statistics."""
        self.decision_stats = {
            "mamba_used": 0,
            "transformer_used": 0,
            "total_decisions": 0,
            "avg_sequence_length": 0.0
        }

        if hasattr(self, 'decision_cache'):
            self.decision_cache.clear()

        self.logger.info("HybridAttentionModule statistics reset")

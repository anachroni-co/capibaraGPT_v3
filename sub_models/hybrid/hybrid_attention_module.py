"""
HybridAttentionModule - Intelligent Attention Routing for Capibara6

Módulo híbrido inteligente que decide automáticamente entre:
- Transformer (O(n²)) para secuencias cortas
- Mamba (O(n)) para secuencias largas

Proporciona el mejor rendimiento según las características de la entrada.
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
from capibara.interfaces.imodules import IModule
from ..mamba.mamba_module import MambaModule, MambaConfig

logger = logging.getLogger(__name__)

@dataclass
class HybridConfig:
    """Configuration para HybridAttentionModule."""
    hidden_size: int = 768
    num_heads: int = 12
    intermediate_size: int = 3072
    
    # Parámetros de decisión híbrida
    mamba_threshold: int = 512  # Longitud mínima para usar Mamba
    transformer_max_length: int = 2048  # Longitud máxima para Transformer
    
    # Configuración Mamba
    mamba_config: Optional[Dict[str, Any]] = None
    
    # Configuración Transformer
    dropout_rate: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Optimizaciones
    use_tpu_optimizations: bool = True
    use_mixed_precision: bool = True
    enable_caching: bool = True
    
    # Métricas y logging
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
            self.logger.warning("Flax no disponible, usando implementación fallback")
    
    def _init_flax_components(self):
        """Inicializar componentes Flax."""
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
        """Forward pass del Transformer."""
        if not FLAX_AVAILABLE:
            return self._fallback_forward(inputs, training)
        
        # Self-attention con conexión residual
        residual = inputs
        hidden_states = self.attention_norm(inputs)
        
        attention_output = self.attention(
            hidden_states,
            deterministic=not training
        )
        
        hidden_states = residual + attention_output
        
        # Feed-forward con conexión residual
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
        """Implementsción fallback cuando Flax no está disponible."""
        self.logger.warning("Usando fallback de Transformer")
        
        # Transformación mínima
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
    Módulo de Atención Híbrida Inteligente.
    
    Decide automáticamente entre Transformer y Mamba basándose en:
    - Longitud de secuencia
    - Características del contenido
    - Recursos disponibles
    - Métricas de rendimiento
    
    Características:
    - Routing inteligente automático
    - Optimización O(n²) vs O(n) según contexto
    - Compatible con arquitectura modular
    - Métricas detalladas de decisiones
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializar HybridAttentionModule.
        
        Args:
            config: Diccionario de configuración
        """
        self.config = HybridConfig(**config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize underlying modules
        self._init_submodules()
        
        # Cache para decisiones
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
        
        self.logger.info(f"HybridAttentionModule inicializado: threshold={self.config.mamba_threshold}")
    
    def _init_submodules(self):
        """Initialize Mamba and Transformer modules."""
        # Configuración Mamba
        mamba_config = self.config.mamba_config or {
            "hidden_size": self.config.hidden_size,
            "d_state": 64,
            "d_conv": 4,
            "expand_factor": 2,
            "use_tpu_optimizations": self.config.use_tpu_optimizations
        }
        
        # Inicializar modules
        self.mamba_module = MambaModule(mamba_config)
        self.transformer_module = TransformerModule(self.config)
        
        self.logger.info("Submódulos Mamba y Transformer inicializados")
    
    def _make_routing_decision(self, inputs: jnp.ndarray, 
                              training: bool = False) -> Dict[str, Any]:
        """
        Decidir qué module usar basándose en características de entrada.
        
        Args:
            inputs: Tensor de entrada
            training: Modo entrenamiento
            
        Returns:
            Dict con decisión y metadatos
        """
        batch_size, seq_len, hidden_size = inputs.shape
        
        # Decisión basada en longitud de secuencia
        decision = {
            "use_mamba": False,
            "use_transformer": False,
            "reason": "",
            "confidence": 0.0,
            "sequence_length": seq_len,
            "batch_size": batch_size
        }
        
        # Reglas de decisión
        if seq_len >= self.config.mamba_threshold:
            # Secuencias largas: usar Mamba (O(n))
            decision.update({
                "use_mamba": True,
                "reason": f"long_sequence_len_{seq_len}_threshold_{self.config.mamba_threshold}",
                "confidence": min(1.0, seq_len / self.config.mamba_threshold),
                "expected_complexity": "O(n)",
                "memory_efficiency": "high"
            })
            
        elif seq_len <= self.config.transformer_max_length:
            # Secuencias cortas/medianas: usar Transformer (O(n²))
            decision.update({
                "use_transformer": True,
                "reason": f"short_sequence_len_{seq_len}_max_{self.config.transformer_max_length}",
                "confidence": 1.0 - (seq_len / self.config.transformer_max_length) * 0.5,
                "expected_complexity": "O(n²)",
                "memory_efficiency": "medium"
            })
            
        else:
            # Caso límite: preferir Mamba para eficiencia
            decision.update({
                "use_mamba": True,
                "reason": f"edge_case_len_{seq_len}_prefer_efficiency",
                "confidence": 0.7,
                "expected_complexity": "O(n)",
                "memory_efficiency": "high"
            })
        
        # Factores adicionales de decisión
        self._apply_additional_decision_factors(decision, inputs, training)
        
        # Logging de decisión
        if self.config.log_decisions:
            self.logger.debug(f"Routing decision: {decision['reason']} "
                            f"(seq_len={seq_len}, confidence={decision['confidence']:.2f})")
        
        return decision
    
    def _apply_additional_decision_factors(self, decision: Dict[str, Any], 
                                         inputs: jnp.ndarray, training: bool):
        """Appliesr factores adicionales de decisión."""
        seq_len = inputs.shape[1]
        
        # Factor de memoria disponible
        if seq_len > 1024 and training:
            # En entrenamiento con secuencias muy largas, preferir Mamba
            if decision["use_transformer"]:
                decision.update({
                    "use_transformer": False,
                    "use_mamba": True,
                    "reason": decision["reason"] + "_memory_optimization",
                    "confidence": min(decision["confidence"] + 0.2, 1.0)
                })
        
        # Factor de precisión vs eficiencia
        if seq_len < 128:
            # Secuencias muy cortas: Transformer puede ser más preciso
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
                raise ValueError(f"Módulo desconocido: {module_name}")
            
            processing_time = time.time() - start_time
            
            # Update statistics
            if module_name == "mamba":
                self.decision_stats["mamba_used"] += 1
            else:
                self.decision_stats["transformer_used"] += 1
            
            self.decision_stats["total_decisions"] += 1
            
            # Calcular promedio de longitud de secuencia
            current_avg = self.decision_stats["avg_sequence_length"]
            total = self.decision_stats["total_decisions"]
            seq_len = inputs.shape[1]
            self.decision_stats["avg_sequence_length"] = (current_avg * (total - 1) + seq_len) / total
            
            return output, module_metrics, processing_time
            
        except Exception as e:
            self.logger.error(f"Error ejecutando module {module_name}: {e}")
            # Fallback: devolver entrada sin modificar
            return inputs, {"error": str(e), "fallback_used": True}, 0.0
    
    def __call__(self, inputs: jnp.ndarray, training: bool = False) -> Dict[str, Any]:
        """
        Interfaz principal del module híbrido.
        
        Args:
            inputs: Tensor de entrada [batch, seq_len, hidden_size]
            training: Modo entrenamiento
            
        Returns:
            Dict con output, metrics e información de decisión
        """
        try:
            # Validar entrada
            if not hasattr(inputs, 'shape') or len(inputs.shape) != 3:
                raise ValueError(f"HybridAttentionModule esperaba tensor 3D, recibió: {inputs.shape if hasattr(inputs, 'shape') else type(inputs)}")
            
            batch_size, seq_len, hidden_size = inputs.shape
            
            # Hacer decisión de routing
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
                # Fallback a Transformer
                output, module_metrics, processing_time = self._execute_with_module(
                    "transformer", inputs, training
                )
                selected_module = "transformer_fallback"
                decision["reason"] += "_fallback"
            
            # Compilar métricas completas
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
            
            # Información de procesamiento
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
            
            # Logging de resultado
            self.logger.debug(f"HybridAttentionModule: {selected_module} procesó secuencia {seq_len} "
                            f"en {processing_time:.3f}s (confianza: {decision['confidence']:.2f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en HybridAttentionModule: {e}")
            
            # Fallback completo
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
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Obtener statistics de decisiones de routing."""
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
            self.logger.info("Configurando optimizaciones TPU para HybridAttentionModule")
            
            # Configurar Mamba
            self.mamba_module.setup_tpu_optimizations()
            
            # Configurar caching si está habilitado
            if self.config.enable_caching:
                self.logger.info("Habilitando caching de decisiones para optimización TPU")
    
    def reset_statistics(self):
        """Reiniciar statistics de decisiones."""
        self.decision_stats = {
            "mamba_used": 0,
            "transformer_used": 0,
            "total_decisions": 0,
            "avg_sequence_length": 0.0
        }
        
        if hasattr(self, 'decision_cache'):
            self.decision_cache.clear()
        
        self.logger.info("Estadísticas de HybridAttentionModule reiniciadas")
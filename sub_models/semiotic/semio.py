"""
Módulo semiótico configurable for CapibaraGPT.

Este file implementa el submódulo SemioModule, encargado del analysis semiótico advanced
with soporte for interpretación literal, cultural and simbólica, integration de atención multi-cabeza,
proyección polisemántica and métricas especializadas. Optimizado for execution en tpu and gpu.

Características principales:
- analysis semiótico configurable and extensible.
- Soporte for diferentes tipos de interpretación (literal, cultural, simbólica).
- integration de atención multi-cabeza and proyecciones polisemánticas.
- Métricas detalladas for seguimiento de interpretaciones and uso de contexto.
- optimization for hardware acelerado (tpu/gpu).

example de uso:
    >>> model = SemioModule(hidden_size=256, tpu_optimized=True)
    >>> x = jnp.ones((1, 10, 256))
    >>> context = jnp.ones((1, 5, 256))
    >>> params = model.init(rng, x, context)
    >>> outputs = model.apply(params, x, context, training=True)
"""

import os
import sys
import jax
import logging
logger = logging.getLogger(__name__)

# Obtiene la path del directory current (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Sube un level for obtain la raíz del proyecto -> /.../CapibaraGPT v3
project_root = os.path.dirname(script_dir)
# Añade la raíz del proyecto a sys.path
if project_root not in sys.path:
    pass  # Using proper imports instead of sys.path manipulation

from flax import linen as nn
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from capibara.interfaces.isub_models import ISubModel
from typing import Dict, Any, Optional, Tuple, Set, Callable
from capibara.core.arm_optimizations import ARM_OPTIMIZATIONS, HARDWARE_INFO

METRICS_KEY = "intermediates"

class SemioModule(nn.Module, ISubModel):
    """
    Módulo de analysis semiótico configurable and optimizado for tpu/gpu.

    Permite perform interpretaciones literales, culturales and simbólicas about secuencias de input,
    integrando atención multi-cabeza, proyecciones polisemánticas and métricas especializadas.

    Args:
        hidden_size (int): dimension del espacio oculto.
        dropout_rate (float): Tasa de dropout.
        num_heads (int): number of cabezas de atención.
        tpu_optimized (bool): if True, usa optimizaciones for tpu.
        log_dir (str): directory for logs.
        max_log_size (int): size maximum de log.
        interpretation_types (Optional[Set[str]]): Tipos de interpretación a use.
        attention_strategy (Optional[Callable]): Estrategia de atención personalizada.
    """
    hidden_size: int = 256
    dropout_rate: float = 0.1
    num_heads: int = 4
    tpu_optimized: bool = False
    log_dir: str = ""
    max_log_size: int = 10*1024*1024
    interpretation_types: Optional[Set[str]] = None
    attention_strategy: Optional[Callable] = None

    def setup(self):
        self.interpretation_types = self.interpretation_types or {"literal", "cultural", "simbólica"}
        self.context_encoder = nn.Dense(self.hidden_size)
        self.input_encoder = nn.Dense(self.hidden_size)
        self.attention = self.attention_strategy or nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size
        )
        self.interpretation_heads = {
            t: nn.Dense(self.hidden_size)
            for t in self.interpretation_types
        }
        self.polysemy_projection = nn.Dense(self.hidden_size)
        self.polysemy_weight = nn.Dense(len(self.interpretation_types))
        self.norm = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_rate)
        if self.tpu_optimized:
            # Specific logic for tpu (e.g., use bfloat16)
            self.dtype = jnp.bfloat16
        else:
            self.dtype = jnp.float32

    def __call__(
        self,
        x: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        training: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Ejecuta el analysis semiótico about la input usando optimizaciones ARM Axion.

        Args:
            x (jnp.ndarray): tensor de input de forma (batch, seq, features).
            context (Optional[jnp.ndarray]): Contexto optional for atención cruzada.
            training (bool): if True, activa modo entrenamiento.
            **kwargs: Argumentos adicionales.

        Returns:
            Dict[str, Any]: Diccionario with interpretaciones, pesos, output atendida, 
                           proyección semántica and métricas.
        """
        # detect ARM Axion
        use_arm = not training and HARDWARE_INFO["is_axion"]
        
        self.validate_input(x)
        
        # Input encoding with optimizaciones ARM
        if use_arm and ARM_OPTIMIZATIONS.get("arm_kleidi"):
            query = ARM_OPTIMIZATIONS["arm_optimization_suite"].kleidi_forward(
                x.astype(self.dtype),
                self.input_encoder.weight
            )
        else:
            query = self.input_encoder(x.astype(self.dtype))
        
        # Context encoding with optimizaciones ARM
        if context is not None:
            if use_arm and ARM_OPTIMIZATIONS.get("arm_kleidi"):
                key_value = ARM_OPTIMIZATIONS["arm_optimization_suite"].kleidi_forward(
                    context.astype(self.dtype),
                    self.context_encoder.weight
                )
            else:
                key_value = self.context_encoder(context.astype(self.dtype))
        else:
            key_value = jnp.zeros_like(query)
        
        # Atención optimizada with SVE2
        if use_arm and ARM_OPTIMIZATIONS.get("sve2_vectorization"):
            attended = ARM_OPTIMIZATIONS["arm_optimization_suite"].sve2_attention(
                inputs_q=query,
                inputs_kv=key_value,
                causal=True
            )
        else:
            attended = self.attention(
                inputs_q=query,
                inputs_kv=key_value,
                deterministic=not training
            )
        
        attended = self.norm(attended + query)
        
        # Interpretaciones with optimizaciones ARM
        interpretations = {}
        for t, head in self.interpretation_heads.items():
            if use_arm and ARM_OPTIMIZATIONS.get("arm_quantization_available"):
                x = ARM_OPTIMIZATIONS["arm_optimization_suite"].quantized_matmul(
                    attended,
                    head.weight
                )
            else:
                x = head(attended)
            interpretations[t] = self.dropout(x, deterministic=not training)
        
        # Proyección semántica with optimizaciones ARM
        if use_arm and ARM_OPTIMIZATIONS.get("arm_kleidi"):
            semantic_projection = ARM_OPTIMIZATIONS["arm_optimization_suite"].kleidi_forward(
                attended,
                self.polysemy_projection.weight
            )
        else:
            semantic_projection = self.polysemy_projection(attended)
        
        # Pesos de polisemia
        if use_arm and ARM_OPTIMIZATIONS.get("arm_quantization_available"):
            weights = ARM_OPTIMIZATIONS["arm_optimization_suite"].quantized_matmul(
                semantic_projection,
                self.polysemy_weight.weight
            )
        else:
            weights = self.polysemy_weight(semantic_projection)
        weights = nn.softmax(weights.mean(axis=0), axis=-1)
        
        # Métricas
        self.sow(METRICS_KEY, "polysemy_weights", weights)
        self.sow(METRICS_KEY, "attention_output", attended.mean(axis=0))
        self.sow(METRICS_KEY, "interpretation_weights", {
            k: float(v.mean()) for k, v in interpretations.items()
        })
        self.sow(METRICS_KEY, "context_used", context is not None)
        self.sow(METRICS_KEY, "arm_optimizations_used", use_arm)
        self.sow(METRICS_KEY, "arm_features", list(ARM_OPTIMIZATIONS.keys()) if use_arm else [])
        
        return {
            "interpretations": interpretations,
            "weights": weights,
            "attended": attended,
            "semantic_projection": semantic_projection,
            "metrics": {
                "arm_optimizations_used": use_arm,
                "arm_features": list(ARM_OPTIMIZATIONS.keys()) if use_arm else []
            }
        }

    def validate_input(self, x: jnp.ndarray) -> None:
        """
        Valida que la input tenga la forma and dimensiones correctas.

        Args:
            x (jnp.ndarray): tensor de input.

        Raises:
            ValueError: if la input not es 3D or not coincide la dimension de features.
        """
        if x.ndim != 3:
            raise ValueError(f"Input must be 3D [batch, seq, features], got {x.ndim}D")
        if x.shape[-1] != self.hidden_size:
            raise ValueError(f"Input must have {self.hidden_size} features")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Recupera las métricas intermedias del analysis semiótico.

        Returns:
            Dict[str, Any]: Métricas calculadas during el forward.
        """
        return {
            "polysemy_weights": self.sow(METRICS_KEY, "polysemy_weights"),
            "attention_output": self.sow(METRICS_KEY, "attention_output"),
            "interpretation_weights": self.sow(METRICS_KEY, "interpretation_weights"),
            "context_used": self.sow(METRICS_KEY, "context_used"),
            "tpu_load": self.sow(METRICS_KEY, "tpu_load") if self.tpu_optimized else None,
            "tpu_overhead": self.sow(METRICS_KEY, "tpu_overhead") if self.tpu_optimized else None
        }

    def _safe_log_dir(self, path: str) -> str:
        return os.path.abspath(os.path.expanduser(path))

    def _setup_fallback_device(self):
        if not self.tpu_optimized:
            self.logger.info("Using fallback device (GPU/CPU)")
            # Configure mesh or device for cpu/gpu

# Full usage example
if __name__ == "__main__":
    import jax
    from jax import numpy as jnp
    logging.basicConfig(level=logging.INFO)
    model = SemioModule(
        hidden_size=256,
        tpu_optimized=True,
        interpretation_types={"literal", "cultural", "symbolic"}
    )
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((1, 10, 256))  # batch=1, seq=10, features=256
    context = jnp.ones((1, 5, 256))
    params = model.init(rng, x, context)
    outputs = model.apply(params, x, context, training=True)
    logger.info(outputs)

def test_validar_entrada():
    modelo = SemioModule(hidden_size=256)
    x = jnp.ones((1, 10, 256))
    modelo.validate_input(x)  # not debe lanzar excepción

def test_adaptive_noise():
    handler = AdaptiveNoiseHandler()
    state = jnp.array([1.0, 0.0])
    noisy_state = handler.apply_noise(state)
    assert jnp.allclose(jnp.linalg.norm(noisy_state), 1.0, atol=1e-6) 

"""
implementation mejorada de interacción semiótica with optimizaciones avanzadas for producción.
"""

import os
import sys
import time
import logging
import numpy as np
from dataclasses import dataclass, fieldield
# Obtiene la path del directory current (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Sube un level for obtain la raíz del proyecto -> /.../CapibaraGPT v3
project_root = os.path.dirname(script_dir)
# Añade la raíz del proyecto a sys.path
if project_root not in sys.path:
    pass  # Using proper imports instead of sys.path manipulation

import jax
from flax import linen as nn
from functools import partial
from jax import numpy as jnp
from flax.core import freeze, unfreezeze

# Imports with fallbacks for dependencias experimentales
try:
    from jax.experimental import Mesh
    from jax.experimental import Partition
    from jax.experimental import mesh_utils
except ImportError:
    # Fallbacks for imports experimentales
    class Mesh:
        pass
    class mesh_utils:
        pass
    class PartitionSpec:
        pass

try:
    from capibara.adaptive import AdaptiveEmbedding
except ImportError:
    class AdaptiveEmbedding(nn.Module):
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

from logging.handlers import RotatingFileHandler

try:
    from capibara.interfaces.isub_models import ISubModel
except ImportError:
    class ISubModel:
        pass

from typing import Dict, Any, Optional, Tuple, List, Set

try:
    from sub_models.experimental.spike_ssm import SpikeSSM
except ImportError:
    class SpikeSSM(nn.Module):
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

try:
    from jax.experimental.profiler import device_memory_stats, start_trace, stop_trace
except ImportError:
    def device_memory_stats():
        return {}
    def start_trace():
        pass
    def stop_trace():
        pass

from capibara.core.arm_optimizations import ARM_OPTIMIZATIONS, HARDWARE_INFO

class TPUError(Exception):
    """error específico for operaciones tpu."""
    pass

@dataclass
class QuantizationConfig:
    """setup de cuantización."""
    enabled: bool = False
    bits: int = 8
    symmetric: bool = True
    per_channel: bool = True
    min_value: float = -1.0
    max_value: float = 1.0

@dataclass
class ScalingConfig:
    """setup de escalado dinámico."""
    load_threshold: float = 0.85
    comm_threshold: float = 0.2
    min_shards: int = 2
    max_shards: int = 8
    scaling_cooldown: float = 300.0  # 5 minutos
    scaling_threshold: float = 0.75  # Umbral for decisión de escalado

@dataclass
class InterpretationMetrics:
    """Métricas de interpretación semiótica."""
    polysemy_weights: Dict[str, float]
    attention_patterns: Dict[str, jnp.ndarray]
    head_activations: Dict[str, jnp.ndarray]
    context_usage: float
    divergence: float
    hotspots: List[Tuple[int, int]] = field(default_factory=list)
    cross_modal_alignment: Optional[float] = None
    semantic_density: Optional[float] = None

@dataclass
class TPUMetrics:
    """Métricas de rendimiento tpu."""
    load_imbalance: float = 0.0
    comm_overhead: float = 0.0
    scaling_events: int = 0
    last_update: float = field(default_factory=time.time)
    load_history: List[float] = field(default_factory=list)
    comm_history: List[float] = field(default_factory=list)
    topology_map: Optional[jnp.ndarray] = None
    hotspots: List[Tuple[int, int]] = field(default_factory=list)
    last_scaling: float = field(default_factory=time.time)

@dataclass
class SemioticMetrics:
    """Métricas semióticas with monitoreo extendido."""
    divergence: float  # Divergencia between interpretaciones
    balance: float  # Balance between tipos interpretativos
    coherence: float  # Coherencia interna
    adaptive_entropy: float  # Entropía cuántica
    ssm_consistency: float  # Consistencia SSM
    numerical_stability: float  # Estabilidad numérica
    sharding_overhead: float  # Coste de sharding
    memory_usage: Dict[str, float]  # Uso de memory by component
    latency: float  # Latencia total
    tpu_metrics: TPUMetrics = field(default_factory=TPUMetrics)
    interpretation_metrics: Optional[InterpretationMetrics] = None
    cross_modal_alignment: Optional[float] = None  # Alineación between modos
    semantic_density: Optional[float] = None  # Densidad semántica

class SemioticInteraction(nn.Module, ISubModel):
    """
    Módulo de interacción semiótica with optimizaciones avanzadas:
    1. Eficiencia de memory
    2. optimization de cómputo
    3. Paralelismo
    4. Monitoreo advanced
    5. setup dinámica
    6. Escalado automático
    7. Cuantización
    8. analysis semiótico extendido
    """
    hidden_size: int
    dropout_rate: float
    num_heads: int
    ssm_dim: int
    adaptive_enabled: bool
    ssm_enabled: bool
    tpu_optimized: bool
    semiotic_weight: float
    ssm_weight: float
    stability_threshold: float
    use_mixed_precision: bool
    profile_interval: int
    max_load: float
    comm_penalty: float
    log_dir: str
    max_log_size: int
    backup_count: int
    interpretation_types: Set[str]
    quantization_config: QuantizationConfig
    scaling_config: ScalingConfig

    def setup(self):
        """
        Inicializa los componentes del módulo.
        - if ssm_enabled es True, se instance SpikeSSM for dotar al módulo de memory and dinámica temporary avanzada.
        - Esta integration permite que el analysis semiótico tenga acceso a mecanismos de memory recurrente and procesamiento secuencial, fundamentales for tareas de interpretación contextual.
        """
        # configure logging estructurado
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        handler = RotatingFileHandler(
            os.path.join(self.log_dir, 'capibara.log'),
            maxBytes=self.max_log_size,
            backupCount=self.backup_count
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # setup de precision mixta
        if self.use_mixed_precision:
            self.policy = jax.policy.Policy('float32')
        
        # Proyección compartida for atención
        self.shared_proj = nn.Dense(self.hidden_size)
        
        # Capas especializadas by level semiótico
        self.interpretation_heads = {
            t: nn.Dense(self.hidden_size) 
            for t in self.interpretation_types
        }
        
        # Mecanismo de atención optimizado
        self.attention = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            kernel_init=nn.initializers.xavier_uniform()
        )
        
        # memory compartida (buffer de contexto)
        self.memory_proj = nn.Dense(self.hidden_size * len(self.interpretation_types))
        
        # Componentes cuánticos with checkpointing
        if self.adaptive_enabled:
            self.adaptive_embedding = AdaptiveEmbedding(
                hidden_size=self.hidden_size,
                embedding_dim=4
            )
            self.adaptive_noise_handler = AdaptiveNoiseHandler()
            # Vectorizar aplicación de ruido
            self.vmap_apply_noise = jax.vmap(
                self.adaptive_noise_handler.apply_noise,
                in_axes=(0, None)
            )
        
        # Componentes SSM with memory de contexto
        if self.ssm_enabled:
            self.ssm = SpikeSSM(
                hidden_size=self.hidden_size,
                ssm_dim=self.ssm_dim,
                context_dim=self.hidden_size * len(self.interpretation_types)
            )
        
        # Normalización and dropout
        self.norm = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # setup tpu with sharding dinámico
        if self.tpu_optimized:
            try:
                devices = jax.devices('tpu')
                if len(devices) >= self.scaling_config.min_shards:
                    self.mesh = Mesh(devices, ('batch', 'hidden'))
                    self.sharding = nn.Partitioned(
                        {'interpretations': PartitionSpec(('batch', 'hidden'))},
                        mesh=self.mesh
                    )
                    # Inicializar métricas tpu and mapa de topología
                    self.tpu_metrics = TPUMetrics()
                    self.tpu_metrics.topology_map = self._build_topology_map()
                else:
                    self.logger.warning(f"TPU devices < {self.scaling_config.min_shards}, disabling sharding")
                    self.tpu_optimized = False
            except Exception as e:
                self.logger.warning(f"TPU setup failed: {str(e)}")
                self.tpu_optimized = False
        
        # Métricas de estado
        self.metrics = SemioticMetrics(
            divergence=0.0,
            balance=0.0,
            coherence=0.0,
            adaptive_entropy=0.0,
            ssm_consistency=0.0,
            numerical_stability=1.0,
            sharding_overhead=0.0,
            memory_usage={},
            latency=0.0,
            tpu_metrics=self.tpu_metrics if self.tpu_optimized else None,
            interpretation_metrics=None
        )

    def _build_topology_map(self) -> jnp.ndarray:
        """
        Construye mapa de topología for optimization de communication.
        
        Returns:
            matrix de distancias between cores
        """
        try:
            return jnp.array([
                [abs(i-j) + abs(k-l) 
                for j in range(self.mesh.shape[0])
                for l in range(self.mesh.shape[1])]
                for i in range(self.mesh.shape[0])
                for k in range(self.mesh.shape[1])
            ])
        except Exception as e:
            self.logger.error(f"Topology map build failed: {str(e)}")
            return jnp.ones((self.mesh.shape[0], self.mesh.shape[1]))

    def validate_input(self, x: jnp.ndarray, context: Optional[jnp.ndarray] = None) -> None:
        """
        Valida dimensiones and tipos de input with mensajes descriptivos.
        
        Args:
            x: tensor de input
            context: tensor de contexto optional
            
        Raises:
            ValueError: if las dimensiones son inválidas
            TypeError: if los tipos son incorrectos
        """
        if x.ndim != 3:
            raise ValueError(
                f"Input must be 3D [batch, seq, features], got {x.ndim}D. "
                f"Shape: {x.shape}. Consider using np.expand_dims() if needed."
            )
            
        if not isinstance(x, (jnp.ndarray, np.ndarray)):
            raise TypeError(
                f"Input must be jnp.ndarray or np.ndarray, got {type(x)}"
            )
            
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Input features dimension mismatch: {x.shape[-1]} != {self.hidden_size}. "
                "Input must match hidden_size."
            )
            
        if context is not None:
            if not isinstance(context, (jnp.ndarray, np.ndarray)):
                raise TypeError(
                    f"Context must be jnp.ndarray or np.ndarray, got {type(context)}"
                )
                
            if context.shape[0] != x.shape[0]:
                raise ValueError(
                    f"Context batch size mismatch: {context.shape[0]} != {x.shape[0]}. "
                    "Context must match input batch dimension."
                )
                
            if context.shape[-1] != self.hidden_size:
                raise ValueError(
                    f"Context features dimension mismatch: {context.shape[-1]} != {self.hidden_size}. "
                    "Context must match hidden_size."
                )

    def quantize_input(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Cuantiza el input according to la setup.
        
        Args:
            x: tensor de input
            
        Returns:
            tensor cuantizado
        """
        if not self.quantization_config.enabled:
            return x
            
        try:
            scale = (self.quantization_config.max_value - 
                    self.quantization_config.min_value) / (2**self.quantization_config.bits - 1)
            
            if self.quantization_config.symmetric:
                x = jnp.clip(x, -self.quantization_config.max_value, 
                           self.quantization_config.max_value)
                x = jnp.round(x / scale) * scale
            else:
                x = jnp.clip(x, self.quantization_config.min_value, 
                           self.quantization_config.max_value)
                x = jnp.round((x - self.quantization_config.min_value) / scale) * scale + \
                    self.quantization_config.min_value
                    
            return x
            
        except Exception as e:
            self.logger.warning(f"Quantization failed: {str(e)}")
            return x

    def detect_hotspots(self, load: jnp.ndarray) -> List[Tuple[int, int]]:
        """
        Detecta hotspots en la load tpu.
        
        Args:
            load: matrix de load
            
        Returns:
            list de coordenadas de hotspots
        """
        try:
            threshold = self.scaling_config.load_threshold
            hotspots = []
            
            for i in range(load.shape[0]):
                for j in range(load.shape[1]):
                    if load[i,j] > threshold:
                        hotspots.append((i,j))
                        
            return hotspots
            
        except Exception as e:
            self.logger.warning(f"Hotspot detection failed: {str(e)}")
            return []

    def optimize_sharding(self) -> PartitionSpec:
        """
        Optimiza la estrategia de sharding basada en hotspots.
        
        Returns:
            Nueva especificación de particionamiento
        """
        try:
            if not self.tpu_metrics.hotspots:
                return self.sharding.partition_spec
                
            # group hotspots by dimension
            batch_hotspots = set(h[0] for h in self.tpu_metrics.hotspots)
            hidden_hotspots = set(h[1] for h in self.tpu_metrics.hotspots)
            
            # create mesh correctamente
            devices = mesh_utils.create_device_mesh((4, 8))  # for tpu v4-32
            mesh = Mesh(devices, axis_names=('data', 'model'))
            
            # adjust sharding for balancear hotspots
            if len(batch_hotspots) > len(hidden_hotspots):
                return PartitionSpec(
                    data=('data', 'model'),
                    model=None
                )
            else:
                return PartitionSpec(
                    data=None,
                    model=('data', 'model')
                )
                
        except Exception as e:
            self.logger.error(f"Sharding optimization failed: {str(e)}")
            return self.sharding.partition_spec

    def _process_adaptive(self, x: jnp.ndarray) -> jnp.ndarray:
        """Procesa operaciones cuánticas with gestión de memory optimizada."""
        try:
            # use jax.lax.scan for evitar materialización innecesaria
            def adaptive_step(carry, x):
                state, _ = carry
                new_state = self.adaptive_embedding(x)
                return (new_state, None), new_state
            
            init_carry = (x, None)
            (final_state, _), _ = jax.lax.scan(adaptive_step, init_carry, x)
            
            # clean memory de forma segura
            if self.tpu_optimized:
                jax.device_sync()  # synchronize before de clean
                jax.clear_caches()
            
            return final_state
            
        except Exception as e:
            self.logger.error(f"Adaptive processing failed: {str(e)}")
            return x

    def profile_tpu_usage(self) -> Dict[str, Any]:
        """Perfila el uso de tpu de forma segura."""
        try:
            if not self.tpu_optimized:
                return {}
                
            with jax.profiler.trace("/tmp/tpu_trace"):
                # obtain métricas de memory
                memory_stats = jax.device_memory_allocated()
                memory_peak = jax.device_memory_peak()
                
                # obtain métricas de rendimiento
                device_count = len(jax.devices('tpu'))
                device_utilization = jax.device_get(jax.device_count())
                
                return {
                    'memory_allocated': memory_stats,
                    'memory_peak': memory_peak,
                    'device_count': device_count,
                    'device_utilization': device_utilization
                }
                
        except Exception as e:
            self.logger.error(f"TPU profiling failed: {str(e)}")
            return {}

    def calculate_comm_cost(self, load: jnp.ndarray) -> float:
        """
        Calcula costo de communication basado en topología.
        
        Args:
            load: matrix de load current
            
        Returns:
            Costo de communication normalizado
        """
        try:
            if self.tpu_metrics.topology_map is None:
                return 0.0
                
            # Penalizar communication between cores distantes
            comm_cost = jnp.mean(self.tpu_metrics.topology_map * load)
            return float(comm_cost)
            
        except Exception as e:
            self.logger.warning(f"Comm cost calculation failed: {str(e)}")
            return 0.0

    def forecast_load(self) -> jnp.ndarray:
        """
        Predice load futura usando model AR simple.
        
        Returns:
            prediction de load
        """
        try:
            if len(self.tpu_metrics.load_history) < 3:
                return jnp.mean(self.tpu_metrics.load_history)
                
            # model AR(3)
            weights = jnp.array([0.6, 0.3, 0.1])
            history = jnp.array(self.tpu_metrics.load_history[-3:])
            return jnp.sum(weights * history)
            
        except Exception as e:
            self.logger.warning(f"Load forecasting failed: {str(e)}")
            return 0.5

    def update_sharding(self, current_load: jnp.ndarray) -> jnp.ndarray:
        """
        Actualiza sharding with optimization de communication.
        
        Args:
            current_load: load current
            
        Returns:
            Nueva distribución de load
        """
        try:
            # calculate costo de communication
            comm_cost = self.calculate_comm_cost(current_load)
            
            # predict load futura
            forecast = self.forecast_load()
            
            # adjust target load with penalización de communication
            target_load = jnp.clip(
                forecast * (1 - self.comm_penalty * comm_cost),
                0.1,
                self.max_load
            )
            
            # update métricas
            self.tpu_metrics.scaling_events += 1
            self.tpu_metrics.comm_history.append(comm_cost)
            
            # Logging estructurado
            self.logger.info(
                f"Sharding updated - Load: {float(forecast):.2f}, "
                f"Comm cost: {float(comm_cost):.2f}, "
                f"Target: {float(target_load):.2f}"
            )
            
            return target_load
            
        except Exception as e:
            self.logger.error(f"Sharding update failed: {str(e)}")
            return current_load

    def should_scale(self) -> bool:
        """
        Determina if se debe scale basado en métricas.
        
        Returns:
            True if se debe scale
        """
        try:
            current_time = time.time()
            time_since_scaling = current_time - self.tpu_metrics.last_scaling
            
            if time_since_scaling < self.scaling_config.scaling_cooldown:
                return False
                
            avg_load = jnp.mean(self.tpu_metrics.load_history)
            comm_overhead = self.tpu_metrics.comm_overhead
            
            should_scale = (
                (avg_load > self.scaling_config.load_threshold or
                 comm_overhead > self.scaling_config.comm_threshold) and
                self.tpu_metrics.scaling_events < self.scaling_config.max_shards
            )
            
            if should_scale:
                self.logger.info(
                    f"Scaling triggered - Load: {float(avg_load):.2f}, "
                    f"Comm: {float(comm_overhead):.2f}, "
                    f"Time since last: {time_since_scaling:.0f}s"
                )
                self.tpu_metrics.last_scaling = current_time
                
            return should_scale
                   
        except Exception as e:
            self.logger.warning(f"Scale check failed: {str(e)}")
            return False

    def calculate_divergence(self, interpretations: Dict[str, jnp.ndarray]) -> float:
        """
        Calcula divergencia between interpretaciones.
        
        Args:
            interpretations: Dict with interpretaciones
            
        Returns:
            Divergencia normalizada
        """
        try:
            # calculate divergencia between pares
            divergences = []
            for i, t1 in enumerate(self.interpretation_types):
                for t2 in list(self.interpretation_types)[i+1:]:
                    div = jnp.mean(jnp.abs(
                        interpretations[t1] - interpretations[t2]
                    ))
                    divergences.append(div)
            
            return float(jnp.mean(divergences))
            
        except Exception as e:
            self.logger.warning(f"Divergence calculation failed: {str(e)}")
            return 0.0

    def calculate_polysemy(self, interpretations: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """
        Calcula pesos de polisemia by interpretación.
        
        Args:
            interpretations: Dict with interpretaciones
            
        Returns:
            Dict with pesos de polisemia
        """
        try:
            return {
                t: float(jnp.mean(jnp.abs(v)))
                for t, v in interpretations.items()
            }
        except Exception as e:
            self.logger.warning(f"Polysemy calculation failed: {str(e)}")
            return {t: 0.0 for t in self.interpretation_types}

    def calculate_cross_modal_alignment(
        self,
        interpretations: Dict[str, jnp.ndarray]
    ) -> float:
        """
        Calcula alineación between modos interpretativos.
        
        Args:
            interpretations: Dict with interpretaciones
            
        Returns:
            Score de alineación normalizado
        """
        try:
            # calculate similitud coseno between pares
            similarities = []
            for i, t1 in enumerate(self.interpretation_types):
                for t2 in list(self.interpretation_types)[i+1:]:
                    sim = jnp.mean(
                        jnp.sum(interpretations[t1] * interpretations[t2], axis=-1) /
                        (jnp.linalg.norm(interpretations[t1], axis=-1) * 
                         jnp.linalg.norm(interpretations[t2], axis=-1))
                    )
                    similarities.append(sim)
            
            return float(jnp.mean(similarities))
            
        except Exception as e:
            self.logger.warning(f"Cross-modal alignment calculation failed: {str(e)}")
            return 0.0

    def calculate_semantic_density(
        self,
        interpretations: Dict[str, jnp.ndarray]
    ) -> float:
        """
        Calcula densidad semántica de las interpretaciones.
        
        Args:
            interpretations: Dict with interpretaciones
            
        Returns:
            Score de densidad normalizado
        """
        try:
            # calculate entropía de Shannon by interpretación
            entropies = []
            for t in self.interpretation_types:
                probs = jnp.abs(interpretations[t])
                probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
                entropy = -jnp.sum(probs * jnp.log(probs + self.stability_threshold))
                entropies.append(entropy)
            
            # Normalizar and combine
            return float(jnp.mean(entropies))
            
        except Exception as e:
            self.logger.warning(f"Semantic density calculation failed: {str(e)}")
            return 0.0

    def dynamic_scaling(self) -> bool:
        """
        Decide if scale basado en múltiples métricas usando un model de decisión.
        
        Returns:
            True if se recomienda scale los recursos
        """
        try:
            if len(self.tpu_metrics.load_history) < 10:
                return False
                
            load_factor = jnp.mean(self.tpu_metrics.load_history[-10:])
            comm_factor = jnp.mean(self.tpu_metrics.comm_history[-10:])
            scaling_score = 0.7*load_factor + 0.3*comm_factor
            
            current_time = time.time()
            time_since_scaling = current_time - self.tpu_metrics.last_scaling
            
            should_scale = (
                scaling_score > self.scaling_config.scaling_threshold and
                time_since_scaling > self.scaling_config.scaling_cooldown and
                self.tpu_metrics.scaling_events < self.scaling_config.max_shards
            )
            
            if should_scale:
                self.logger.info(
                    f"Dynamic scaling triggered - Score: {float(scaling_score):.2f}, "
                    f"Load: {float(load_factor):.2f}, Comm: {float(comm_factor):.2f}"
                )
                self.tpu_metrics.last_scaling = current_time
                
            return should_scale
            
        except Exception as e:
            self.logger.warning(f"Dynamic scaling check failed: {str(e)}")
            return False

    def __call__(
        self,
        x: jnp.ndarray,
        context: Optional[jnp.ndarray] = None,
        training: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass with optimizaciones ARM Axion.
        - Usa optimizaciones SVE2 for atención vectorizada
        - Usa Kleidi for matmul
        - Usa quantización ARM for proyecciones
        """
        start_time = time.time()
        
        try:
            # 1. validate input
            self.validate_input(x, context)
            
            # detect ARM Axion
            use_arm = not training and HARDWARE_INFO["is_axion"]
            
            # 2. Cuantización if está habilitada
            x = self.quantize_input(x)
            if context is not None:
                context = self.quantize_input(context)
            
            # 3. apply precision mixta if está habilitada
            if self.use_mixed_precision:
                x = self.policy.cast_to_compute(x)
                if context is not None:
                    context = self.policy.cast_to_compute(context)
            
            # 4. Proyección compartida with optimizaciones ARM
            if use_arm and ARM_OPTIMIZATIONS.get("arm_kleidi"):
                shared_features = ARM_OPTIMIZATIONS["arm_optimization_suite"].kleidi_forward(
                    x,
                    self.shared_proj.weight
                )
            else:
                shared_features = self.shared_proj(x)
            
            # 5. Procesamiento paralelo de interpretaciones
            outputs = {}
            for t, head in self.interpretation_heads.items():
                if use_arm and ARM_OPTIMIZATIONS.get("arm_quantization_available"):
                    outputs[t] = ARM_OPTIMIZATIONS["arm_optimization_suite"].quantized_matmul(
                        shared_features,
                        head.weight
                    )
                else:
                    outputs[t] = head(shared_features)
            
            # 6. Atención optimizada with SVE2
            keys_values = jnp.stack(list(outputs.values()))
            queries = jnp.stack([outputs[t] for t in self.interpretation_types])
            
            if use_arm and ARM_OPTIMIZATIONS.get("sve2_vectorization"):
                attended = ARM_OPTIMIZATIONS["arm_optimization_suite"].sve2_attention(
                    queries=queries,
                    keys=keys_values,
                    values=keys_values,
                    causal=True
                )
            else:
                attended = self.attention(
                    queries=queries,
                    keys=keys_values,
                    values=keys_values
                )
            
            # 7. update with memory usando optimizaciones ARM
            if use_arm and ARM_OPTIMIZATIONS.get("arm_kleidi"):
                memory = ARM_OPTIMIZATIONS["arm_optimization_suite"].kleidi_forward(
                    jnp.concatenate(attended),
                    self.memory_proj.weight
                )
            else:
                memory = self.memory_proj(jnp.concatenate(attended))
            
            # 8. calculate métricas de interpretación
            cross_modal = self.calculate_cross_modal_alignment(outputs)
            semantic_density = self.calculate_semantic_density(outputs)
            
            interpretation_metrics = InterpretationMetrics(
                polysemy_weights=self.calculate_polysemy(outputs),
                attention_patterns={
                    t: jnp.mean(attended[i], axis=0)
                    for i, t in enumerate(self.interpretation_types)
                },
                head_activations={
                    t: jnp.mean(outputs[t], axis=0)
                    for t in self.interpretation_types
                },
                context_usage=float(jnp.mean(jnp.abs(memory))),
                divergence=self.calculate_divergence(outputs),
                cross_modal_alignment=cross_modal,
                semantic_density=semantic_density
            )
            
            # 9. Regularización semiótica en entrenamiento
            if training:
                sem_loss = self.semiotic_weight * interpretation_metrics.divergence
                self.sow('intermediates', 'semiotic_loss', sem_loss)
                self.sow('intermediates', 'interpretation_divergence', 
                        interpretation_metrics.divergence)
            
            # 10. Procesamiento cuántico optimizado
            adaptive_output = None
            if self.adaptive_enabled:
                try:
                    adaptive_output = self._process_adaptive(
                        outputs['symbolic']
                    )
                    self.metrics.adaptive_entropy = float(jnp.mean(adaptive_output))
                    
                except Exception as e:
                    self.logger.warning(f"Adaptive processing failed: {str(e)}")
                    adaptive_output = outputs['symbolic']
            
            # 11. Procesamiento SSM optimizado
            ssm_output = None
            if self.ssm_enabled:
                try:
                    ssm_input = jnp.concatenate([
                        outputs[t] for t in self.interpretation_types
                    ], axis=-1)
                    ssm_output = self.ssm(ssm_input)
                    # validation de la output de SpikeSSM
                    if not isinstance(ssm_output, jnp.ndarray):
                        raise TypeError("SpikeSSM output must be a jnp.ndarray")
                    if ssm_output.shape[0] != x.shape[0]:
                        raise ValueError("SpikeSSM output batch size mismatch")
                    self.metrics.ssm_consistency = float(jnp.mean(ssm_output))
                    if training:
                        ssm_output = ssm_output - self.ssm_weight * jax.grad(
                            lambda x: self.metrics.ssm_consistency,
                            allow_int=True
                        )(ssm_output)
                except Exception as e:
                    self.logger.error(f"SSM processing failed: {str(e)}")
                    ssm_output = outputs['symbolic']
            
            # 12. apply sharding tpu with optimizaciones
            if self.tpu_optimized:
                try:
                    # Perfilar uso current
                    current_load = self.profile_tpu_usage()
                    
                    # detect hotspots
                    self.tpu_metrics.hotspots = self.detect_hotspots(current_load)
                    
                    # update sharding if es necessary
                    if self.dynamic_scaling():
                        new_spec = self.optimize_sharding()
                        self.sharding = nn.Partitioned(
                            {'interpretations': new_spec},
                            mesh=self.mesh
                        )
                        
                        # apply new sharding
                        start_sharding = time.time()
                        outputs = self.sharding(outputs)
                        if ssm_output is not None:
                            ssm_output = jax.lax.with_sharding_constraint(
                                ssm_output,
                                new_spec
                            )
                        self.metrics.sharding_overhead = time.time() - start_sharding
                        
                except TPUError as e:
                    self.logger.error(f"TPU operation failed: {str(e)}")
                    # Fallback a cpu
                    self.tpu_optimized = False
                except Exception as e:
                    self.logger.error(f"TPU optimization failed: {str(e)}")
            
            # 13. update métricas
            self.metrics.divergence = interpretation_metrics.divergence
            self.metrics.balance = float(jnp.std([
                interpretation_metrics.polysemy_weights[t]
                for t in self.interpretation_types
            ]))
            self.metrics.coherence = float(jnp.mean([
                jnp.mean(jnp.abs(outputs[t]))
                for t in self.interpretation_types
            ]))
            self.metrics.interpretation_metrics = interpretation_metrics
            self.metrics.cross_modal_alignment = cross_modal
            self.metrics.semantic_density = semantic_density
            
            # 14. calculate estabilidad numérica
            numerical_stability = jnp.min([
                jnp.min(jnp.abs(outputs[t]))
                for t in self.interpretation_types
            ])
            self.metrics.numerical_stability = float(numerical_stability)
            
            # 15. Métricas de rendimiento
            self.metrics.latency = time.time() - start_time
            try:
                self.metrics.memory_usage = device_memory_stats()
            except Exception as e:
                self.logger.warning(f"Memory profiling failed: {str(e)}")
                self.metrics.memory_usage = {}
            
            # 16. Métricas extendidas
            metrics = {
                'interpretation_weights': interpretation_metrics.polysemy_weights,
                'semiotic_metrics': {
                    'divergence': self.metrics.divergence,
                    'balance': self.metrics.balance,
                    'coherence': self.metrics.coherence,
                    'cross_modal_alignment': self.metrics.cross_modal_alignment,
                    'semantic_density': self.metrics.semantic_density
                },
                'adaptive_metrics': {
                    'entropy': self.metrics.adaptive_entropy
                } if self.adaptive_enabled else None,
                'ssm_metrics': {
                    'consistency': self.metrics.ssm_consistency
                } if self.ssm_enabled else None,
                'stability_metrics': {
                    'numerical_stability': self.metrics.numerical_stability,
                    'sharding_overhead': self.metrics.sharding_overhead
                },
                'performance_metrics': {
                    'latency': self.metrics.latency,
                    'memory_usage': self.metrics.memory_usage
                },
                'tpu_metrics': {
                    'load_imbalance': self.tpu_metrics.load_imbalance,
                    'comm_overhead': self.tpu_metrics.comm_overhead,
                    'scaling_events': self.tpu_metrics.scaling_events,
                    'hotspots': self.tpu_metrics.hotspots
                } if self.tpu_optimized else None,
                'interpretation_metrics': {
                    'attention_patterns': {
                        k: v.tolist() for k, v in interpretation_metrics.attention_patterns.items()
                    },
                    'head_activations': {
                        k: v.tolist() for k, v in interpretation_metrics.head_activations.items()
                    },
                    'context_usage': interpretation_metrics.context_usage,
                    'cross_modal_alignment': interpretation_metrics.cross_modal_alignment,
                    'semantic_density': interpretation_metrics.semantic_density
                },
                'tpu_shards': 1 if self.tpu_optimized else 0
            }
            
            # Logging estructurado de métricas
            self.logger.info(
                f"Forward pass completed - "
                f"Latency: {self.metrics.latency:.3f}s, "
                f"Divergence: {self.metrics.divergence:.3f}, "
                f"Stability: {self.metrics.numerical_stability:.3f}, "
                f"Cross-modal: {self.metrics.cross_modal_alignment:.3f}, "
                f"Density: {self.metrics.semantic_density:.3f}"
            )
            
            return {
                'interpretations': outputs,
                'adaptive_state': adaptive_output,
                'ssm_output': ssm_output,
                'metrics': metrics
            }
            
        except Exception as e:
            self.logger.error(f"Forward pass failed: {str(e)}")
            raise

class AdaptiveNoiseHandler:
    """Manejador de ruido cuántico with optimizaciones and monitoreo."""
    
    def __init__(self, embedding_dim: int = 4):
        self.t1 = 50e-6  # Tiempo de decoherencia
        self.t2 = 70e-6
        self.embedding_dim = embedding_dim
        self.fidelity_history: List[float] = []
        self.stability_threshold = 1e-6
        
    def validate_state(self, state: jnp.ndarray) -> None:
        """
        Valida estado cuántico with manejo de errores.
        
        Args:
            state: Estado cuántico a validate
            
        Raises:
            ValueError: if el estado not es valid
        """
        try:
            norm = jnp.linalg.norm(state)
            if not jnp.allclose(norm, 1.0, atol=self.stability_threshold):
                raise ValueError(f"Estado no normalizado: {norm}")
        except Exception as e:
            logging.error(f"Error validando estado cuántico: {str(e)}")
            raise
        
    @jax.jit
    def apply_noise(
        self,
        state: jnp.ndarray,
        t_gate: float = 1e-6
    ) -> jnp.ndarray:
        """
        Aplica ruido adaptativo with jit and monitoreo.
        
        Args:
            state: Estado cuántico
            t_gate: Tiempo de operation de puerta
            
        Returns:
            Estado cuántico with ruido
        """
        try:
            # validate estado
            self.validate_state(state)
            
            # adjust ruido basado en fidelidad histórica
            avg_fidelity = jnp.mean(self.fidelity_history) if self.fidelity_history else 0.9
            adaptive_gamma = jnp.clip(self.t1 / (t_gate + self.stability_threshold), 0, 1)
            adaptive_gamma *= avg_fidelity
            
            # Probabilidades de decoherencia adaptativas
            p_decay = 1 - jnp.exp(-t_gate * adaptive_gamma)
            p_dephase = 1 - jnp.exp(-t_gate * adaptive_gamma * 0.5)
            
            # Ruido dependiente del estado with clipping
            gamma = 1 - jnp.exp(-t_gate * adaptive_gamma)
            p = jnp.clip(jnp.abs(state) ** 2 * gamma, 0, 1)
            noise = jnp.random.bernoulli(p) * jnp.sqrt(gamma)
            state = state * (1 - noise)
            
            # Operadores de ruido adaptativos
            decay_op = jnp.sqrt(p_decay) * jnp.array([[0, 1], [0, 0]])
            phase_op = jnp.sqrt(p_dephase/2) * jnp.array([[1, 0], [0, -1]])
            
            # apply a each qubit with slicing for eficiencia
            noisy_state = state
            for q in range(self.embedding_dim):
                if q < 4:  # use kron for few qubits
                    noisy_state = jnp.kron(
                        noisy_state,
                        (jnp.eye(2) + decay_op + phase_op)
                    )
                else:  # use slicing for more qubits
                    noisy_state = noisy_state * (1 + decay_op[q%2] + phase_op[q%2])
            
            # Normalizar estado end
            noisy_state = noisy_state / jnp.linalg.norm(noisy_state)
            
            # validate and update fidelidad
            self.validate_state(noisy_state)
            fidelity = jnp.abs(jnp.vdot(state, noisy_state)) ** 2
            self.fidelity_history.append(float(fidelity))
            if len(self.fidelity_history) > 100:
                self.fidelity_history.pop(0)
            
            return noisy_state
            
        except Exception as e:
            logging.warning(f"Adaptive noise failed: {str(e)}")
            return state  # Fallback secure 

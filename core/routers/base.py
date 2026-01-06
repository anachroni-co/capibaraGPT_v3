"""
Interfaz base para routers en CapibaraGPT.
Usa directamente las implementaciones nativas de JAX y optimizaciones TPU v4-32.
"""

import os
import psutil
import logging
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Protocol, runtime_checkable, List, Tuple

from capibara.jax import nn
from capibara.jax import numpy as jnp

from capibara.utils.monitoring import MemoryMonitor
from capibara.core.config import RouterConfig, ModularModelConfig

# Direct imports of native implementations
try:
    from capibara.jax.tpu_v4.backend import (
        TpuV4LinalgOps,
        TpuV4SparseOps,
        TpuV4NeuralOps,
        TPU_V4_AVAILABLE,
    )   
    from capibara.jax.tpu_v4.optimizations import tpu_optimized_gemm
    NATIVE_TPU_AVAILABLE = True
except ImportError:
    NATIVE_TPU_AVAILABLE = False
    TPU_V4_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class FallbackConfig:
    """Configuration for the fallback system."""
    memory_threshold: float = 0.85  # 85% uso de memoria
    latency_threshold_ms: float = 100.0  # 100ms máximo
    batch_size_reduction: float = 0.5  # Reducir batch size 50%
    min_batch_size: int = 1
    enable_auto_recovery: bool = True
    recovery_wait_time: int = 300  # 5 minutos

@runtime_checkable
class RouterProtocol(Protocol):
    """Protocolo que define la interfaz mínima para routers"""
    def route(self, x: jnp.ndarray, context: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        ...

class BaseRouter(nn.Module, ABC):
    """Abstract base class for all routers"""
    config: RouterConfig
    
    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config
        
    def setup(self) -> None:
        """Configuration común para todos los routers"""
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_heads
        self.dropout_rate = self.config.dropout_rate
        
        # Configuración de fallbacks
        self.fallback_config = FallbackConfig()
        self.current_batch_size = self.config.batch_size
        self.in_fallback_mode = False
        self.last_fallback_time = 0
        
        # Monitor de memoria
        self.memory_monitor = MemoryMonitor()
        
        # Inicializar operaciones TPU directamente si están disponibles
        if NATIVE_TPU_AVAILABLE and TPU_V4_AVAILABLE:
            self._initialize_tpu_ops()
        else:
            self._initialize_cpu_ops()
            
    def _initialize_tpu_ops(self):
        """Inicializar operaciones TPU."""
        try:
            self.linalg_ops = TpuV4LinalgOps()
            self.sparse_ops = TpuV4SparseOps()
            self.neural_ops = TpuV4NeuralOps()
            logger.info("✅ Native TPU v4-32 operations initialized")
        except Exception as e:
            logger.error(f"❌ Error initializing TPU ops: {e}")
            self._initialize_cpu_ops()
    
    def _initialize_cpu_ops(self):
        """Inicializar operaciones CPU."""
        self.linalg_ops = None
        self.sparse_ops = None
        self.neural_ops = None
        logger.info("ℹ️ Using CPU fallback operations")
    
    def _check_memory_usage(self) -> bool:
        """Verifiesr uso de memoria."""
        memory_usage = self.memory_monitor.get_memory_usage()
        return memory_usage > self.fallback_config.memory_threshold
    
    def _check_latency(self, start_time: float) -> bool:
        """Verifiesr latencia."""
        import time
        latency = (time.time() - start_time) * 1000
        return latency > self.fallback_config.latency_threshold_ms
    
    def _activate_fallback(self):
        """Activar modo fallback."""
        import time
        if not self.in_fallback_mode:
            self.in_fallback_mode = True
            self.last_fallback_time = time.time()
            
            # Reducir batch size
            new_batch_size = max(
                int(self.current_batch_size * self.fallback_config.batch_size_reduction),
                self.fallback_config.min_batch_size
            )
            self.current_batch_size = new_batch_size
            
            logger.warning(f"⚠️ Activating fallback mode - Reducing batch size to {new_batch_size}")
    
    def _check_recovery(self):
        """Verifiesr si podemos recuperar modo normal."""
        import time
        if not self.fallback_config.enable_auto_recovery:
            return
            
        current_time = time.time()
        if (current_time - self.last_fallback_time > self.fallback_config.recovery_wait_time and
            not self._check_memory_usage()):
            self.in_fallback_mode = False
            self.current_batch_size = self.config.batch_size
            logger.info("✅ Recovered from fallback mode")
    
    def _optimized_matmul(self, a: jnp.ndarray, b: jnp.ndarray, 
                         transpose_a: bool = False, transpose_b: bool = False) -> jnp.ndarray:
        """Multiplicación de matrices optimizada con fallback automático."""
        import time
        start_time = time.time()
        
        # Verificar memoria y latencia
        if self._check_memory_usage() or self._check_latency(start_time):
            self._activate_fallback()
        else:
            self._check_recovery()
        
        try:
            if self.linalg_ops is not None and not self.in_fallback_mode:
                return self.linalg_ops.matrix_multiply(
                    a, b, transpose_a=transpose_a, transpose_b=transpose_b
                )
            elif NATIVE_TPU_AVAILABLE and not self.in_fallback_mode:
                return tpu_optimized_gemm(a, b, transpose_a, transpose_b)
        except Exception as e:
            logger.warning(f"TPU matmul failed, using CPU fallback: {e}")
            self._activate_fallback()
        
        # Fallback estándar
        if transpose_a:
            a = jnp.swapaxes(a, -2, -1)
        if transpose_b:
            b = jnp.swapaxes(b, -2, -1)
        return jnp.matmul(a, b)
    
    def _optimized_attention(self, query: jnp.ndarray, key: jnp.ndarray, 
                            value: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Atención optimizada con fallback automático."""
        import time
        start_time = time.time()
        
        # Verificar memoria y latencia
        if self._check_memory_usage() or self._check_latency(start_time):
            self._activate_fallback()
        else:
            self._check_recovery()
        
        try:
            if self.neural_ops is not None and not self.in_fallback_mode:
                return self.neural_ops.attention(query, key, value, mask)
        except Exception as e:
            logger.warning(f"TPU attention failed: {e}")
            self._activate_fallback()
        
        # Implementación estándar con optimizaciones de memoria
        scores = self._optimized_matmul(query, key, transpose_b=True)
        scores = scores / jnp.sqrt(query.shape[-1])
        
        if mask is not None:
            scores = jnp.where(mask[:, None, None, :], scores, float('-inf'))
        
        # Usar menos memoria en softmax
        weights = nn.softmax(scores, axis=-1)
        del scores  # Liberar memoria
        
        output = self._optimized_matmul(weights, value)
        del weights  # Liberar memoria
        
        return output
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de rendimiento."""
        return {
            "in_fallback_mode": self.in_fallback_mode,
            "current_batch_size": self.current_batch_size,
            "memory_usage": self.memory_monitor.get_memory_usage(),
            "tpu_available": NATIVE_TPU_AVAILABLE and TPU_V4_AVAILABLE
        }
    
    @abstractmethod
    def route(self, x: jnp.ndarray, context: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Main routing method"""
        pass

    def combine_outputs(self, outputs: Dict[str, Any]) -> Any:
        """Base method to combine outputs. Must be implemented by subclasses."""
        raise NotImplementedError("El method combine_outputs debe ser implementado por las subclases")

class BaseRouterV2(BaseRouter):
    """Router base versión 2 con funcionalidades extendidas."""
    
    def __init__(self):
        """Initializes the router."""
        super().__init__()
        self.memory_monitor = None
    
    def route(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation base of routing."""
        return inputs
        
    def combine_outputs(self, outputs: Dict[str, Any]) -> Any:
        """Base method to combine outputs. Must be implemented by subclasses."""
        raise NotImplementedError("El method combine_outputs debe ser implementado por las subclases.") 
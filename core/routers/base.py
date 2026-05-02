"""
Base interface for routers in CapibaraGPT.
Uses JAX native implementations directly with TPU v4-32 optimizations.
"""

import os
import logging
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Protocol, runtime_checkable, List, Tuple

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False

try:
    from jax import nn
    from jax import numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp  # type: ignore
    nn = None  # type: ignore
    JAX_AVAILABLE = False

try:
    from capibara.utils.monitoring import MemoryMonitor
except ImportError:
    MemoryMonitor = None  # type: ignore

try:
    from capibara.core.config import RouterConfig, ModularModelConfig
except ImportError:
    RouterConfig = None  # type: ignore
    ModularModelConfig = None  # type: ignore

# Direct imports of native implementations
try:
    from capibara.jax_ext.tpu_v4.backend import (
        TpuV4LinalgOps,
        TpuV4SparseOps,
        TpuV4NeuralOps,
        TPU_V4_AVAILABLE,
    )   
    from capibara.jax_ext.tpu_v4.optimizations import tpu_optimized_gemm
    NATIVE_TPU_AVAILABLE = True
except ImportError:
    NATIVE_TPU_AVAILABLE = False
    TPU_V4_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class FallbackConfig:
    """Configuration for the fallback system."""
    memory_threshold: float = 0.85  # 85% memory usage
    latency_threshold_ms: float = 100.0  # 100ms maximum
    batch_size_reduction: float = 0.5  # Reduce batch size by 50%
    min_batch_size: int = 1
    enable_auto_recovery: bool = True
    recovery_wait_time: int = 300  # 5 minutes

@runtime_checkable
class RouterProtocol(Protocol):
    """Protocol that defines the minimum interface for routers."""
    def route(self, x: jnp.ndarray, context: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        ...

if JAX_AVAILABLE and nn is not None:
    _FlaxBase = nn.Module
else:
    class _FlaxBase:  # type: ignore[no-redef]
        pass

class BaseRouter(_FlaxBase, ABC):  # type: ignore[misc]
    """Abstract base class for all routers"""
    config: RouterConfig
    
    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config
        
    def setup(self) -> None:
        """Common configuration for all routers."""
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_heads
        self.dropout_rate = self.config.dropout_rate

        # Fallback configuration
        self.fallback_config = FallbackConfig()
        self.current_batch_size = self.config.batch_size
        self.in_fallback_mode = False
        self.last_fallback_time = 0

        # Memory monitor
        self.memory_monitor = MemoryMonitor()

        # Initialize TPU operations directly if available
        if NATIVE_TPU_AVAILABLE and TPU_V4_AVAILABLE:
            self._initialize_tpu_ops()
        else:
            self._initialize_cpu_ops()
            
    def _initialize_tpu_ops(self):
        """Initialize TPU operations."""
        try:
            self.linalg_ops = TpuV4LinalgOps()
            self.sparse_ops = TpuV4SparseOps()
            self.neural_ops = TpuV4NeuralOps()
            logger.info(" Native TPU v4-32 operations initialized")
        except Exception as e:
            logger.error(f" Error initializing TPU ops: {e}")
            self._initialize_cpu_ops()

    def _initialize_cpu_ops(self):
        """Initialize CPU operations."""
        self.linalg_ops = None
        self.sparse_ops = None
        self.neural_ops = None
        logger.info("️ Using CPU fallback operations")
    
    def _check_memory_usage(self) -> bool:
        """Verify memory usage."""
        memory_usage = self.memory_monitor.get_memory_usage()
        return memory_usage > self.fallback_config.memory_threshold

    def _check_latency(self, start_time: float) -> bool:
        """Verify latency."""
        import time
        latency = (time.time() - start_time) * 1000
        return latency > self.fallback_config.latency_threshold_ms

    def _activate_fallback(self):
        """Activate fallback mode."""
        import time
        if not self.in_fallback_mode:
            self.in_fallback_mode = True
            self.last_fallback_time = time.time()

            # Reduce batch size
            new_batch_size = max(
                int(self.current_batch_size * self.fallback_config.batch_size_reduction),
                self.fallback_config.min_batch_size
            )
            self.current_batch_size = new_batch_size

            logger.warning(f"️ Activating fallback mode - Reducing batch size to {new_batch_size}")

    def _check_recovery(self):
        """Verify if we can recover normal mode."""
        import time
        if not self.fallback_config.enable_auto_recovery:
            return

        current_time = time.time()
        if (current_time - self.last_fallback_time > self.fallback_config.recovery_wait_time and
            not self._check_memory_usage()):
            self.in_fallback_mode = False
            self.current_batch_size = self.config.batch_size
            logger.info(" Recovered from fallback mode")
    
    def _optimized_matmul(self, a: jnp.ndarray, b: jnp.ndarray,
                         transpose_a: bool = False, transpose_b: bool = False) -> jnp.ndarray:
        """Optimized matrix multiplication with automatic fallback."""
        import time
        start_time = time.time()
        
        # Check memory and latency
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

        # Standard fallback
        if transpose_a:
            a = jnp.swapaxes(a, -2, -1)
        if transpose_b:
            b = jnp.swapaxes(b, -2, -1)
        return jnp.matmul(a, b)
    
    def _optimized_attention(self, query: jnp.ndarray, key: jnp.ndarray,
                            value: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Optimized attention with automatic fallback."""
        import time
        start_time = time.time()
        
        # Check memory and latency
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
        
        # Standard implementation with memory optimizations
        scores = self._optimized_matmul(query, key, transpose_b=True)
        scores = scores / jnp.sqrt(query.shape[-1])
        
        if mask is not None:
            scores = jnp.where(mask[:, None, None, :], scores, float('-inf'))
        
        # Use less memory in softmax
        if JAX_AVAILABLE and nn is not None:
            weights = nn.softmax(scores, axis=-1)
        else:
            _e = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
            weights = _e / jnp.sum(_e, axis=-1, keepdims=True)
        del scores  # Free memory
        
        output = self._optimized_matmul(weights, value)
        del weights  # Free memory
        
        return output
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
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
        """Combine outputs from routed experts with a safe default."""
        if not isinstance(outputs, dict) or not outputs:
            return outputs

        values = list(outputs.values())
        try:
            shapes = [getattr(v, "shape", None) for v in values]
            if shapes and all(shape == shapes[0] for shape in shapes):
                stacked = jnp.stack(values, axis=0)
                return jnp.mean(stacked, axis=0)
        except Exception:
            pass

        return outputs

class BaseRouterV2(BaseRouter):
    """Base router version 2 with extended functionality."""
    
    def __init__(self, config: Optional[RouterConfig] = None):
        """Initializes the router."""
        super().__init__(config or RouterConfig())
        self.memory_monitor = MemoryMonitor()
    
    def route(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation base of routing."""
        return inputs
        
    def combine_outputs(self, outputs: Dict[str, Any]) -> Any:
        """Default combine behavior for v2 routers."""
        return super().combine_outputs(outputs)

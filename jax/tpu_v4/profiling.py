"""Profiling y métricas para TPU v4-32."""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from functools import partial

# Gets the current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to obtain project root -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Add project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import jax
    import jax.numpy as jnp
    from capibara.jax.experimental import checkpoint
    from capibara.jax.sharding import PartitionSpec as P
    JAX_AVAILABLE = True
    # Type alias for annotations
    ArrayType = jnp.ndarray
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    # Fallback type alias for annotations
    from typing import Any
    ArrayType = Any

logger = logging.getLogger(__name__)

# Define checkpoint function for compatibility
def checkpoint_fn(fn):
    """Checkpoint decorator - using capibara's checkpoint."""
    if JAX_AVAILABLE:
        return checkpoint.checkpoint(fn)
    else:
        # Fallback: just return the function without checkpointing
        return fn

class TpuProfiler:
    """Profiler específico para TPU v4-32."""
    
    def __init__(self, model):
        self.model = model
        self.metrics_collector = model.metrics_collector
        self.config = model.config
    
    def profile_forward_pass(self, inputs: Dict[str, ArrayType], 
                           context: Optional[ArrayType] = None) -> Dict[str, Any]:
        """Profile of forward pass for optimization."""
        with jax.profiler.trace("/tmp/capibara_profile"):
            output = self.model(inputs, context, training=False)
            output["output"].block_until_ready()
        return output
    
    def get_tpu_metrics(self) -> Dict[str, float]:
        """Get TPU-specific metrics."""
        metrics = {}
        for device in jax.devices():
            if device.platform == 'tpu':
                stats = device.memory_stats()
                metrics[f"tpu_{device.id}_memory_used_gb"] = stats.get('bytes_in_use', 0) / (1024**3)
                metrics[f"tpu_{device.id}_memory_limit_gb"] = stats.get('bytes_limit', 0) / (1024**3)
                metrics[f"tpu_{device.id}_memory_utilization"] = (
                    stats.get('bytes_in_use', 0) / stats.get('bytes_limit', 1)
                )
        return metrics
    
    def benchmark_tpu_v4_32(self) -> List[Dict[str, Any]]:
        """Benchmark específico para TPU v4-32."""
        # Configuraciones de prueba optimizadas para TPU
        batch_sizes = [32, 64, 128, 256]  # TPU works better with large batches
        seq_lengths = [512, 1024, 2048]
        
        results = []
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                # Create test data
                inputs = {
                    "text": jax.random.randint(
                        jax.random.PRNGKey(42),
                        (batch_size, seq_len),
                        0, self.config.vocab_size
                    )
                }
                
                # Warmup
                for _ in range(5):
                    _ = self.model(inputs, training=False)
                
                # Benchmark
                start = time.time()
                for _ in range(100):
                    output = self.model(inputs, training=False)
                    output["output"].block_until_ready()
                duration = time.time() - start
                
                # Calcular métricas
                total_tokens = batch_size * seq_len * 100
                throughput = total_tokens / duration
                
                # Estimar FLOPS (aproximado)
                flops_per_token = 6 * self.config.hidden_size * self.config.hidden_size
                tflops = (total_tokens * flops_per_token) / (duration * 1e12)
                
                # Obtener métricas de memoria
                memory_metrics = self.get_tpu_metrics()
                
                results.append({
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "throughput_tokens/sec": throughput,
                    "latency_ms": (duration / 100) * 1000,
                    "estimated_tflops": tflops,
                    "memory_metrics": memory_metrics
                })
                
                # Registrar métricas
                self.metrics_collector.record(
                    f"benchmark_b{batch_size}_s{seq_len}_throughput", 
                    throughput
                )
                self.metrics_collector.record(
                    f"benchmark_b{batch_size}_s{seq_len}_tflops", 
                    tflops
                )
        
        return results

@checkpoint_fn
def checkpointed_transformer_block(block, x: ArrayType, training: bool) -> ArrayType:
    """Transformer block con gradient checkpointing."""
    return block(x, training=training)

def _uniform_fallback_weights(x: ArrayType, num_modules: int, 
                            dtype, mesh) -> ArrayType:
    """Fallback weights optimizado para TPU."""
    batch_size = x.shape[0]
    
    # Create uniform weights with correct dtype
    uniform_weights = jnp.full(
        (batch_size, num_modules), 
        1.0 / num_modules,
        dtype=dtype  # Use bfloat16 for TPU
    )
    
    # Aplicar sharding
    return jax.lax.with_sharding_constraint(
        uniform_weights, 
        P("data", None)
    )

def _expert_weights_with_cache(x: ArrayType, context: Optional[ArrayType], 
                             training: bool, cache: Dict) -> ArrayType:
    """Compute expert weights con cache para TPU."""
    # Añadir cache key based en shape para reutilizar compilaciones
    cache_key = (x.shape, context.shape if context is not None else None, training)
    
    if cache_key in cache:
        return cache[cache_key]
    
    # Rest of routing code...
    # [Router-specific implementation]
    
    return weights 
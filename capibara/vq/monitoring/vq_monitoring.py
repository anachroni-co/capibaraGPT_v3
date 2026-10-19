"""
Sistema de monitoreo for vector Quantization en tpu v6.
Proporciona métricas detalladas de rendimiento, memory and costos.
"""

import jax
import time
import logging
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

@dataclass
class VQMonitoringConfig:
    """setup for el sistema de monitoreo VQ."""
    enable_performance_tracking: bool = True
    enable_memory_tracking: bool = True
    enable_cost_tracking: bool = True
    enable_diversity_tracking: bool = True
    metrics_window_size: int = 1000
    log_interval_steps: int = 100
    alert_memory_threshold: float = 0.95  # 95% uso memory
    alert_performance_threshold: float = 0.5  # 50% rendimiento esperado
    alert_cost_threshold: float = 1000.0  # USD by hora
    alert_diversity_threshold: float = 0.8  # 80% diversidad mínima


class VQPerformanceMetrics:
    """Métricas de rendimiento for operaciones VQ."""
    
    def __init__(self, config: VQMonitoringConfig):
        self.config = config
        self.reset_metrics()
        
    def reset_metrics(self):
        """Reset all performance metrics."""
        self.metrics = {
            'vq_encode_time': [],
            'ml_layers_time': [],
            'attention_time': [],
            'total_time': [],
            'tflops_utilization': [],
            'operations_per_second': [],
            'batch_throughput': []
        }
        
    def update_metrics(self, 
                      operation_type: str,
                      duration: float,
                      batch_size: int,
                      num_operations: int):
        """Update performance metrics for an operation."""
        # Add duration to appropriate metric
        if operation_type in self.metrics:
            self.metrics[operation_type].append(duration)
            
            # Keep only recent metrics
            if len(self.metrics[operation_type]) > self.config.metrics_window_size:
                self.metrics[operation_type] = self.metrics[operation_type][-self.config.metrics_window_size:]
        
        # Calculate operations per second
        ops_per_second = num_operations / duration
        self.metrics['operations_per_second'].append(ops_per_second)
        
        # Calculate batch throughput
        throughput = batch_size / duration
        self.metrics['batch_throughput'].append(throughput)
        
        # Calculate TFLOPS utilization
        theoretical_peak = 1293.0  # tpu v6 peak TFLOPS
        actual_tflops = (num_operations * 2) / (duration * 1e12)  # *2 for FMA
        utilization = actual_tflops / theoretical_peak
        self.metrics['tflops_utilization'].append(utilization)
        
        # Check for performance alerts
        if utilization < self.config.alert_performance_threshold:
            logger.warning(
                f"Low performance alert - Utilization: {utilization:.2%}, "
                f"Operation: {operation_type}"
            )
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of recent performance metrics."""
        summary = {}
        for metric, values in self.metrics.items():
            if values:
                summary[f"avg_{metric}"] = np.mean(values[-100:])
                summary[f"max_{metric}"] = np.max(values[-100:])
                summary[f"min_{metric}"] = np.min(values[-100:])
        return summary


class VQMemoryTracker:
    """Monitor memory usage for operaciones VQ."""
    
    def __init__(self, config: VQMonitoringConfig):
        self.config = config
        self.reset_tracking()
        
    def reset_tracking(self):
        """Reset memory tracking metrics."""
        self.metrics = {
            'total_allocated': [],
            'peak_allocated': [],
            'fragmentation': [],
            'cache_size': [],
            'cache_hit_rate': []
        }
        self.peak_memory = 0
        
    def update_tracking(self, 
                       current_allocated: int,
                       cache_size: int,
                       cache_hits: int,
                       cache_misses: int):
        """Update memory tracking metrics."""
        # Update allocated memory metrics
        self.metrics['total_allocated'].append(current_allocated)
        self.peak_memory = max(self.peak_memory, current_allocated)
        self.metrics['peak_allocated'].append(self.peak_memory)
        
        # Calculate memory fragmentation
        fragmentation = 1.0 - (current_allocated / self.peak_memory)
        self.metrics['fragmentation'].append(fragmentation)
        
        # Update cache metrics
        self.metrics['cache_size'].append(cache_size)
        hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
        self.metrics['cache_hit_rate'].append(hit_rate)
        
        # Check for memory alerts
        total_memory = jax.device_get(jax.devices()[0].memory_stats()['bytes_in_use'])
        memory_utilization = current_allocated / total_memory
        if memory_utilization > self.config.alert_memory_threshold:
            logger.warning(
                f"High memory usage alert - Utilization: {memory_utilization:.2%}, "
                f"Allocated: {current_allocated / 1e9:.2f}GB"
            )
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of memory metrics."""
        return {
            'current_memory_gb': self.metrics['total_allocated'][-1] / 1e9,
            'peak_memory_gb': self.peak_memory / 1e9,
            'avg_fragmentation': np.mean(self.metrics['fragmentation'][-100:]),
            'cache_hit_rate': np.mean(self.metrics['cache_hit_rate'][-100:])
        }


class VQCostTracker:
    """Monitor costos de operaciones VQ."""
    
    def __init__(self, config: VQMonitoringConfig):
        self.config = config
        self.reset_tracking()
        
        # Costos by hora tpu v6 (estimados)
        self.tpu_cost_per_hour = 100.0  # USD
        self.network_cost_per_gb = 0.12  # USD
        
    def reset_tracking(self):
        """Reset cost tracking metrics."""
        self.metrics = {
            'compute_cost': [],
            'memory_cost': [],
            'network_cost': [],
            'total_cost': [],
            'cost_per_operation': []
        }
        self.start_time = time.time()
        
    def update_tracking(self,
                       duration: float,
                       num_operations: int,
                       memory_used_gb: float,
                       network_transfer_gb: float):
        """Update cost tracking metrics."""
        # Calculate costs for this period
        hours = duration / 3600.0
        compute_cost = hours * self.tpu_cost_per_hour
        memory_cost = memory_used_gb * 0.05  # USD per GB-hour
        network_cost = network_transfer_gb * self.network_cost_per_gb
        
        # Update metrics
        self.metrics['compute_cost'].append(compute_cost)
        self.metrics['memory_cost'].append(memory_cost)
        self.metrics['network_cost'].append(network_cost)
        
        total_cost = compute_cost + memory_cost + network_cost
        self.metrics['total_cost'].append(total_cost)
        
        cost_per_op = total_cost / num_operations
        self.metrics['cost_per_operation'].append(cost_per_op)
        
        # Check for cost alerts
        hourly_rate = total_cost / hours
        if hourly_rate > self.config.alert_cost_threshold:
            logger.warning(
                f"High cost alert - Rate: ${hourly_rate:.2f}/hour, "
                f"Cost per op: ${cost_per_op:.6f}"
            )
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of cost metrics."""
        total_time = time.time() - self.start_time
        return {
            'total_cost_usd': sum(self.metrics['total_cost']),
            'cost_per_hour_usd': sum(self.metrics['total_cost']) / (total_time / 3600),
            'avg_cost_per_op_usd': np.mean(self.metrics['cost_per_operation'][-100:])
        }


class VQDiversityTracker:
    """Monitor diversidad and coherencia de códigos VQ."""
    
    def __init__(self, config: VQMonitoringConfig):
        self.config = config
        self.reset_tracking()
        
    def reset_tracking(self):
        """Reset diversity tracking metrics."""
        self.metrics = {
            'code_usage': [],
            'entropy': [],
            'perplexity': [],
            'coherence': []
        }
        
    def update_tracking(self,
                       code_usage: jnp.ndarray,
                       encoded_states: jnp.ndarray,
                       original_states: jnp.ndarray):
        """Update diversity tracking metrics."""
        # Calculate code usage distribution
        usage_dist = jnp.mean(code_usage, axis=0)
        self.metrics['code_usage'].append(usage_dist)
        
        # Calculate entropy of code usage
        entropy = -jnp.sum(usage_dist * jnp.log(usage_dist + 1e-10))
        self.metrics['entropy'].append(float(entropy))
        
        # Calculate perplexity
        perplexity = jnp.exp(entropy)
        self.metrics['perplexity'].append(float(perplexity))
        
        # Calculate coherence (similarity between estados originales and codificados)
        coherence = jnp.mean(
            jnp.sum(original_states * encoded_states, axis=-1) /
            (jnp.linalg.norm(original_states, axis=-1) * 
             jnp.linalg.norm(encoded_states, axis=-1) + 1e-10)
        )
        self.metrics['coherence'].append(float(coherence))
        
        # Check for diversity alerts
        normalized_entropy = entropy / jnp.log(len(usage_dist))
        if normalized_entropy < self.config.alert_diversity_threshold:
            logger.warning(
                f"Low diversity alert - Normalized entropy: {normalized_entropy:.2%}, "
                f"Perplexity: {perplexity:.2f}"
            )
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of diversity metrics."""
        return {
            'avg_entropy': np.mean(self.metrics['entropy'][-100:]),
            'avg_perplexity': np.mean(self.metrics['perplexity'][-100:]),
            'avg_coherence': np.mean(self.metrics['coherence'][-100:]),
            'active_codes_ratio': np.mean(
                [np.sum(u > 0.01) / len(u) for u in self.metrics['code_usage'][-100:]]
            )
        }


class VQMonitoringSystem:
    """Sistema complete de monitoreo VQ."""
    
    def __init__(self, config: Optional[VQMonitoringConfig] = None):
        self.config = config or VQMonitoringConfig()
        self.performance = VQPerformanceMetrics(self.config)
        self.memory = VQMemoryTracker(self.config)
        self.cost = VQCostTracker(self.config)
        self.diversity = VQDiversityTracker(self.config)
        
    def start_operation(self) -> float:
        """Start tracking a new operation."""
        return time.time()
        
    def end_operation(self,
                     start_time: float,
                     operation_type: str,
                     batch_size: int,
                     num_operations: int,
                     memory_used: int,
                     cache_info: Dict[str, int],
                     code_usage: jnp.ndarray,
                     encoded_states: jnp.ndarray,
                     original_states: jnp.ndarray,
                     network_transfer: float = 0.0):
        """Track completion of an operation."""
        duration = time.time() - start_time
        
        # Update all tracking systems
        self.performance.update_metrics(
            operation_type, duration, batch_size, num_operations
        )
        
        self.memory.update_tracking(
            memory_used,
            cache_info['size'],
            cache_info['hits'],
            cache_info['misses']
        )
        
        self.cost.update_tracking(
            duration,
            num_operations,
            memory_used / 1e9,  # Convert to GB
            network_transfer
        )
        
        self.diversity.update_tracking(
            code_usage,
            encoded_states,
            original_states
        )
        
        # Log periodic summaries
        self._log_summaries()
        
    def _log_summaries(self):
        """Log summaries of all metrics periodically."""
        total_ops = len(self.performance.metrics['total_time'])
        if total_ops % self.config.log_interval_steps == 0:
            perf_summary = self.performance.get_summary()
            mem_summary = self.memory.get_summary()
            cost_summary = self.cost.get_summary()
            div_summary = self.diversity.get_summary()
            
            logger.info(
                f"VQ Monitoring Summary (Step {total_ops}):\n"
                f"Performance: {perf_summary}\n"
                f"Memory: {mem_summary}\n"
                f"Cost: {cost_summary}\n"
                f"Diversity: {div_summary}"
            )
    
    def get_full_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all metrics."""
        return {
            'performance': self.performance.get_summary(),
            'memory': self.memory.get_summary(),
            'cost': self.cost.get_summary(),
            'diversity': self.diversity.get_summary()
        } 

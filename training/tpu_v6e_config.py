"""
Specific configurations for TPU v6e 8x8 with preemptible instances.

This module contains all optimized configurations for:
- TPU v6e with 8x8 topology (64 chips)
- Google Cloud preemptible instances
- Robust checkpointing and recovery
- Optimal distributed parallelization
"""

import os
import logging
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

@dataclass
class TPUv6eHardwareConfig:
    """Configuration specific for hardware TPU v6e."""
    # Especificaciones TPU v6e
    accelerator_type: str = "v6e-64"
    machine_type: str = "ct6e-standard-4t"
    
    # Topología 8x8
    mesh_rows: int = 8
    mesh_cols: int = 8
    total_chips: int = 64
    
    # Distribución física
    hosts: int = 8
    vms: int = 16
    chips_per_host: int = 8
    chips_per_vm: int = 4
    
    # Especificaciones de memoria
    hbm_per_chip_gb: int = 32  # 32GB HBM por chip TPU v6e
    total_hbm_gb: int = 2048   # 64 chips × 32GB
    
    # Network y bandwidth
    interconnect_bandwidth_gbps: int = 4800  # ICI bandwidth
    host_bandwidth_gbps: int = 1600
    
@dataclass
class PreemptibleConfig:
    """Configuration for handling instancias preemptibles."""
    # Checkpoint frequency (muy importante para preemptibles)
    checkpoint_every_steps: int = 100
    emergency_checkpoint_steps: int = 50
    quick_save_steps: int = 25
    
    # Estrategias de respaldo
    keep_last_n_checkpoints: int = 5
    keep_emergency_checkpoints: int = 10
    backup_to_gcs: bool = True
    gcs_bucket: str = "capibara-tpu-checkpoints"
    
    # Monitoring de preemption
    preemption_warning_seconds: int = 30
    max_preemption_retries: int = 5
    retry_delay_base_seconds: float = 60.0
    exponential_backoff: bool = True
    
    # Health checks
    health_check_interval_steps: int = 10
    device_health_timeout_seconds: int = 30

@dataclass
class ParallelizationConfig:
    """Configuration for optimized parallelization para 8x8."""
    # Estrategias parallelization
    data_parallel_size: int = 16    # A través de VMs
    model_parallel_size: int = 4    # Dentro de cada VM
    pipeline_parallel_size: int = 1 # No pipeline por defecto
    
    # Sharding strategies
    parameter_sharding: str = "fsdp"  # Fully Sharded Data Parallel
    optimizer_sharding: bool = True
    gradient_sharding: bool = True
    
    # Batch configuration
    global_batch_size: int = 2048
    micro_batch_size: int = 32
    gradient_accumulation_steps: int = 8
    
    # Communication optimization
    gradient_compression: bool = True
    all_reduce_algorithm: str = "ring"  # ring, tree, or nccl
    overlap_communication: bool = True

@dataclass
class OptimizationConfig:
    """Optimizations specific for TPU v6e."""
    # Precision
    use_bf16: bool = True
    use_mixed_precision: bool = True
    loss_scale: str = "dynamic"
    
    # Memory optimization
    activation_checkpointing: bool = True
    gradient_checkpointing: bool = True
    offload_optimizer: bool = False  # TPU v6e tiene suficiente HBM
    
    # Compute optimization
    xla_optimization_level: int = 3
    enable_async_collective_fusion: bool = True
    enable_multiple_steps_fusion: bool = True
    
    # JAX specific
    jax_enable_x64: bool = False
    jax_default_matmul_precision: str = "bfloat16"
    jax_spmd_mode: bool = True

@dataclass
class MonitoringConfig:
    """Configuration for monitoring and logging."""
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "capibara-tpu-v6e"
    wandb_entity: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_interval_steps: int = 10
    save_metrics_interval_steps: int = 100
    
    # Performance monitoring
    profile_steps: List[int] = field(default_factory=lambda: [100, 500, 1000])
    monitor_memory: bool = True
    monitor_network: bool = True
    
    # Alertas
    alert_on_high_loss: bool = True
    loss_spike_threshold: float = 2.0
    alert_webhook: Optional[str] = None

class TPUv6eConfigManager:
    """Configuration manager para TPU v6e."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else None
        self.hardware = TPUv6eHardwareConfig()
        self.preemptible = PreemptibleConfig()
        self.parallelization = ParallelizationConfig()
        self.optimization = OptimizationConfig()
        self.monitoring = MonitoringConfig()
        
        if self.config_path and self.config_path.exists():
            self.load_config()
    
    def load_config(self):
        """Load configuration from archivo YAML."""
        if not self.config_path or not self.config_path.exists():
            return
            
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Actualizar configuraciones
        if 'hardware' in config_data:
            self._update_dataclass(self.hardware, config_data['hardware'])
        if 'preemptible' in config_data:
            self._update_dataclass(self.preemptible, config_data['preemptible'])
        if 'parallelization' in config_data:
            self._update_dataclass(self.parallelization, config_data['parallelization'])
        if 'optimization' in config_data:
            self._update_dataclass(self.optimization, config_data['optimization'])
        if 'monitoring' in config_data:
            self._update_dataclass(self.monitoring, config_data['monitoring'])
    
    def save_config(self, output_path: str):
        """Save current configuration to archivo YAML."""
        config_data = {
            'hardware': self._dataclass_to_dict(self.hardware),
            'preemptible': self._dataclass_to_dict(self.preemptible),
            'parallelization': self._dataclass_to_dict(self.parallelization),
            'optimization': self._dataclass_to_dict(self.optimization),
            'monitoring': self._dataclass_to_dict(self.monitoring)
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def setup_environment_variables(self):
        """Configures environment variables para TPU v6e."""
        # JAX configuration
        os.environ['JAX_PLATFORMS'] = 'tpu'
        os.environ['JAX_ENABLE_X64'] = str(self.optimization.jax_enable_x64).lower()
        
        # XLA optimization
        xla_flags = [
            f'--xla_tpu_spmd_threshold_for_allgather_cse=10000',
            f'--xla_tpu_spmd_rewrite_einsum_with_reshape=true',
        ]
        
        if self.optimization.enable_async_collective_fusion:
            xla_flags.append('--xla_tpu_enable_async_collective_fusion=true')
            
        if self.optimization.enable_multiple_steps_fusion:
            xla_flags.append('--xla_tpu_enable_async_collective_fusion_multiple_steps=true')
        
        os.environ['XLA_FLAGS'] = ' '.join(xla_flags)
        
        # LIBTPU configuration
        libtpu_flags = []
        if self.optimization.enable_multiple_steps_fusion:
            libtpu_flags.append('--xla_tpu_enable_async_collective_fusion_multiple_steps=true')
        
        if libtpu_flags:
            os.environ['LIBTPU_INIT_ARGS'] = ' '.join(libtpu_flags)
        
        # Memory optimization
        if self.optimization.activation_checkpointing:
            os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    
    def get_optimal_batch_size(self, model_size_b: float) -> int:
        """Calcular batch size óptimo based en el tamaño del modelo."""
        # Memoria disponible por chip (reservando espacio para activaciones)
        available_memory_gb = self.hardware.hbm_per_chip_gb * 0.7  # 70% utilization
        
        # Memory estimation per parameter (bf16 + gradients + optimizer states)
        memory_per_param_bytes = 8  # 2 (bf16) + 2 (grad) + 4 (optimizer states)
        
        # Model memory
        model_memory_gb = (model_size_b * 1e9 * memory_per_param_bytes) / (1024**3)
        
        # Memoria por sample (estimación conservadora)
        sequence_length = 2048
        hidden_size = int((model_size_b * 1e9 / (12 * sequence_length))**0.5)  # Rough estimate
        memory_per_sample_gb = (sequence_length * hidden_size * 4) / (1024**3)  # bf16 activations
        
        # Calcular batch size por chip
        memory_for_batch = available_memory_gb - (model_memory_gb / self.parallelization.model_parallel_size)
        samples_per_chip = max(1, int(memory_for_batch / memory_per_sample_gb))
        
        # Global batch size
        global_batch = samples_per_chip * self.hardware.total_chips // self.parallelization.model_parallel_size
        
        # Redondear a múltiplo de data parallel size
        global_batch = (global_batch // self.parallelization.data_parallel_size) * self.parallelization.data_parallel_size
        
        return max(self.parallelization.data_parallel_size, global_batch)
    
    def validate_configuration(self) -> List[str]:
        """Validates la configuration and return warnings/errors."""
        warnings = []
        
        # Validar topología
        if self.hardware.mesh_rows * self.hardware.mesh_cols != self.hardware.total_chips:
            warnings.append(f"Mesh topology mismatch: {self.hardware.mesh_rows}x{self.hardware.mesh_cols} != {self.hardware.total_chips}")
        
        # Validar paralelización
        total_parallel = (self.parallelization.data_parallel_size * 
                         self.parallelization.model_parallel_size * 
                         self.parallelization.pipeline_parallel_size)
        if total_parallel > self.hardware.total_chips:
            warnings.append(f"Parallelization exceeds available chips: {total_parallel} > {self.hardware.total_chips}")
        
        # Validar batch size
        if self.parallelization.global_batch_size % self.parallelization.data_parallel_size != 0:
            warnings.append(f"Global batch size {self.parallelization.global_batch_size} not divisible by data parallel size {self.parallelization.data_parallel_size}")
        
        # Validar checkpointing para preemptibles
        if self.preemptible.checkpoint_every_steps > 500:
            warnings.append(f"Checkpoint interval {self.preemptible.checkpoint_every_steps} may be too high for preemptible instances")
        
        return warnings
    
    @staticmethod
    def _update_dataclass(obj, data: Dict[str, Any]):
        """Updatesr dataclass con datos del diccionario."""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    @staticmethod
    def _dataclass_to_dict(obj) -> Dict[str, Any]:
        """Convertir dataclass a diccionario."""
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):
                result[key] = value
        return result

# Predefined configurations for different model scales
def get_config_for_model_scale(model_scale: str) -> TPUv6eConfigManager:
    """Get optimized configuration for a specific model scale."""
    config = TPUv6eConfigManager()
    
    if model_scale == "300M":
        # Modelo pequeño - máximo throughput
        config.parallelization.data_parallel_size = 32
        config.parallelization.model_parallel_size = 2
        config.parallelization.global_batch_size = 4096
        config.parallelization.micro_batch_size = 64
        
    elif model_scale == "1B":
        # Modelo mediano
        config.parallelization.data_parallel_size = 16
        config.parallelization.model_parallel_size = 4
        config.parallelization.global_batch_size = 2048
        config.parallelization.micro_batch_size = 32
        
    elif model_scale == "3B":
        # Modelo grande
        config.parallelization.data_parallel_size = 16
        config.parallelization.model_parallel_size = 4
        config.parallelization.global_batch_size = 1024
        config.parallelization.micro_batch_size = 16
        
    elif model_scale == "7B":
        # Modelo muy grande
        config.parallelization.data_parallel_size = 8
        config.parallelization.model_parallel_size = 8
        config.parallelization.global_batch_size = 512
        config.parallelization.micro_batch_size = 8
        
    elif model_scale == "13B":
        # Modelo extra grande
        config.parallelization.data_parallel_size = 4
        config.parallelization.model_parallel_size = 16
        config.parallelization.global_batch_size = 256
        config.parallelization.micro_batch_size = 4
        
    # Ajustar checkpointing para modelos más grandes (más costosos de restart)
    if model_scale in ["7B", "13B"]:
        config.preemptible.checkpoint_every_steps = 75
        config.preemptible.emergency_checkpoint_steps = 25
    
    return config

# Configuración de ejemplo for development/testing
DEVELOPMENT_CONFIG = {
    'hardware': {
        'mesh_rows': 2,
        'mesh_cols': 2,
        'total_chips': 4  # Para testing local/simulación
    },
    'preemptible': {
        'checkpoint_every_steps': 10,
        'emergency_checkpoint_steps': 5
    },
    'parallelization': {
        'data_parallel_size': 2,
        'model_parallel_size': 2,
        'global_batch_size': 32
    },
    'monitoring': {
        'use_wandb': False,
        'log_interval_steps': 1
    }
}

def create_development_config() -> TPUv6eConfigManager:
    """Createsr configuración for development/testing."""
    config = TPUv6eConfigManager()
    config._update_dataclass(config.hardware, DEVELOPMENT_CONFIG['hardware'])
    config._update_dataclass(config.preemptible, DEVELOPMENT_CONFIG['preemptible'])
    config._update_dataclass(config.parallelization, DEVELOPMENT_CONFIG['parallelization'])
    config._update_dataclass(config.monitoring, DEVELOPMENT_CONFIG['monitoring'])
    return config
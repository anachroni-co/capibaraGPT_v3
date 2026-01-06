"""
setup avanzada for el module Chain-of-Thought.

This module defines las configuraciones específicas for el module CoT,
incluyendo optimizaciones de memory, paralelismo and gestión de recursos.
"""

import logging

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable

from .memory_config import MemoryOptimizationConfig
from .unified_model_config import ModularModelConfig

class ThoughtProcessingStrategy(Enum):
    """Estrategias de procesamiento de pensamientos."""
    SEQUENTIAL = "sequential"  # Procesamiento secuencial tradicional
    PARALLEL = "parallel"      # Procesamiento paralelo de pensamientos
    HIERARCHICAL = "hierarchical"  # Procesamiento jerárquico with dependencias
    ADAPTIVE = "adaptive"      # Adaptativo based en complejidad

@dataclass
class CoreConfig:
    """setup for núcleos de conocimiento."""
    enabled: bool = True
    temperature: float = 0.7
    max_tokens: int = 150
    confidence_threshold: float = 0.6
    priority: int = 1
    cache_size: int = 1000
    batch_size: int = 32
    use_quantization: bool = True
    pruning_threshold: float = 0.1

@dataclass
class RoutingConfig:
    """setup for el routing de conocimiento."""
    strategy: str = "dynamic"  # dynamic, static, hybrid
    cache_size: int = 1000
    context_window: int = 512
    attention_heads: int = 8
    dropout_rate: float = 0.1
    use_sparse_attention: bool = True
    routing_temperature: float = 0.7
    min_confidence: float = 0.6
    max_parallel_routes: int = 4

@dataclass
class ProcessingConfig:
    """setup for el procesamiento de pensamientos."""
    strategy: ThoughtProcessingStrategy = ThoughtProcessingStrategy.ADAPTIVE
    max_steps: int = 8
    min_steps: int = 2
    early_stopping_patience: int = 2
    convergence_threshold: float = 0.01
    max_branching_factor: int = 3
    max_depth: int = 4
    use_beam_search: bool = True
    beam_width: int = 4

@dataclass
class OptimizationConfig:
    """setup for optimizaciones de rendimiento."""
    use_gradient_checkpointing: bool = True
    checkpoint_every_n_steps: int = 2
    mixed_precision: bool = True
    compute_dtype: str = "bfloat16"
    param_dtype: str = "float32"
    output_dtype: str = "float32"
    use_kernel_fusion: bool = True
    max_memory_gb: float = 32.0
    cleanup_threshold: float = 0.85
    batch_size: int = 32
    gradient_accumulation_steps: int = 4

@dataclass
class MonitoringConfig:
    """setup for monitoreo and logging."""
    enable_memory_tracking: bool = True
    memory_log_interval: float = 60.0
    performance_metrics_interval: float = 300.0
    log_thought_stats: bool = True
    log_core_usage: bool = True
    log_memory_usage: bool = True
    profiling_enabled: bool = False
    trace_enabled: bool = False

@dataclass
class AdvancedCoTConfig:
    """Complete advanced setup for the CoT module."""
    
    # setup base del model
    model_config: ModularModelConfig
    
    # setup de memory
    memory_config: MemoryOptimizationConfig
    
    # Configuraciones específicas
    core_config: CoreConfig = field(default_factory=CoreConfig)
    routing_config: RoutingConfig = field(default_factory=RoutingConfig)
    processing_config: ProcessingConfig = field(default_factory=ProcessingConfig)
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)
    monitoring_config: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # setup de paralelismo
    enable_model_parallel: bool = True
    model_parallel_size: int = 2
    pipeline_parallel_size: int = 2
    data_parallel_size: int = 2
    
    # setup de núcleos
    knowledge_core_configs: Dict[str, CoreConfig] = field(default_factory=dict)
    enable_dynamic_routing: bool = True
    enable_hierarchical_reasoning: bool = True
    enable_cross_core_communication: bool = True
    
    # Layers setup
    layer_configs: Dict[str, Any] = field(default_factory=dict)
    layer_selection_strategy: str = "dynamic"
    
    def __post_init__(self):
        """initialization posterior a la creation."""
        if not self.knowledge_core_configs:
            self.knowledge_core_configs = self._create_default_core_configs()
        
        if not self.layer_configs:
            self.layer_configs = self._create_default_layer_configs()
    
    def _create_default_core_configs(self) -> Dict[str, CoreConfig]:
        """Crea configuraciones by defect for los núcleos."""
        return {
            "logical": CoreConfig(
                temperature=0.3,
                confidence_threshold=0.8,
                priority=1
            ),
            "mathematical": CoreConfig(
                temperature=0.2,
                confidence_threshold=0.9,
                priority=1
            ),
            "cultural": CoreConfig(
                temperature=0.8,
                confidence_threshold=0.6,
                priority=2
            ),
            "creative": CoreConfig(
                temperature=0.9,
                confidence_threshold=0.5,
                priority=3,
                use_quantization=False  # maintain precision for creatividad
            )
        }
    
    def _create_default_layer_configs(self) -> Dict[str, Any]:
        """Crea configuraciones by defect for los layers."""
        return {
            "attention": {
                "num_heads": 8,
                "key_size": 64,
                "attention_dropout": 0.1,
                "use_sparse": True
            },
            "feedforward": {
                "hidden_size": self.model_config.hidden_size * 4,
                "activation": "gelu",
                "dropout": 0.1
            },
            "reasoning": {
                "num_layers": 2,
                "intermediate_size": self.model_config.hidden_size * 2,
                "activation": "gelu"
            }
        }
    
    def optimize_for_device(self, device_type: str = "tpu"):
        """Optimiza la setup for un type específico de dispositivo."""
        if device_type == "tpu":
            self.optimization_config.compute_dtype = "bfloat16"
            self.optimization_config.use_kernel_fusion = True
            self.enable_model_parallel = True
        elif device_type == "gpu":
            self.optimization_config.compute_dtype = "float16"
            self.optimization_config.use_kernel_fusion = True
            self.model_parallel_size = min(self.model_parallel_size, 2)
        else:  # cpu
            self.optimization_config.mixed_precision = False
            self.optimization_config.compute_dtype = "float32"
            self.enable_model_parallel = False
        
        return self
    
    def enable_debug_mode(self):
        """Activa configuraciones for debugging."""
        self.monitoring_config.profiling_enabled = True
        self.monitoring_config.trace_enabled = True
        self.monitoring_config.log_thought_stats = True
        self.monitoring_config.memory_log_interval = 10.0
        return self
    
    def enable_production_mode(self):
        """Activa configuraciones optimizadas for producción."""
        self.monitoring_config.profiling_enabled = False
        self.monitoring_config.trace_enabled = False
        self.optimization_config.use_kernel_fusion = True
        self.routing_config.use_sparse_attention = True
        return self 
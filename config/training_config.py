"""
Centralized Training Configurations for CapibaraGPT-v2

This module centralizes all training configurations, including
automatic consensus distilling for 3B+ models and TPU v4-32 optimizations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import os

class ModelScale(Enum):
    """Supported model scales."""
    MICRO_300M = "300M"
    MICRO_600M = "600M"
    MICRO_1_2B = "1.2B"
    SMALL_3B = "3B"
    SMALL_7B = "7B"
    SMALL_13B = "13B"
    MEDIUM_30B = "30B"
    MEDIUM_50B = "50B"
    MEDIUM_65B = "65B"
    LARGE_130B = "130B"
    LARGE_300B = "300B"
    LARGE_650B = "650B"
    XLARGE_1T = "1T"

class TrainingPhase(Enum):
    """Training phases according to scale."""
    MICRO_PHASE = "micro_phase"     # 300M - 1.2B
    SMALL_PHASE = "small_phase"     # 3B - 13B (consensus distilling AUTO)
    MEDIUM_PHASE = "medium_phase"   # 30B - 65B (full consensus + critics)
    LARGE_PHASE = "large_phase"     # 130B - 1T (ensemble + synthetic reasoning)

@dataclass
class HardwareConfig:
    """Hardware configuration."""
    # TPU Configuration
    use_tpu: bool = True
    tpu_cores: int = 32
    mesh_shape: Tuple[int, int] = (4, 8)
    memory_per_core_gb: int = 32
    dtype: str = "bfloat16"
    param_dtype: str = "float32"
    
    # GPU Configuration (fallback)
    use_gpu: bool = False
    gpu_count: int = 8
    mixed_precision: bool = True
    
    # General
    enable_xla: bool = True
    enable_sharding: bool = True
    optimization_level: int = 3

@dataclass
class ConsensusConfig:
    """Configuration for consensus distilling."""
    # Voting Configuration
    n_teachers: int = 5
    n_critics: int = 3
    alpha: float = 0.1
    confidence_threshold: float = 0.7
    teacher_reset_every: int = 100
    critic_reset_every: int = 25
    
    # Distillation Configuration
    distillation_weight: float = 0.3
    temperature: float = 4.0
    use_soft_targets: bool = True
    
    # Quality Control
    quality_threshold: float = 0.7
    tie_break_method: str = "critic_arbitration"
    
    # Teacher Models (by phase)
    teacher_models: Dict[str, List[str]] = field(default_factory=lambda: {
        "micro_phase": [
            "microsoft/CodeT5p-220m",
            "Salesforce/codet5-base", 
            "microsoft/codebert-base"
        ],
        "small_phase": [
            "microsoft/CodeT5p-770m",
            "Salesforce/codet5-large",
            "microsoft/unixcoder-base"
        ],
        "medium_phase": [
            "bigcode/starcoder-1b",
            "microsoft/CodeT5p-2b",
            "Salesforce/codet5-large"
        ],
        "large_phase": [
            "bigcode/starcoder-3b",
            "microsoft/CodeT5p-6b",
            "facebook/incoder-6B"
        ]
    })

@dataclass
class DatasetConfig:
    """Dataset configuration per phase."""
    # Dataset Sizing
    size_range: Tuple[int, int] = (20_000_000, 80_000_000)
    quality_threshold: float = 0.7
    synthetic_ratio: float = 0.0
    
    # Batch Configuration
    batch_size: int = 512
    seq_length: int = 2048
    cache_enabled: bool = True
    
    # Data Sources
    data_sources: List[str] = field(default_factory=lambda: ["common_crawl", "code", "books"])
    preprocessing_steps: List[str] = field(default_factory=lambda: ["tokenize", "filter", "deduplicate"])

@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    optimizer_type: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Scheduling
    scheduler_type: str = "cosine_annealing"
    warmup_steps: int = 10000
    warmup_ratio: float = 0.1
    
    # Gradient Control
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1

@dataclass
class ModelArchConfig:
    """Architecture configuration per scale."""
    scale: ModelScale
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    vocab_size: int = 50257
    max_length: int = 2048
    dropout_rate: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Activations
    activation_function: str = "gelu"
    use_remat: bool = True
    use_gradient_checkpointing: bool = True
    
    def get_parameter_count(self) -> int:
        """Estimate number of parameters."""
        # Approximate formula for transformer
        embedding_params = self.vocab_size * self.hidden_size
        transformer_params = self.num_layers * (
            # Attention
            4 * self.hidden_size * self.hidden_size +  # Q, K, V, O
            # FFN  
            2 * self.hidden_size * self.intermediate_size +
            # Layer norms
            2 * self.hidden_size
        )
        output_params = self.vocab_size * self.hidden_size
        
        total = embedding_params + transformer_params + output_params
        return total

class TrainingConfigFactory:
    """Factory for creating optimized configurations according to scale."""

    @staticmethod
    def should_use_consensus_distilling(model_scale: ModelScale) -> bool:
        """Determine whether to use consensus distilling automatically."""
        configs = {
            ModelScale.MICRO_300M: False,
            ModelScale.MICRO_600M: False,
            ModelScale.MICRO_1_2B: False,
            ModelScale.SMALL_3B: True,   # AUTO para 3B+
            ModelScale.SMALL_7B: True,
            ModelScale.SMALL_13B: True,
            ModelScale.MEDIUM_30B: True,
            ModelScale.MEDIUM_50B: True,
            ModelScale.MEDIUM_65B: True,
            ModelScale.LARGE_130B: True,
            ModelScale.LARGE_300B: True,
            ModelScale.LARGE_650B: True,
            ModelScale.XLARGE_1T: True
        }
        return configs[model_scale]

# Predefined configurations for quick use
def get_config_for_scale(model_scale: ModelScale) -> Dict[str, Any]:
    """Get complete configuration for a specific scale."""
    use_consensus = TrainingConfigFactory.should_use_consensus_distilling(model_scale)
    
    return {
        "model_scale": model_scale,
        "use_consensus_distilling": use_consensus,
        "consensus_activated_automatically": use_consensus and model_scale.value in ["3B", "7B", "13B", "30B", "50B", "65B", "130B", "300B", "650B", "1T"]
    }

# Quick configurations
MICRO_300M_CONFIG = get_config_for_scale(ModelScale.MICRO_300M)
SMALL_3B_CONFIG = get_config_for_scale(ModelScale.SMALL_3B)  # AUTO consensus distilling
MEDIUM_30B_CONFIG = get_config_for_scale(ModelScale.MEDIUM_30B)  # AUTO consensus + critics
LARGE_1T_CONFIG = get_config_for_scale(ModelScale.XLARGE_1T)  # AUTO ensemble + synthetic 
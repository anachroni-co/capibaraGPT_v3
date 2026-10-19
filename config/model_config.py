"""
Model configuration classes for CapibaraGPT.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    """Tipos de model disponibles."""
    STANDARD = "standard"
    NEUROADAPTIVE = "neuroadaptive"
    HYBRID = "hybrid"
    ULTRA = "ultra"

@dataclass
class ModelConfig:
    """setup base del model."""
    
    # Basic model parameters
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    
    # Model type
    model_type: ModelType = ModelType.STANDARD
    
    # Advanced features
    use_cache: bool = True
    use_gradient_checkpointing: bool = False
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    
    # Activation function
    activation_function: str = "gelu"
    
    # Layer norm configuration
    layer_norm_epsilon: float = 1e-5
    
    # Initialize weights
    initializer_range: float = 0.02

@dataclass
class NeuroAdaptiveConfig(ModelConfig):
    """setup for capas neuro-adaptativas."""
    
    # NeuroAdaptive specific parameters
    adaptation_rate: float = 0.01
    plasticity_threshold: float = 0.1
    memory_capacity: int = 1000
    learning_decay: float = 0.99
    
    # Adaptive mechanisms
    enable_synaptic_plasticity: bool = True
    enable_homeostatic_scaling: bool = True
    enable_metaplasticity: bool = False
    
    # Neural development
    enable_neurogenesis: bool = False
    pruning_threshold: float = 0.05
    growth_factor: float = 1.1
    
    # Memory management
    short_term_memory_size: int = 100
    long_term_memory_size: int = 1000
    consolidation_rate: float = 0.01
    
    def __post_init__(self):
        self.model_type = ModelType.NEUROADAPTIVE

@dataclass
class HybridModelConfig(ModelConfig):
    """setup for modelos híbridos."""
    
    # Hybrid architecture
    ssm_layers: Optional[List[int]] = None  # Which layers use SSM
    attention_layers: Optional[List[int]] = None  # Which layers use attention
    
    # SSM parameters
    ssm_state_size: int = 64
    ssm_rank: int = 16
    ssm_conv_kernel_size: int = 4
    
    # Mixing strategies
    layer_mixing_strategy: str = "interleaved"  # interleaved, alternating, custom
    cross_layer_connections: bool = False
    
    def __post_init__(self):
        self.model_type = ModelType.HYBRID
        if self.ssm_layers is None:
            # Default: alternate between SSM and attention
            self.ssm_layers = list(range(0, self.num_layers, 2))
        if self.attention_layers is None:
            self.attention_layers = list(range(1, self.num_layers, 2))

@dataclass
class UltraModelConfig(ModelConfig):
    """setup for modelos ultra-avanzados."""
    
    # Ultra features
    enable_quantum_layers: bool = False
    enable_vq_compression: bool = True
    enable_expert_routing: bool = True
    enable_dynamic_sparsity: bool = True
    
    # Expert routing
    num_experts: int = 8
    expert_capacity_factor: float = 1.25
    routing_algorithm: str = "learned"  # learned, hash, random
    
    # VQ parameters
    vq_codebook_size: int = 1024
    vq_commitment_cost: float = 0.25
    vq_decay: float = 0.99
    
    # Dynamic sparsity
    sparsity_target: float = 0.9
    sparsity_schedule: str = "polynomial"  # linear, polynomial, exponential
    
    # Quantum-ready features
    quantum_entanglement_layers: Optional[List[int]] = None
    quantum_superposition_enabled: bool = False
    
    def __post_init__(self):
        self.model_type = ModelType.ULTRA
        if self.quantum_entanglement_layers is None:
            self.quantum_entanglement_layers = []

# Factory functions
def create_model_config(
    model_type: Union[str, ModelType] = "standard",
    **kwargs
) -> ModelConfig:
    """create setup de model basada en type."""
    
    if isinstance(model_type, str):
        model_type = ModelType(model_type)
    
    config_classes = {
        ModelType.STANDARD: ModelConfig,
        ModelType.NEUROADAPTIVE: NeuroAdaptiveConfig,
        ModelType.HYBRID: HybridModelConfig,
        ModelType.ULTRA: UltraModelConfig
    }
    
    config_class = config_classes.get(model_type, ModelConfig)
    return config_class(**kwargs)

def get_preset_config(preset: str) -> ModelConfig:
    """obtain setup predefinida."""
    
    presets = {
        "small": ModelConfig(
            hidden_size=512,
            num_layers=6,
            num_heads=8,
            max_position_embeddings=512
        ),
        "medium": ModelConfig(
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            max_position_embeddings=1024
        ),
        "large": ModelConfig(
            hidden_size=1024,
            num_layers=24,
            num_heads=16,
            max_position_embeddings=2048
        ),
        "ultra": UltraModelConfig(
            hidden_size=1536,
            num_layers=48,
            num_heads=24,
            max_position_embeddings=4096,
            enable_vq_compression=True,
            enable_expert_routing=True,
            num_experts=16
        ),
        "neuroadaptive": NeuroAdaptiveConfig(
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            adaptation_rate=0.01,
            enable_synaptic_plasticity=True,
            enable_neurogenesis=True
        )
    }
    
    return presets.get(preset, ModelConfig())

__all__ = [
    'ModelConfig',
    'NeuroAdaptiveConfig', 
    'HybridModelConfig',
    'UltraModelConfig',
    'ModelType',
    'create_model_config',
    'get_preset_config'
]
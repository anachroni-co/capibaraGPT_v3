# Content of config_types.py
"""
Configuration types for CapibaraModel using Pydantic for validation.
"""

import logging
# from pathlib import nore  # Fixed: removed incorrect import
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, ValidationInfo # type: ignore

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

class ModelConfig(BaseModel):
    """Model architecture configuration."""
    hidden_size: int = Field(..., gt=0, description="Hidden layer size")
    seq_len: int = Field(..., gt=0, description="Maximum sequence length")
    num_layers: int = Field(..., gt=0, description="Number of model layers")
    num_heads: int = Field(..., gt=0, description="Number of attention heads")
    dropout_rate: float = Field(..., ge=0.0, le=1.0, description="Dropout rate")

    # Optional attributes
    input_dim: int = Field(default=768, gt=0, description="Input dimension")
    use_self_attention: bool = Field(default=False, description="Use self-attention")
    use_sparse: bool = Field(default=False, description="Use sparse layers")
    use_meta_la: bool = Field(default=False, description="Use meta-learning")
    use_mixture: bool = Field(default=False, description="Use mixture of experts")
    use_liquid: bool = Field(default=False, description="Use liquid state machine")
    use_meta_bamdp: bool = Field(default=False, description="Use meta-BAMDP")
    use_snns_li_cell: bool = Field(default=False, description="Use SNNs Li cell")
    use_spike_ssm: bool = Field(default=False, description="Use spike SSM")
    use_platonic: bool = Field(default=False, description="Use platonic attention")
    use_quineana: bool = Field(default=False, description="Use quineana attention")
    use_aleph_tilde: bool = Field(default=False, description="Use Aleph-tilde")
    use_bitnet_quantizer: bool = Field(default=False, description="Use BitNet quantizer")
    bit_width: int = Field(default=8, ge=1, le=32, description="Bit width for quantization")
    symmetric: bool = Field(default=True, description="Symmetric quantization")
    use_bitnet_in_mixture: bool = Field(default=False, description="Use BitNet in mixture")
    mixture_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Threshold for mixture")
    mixture_sparsity: float = Field(default=0.1, ge=0.0, le=1.0, description="Sparsity for mixture")
    use_flax_deep_dialog: bool = Field(default=False, description="Use Flax Deep Dialog")
    num_bits: int = Field(default=8, description="Number of bits for quantization")
    use_quantization: bool = Field(default=False, description="Enable quantization")
    quant_min: float = Field(default=0.0, description="Minimum value for quantization")
    quant_max: float = Field(default=255.0, description="Maximum value for quantization")

    @field_validator('num_heads')
    def validate_num_heads(cls, v, info: ValidationInfo):
        """Validate that num_heads divides hidden_size."""
        hidden_size = info.data.get('hidden_size')
        if hidden_size is not None and hidden_size % v != 0:
            raise ValueError(f"num_heads ({v}) must divide hidden_size ({hidden_size})")
        return v

    @field_validator('bit_width')
    def validate_bit_width(cls, v, info: ValidationInfo):
        """Validate bit_width when quantization is used."""
        if info.data.get('use_bitnet_quantizer', False) and v not in [4, 8, 16]:
            raise ValueError("bit_width must be 4, 8, or 16 for quantization")
        return v

class TrainingConfig(BaseModel):
    """Training configuration."""
    train_data_path: str = Field(..., description="Path to training data")
    val_data_path: str = Field(..., description="Path to validation data")

    # Optional attributes
    seed: int = Field(default=42, ge=0, description="Random seed")
    batch_size: int = Field(default=32, gt=0, description="Batch size")
    learning_rate: float = Field(default=0.001, gt=0, description="Learning rate")
    num_epochs: int = Field(default=10, gt=0, description="Number of epochs")
    vocab_size: int = Field(default=32000, gt=0, description="Vocabulary size")

    @field_validator('train_data_path', 'val_data_path')
    def validate_data_paths(cls, v):
        """Validate that data paths exist."""
        path = Path(v)
        if not path.exists():
            logger.warning(f"Data path not found: {v}")
        return v

class PruningConfig(BaseModel):
    """Pruning configuration."""
    enabled: bool = Field(default=False, description="Enable pruning")
    threshold: float = Field(default=0.2, ge=0.0, le=1.0, description="Pruning threshold")

class WandbConfig(BaseModel):
    """Weights & Biases configuration."""
    project: str = Field(..., description="Project name in W&B")
    entity: str = Field(..., description="Entity in W&B")

class ModulesConfig(BaseModel):
    """Module configuration."""
    coherence_detector: bool = Field(default=False, description="Enable coherence detector")
    contextual_activation: bool = Field(default=False, description="Enable contextual activation")
    personality_manager: bool = Field(default=False, description="Enable personality manager")
    ethics_module: bool = Field(default=False, description="Enable ethics module")

class PathsConfig(BaseModel):
    """Paths configuration."""
    checkpoints: str = Field(default="checkpoints", description="Path for checkpoints")
    logs: str = Field(default="logs", description="Path for logs")
    data: str = Field(default="data", description="Path for data")

    @field_validator('checkpoints', 'logs', 'data')
    def validate_paths(cls, v):
        """Validate and create paths if they don't exist."""
        path = Path(v)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directory {v}: {str(e)}")
        return v

class CapibaraConfig(BaseModel):
    """Main configuration container."""
    model: ModelConfig
    training: TrainingConfig
    pruning: PruningConfig
    wandb: WandbConfig
    modules: ModulesConfig
    paths: PathsConfig

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CapibaraConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> 'CapibaraConfig':
        """Load config from YAML file."""
        path = Path(yaml_path)
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML config files.")
        with path.open() as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def validate(self) -> List[str]:
        """Perform additional validations of the complete configuration."""
        warnings = []

        # Validate required memory
        try:
            from .config_validators import estimate_model_memory
            model_mem = estimate_model_memory(self.model_dump())
            logger.info(f"Estimated model memory: {model_mem/1e9:.2f}GB")
        except Exception as e:
            warnings.append(f"Error estimating memory: {str(e)}")

        # Validate module compatibility
        if self.modules.coherence_detector and not self.modules.contextual_activation:
            warnings.append("Coherence detector requires contextual activation")

        return warnings

# Content of schemas.py
"""
This file has been moved to capibara/config/__init__.py to have a single source of truth.
The content has been unified in capibara/config/__init__.py
"""

# from typing import Dict, Any, Optional, List, Union
# from pydantic import BaseModel, Field, validator # type: ignore
# import yaml # type: ignore
# from capibaranum import Enum

# ... rest of commented code ... 

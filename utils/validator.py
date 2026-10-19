"""Configuration validator for CapibaraModel using Pydantic."""

import yaml # type: ignore
from pathlib import Path
from typing import Dict, Any, Optional, TypeVar, Callable, cast
from pydantic import BaseModel, Field, validator # type: ignore
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

def typed_validator(field: str) -> Callable[[Callable[[Any, T], T]], Callable[[Any, T], T]]:
    def decorator(func: Callable[[Any, T], T]) -> Callable[[Any, T], T]:
        return cast(Callable[[Any, T], T], validator(field)(func))
    return decorator

class TPUConfig(BaseModel):
    """tpu configuration."""
    cores: int = Field(..., ge=1, le=8)
    memory_gb: int = Field(..., ge=8, le=128)
    dtype: str = Field(..., pattern="^(float32|bfloat16|float16)$")
    precision: str = Field(..., pattern="^(low|medium|high)$")
    optimization_level: int = Field(..., ge=0, le=3)
    enable_xla: bool = True
    enable_auto_mixed_precision: bool = True

class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    enabled: bool = True
    log_interval: int = Field(..., gt=0)
    metrics: list[str] = Field(default_factory=list)
    wandb: Dict[str, Any] = Field(default_factory=dict)

class QuantizationConfig(BaseModel):
    """Quantization configuration."""
    enabled: bool = True
    bits: int = Field(..., ge=4, le=16)
    calibration: Dict[str, Any] = Field(default_factory=dict)
    qat: Dict[str, Any] = Field(default_factory=dict)

    @typed_validator('calibration')
    def validate_calibration(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate calibration configuration."""
        if 'samples' in v:
            assert v['samples'] > 0, "Number of samples must be positive"
        if 'percentile' in v:
            assert 0 < v['percentile'] <= 100, "Percentile must be between 0 and 100"
        return v

class TrainingConfig(BaseModel):
    """Training configuration."""
    batch_size: int = Field(..., gt=0)
    learning_rate: float = Field(..., gt=0)
    weight_decay: float = Field(..., ge=0)
    warmup_steps: int = Field(..., ge=0)
    max_steps: int = Field(..., gt=0)
    gradient_clip: float = Field(..., gt=0)
    optimizer: str = Field(..., pattern="^(adam|adamw|sgd)$")
    scheduler: str = Field(..., pattern="^(cosine|linear|constant)$")
    mixed_precision: bool = True
    use_gradient_centralization: bool = Field(True, description="Use Gradient Centralization")
    gradient_accumulation: int = Field(..., ge=1)
    checkpointing: Dict[str, Any] = Field(default_factory=dict)

class ModelConfig(BaseModel):
    """Model configuration."""
    vocab_size: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)
    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(..., gt=0)
    intermediate_size: int = Field(..., gt=0)
    max_position_embeddings: int = Field(..., gt=0)
    dropout: float = Field(..., ge=0, le=1)
    layer_norm_eps: float = Field(..., gt=0)
    activation: str = Field(..., pattern="^(gelu|relu|tanh)$")
    pad_token_id: int = Field(..., ge=0)
    bos_token_id: int = Field(..., ge=0)
    eos_token_id: int = Field(..., ge=0)

class CapibaraConfig(BaseModel):
    """Complete CapibaraModel configuration."""
    tpu: TPUConfig
    monitoring: MonitoringConfig
    quantization: QuantizationConfig
    training: TrainingConfig
    model: ModelConfig

    class Config:
        """Pydantic configuration."""
        extra = "forbid"
        validate_assignment = True

class ConfigValidator:
    """Configuration validator for CapibaraModel."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration validator.
        
        Args:
            config_path: Optional path to custom configuration file.
        """
        self.config_path = config_path
        self.config: Optional[CapibaraConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load and validate configuration."""
        try:
            # Load default configuration
            default_config_path = Path(__file__).parent / "default.yaml"
            with open(default_config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # If there's custom configuration, update it
            if self.config_path:
                custom_config_path = Path(self.config_path)
                if not custom_config_path.exists():
                    raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
                
                with open(custom_config_path, 'r') as f:
                    custom_config = yaml.safe_load(f)
                    self._update_dict(config_dict, custom_config)
            
            # Validate with Pydantic
            self.config = CapibaraConfig(**config_dict)
            logger.info("Configuration loaded and validated successfully")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _update_dict(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively update a base dictionary with an update dictionary."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._update_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get_config(self) -> CapibaraConfig:
        """Get validated configuration.
        
        Returns:
            Validated configuration as CapibaraConfig object.
            
        Raises:
            RuntimeError: If configuration has not been loaded.
        """
        if self.config is None:
            raise RuntimeError("Configuration has not been loaded")
        return self.config 
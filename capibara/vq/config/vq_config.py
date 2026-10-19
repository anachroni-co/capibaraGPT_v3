"""
Configurations for VQ (Vector Quantization) components of CapibaraGPT.
Includes configurations for blocks, VQ codes, and general parameters.
"""

import os
import sys
import tomli
import logging
import numpy as np
from pathlib import Path

from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
from typing import Literal, Dict, Optional, Union, Any

logger = logging.getLogger(__name__)

@dataclass
class ModalityConfig:
    """Configuration for a specific modality."""
    codebook_size: int
    embedding_dim: int
    adaptive_enabled: bool
    backend: str
    
    def __post_init__(self):
        """validation post-initialization."""
        if self.codebook_size <= 0:
            raise ValueError("codebook_size must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.backend not in ["jax", "torch", "flax"]:
            raise ValueError("backend must be one of: jax, torch, flax")

@dataclass
class BackendConfig:
    """Configuration for a specific VQ backend."""
    batch_size: int = 32
    optimization_level: int = 3
    precision: str = "bfloat16"
    memory_efficient: bool = True
    device: Optional[str] = None
    compiled: Optional[bool] = None
    diff_method: Optional[str] = None
    
    def __post_init__(self):
        """validation post-initialization."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 <= self.optimization_level <= 3:
            raise ValueError("optimization_level must be between 0 and 3")
        if self.precision not in ["bfloat16", "float32", "float16", "mixed"]:
            raise ValueError("precision must be one of: bfloat16, float32, float16, mixed")

class VQConfig(BaseModel):
    """Complete configuration for the VQ module."""
    
    # General configuration
    experimental: bool = True
    debug_mode: bool = False
    log_level: str = "INFO"

    # VQbit configuration
    beta: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Commitment factor for VQbit"
    )
    decay: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Decay rate for EMA"
    )
    epsilon: float = Field(
        default=1e-5,
        gt=0.0,
        description="Epsilon for numerical stability"
    )

    # Modality configuration
    modalities: Dict[str, ModalityConfig]

    # Backend configuration
    backends: Dict[str, BackendConfig]

    # Training configuration
    learning_rate: float = Field(
        default=1e-4,
        gt=0.0,
        description="Learning rate"
    )
    batch_size: int = Field(
        default=32,
        gt=0,
        description="Batch size"
    )
    max_epochs: int = Field(
        default=100,
        gt=0,
        description="Maximum number of epochs"
    )
    early_stopping_patience: int = Field(
        default=10,
        gt=0,
        description="Patience for early stopping"
    )
    gradient_clip: float = Field(
        default=1.0,
        gt=0.0,
        description="Maximum gradient value"
    )

    # Metrics configuration
    track_vq_metrics: bool = True
    track_vqbit_metrics: bool = True
    track_performance_metrics: bool = True
    save_metrics_interval: int = Field(
        default=100,
        gt=0,
        description="Interval for saving metrics"
    )

    # Optimization configuration
    use_mixed_precision: bool = True
    use_gradient_accumulation: bool = True
    accumulation_steps: int = Field(
        default=4,
        gt=0,
        description="Gradient accumulation steps"
    )
    use_amp: bool = True

    # Memory configuration
    max_memory_usage: str = Field(
        default="16GB",
        description="Maximum memory usage"
    )
    clear_cache_interval: int = Field(
        default=1000,
        gt=0,
        description="Interval for clearing cache"
    )
    use_memory_efficient_attention: bool = True

    # Dimension configuration
    hidden_size: int = Field(
        default=512,
        gt=0,
        description="Hidden space size"
    )
    output_size: int = Field(
        default=512,
        gt=0,
        description="Output size"
    )
    
    @validator('max_memory_usage')
    def validate_memory_usage(cls, v):
        """Validates memory usage format."""
        if not v.endswith(('GB', 'MB')):
            raise ValueError("max_memory_usage must end with GB or MB")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validates the logging level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()
    
    @classmethod
    def from_toml(cls, config_path: str = "configs_toml/vq.toml") -> "VQConfig":
        """Loads the configuration from a TOML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, "rb") as f:
            config_dict = tomli.load(f)
            
        # Load general configuration
        general = config_dict.get("general", {})
        vqbit = config_dict.get("vqbit", {})

        # Load modality configuration
        modalities = {}
        for name, mod_config in config_dict.get("modalities", {}).items():
            modalities[name] = ModalityConfig(
                codebook_size=mod_config["codebook_size"],
                embedding_dim=mod_config["embedding_dim"],
                adaptive_enabled=mod_config["adaptive_enabled"],
                backend=mod_config["backend"]
            )

        # Load backend configuration
        backends = {}
        for name, backend_config in config_dict.get("vq", {}).get("backends", {}).items():
            backends[name] = BackendConfig(**backend_config)

        # Load training configuration
        training = config_dict.get("training", {})

        # Load metrics configuration
        metrics = config_dict.get("metrics", {})

        # Load optimization configuration
        optimization = config_dict.get("optimization", {})

        # Load memory configuration
        memory = config_dict.get("memory", {})

        # Build complete configuration
        config_data = {
            **general,
            **vqbit,
            **training,
            **metrics,
            **optimization,
            **memory,
            "modalities": modalities,
            "backends": backends
        }
        
        return cls(**config_data)
    
    def get_vqbit_config(self) -> Dict[str, Any]:
        """Gets the VQbit configuration."""
        return {
            "beta": self.beta,
            "decay": self.decay,
            "epsilon": self.epsilon,
            "track_vqbit_metrics": self.track_vqbit_metrics
        }
    
    def get_vq_config(self) -> Dict[str, Any]:
        """Gets the VQ configuration."""
        return {
            "modalities": self.modalities,
            "backends": self.backends,
            "track_vq_metrics": self.track_vq_metrics
        }
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Gets the memory configuration."""
        return {
            "max_memory_usage": self.max_memory_usage,
            "clear_cache_interval": self.clear_cache_interval,
            "use_memory_efficient_attention": self.use_memory_efficient_attention,
            "use_mixed_precision": self.use_mixed_precision
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Gets the training configuration."""
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "gradient_clip": self.gradient_clip
        }
    
    def get_metrics_config(self) -> Dict[str, Any]:
        """Gets the metrics configuration."""
        return {
            "track_vq_metrics": self.track_vq_metrics,
            "track_vqbit_metrics": self.track_vqbit_metrics,
            "track_performance_metrics": self.track_performance_metrics,
            "save_metrics_interval": self.save_metrics_interval
        }

# Default configurations
DEFAULT_VQ_CONFIG = VQConfig(
    modalities={
        "text": ModalityConfig(
            codebook_size=512,
            embedding_dim=128,
            adaptive_enabled=True,
            backend="jax"
        ),
        "image": ModalityConfig(
            codebook_size=1024,
            embedding_dim=256,
            adaptive_enabled=True,
            backend="jax"
        ),
        "audio": ModalityConfig(
            codebook_size=256,
            embedding_dim=64,
            adaptive_enabled=True,
            backend="jax"
        )
    },
    backends={
        "jax": BackendConfig(
            batch_size=32,
            optimization_level=3,
            precision="bfloat16",
            memory_efficient=True,
            device="tpu",
            compiled=True
        ),
        "torch": BackendConfig(
            batch_size=32,
            optimization_level=2,
            precision="float32",
            memory_efficient=True,
            device="cuda",
            compiled=False
        )
    }
)

# Legacy configuration for compatibility
AdaptiveConfig = VQConfig  # Alias for backward compatibility
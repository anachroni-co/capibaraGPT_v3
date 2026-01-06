"""
Configuration for Convexity-Aware Learning Rate Control.

This module provides configuration classes for the ConvexityController
and ConvexityTrainingStrategy, integrating with the existing unified
configuration system.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


@dataclass
class ConvexityControllerConfig:
    """Configuration for convexity-aware learning rate control."""
    
    # Core parameters
    window_size: int = 5                    # Window size for second difference (>=3)
    ema_alpha: float = 0.2                  # EMA smoothing factor [0,1]
    tolerance: float = 1e-5                 # Convexity tolerance (>=0)
    
    # Adjustment factors
    down_multiplier: float = 0.90           # Factor when losing convexity
    up_multiplier: float = 1.01             # Factor when maintaining convexity
    
    # Safety bounds
    min_gain: float = 0.1                   # Minimum LR gain
    max_gain: float = 3.0                   # Maximum LR gain
    
    # Control parameters
    cooldown_steps: int = 50                # Steps between reductions
    warmup_enable_steps: int = 500          # Steps before enabling increases
    
    # Advanced options
    enable_logging: bool = True             # Enable detailed logging
    stability_threshold: float = 1e-8       # Stability threshold for metrics
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.window_size < 3:
            raise ValueError("window_size must be >= 3")
        if not 0 <= self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in [0,1]")
        if self.tolerance < 0:
            raise ValueError("tolerance must be >= 0")
        if self.min_gain <= 0 or self.max_gain <= self.min_gain:
            raise ValueError("Invalid gain bounds")
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be >= 0")


@dataclass
class ConvexityTrainingConfig:
    """Configuration for convexity-aware training."""
    
    # Base training parameters
    total_steps: int = 10000
    base_lr: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    
    # Optimizer configuration
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0
    
    # Convexity controller configuration
    convexity_config: Optional[ConvexityControllerConfig] = None
    
    # Logging and monitoring
    log_every: int = 50
    checkpoint_every: int = 1000
    enable_wandb: bool = False
    
    # Advanced options
    min_lr_factor: float = 0.1  # Minimum LR as fraction of base_lr
    
    def __post_init__(self):
        """Initialize default convexity config if not provided."""
        if self.convexity_config is None:
            self.convexity_config = ConvexityControllerConfig()


class ConvexityConfigPydantic(BaseModel):
    """Pydantic version of ConvexityControllerConfig for validation."""
    
    # Core parameters
    window_size: int = Field(5, ge=3, description="Window size for second difference")
    ema_alpha: float = Field(0.2, ge=0.0, le=1.0, description="EMA smoothing factor")
    tolerance: float = Field(1e-5, ge=0.0, description="Convexity tolerance")
    
    # Adjustment factors
    down_multiplier: float = Field(0.90, gt=0.0, le=1.0, description="Factor when losing convexity")
    up_multiplier: float = Field(1.01, ge=1.0, description="Factor when maintaining convexity")
    
    # Safety bounds
    min_gain: float = Field(0.1, gt=0.0, description="Minimum LR gain")
    max_gain: float = Field(3.0, gt=0.0, description="Maximum LR gain")
    
    # Control parameters
    cooldown_steps: int = Field(50, ge=0, description="Steps between reductions")
    warmup_enable_steps: int = Field(500, ge=0, description="Steps before enabling increases")
    
    # Advanced options
    enable_logging: bool = Field(True, description="Enable detailed logging")
    stability_threshold: float = Field(1e-8, gt=0.0, description="Stability threshold")
    
    @validator('max_gain')
    def validate_gain_bounds(cls, v, values):
        """Validate that max_gain > min_gain."""
        if 'min_gain' in values and v <= values['min_gain']:
            raise ValueError('max_gain must be greater than min_gain')
        return v
    
    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"


class ConvexityTrainingConfigPydantic(BaseModel):
    """Pydantic version of ConvexityTrainingConfig for validation."""
    
    # Base training parameters
    total_steps: int = Field(10000, gt=0, description="Total training steps")
    base_lr: float = Field(3e-4, gt=0.0, description="Base learning rate")
    warmup_steps: int = Field(1000, ge=0, description="Warmup steps")
    weight_decay: float = Field(0.1, ge=0.0, description="Weight decay")
    
    # Optimizer configuration
    adam_b1: float = Field(0.9, ge=0.0, lt=1.0, description="Adam beta1")
    adam_b2: float = Field(0.95, ge=0.0, lt=1.0, description="Adam beta2")
    adam_eps: float = Field(1e-8, gt=0.0, description="Adam epsilon")
    grad_clip_norm: float = Field(1.0, gt=0.0, description="Gradient clipping norm")
    
    # Convexity controller configuration
    convexity_config: Optional[ConvexityConfigPydantic] = None
    
    # Logging and monitoring
    log_every: int = Field(50, gt=0, description="Logging frequency")
    checkpoint_every: int = Field(1000, gt=0, description="Checkpointing frequency")
    enable_wandb: bool = Field(False, description="Enable WandB logging")
    
    # Advanced options
    min_lr_factor: float = Field(0.1, gt=0.0, le=1.0, description="Minimum LR factor")
    
    @validator('warmup_steps')
    def validate_warmup_steps(cls, v, values):
        """Validate that warmup_steps <= total_steps."""
        if 'total_steps' in values and v > values['total_steps']:
            raise ValueError('warmup_steps must be <= total_steps')
        return v
    
    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"


# Configuration presets

def get_default_convexity_config() -> ConvexityControllerConfig:
    """Get default convexity controller configuration."""
    return ConvexityControllerConfig()


def get_conservative_convexity_config() -> ConvexityControllerConfig:
    """Get conservative convexity controller configuration."""
    return ConvexityControllerConfig(
        window_size=7,
        ema_alpha=0.15,
        tolerance=3e-5,
        down_multiplier=0.95,
        up_multiplier=1.005,
        min_gain=0.2,
        max_gain=2.0,
        cooldown_steps=75,
        warmup_enable_steps=1000
    )


def get_aggressive_convexity_config() -> ConvexityControllerConfig:
    """Get aggressive convexity controller configuration."""
    return ConvexityControllerConfig(
        window_size=3,
        ema_alpha=0.3,
        tolerance=1e-6,
        down_multiplier=0.85,
        up_multiplier=1.02,
        min_gain=0.05,
        max_gain=5.0,
        cooldown_steps=25,
        warmup_enable_steps=200
    )


def get_tpu_v6e_convexity_config() -> ConvexityControllerConfig:
    """Get TPU v6e-64 optimized convexity controller configuration."""
    return ConvexityControllerConfig(
        window_size=5,
        ema_alpha=0.2,
        tolerance=1e-5,
        down_multiplier=0.90,
        up_multiplier=1.01,
        min_gain=0.1,
        max_gain=3.0,
        cooldown_steps=50,
        warmup_enable_steps=500,
        enable_logging=True
    )


def get_training_config_for_model_size(
    model_size: str,
    total_steps: int = 50000,
    base_lr: Optional[float] = None
) -> ConvexityTrainingConfig:
    """
    Get training configuration optimized for specific model size.
    
    Args:
        model_size: Model size ("300M", "15B", "30B")
        total_steps: Total training steps
        base_lr: Base learning rate (auto-scaled if None)
        
    Returns:
        ConvexityTrainingConfig optimized for the model size
    """
    
    # Auto-scale learning rate based on model size
    lr_scales = {
        "300M": 1.0,      # Base: 3e-4
        "15B": 0.5,       # Reduced for large model: 1.5e-4
        "30B": 0.3        # Further reduced for massive model: 9e-5
    }
    
    if base_lr is None:
        base_lr = 3e-4 * lr_scales.get(model_size, 1.0)
    
    # Scale other parameters based on model size
    warmup_steps = max(1000, total_steps // 50)
    convexity_warmup = max(500, total_steps // 100)
    
    # Adjust convexity parameters for larger models
    convexity_params = {
        "300M": {
            "tolerance": 1e-5,
            "cooldown_steps": 50,
            "up_multiplier": 1.01,
            "down_multiplier": 0.90
        },
        "15B": {
            "tolerance": 3e-5,      # More conservative for large model
            "cooldown_steps": 75,    # Longer cooldown
            "up_multiplier": 1.005,  # Slower increases
            "down_multiplier": 0.95  # Gentler decreases
        },
        "30B": {
            "tolerance": 5e-5,      # Most conservative
            "cooldown_steps": 100,   # Longest cooldown
            "up_multiplier": 1.002,  # Minimal increases
            "down_multiplier": 0.97  # Very gentle decreases
        }
    }
    
    params = convexity_params.get(model_size, convexity_params["300M"])
    
    convexity_config = ConvexityControllerConfig(
        warmup_enable_steps=convexity_warmup,
        tolerance=params["tolerance"],
        cooldown_steps=params["cooldown_steps"],
        up_multiplier=params["up_multiplier"],
        down_multiplier=params["down_multiplier"]
    )
    
    return ConvexityTrainingConfig(
        total_steps=total_steps,
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        convexity_config=convexity_config,
        log_every=50,
        checkpoint_every=max(1000, total_steps // 50)
    )


def create_config_from_dict(config_dict: Dict[str, Any]) -> ConvexityTrainingConfig:
    """Create configuration from dictionary with validation."""
    
    # Extract convexity config if present
    convexity_dict = config_dict.pop('convexity_config', {})
    if convexity_dict:
        convexity_config = ConvexityControllerConfig(**convexity_dict)
        config_dict['convexity_config'] = convexity_config
    
    return ConvexityTrainingConfig(**config_dict)


def validate_config_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration dictionary using Pydantic."""
    
    # Validate convexity config if present
    if 'convexity_config' in config_dict:
        convexity_validated = ConvexityConfigPydantic(**config_dict['convexity_config'])
        config_dict['convexity_config'] = convexity_validated.dict()
    
    # Validate main config
    validated = ConvexityTrainingConfigPydantic(**config_dict)
    return validated.dict()


# Configuration export utilities

def export_config_to_dict(config: ConvexityTrainingConfig) -> Dict[str, Any]:
    """Export configuration to dictionary format."""
    
    result = {
        'total_steps': config.total_steps,
        'base_lr': config.base_lr,
        'warmup_steps': config.warmup_steps,
        'weight_decay': config.weight_decay,
        'adam_b1': config.adam_b1,
        'adam_b2': config.adam_b2,
        'adam_eps': config.adam_eps,
        'grad_clip_norm': config.grad_clip_norm,
        'log_every': config.log_every,
        'checkpoint_every': config.checkpoint_every,
        'enable_wandb': config.enable_wandb,
        'min_lr_factor': config.min_lr_factor
    }
    
    if config.convexity_config:
        result['convexity_config'] = {
            'window_size': config.convexity_config.window_size,
            'ema_alpha': config.convexity_config.ema_alpha,
            'tolerance': config.convexity_config.tolerance,
            'down_multiplier': config.convexity_config.down_multiplier,
            'up_multiplier': config.convexity_config.up_multiplier,
            'min_gain': config.convexity_config.min_gain,
            'max_gain': config.convexity_config.max_gain,
            'cooldown_steps': config.convexity_config.cooldown_steps,
            'warmup_enable_steps': config.convexity_config.warmup_enable_steps,
            'enable_logging': config.convexity_config.enable_logging,
            'stability_threshold': config.convexity_config.stability_threshold
        }
    
    return result


# Module exports
__all__ = [
    'ConvexityControllerConfig',
    'ConvexityTrainingConfig',
    'ConvexityConfigPydantic',
    'ConvexityTrainingConfigPydantic',
    'get_default_convexity_config',
    'get_conservative_convexity_config',
    'get_aggressive_convexity_config',
    'get_tpu_v6e_convexity_config',
    'get_training_config_for_model_size',
    'create_config_from_dict',
    'validate_config_dict',
    'export_config_to_dict'
]
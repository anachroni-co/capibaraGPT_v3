"""
Configuration for adaptive computation and routing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class AdaptiveConfig:
    """Configuration for adaptive computation and routing."""
    
    # Model dimensions
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    
    # Routing configuration
    num_experts: int = 8
    routing_type: Literal['top_k', 'gating', 'mixture'] = 'top_k'
    
    # Training hyperparameters
    aux_loss_weight: float = 0.01
    load_balancing_weight: float = 0.01
    
    # Thresholds
    early_exit_threshold: float = 0.5
    compute_threshold: float = 0.1
    
    # Computation parameters
    max_depth: int = 24
    compute_budget: float = 1.0
    enable_moe_integration: bool = True
    adaptive_routing: bool = True
    
    # Hardware configuration
    device: str = "tpu"  # ["tpu", "gpu", "cpu"]
    precision: str = "bfloat16"  # ["float32", "bfloat16", "float16"]
    
    def __post_init__(self):
        """Validate config values."""
        assert self.hidden_size > 0, "hidden_size must be positive"
        assert self.intermediate_size > 0, "intermediate_size must be positive"
        assert self.num_attention_heads > 0, "num_attention_heads must be positive"
        assert self.num_experts > 0, "num_experts must be positive"
        assert self.aux_loss_weight >= 0, "aux_loss_weight must be non-negative"
        assert self.load_balancing_weight >= 0, "load_balancing_weight must be non-negative"
        assert 0 <= self.early_exit_threshold <= 1, "early_exit_threshold must be between 0 and 1"
        assert 0 <= self.compute_threshold <= 1, "compute_threshold must be between 0 and 1"
        assert self.max_depth > 0, "max_depth must be positive"
        assert self.compute_budget > 0, "compute_budget must be positive"
        assert self.routing_type in ['top_k', 'gating', 'mixture'], "Invalid routing_type"
        assert self.device in ['tpu', 'gpu', 'cpu'], "Invalid device"
        assert self.precision in ['float32', 'bfloat16', 'float16'], "Invalid precision"
    
    @classmethod
    def from_json(cls, json_path: str) -> "AdaptiveConfig":
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict) 
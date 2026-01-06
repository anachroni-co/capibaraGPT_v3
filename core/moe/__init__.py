"""
Dynamic Mixture of Experts (MoE) System for Capibara-6

This module provides a complete MoE implementation optimized for TPU v6e-64
with advanced routing, load balancing, and expert specialization.
"""

from .dynamic_moe import (
    MoEConfig,
    ExpertLayer,
    DynamicRouter,
    DynamicMoELayer,
    MoEManager,
    create_moe_config,
    create_dynamic_moe_layer,
    create_moe_manager
)

__all__ = [
    "MoEConfig",
    "ExpertLayer", 
    "DynamicRouter",
    "DynamicMoELayer",
    "MoEManager",
    "create_moe_config",
    "create_dynamic_moe_layer",
    "create_moe_manager"
]
"""
Factory functions for the Chain-of-Thought module.

This file provides convenience functions to create and configure
instances of the CoT module.
"""

from typing import Any, Callable, Dict, Optional

from .module import EnhancedChainOfThoughtModule
from config import (
    CoreConfig,
    RoutingConfig,
    ProcessingConfig,
    MonitoringConfig,
    AdvancedCoTConfig,
    OptimizationConfig,
)

def create_enhanced_cot_config(
    core_model_generate_fn: Callable,
    hidden_size: int = 768,
    max_steps: int = 8,
    enable_dynamic_routing: bool = True,
    enable_hierarchical_reasoning: bool = True,
    enable_cross_core_communication: bool = True,
    enable_submodel_fusion: bool = True,
    device_type: str = "tpu",
    debug_mode: bool = False,
    **kwargs
) -> AdvancedCoTConfig:
    """
    Factory function to create an enhanced CoT module configuration.

    Args:
        core_model_generate_fn: Generation function of the base model
        hidden_size: Size of the embeddings
        max_steps: Maximum number of reasoning steps
        enable_dynamic_routing: Enable dynamic routing
        enable_hierarchical_reasoning: Enable hierarchical reasoning
        enable_cross_core_communication: Enable cross-core communication
        enable_submodel_fusion: Enable sub-model fusion
        device_type: Device type ("tpu", "gpu", "cpu")
        debug_mode: Enable debug mode
        **kwargs: Additional arguments for the configuration

    Returns:
        AdvancedCoTConfig: Optimized configuration for the CoT module
    """

    # create base configuration
    config = AdvancedCoTConfig(
        core_model_generate_fn=core_model_generate_fn,
        hidden_size=hidden_size,
        max_steps=max_steps,
        enable_dynamic_routing=enable_dynamic_routing,
        enable_hierarchical_reasoning=enable_hierarchical_reasoning,
        enable_cross_core_communication=enable_cross_core_communication,
        enable_submodel_fusion=enable_submodel_fusion,
        **kwargs
    )

    # optimize for the specific device
    config = config.optimize_for_device(device_type)

    # configure debug mode if enabled
    if debug_mode:
        config = config.enable_debug_mode()
    else:
        config = config.enable_production_mode()

    return config

def create_enhanced_cot_module(
    config: AdvancedCoTConfig,
    cache_size: int = 128
) -> EnhancedChainOfThoughtModule:
    """
    Factory function to create an enhanced CoT module.

    Args:
        config: Module configuration
        cache_size: Size of the reasoning cache

    Returns:
        EnhancedChainOfThoughtModule: Configured CoT module
    """
    return EnhancedChainOfThoughtModule(
        config=config,
        cache_size=cache_size
    )

def enhanced_chain_of_thought(
    query: str,
    core_model_generate_fn: Callable,
    initial_context: Optional[str] = None,
    device_type: str = "tpu",
    debug_mode: bool = False,
    **config_kwargs
) -> Dict[str, Any]:
    """
    Convenience function to execute enhanced reasoning.

    Args:
        query: Query to process
        core_model_generate_fn: Generation function of the base model
        initial_context: Optional initial context
        device_type: Device type ("tpu", "gpu", "cpu")
        debug_mode: Enable debug mode
        **config_kwargs: Additional arguments for the configuration

    Returns:
        Dict[str, Any]: Reasoning result
    """

    # create optimized configuration
    config = create_enhanced_cot_config(
        core_model_generate_fn=core_model_generate_fn,
        device_type=device_type,
        debug_mode=debug_mode,
        **config_kwargs
    )

    # create and execute module
    module = create_enhanced_cot_module(config)
    return module(query, initial_context)

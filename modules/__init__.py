"""
Modules Package - CapibaraGPT v3

Provides modular neural network components including:
- Attention mechanisms (shared, multi-scale)
- Hierarchical reasoning
- Adaptive routing
- Specialized processors
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Core module imports with fallbacks
try:
    from .shared_attention import (
        OptimizedSharedAttention,
        MultiScaleSharedAttention,
        create_shared_attention,
    )
    ATTENTION_AVAILABLE = True
except Exception as e:
    ATTENTION_AVAILABLE = False
    OptimizedSharedAttention = None
    MultiScaleSharedAttention = None
    create_shared_attention = None
    logger.warning("Attention modules unavailable: %s", e)

try:
    from .hierarchical_reasoning import HierarchicalReasoning
    REASONING_AVAILABLE = True
except Exception as e:
    REASONING_AVAILABLE = False
    HierarchicalReasoning = None
    logger.warning("Hierarchical reasoning unavailable: %s", e)

try:
    from .capibara_adaptive_router import AdaptiveRouter
    from .capibara_adaptive_router import (
        OptimizedAdaptiveRouter,
        ContextualRouterOptimized,
        VQbitLayerOptimized,
        ExpertLayer,
        create_router_for_tpu_v4_32,
        distributed_router_forward,
    )
    ROUTER_AVAILABLE = True
except Exception as e:
    ROUTER_AVAILABLE = False
    AdaptiveRouter = None
    OptimizedAdaptiveRouter = None
    ContextualRouterOptimized = None
    VQbitLayerOptimized = None
    ExpertLayer = None
    create_router_for_tpu_v4_32 = None
    distributed_router_forward = None
    logger.warning("Adaptive router unavailable: %s", e)

try:
    from .specialized_processors import (
        SpecializedProcessorManager,
        ProcessorType,
        ProcessorConfig,
        TextAnalysisProcessor,
        SentimentAnalysisProcessor,
        EntityExtractionProcessor,
        CodeAnalysisProcessor,
        MultimodalFusionProcessor,
        create_processor_manager,
        create_default_processors,
        get_global_processor_manager,
    )
    PROCESSORS_AVAILABLE = True
    SpecializedProcessor = SpecializedProcessorManager
except Exception as e:
    PROCESSORS_AVAILABLE = False
    SpecializedProcessor = None
    SpecializedProcessorManager = None
    ProcessorType = None
    ProcessorConfig = None
    TextAnalysisProcessor = None
    SentimentAnalysisProcessor = None
    EntityExtractionProcessor = None
    CodeAnalysisProcessor = None
    MultimodalFusionProcessor = None
    create_processor_manager = None
    create_default_processors = None
    get_global_processor_manager = None
    logger.warning("Specialized processors unavailable: %s", e)

try:
    from .ultra_module_orchestrator import (
        UltraModuleOrchestrator,
        create_ultra_module_system,
        create_ultra_module_config,
        ModuleType,
        OrchestrationStrategy,
        UltraModuleConfig,
    )
    ORCHESTRATOR_AVAILABLE = True
except Exception as e:
    ORCHESTRATOR_AVAILABLE = False
    UltraModuleOrchestrator = None
    create_ultra_module_system = None
    create_ultra_module_config = None
    ModuleType = None
    OrchestrationStrategy = None
    UltraModuleConfig = None
    logger.warning("Ultra module orchestrator unavailable: %s", e)


def get_module_status() -> Dict[str, bool]:
    """Get availability status of all modules."""
    return {
        "attention": ATTENTION_AVAILABLE,
        "reasoning": REASONING_AVAILABLE,
        "router": ROUTER_AVAILABLE,
        "processors": PROCESSORS_AVAILABLE,
        "orchestrator": ORCHESTRATOR_AVAILABLE,
    }


__all__ = [
    # Attention
    "OptimizedSharedAttention",
    "MultiScaleSharedAttention",
    "create_shared_attention",
    # Reasoning
    "HierarchicalReasoning",
    # Router
    "AdaptiveRouter",
    "OptimizedAdaptiveRouter",
    "ContextualRouterOptimized",
    "VQbitLayerOptimized",
    "ExpertLayer",
    "create_router_for_tpu_v4_32",
    "distributed_router_forward",
    # Processors
    "SpecializedProcessor",
    "SpecializedProcessorManager",
    "ProcessorType",
    "ProcessorConfig",
    "TextAnalysisProcessor",
    "SentimentAnalysisProcessor",
    "EntityExtractionProcessor",
    "CodeAnalysisProcessor",
    "MultimodalFusionProcessor",
    "create_processor_manager",
    "create_default_processors",
    "get_global_processor_manager",
    # Orchestrator
    "UltraModuleOrchestrator",
    "create_ultra_module_system",
    "create_ultra_module_config",
    "ModuleType",
    "OrchestrationStrategy",
    "UltraModuleConfig",
    # Status
    "get_module_status",
]

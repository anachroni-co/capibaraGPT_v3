"""
Core Module - CapibaraGPT v3

Central integration point for model construction, training, and inference:
- config: Model and training configuration
- modular_model: Dynamic model composition
- router/routing: Intelligent module routing
- encoders: Vision, video, multimodal encoders
- cot: Chain of Thought reasoning
- ultra_core_integration: SSM-hybrid layers and orchestration
"""

import logging

from core.import_utils import safe_import

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe imports — each returns None when the submodule is unavailable
# ---------------------------------------------------------------------------

# Configuration
Config, ModelConfig = safe_import("core.config", "Config", "ModelConfig")
CONFIG_AVAILABLE = Config is not None

# Modular model
ModularCapibaraModel, ModularConfig = safe_import(
    "core.modular_model", "ModularCapibaraModel", "ModularConfig"
)
MODULAR_AVAILABLE = ModularCapibaraModel is not None

# Enhanced router
(EnhancedRouter, AdvancedRouter, create_enhanced_router,
 create_core_integrated_router, RouterCoreConfig, RouterType) = safe_import(
    "core.router",
    "EnhancedRouter", "Router", "create_enhanced_router",
    "create_core_integrated_router", "RouterConfig", "RouterType",
)
ENHANCED_ROUTER_AVAILABLE = EnhancedRouter is not None

# Simple router
SimpleRouter, create_router = safe_import("core.routing", "Router", "create_router")
SIMPLE_ROUTER_AVAILABLE = SimpleRouter is not None

# Legacy routers
BaseRouter, BaseRouterV2 = safe_import("core.routers.base", "BaseRouter", "BaseRouterV2")
AdaptiveRouter = safe_import("core.routers.adaptive_router", "AdaptiveRouter")
LEGACY_ROUTERS_AVAILABLE = BaseRouter is not None

# TTS Router
TTSRouter = safe_import("core.routers.tts_router", "TTSRouter")
TTS_ROUTER_AVAILABLE = TTSRouter is not None

# Encoders
(VisionEncoder, VisionEncoderConfig, VideoEncoder,
 VideoEncoderConfig, MultimodalCombiner, CombinerConfig) = safe_import(
    "core.encoders",
    "VisionEncoder", "VisionEncoderConfig",
    "VideoEncoder", "VideoEncoderConfig",
    "MultimodalCombiner", "CombinerConfig",
)
ENCODERS_AVAILABLE = VisionEncoder is not None

# Chain of Thought
ChainOfThought, create_cot_handler = safe_import(
    "core.cot", "ChainOfThought", "create_cot_handler"
)
COT_AVAILABLE = ChainOfThought is not None

# Model core
ModelCore, create_model = safe_import("core._model", "ModelCore", "create_model")
MODEL_AVAILABLE = ModelCore is not None

# Ultra Core Integration
(UltraIntelligentCoreRouter, SSMHybridLayer, UltraCoreOrchestrator,
 create_ultra_core_system, create_ultra_config, integrate_with_training,
 validate_ultra_core_system, ULTRA_TRAINING_AVAILABLE,
 SSM_AVAILABLE) = safe_import(
    "core.ultra_core_integration",
    "UltraIntelligentCoreRouter", "SSMHybridLayer", "UltraCoreOrchestrator",
    "create_ultra_core_system", "create_ultra_config", "integrate_with_training",
    "validate_ultra_core_system", "ULTRA_TRAINING_AVAILABLE", "SSM_AVAILABLE",
)
ULTRA_CORE_AVAILABLE = UltraIntelligentCoreRouter is not None
if ULTRA_TRAINING_AVAILABLE is None:
    ULTRA_TRAINING_AVAILABLE = False
if SSM_AVAILABLE is None:
    SSM_AVAILABLE = False


__all__ = [
    # Config
    "Config",
    "ModelConfig",
    "ModularConfig",
    # Enhanced router
    "EnhancedRouter",
    "AdvancedRouter",
    "create_enhanced_router",
    "create_core_integrated_router",
    "RouterCoreConfig",
    "RouterType",
    # Simple router
    "SimpleRouter",
    "create_router",
    # Optimization
    # Legacy routers
    "BaseRouter",
    "BaseRouterV2",
    "AdaptiveRouter",
    "TTSRouter",
    # Optimizers
    # Encoders
    "VisionEncoder",
    "VisionEncoderConfig",
    "VideoEncoder",
    "VideoEncoderConfig",
    "MultimodalCombiner",
    "CombinerConfig",
    # Chain of Thought
    "ChainOfThought",
    "create_cot_handler",
    # Model
    "ModelCore",
    "create_model",
    "ModularCapibaraModel",
    # Ultra Core
    "UltraIntelligentCoreRouter",
    "SSMHybridLayer",
    "UltraCoreOrchestrator",
    "create_ultra_core_system",
    "create_ultra_config",
    "integrate_with_training",
    "validate_ultra_core_system",
    # Availability flags
    "CONFIG_AVAILABLE",
    "MODULAR_AVAILABLE",
    "ENHANCED_ROUTER_AVAILABLE",
    "SIMPLE_ROUTER_AVAILABLE",
    "LEGACY_ROUTERS_AVAILABLE",
    "TTS_ROUTER_AVAILABLE",
    "ENCODERS_AVAILABLE",
    "COT_AVAILABLE",
    "MODEL_AVAILABLE",
    "ULTRA_CORE_AVAILABLE",
    "ULTRA_TRAINING_AVAILABLE",
    "SSM_AVAILABLE",
]

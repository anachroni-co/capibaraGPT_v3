"""Core Module for CapibaraGPT v3 - Ultra Advanced Edition.

This module serves as the central integration point for all core components of the
CapibaraGPT architecture, providing foundational building blocks for model construction,
training, inference, and advanced features.

Architecture Overview:
    The core module integrates multiple sophisticated subsystems:

    - **Configuration Management**: Model, training, and routing configurations
    - **Modular Architecture**: Dynamic model composition with pluggable components
    - **Intelligent Routing**: Multi-strategy routing for optimal module selection
    - **Optimization**: Training state management and metrics tracking
    - **Multimodal Encoders**: Vision, video, and multimodal combination
    - **Chain of Thought**: Multi-step reasoning capabilities
    - **Ultra Core Integration**: Advanced SSM-hybrid layers and intelligent routing

Key Components:
    Configuration:
        - Config: General-purpose configuration manager
        - ModelConfig: Model architecture configuration
        - ModularConfig: Modular model configuration
        - RouterCoreConfig: Router-specific configuration

    Routing Systems:
        - SimpleRouter: Basic routing functionality
        - EnhancedRouter: Advanced routing with caching and verification
        - AdvancedRouter: Alias for Router with extended features
        - BaseRouter/BaseRouterV2: Legacy router implementations
        - AdaptiveRouter: Dynamic routing based on input characteristics
        - TTSRouter: Text-to-speech specific routing
        - UltraIntelligentCoreRouter: Next-generation intelligent routing

    Model Architecture:
        - ModelCore: Base model implementation
        - ModularCapibaraModel: Modular architecture for flexible composition

    Encoders:
        - VisionEncoder: Image encoding for multimodal tasks
        - VideoEncoder: Video sequence encoding
        - MultimodalCombiner: Fusion of multiple modalities

    Advanced Features:
        - ChainOfThought: Multi-step reasoning engine
        - SSMHybridLayer: State-space model hybrid layers
        - UltraCoreOrchestrator: Coordinated system orchestration

Example:
    Basic model setup with routing:

    >>> from capibara.core import Config, ModelConfig, create_model
    >>> from capibara.core import EnhancedRouter, RouterCoreConfig
    >>>
    >>> # Configure model
    >>> config = Config(model_name="capibara-v3")
    >>> model_config = ModelConfig(
    ...     hidden_size=768,
    ...     num_layers=12,
    ...     num_heads=12
    ... )
    >>>
    >>> # Create model
    >>> model = create_model(model_config)
    >>>
    >>> # Setup routing
    >>> router_config = RouterCoreConfig(
    ...     hidden_size=768,
    ...     num_heads=8
    ... )
    >>> router = EnhancedRouter(router_config)

    Advanced modular setup:

    >>> from capibara.core import ModularCapibaraModel, ModularConfig
    >>> from capibara.core import VisionEncoder, VisionEncoderConfig
    >>>
    >>> # Configure vision encoder
    >>> vision_config = VisionEncoderConfig(
    ...     image_size=224,
    ...     patch_size=16,
    ...     hidden_size=768
    ... )
    >>> vision_encoder = VisionEncoder(vision_config)
    >>>
    >>> # Create modular model
    >>> modular_config = ModularConfig(
    ...     base_config=model_config,
    ...     enable_vision=True
    ... )
    >>> modular_model = ModularCapibaraModel(modular_config)

    Ultra core integration (when available):

    >>> from capibara.core import ULTRA_CORE_AVAILABLE
    >>>
    >>> if ULTRA_CORE_AVAILABLE:
    ...     from capibara.core import create_ultra_core_system, create_ultra_config
    ...
    ...     ultra_config = create_ultra_config()
    ...     ultra_system = create_ultra_core_system(ultra_config)
    ...     print("Ultra Core system initialized")

Feature Flags:
    - ULTRA_CORE_AVAILABLE: Whether Ultra Core Integration is available
    - ULTRA_TRAINING_AVAILABLE: Whether Ultra Training features are available
    - SSM_AVAILABLE: Whether State-Space Model layers are available

Note:
    Some advanced features (Ultra Core, SSM) may not be available depending on
    installed dependencies. The module provides graceful fallbacks and feature
    flags to check availability.

See Also:
    - capibara.training: Training infrastructure and strategies
    - capibara.inference: Inference and deployment utilities
    - capibara.data: Data loading and preprocessing
    - capibara.sub_models: Specialized sub-model implementations
"""

from typing import Any

# Base core components
from .config import Config, ModelConfig  # noqa: F401
from .modular_model import ModularCapibaraModel, ModularConfig  # noqa: F401
from .router import (
    EnhancedRouter,
    Router as AdvancedRouter,
    create_enhanced_router,
    create_core_integrated_router,
    RouterConfig as RouterCoreConfig,
    RouterType,
)  # noqa: F401
from .routing import Router as SimpleRouter, create_router  # noqa: F401
from .optimization import TrainingMetrics, TrainingState  # noqa: F401

# Legacy routers (for backwards compatibility)
from .routers.base import BaseRouter, BaseRouterV2  # noqa: F401
from .routers.adaptive_router import AdaptiveRouter  # noqa: F401
# Keep TTSRouter import consistent with project structure if exists
try:
    from .routers.tts_router import TTSRouter  # type: ignore # noqa: F401
except Exception:
    TTSRouter = None  # type: ignore

# Optimizers
from .optimizers.optimizer import OptimizerConfig, create_optimizer  # noqa: F401

# Encoders
from .encoders import (  # noqa: F401
    VisionEncoder,
    VisionEncoderConfig,
    VideoEncoder,
    VideoEncoderConfig,
    MultimodalCombiner,
    CombinerConfig,
)

# Chain of Thought
from .cot import ChainOfThought, create_cot_handler  # noqa: F401

# Modelo base
from ._model import ModelCore, create_model  # noqa: F401

# 🌟 ULTRA CORE INTEGRATION - New ultra-advanced components
try:
    from .ultra_core_integration import (  # noqa: F401
        UltraIntelligentCoreRouter,
        SSMHybridLayer,
        UltraCoreOrchestrator,
        create_ultra_core_system,
        create_ultra_config,
        integrate_with_training,
        validate_ultra_core_system,
        ULTRA_TRAINING_AVAILABLE,
        SSM_AVAILABLE,
    )
    ULTRA_CORE_AVAILABLE = True
except ImportError:
    ULTRA_CORE_AVAILABLE = False
    # Stub implementations for backwards compatibility
    UltraIntelligentCoreRouter = None  # type: ignore
    SSMHybridLayer = None  # type: ignore
    UltraCoreOrchestrator = None  # type: ignore

    def create_ultra_core_system(*args, **kwargs) -> Any:  # type: ignore
        raise ImportError("Ultra Core Integration not available")

    def create_ultra_config(*args, **kwargs) -> Any:  # type: ignore
        raise ImportError("Ultra Core Integration not available")

    def integrate_with_training(*args, **kwargs) -> Any:  # type: ignore
        raise ImportError("Ultra Core Integration not available")

    def validate_ultra_core_system(*args, **kwargs) -> Any:  # type: ignore
        raise ImportError("Ultra Core Integration not available")

    ULTRA_TRAINING_AVAILABLE = False  # type: ignore
    SSM_AVAILABLE = False  # type: ignore

__all__ = [
    # config
    "Config",
    "ModelConfig",
    "ModularConfig",
    # routers
    "SimpleRouter",
    "create_router",
    "EnhancedRouter",
    "AdvancedRouter",
    "create_enhanced_router",
    "create_core_integrated_router",
    "RouterCoreConfig",
    "RouterType",
    # training/optimization
    "TrainingMetrics",
    "TrainingState",
    # encoders
    "VisionEncoder",
    "VisionEncoderConfig",
    "VideoEncoder",
    "VideoEncoderConfig",
    "MultimodalCombiner",
    "CombinerConfig",
    # CoT
    "ChainOfThought",
    "create_cot_handler",
    # models
    "ModelCore",
    "create_model",
    "ModularCapibaraModel",
    "ModularConfig",
    # legacy routers
    "BaseRouter",
    "BaseRouterV2",
    "AdaptiveRouter",
    "TTSRouter",
    # Ultra core (if available)
    "UltraIntelligentCoreRouter",
    "SSMHybridLayer",
    "UltraCoreOrchestrator",
    "create_ultra_core_system",
    "create_ultra_config",
    "integrate_with_training",
    "validate_ultra_core_system",
    "ULTRA_CORE_AVAILABLE",
    "ULTRA_TRAINING_AVAILABLE",
    "SSM_AVAILABLE",
]

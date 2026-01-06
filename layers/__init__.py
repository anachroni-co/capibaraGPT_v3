"""
layers module.

# This module provides functionality for layers operations.
"""

import os
import sys
from pathlib import Path
from typing import Any

def get_project_root():
    """Get the root path of the project."""
    return Path(__file__).parent.parent

# Version information
__version__ = "1.0.0"
__author__ = "CapibaraGPT Team"

# 🌟 SSM HYBRID LAYERS - Ultra-advanced or(n) architectures
try:
    from .ssm_hybrid_layers import (
        UltraSSMLayer, MambaLayer, S4Layer, HybridSSMLayer,
        SSMHybridLayerConfig, create_ssm_layer, create_ssm_config,
        validate_ssm_system, SSM_COMPONENTS_AVAILABLE
    )
    SSM_LAYERS_AVAILABLE = True
except ImportError:
    SSM_LAYERS_AVAILABLE = False
    # Stub implementations for backwards compatibility
    UltraSSMLayer = None
    MambaLayer = None
    S4Layer = None
    HybridSSMLayer = None
    SSMHybridLayerConfig = None
    
    def create_ssm_layer(*args, **kwargs) -> Any:
        raise ImportError("SSM Hybrid Layers not available")
    
    def create_ssm_config(*args, **kwargs) -> Any:
        raise ImportError("SSM Hybrid Layers not available")
    
    def validate_ssm_system(*args, **kwargs) -> Any:
        raise ImportError("SSM Hybrid Layers not available")
    
    SSM_COMPONENTS_AVAILABLE = False

# Passive learning layers
try:
    from .pasive.synthetic_embedding import SyntheticEmbedding
    from .pasive.attention import DistributedAttention
    PASSIVE_LAYERS_AVAILABLE = True
except ImportError:
    SyntheticEmbedding = None
    DistributedAttention = None
    PASSIVE_LAYERS_AVAILABLE = False

# Abstract reasoning layers
try:
    from .abstract_reasoning.platonic import Platonic
    from .abstract_reasoning.quineana import Quineana
    from .abstract_reasoning.game_theory import GameTheory
    ABSTRACT_REASONING_AVAILABLE = True
except ImportError:
    Platonic = None
    Quineana = None
    GameTheory = None
    ABSTRACT_REASONING_AVAILABLE = False

# Sparsity and quantization layers
try:
    from .sparsity.bitnet import Conv1DBlock, BitNet158
    from .sparsity.sparse_capibara import SparseCapibara
    from .sparsity.affine_quantizer import AffineQuantizer
    from .sparsity.mixture_of_rookies import MixtureOfRookies
    SPARSITY_LAYERS_AVAILABLE = True
except ImportError:
    Conv1DBlock = None
    BitNet158 = None
    SparseCapibara = None
    AffineQuantizer = None
    MixtureOfRookies = None
    SPARSITY_LAYERS_AVAILABLE = False

# Additional specialized layers
try:
    from .conv1d_block import Conv1DBlock as Conv1DBlockLayer
    from .meta_la import MetaLA, MetaLAConfig
    from .smb_layer import SMBLayer
    ADDITIONAL_LAYERS_AVAILABLE = True
except ImportError:
    Conv1DBlockLayer = None
    MetaLA = None
    MetaLAConfig = None
    SMBLayer = None
    ADDITIONAL_LAYERS_AVAILABLE = False

# 🚀 ULTRA LAYER INTEGRATION - Complete ecosystem orchestration
try:
    from .ultra_layer_integration import (
        UltraLayerOrchestrator, UltraLayerIntegrationConfig,
        LayerPerformanceMetrics, create_ultra_layer_system,
        create_ultra_layer_config, demonstrate_ultra_layer_integration,
        ULTRA_CORE_AVAILABLE, ULTRA_TRAINING_INTEGRATION
    )
    ULTRA_LAYER_INTEGRATION_AVAILABLE = True
except ImportError:
    ULTRA_LAYER_INTEGRATION_AVAILABLE = False
    # Stub implementations for backwards compatibility
    UltraLayerOrchestrator = None
    UltraLayerIntegrationConfig = None
    LayerPerformanceMetrics = None
    
    def create_ultra_layer_system(*args, **kwargs) -> Any:
        raise ImportError("Ultra Layer Integration not available")
    
    def create_ultra_layer_config(*args, **kwargs) -> Any:
        raise ImportError("Ultra Layer Integration not available")
    
    def demonstrate_ultra_layer_integration(*args, **kwargs) -> Any:
        raise ImportError("Ultra Layer Integration not available")
    
    ULTRA_CORE_AVAILABLE = False
    ULTRA_TRAINING_INTEGRATION = False


# Build __all__ dynamically based on what's available
__all__ = [
    # Status flags (always available)
    "SSM_LAYERS_AVAILABLE",
    "SSM_COMPONENTS_AVAILABLE", 
    "PASSIVE_LAYERS_AVAILABLE",
    "ABSTRACT_REASONING_AVAILABLE",
    "SPARSITY_LAYERS_AVAILABLE",
    "ADDITIONAL_LAYERS_AVAILABLE",
    "ULTRA_LAYER_INTEGRATION_AVAILABLE",
    "ULTRA_CORE_AVAILABLE",
    "ULTRA_TRAINING_INTEGRATION"
]

# Add SSM layers if available
if SSM_LAYERS_AVAILABLE:
    __all__.extend([
        "UltraSSMLayer",
        "MambaLayer", 
        "S4Layer",
        "HybridSSMLayer",
        "SSMHybridLayerConfig",
        "create_ssm_layer",
        "create_ssm_config",
        "validate_ssm_system"
    ])

# Add passive learning layers if available
if PASSIVE_LAYERS_AVAILABLE:
    __all__.extend([
        "SyntheticEmbedding",
        "DistributedAttention"
    ])

# Add abstract reasoning layers if available  
if ABSTRACT_REASONING_AVAILABLE:
    __all__.extend([
        "Platonic",
        "Quineana",
        "GameTheory"
    ])

# Add sparsity layers if available
if SPARSITY_LAYERS_AVAILABLE:
    __all__.extend([
        "BitNet158",
        "Conv1DBlock", 
        "SparseCapibara",
        "AffineQuantizer",
        "MixtureOfRookies"
    ])

# Add additional specialized layers if available
if ADDITIONAL_LAYERS_AVAILABLE:
    __all__.extend([
        "Conv1DBlockLayer",
        "MetaLA",
        "MetaLAConfig", 
        "SMBLayer"
    ])

# Add ultra layer integration if available
if ULTRA_LAYER_INTEGRATION_AVAILABLE:
    __all__.extend([
        "UltraLayerOrchestrator",
        "UltraLayerIntegrationConfig",
        "LayerPerformanceMetrics",
        "create_ultra_layer_system",
        "create_ultra_layer_config",
        "demonstrate_ultra_layer_integration"
    ])

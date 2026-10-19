"""
Layers Module - CapibaraGPT v3

Neural network layer implementations:
- pasive: Synthetic embedding and distributed attention
- abstract_reasoning: Platonic, Quineana, GameTheory
- sparsity: BitNet, sparse layers, quantization

Note: Most layers require JAX/Flax or PyTorch. Check availability flags
before using specific layers.
"""

import logging

logger = logging.getLogger(__name__)

# Passive learning layers (requires JAX/Flax)
try:
    from .pasive.synthetic_embedding import SyntheticEmbedding
    from .pasive.attention import DistributedAttention
    PASSIVE_LAYERS_AVAILABLE = True
except Exception as e:
    logger.debug(f"Passive layers not available: {e}")
    PASSIVE_LAYERS_AVAILABLE = False
    SyntheticEmbedding = None
    DistributedAttention = None

# Abstract reasoning layers (requires JAX/Flax)
try:
    from .abstract_reasoning.platonic import Platonic
    from .abstract_reasoning.quineana import Quineana
    from .abstract_reasoning.game_theory import GameTheory, BasicGameTheory
    ABSTRACT_REASONING_AVAILABLE = True
except Exception as e:
    logger.debug(f"Abstract reasoning layers not available: {e}")
    ABSTRACT_REASONING_AVAILABLE = False
    Platonic = None
    Quineana = None
    GameTheory = None
    BasicGameTheory = None

# Sparsity and quantization layers (requires JAX/Flax)
try:
    from .sparsity.bitnet import Conv1DBlock, BitNet158
    from .sparsity.sparse_capibara import SparseCapibara
    from .sparsity.affine_quantizer import AffineQuantizer
    from .sparsity.mixture_of_rookies import MixtureOfRookies
    SPARSITY_LAYERS_AVAILABLE = True
except Exception as e:
    logger.debug(f"Sparsity layers not available: {e}")
    SPARSITY_LAYERS_AVAILABLE = False
    Conv1DBlock = None
    BitNet158 = None
    SparseCapibara = None
    AffineQuantizer = None
    MixtureOfRookies = None

SSM_LAYERS_AVAILABLE = False
SSM_COMPONENTS_AVAILABLE = False
ADDITIONAL_LAYERS_AVAILABLE = False
ULTRA_LAYER_INTEGRATION_AVAILABLE = False
ULTRA_CORE_AVAILABLE = False
ULTRA_TRAINING_INTEGRATION = False

# Base layers (moved from root)
try:
    from .base import BaseLayer, LayerConfig
    BASE_LAYERS_AVAILABLE = True
except Exception as e:
    logger.debug(f"Base layers not available: {e}")
    BASE_LAYERS_AVAILABLE = False
    BaseLayer = None
    LayerConfig = None

# Core layers (root)
try:
    from .self_attention import SelfAttention, SelfAttentionConfig, TpuAttentionCache
    from .embedding import EmbeddingLayer, EmbeddingConfig
    from .conv1d_block import Conv1DBlock, Conv1DBlockConfig
    from .stack import LayerStack, LayerStackConfig
    from .smb_layer import SMBLayer, SMBLayerConfig
    CORE_LAYERS_AVAILABLE = True
except Exception as e:
    logger.debug(f"Core layers not available: {e}")
    CORE_LAYERS_AVAILABLE = False
    SelfAttention = None
    SelfAttentionConfig = None
    TpuAttentionCache = None
    EmbeddingLayer = None
    EmbeddingConfig = None
    Conv1DBlock = None
    Conv1DBlockConfig = None
    LayerStack = None
    LayerStackConfig = None
    SMBLayer = None
    SMBLayerConfig = None

# SSM/Hybrid layers (moved from root)
try:
    from .ssm_hybrid_layers import SSMHybridLayer
    from .ultra_layer_integration import UltraLayerIntegration
    SSM_HYBRID_AVAILABLE = True
except Exception as e:
    logger.debug(f"SSM/Hybrid layers not available: {e}")
    SSM_HYBRID_AVAILABLE = False
    SSMHybridLayer = None
    UltraLayerIntegration = None


__all__ = [
    # Passive layers
    "SyntheticEmbedding",
    "DistributedAttention",
    # Abstract reasoning
    "Platonic",
    "Quineana",
    "GameTheory",
    "BasicGameTheory",
    # Sparsity
    "Conv1DBlock",
    "BitNet158",
    "SparseCapibara",
    "AffineQuantizer",
    "MixtureOfRookies",
    # Base layers
    "BaseLayer",
    "LayerConfig",
    "SelfAttention",
    "SelfAttentionConfig",
    "TpuAttentionCache",
    "EmbeddingLayer",
    "EmbeddingConfig",
    "Conv1DBlock",
    "Conv1DBlockConfig",
    "LayerStack",
    "LayerStackConfig",
    "SMBLayer",
    "SMBLayerConfig",
    # Availability flags
    "SSM_LAYERS_AVAILABLE",
    "SSM_COMPONENTS_AVAILABLE",
    "PASSIVE_LAYERS_AVAILABLE",
    "ABSTRACT_REASONING_AVAILABLE",
    "SPARSITY_LAYERS_AVAILABLE",
    "ADDITIONAL_LAYERS_AVAILABLE",
    "BASE_LAYERS_AVAILABLE",
    "CORE_LAYERS_AVAILABLE",
    "SSM_HYBRID_AVAILABLE",
]

"""
Training Module - CapibaraGPT v3

Meta-consensus training system with HuggingFace integration:
- unified_consensus: Base consensus coordination
- enhanced_hf_consensus_strategy: HuggingFace-optimized consensus
- hybrid_expert_router: Multi-tier expert routing
- meta_consensus_system: Higher-order consensus coordination
"""

import logging

logger = logging.getLogger(__name__)

# Unified consensus
try:
    from .unified_consensus import UnifiedConsensusStrategy, ConsensusConfig
    UNIFIED_AVAILABLE = True
except ImportError:
    UNIFIED_AVAILABLE = False
    UnifiedConsensusStrategy = None
    ConsensusConfig = None

# Enhanced HuggingFace consensus
try:
    from .enhanced_hf_consensus_strategy import EnhancedHFConsensusStrategy, ServerlessExpertConfig
    HF_CONSENSUS_AVAILABLE = True
except ImportError:
    HF_CONSENSUS_AVAILABLE = False
    EnhancedHFConsensusStrategy = None
    ServerlessExpertConfig = None

# Meta-consensus system
try:
    from .meta_consensus_system import MetaConsensusSystem, MetaConsensusConfig, create_meta_consensus_system
    META_CONSENSUS_AVAILABLE = True
except ImportError:
    META_CONSENSUS_AVAILABLE = False
    MetaConsensusSystem = None
    MetaConsensusConfig = None
    create_meta_consensus_system = None


__all__ = [
    # Unified consensus
    "UnifiedConsensusStrategy",
    "ConsensusConfig",
    # HuggingFace consensus
    "EnhancedHFConsensusStrategy",
    "ServerlessExpertConfig",
    # Hybrid router
    # BTX training
    # Meta-consensus
    "MetaConsensusSystem",
    "MetaConsensusConfig",
    "create_meta_consensus_system",
    # Availability flags
    "UNIFIED_AVAILABLE",
    "HF_CONSENSUS_AVAILABLE",
    "META_CONSENSUS_AVAILABLE",
]

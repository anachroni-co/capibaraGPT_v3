"""Meta-Consensus Training System for CapibaraGPT.

This package provides a comprehensive meta-consensus training system with HuggingFace
integration, hybrid expert routing, and advanced training strategies for large language
models. It combines multiple consensus mechanisms to achieve optimal model performance
through distributed training and expert ensemble techniques.

The training system features:
- Unified consensus strategies for multi-expert coordination
- Enhanced HuggingFace integration with serverless scaling
- Hybrid expert routing with cost-quality optimization
- Branch-Train-MiX (BTX) training methodology
- Meta-consensus coordination across training strategies
- Hierarchical expert organization and selection

Key Components:
    Consensus Strategies:
        - UnifiedConsensusStrategy: Base consensus coordination
        - EnhancedHFConsensusStrategy: HuggingFace-optimized consensus
        - MetaConsensusSystem: Higher-order consensus coordination

    Routing Systems:
        - HybridExpertRouter: Multi-tier expert routing with cost optimization
        - ExpertTier: Expert tier classification (local, serverless, premium)
        - RoutingStrategy: Routing strategy selection (cost vs quality)

    Training Systems:
        - BTXTrainingSystem: Branch-Train-MiX training infrastructure
        - ConsensusConfig: Configuration for consensus mechanisms
        - MetaConsensusConfig: Meta-level consensus configuration

Example:
    Basic meta-consensus setup:

    >>> from capibara.training import MetaConsensusSystem, MetaConsensusConfig
    >>> from capibara.training import create_meta_consensus_system
    >>>
    >>> # Create meta-consensus configuration
    >>> config = MetaConsensusConfig(
    ...     num_experts=8,
    ...     consensus_threshold=0.7,
    ...     use_hf_integration=True
    ... )
    >>>
    >>> # Initialize system
    >>> system = create_meta_consensus_system(config)
    >>>
    >>> # Train with consensus
    >>> results = system.train(training_data)

    Hybrid expert routing:

    >>> from capibara.training import HybridExpertRouter, ExpertTier
    >>> from capibara.training import RoutingStrategy
    >>>
    >>> # Create hybrid router
    >>> router = HybridExpertRouter(
    ...     strategy=RoutingStrategy.COST_OPTIMIZED,
    ...     available_tiers=[ExpertTier.LOCAL, ExpertTier.SERVERLESS]
    ... )
    >>>
    >>> # Route query to optimal expert
    >>> expert = router.route_query(query, context)

    BTX training:

    >>> from capibara.training import BTXTrainingSystem, BTXExpertConfig
    >>>
    >>> # Configure BTX training
    >>> btx_config = BTXExpertConfig(
    ...     branch_factor=4,
    ...     train_epochs=10,
    ...     mix_strategy="weighted_average"
    ... )
    >>>
    >>> # Initialize BTX system
    >>> btx_system = BTXTrainingSystem(btx_config)
    >>> trained_model = btx_system.train(model, data)

Note:
    This package integrates with HuggingFace Transformers for model hosting and
    inference. Optional dependencies include serverless inference endpoints and
    premium API access for high-quality expert routing.

    The meta-consensus approach enables training stability and improved performance
    through multi-expert agreement mechanisms.

See Also:
    - capibara.core.optimization: Low-level training optimization
    - capibara.core.router: Base routing infrastructure
    - capibara.inference: Inference deployment systems
    - HuggingFace Hub: https://huggingface.co/

Version:
    1.0.0
"""

from .unified_consensus import UnifiedConsensusStrategy, ConsensusConfig
from .enhanced_hf_consensus_strategy import EnhancedHFConsensusStrategy, ServerlessExpertConfig
from .hybrid_expert_router import HybridExpertRouter, ExpertTier, RoutingStrategy
from .btx_training_system import BTXTrainingSystem, BTXExpertConfig
from .meta_consensus_system import MetaConsensusSystem, MetaConsensusConfig, create_meta_consensus_system

__version__ = "1.0.0"

__all__ = [
    "UnifiedConsensusStrategy",
    "EnhancedHFConsensusStrategy",
    "HybridExpertRouter",
    "BTXTrainingSystem",
    "MetaConsensusSystem",
    "create_meta_consensus_system",
    # Configuration classes
    "ConsensusConfig",
    "ServerlessExpertConfig",
    "BTXExpertConfig",
    "MetaConsensusConfig",
    # Enums
    "ExpertTier",
    "RoutingStrategy",
]

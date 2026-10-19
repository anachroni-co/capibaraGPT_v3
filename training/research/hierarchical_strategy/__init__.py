"""
Hierarchical Training Strategy Module - CapibaraGPT v3

Complete implementation of hierarchical training strategy:
- Transfer Learning Pipeline: 300M -> 600M -> 1.2B -> 3B -> 7B -> 13B
- Hierarchical MoE with 3 levels and 2.6B router
- Selective Ensemble for complex queries
- Cost optimization: $0.27/1K tokens (vs industry $0.50+)

Components:
- HierarchicalTrainingPipeline: Main training pipeline
- HierarchicalMoERouter: Intelligent routing system
- EnsembleManager: Selective ensemble management
- TransferLearningManager: Transfer learning management
"""

from .training_pipeline import (
    ModelTier,
    ModelConfig,
    DistillationConfig,
    create_training_pipeline,
    validate_training_strategy,
    HierarchicalTrainingPipeline,
)

from ..moe_hierarchical_router import (
    ExpertDomain,
    QueryAnalysis,
    RoutingDecision,
    QueryComplexity,
    HierarchicalMoERouter,
    create_hierarchical_router,
    estimate_routing_efficiency,
)

# NOTE: ensemble_manager and transfer_learning_manager were declared
# here but never implemented; imports removed in the 2026-07 cleanup.

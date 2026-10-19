"""
Unified Consensus Module - Modular Structure

This package provides a modular structure for the unified consensus system.
Components are organized into logical submodules while maintaining
backwards compatibility with the original unified_consensus.py.

Submodules:
    - config: Training configuration (IntegratedTrainingConfig)
    - voting: Voting system (AdvancedVotingSystem, VotingMetrics)
    - distillation: Distillation manager (DistillationManager)
    - trainer: Integrated trainer (IntegratedCapibaraTrainer)

For backwards compatibility, all classes are re-exported from unified_consensus.py:
    from training.consensus.unified_consensus import *
"""

# Re-export from parent module for backwards compatibility
try:
    from ..unified_consensus import (
        IntegratedTrainingConfig,
        EnhancedConsensusDistiller,
        EnhancedRefiner,
        IntegratedCapibaraTrainer,
        DatasetMerger,
        ModelEvaluator,
        VotingMetrics,
        TeacherModel,
        CriticModel,
        AdvancedVotingSystem,
        DistillationManager,
        UnifiedVotingConsensus,
        UnifiedRefinementConsensus,
        UnifiedDistillationConsensus,
        UnifiedCrossTeachingConsensus,
        get_production_integrated_config,
        should_use_consensus_for_scale,
        create_consensus_system_for_scale,
        create_distillation_manager_for_scale,
    )
    UNIFIED_AVAILABLE = True
except ImportError as e:
    UNIFIED_AVAILABLE = False
    import logging
    logging.getLogger(__name__).debug(f"Unified consensus not fully available: {e}")

__all__ = [
    "IntegratedTrainingConfig",
    "EnhancedConsensusDistiller",
    "EnhancedRefiner",
    "IntegratedCapibaraTrainer",
    "DatasetMerger",
    "ModelEvaluator",
    "VotingMetrics",
    "TeacherModel",
    "CriticModel",
    "AdvancedVotingSystem",
    "DistillationManager",
    "UnifiedVotingConsensus",
    "UnifiedRefinementConsensus",
    "UnifiedDistillationConsensus",
    "UnifiedCrossTeachingConsensus",
    "get_production_integrated_config",
    "should_use_consensus_for_scale",
    "create_consensus_system_for_scale",
    "create_distillation_manager_for_scale",
    "UNIFIED_AVAILABLE",
]

"""
Alignment Training for CapibaraGPT.

Includes:
- Multi-Objective RLAIF: AI feedback with multiple objectives
- Constitutional AI: Critique and revision
"""

from .rlaif_multi_objective import (
    RLAIFConfig,
    AlignmentObjective,
    ConstitutionalPrinciple,
    AIFeedbackModel,
    MultiObjectiveRewardModel,
    ConstitutionalReviser,
    RLAIFTrainer,
    DEFAULT_CONSTITUTION,
    create_rlaif_config,
)

__all__ = [
    'RLAIFConfig',
    'AlignmentObjective',
    'ConstitutionalPrinciple',
    'AIFeedbackModel',
    'MultiObjectiveRewardModel',
    'ConstitutionalReviser',
    'RLAIFTrainer',
    'DEFAULT_CONSTITUTION',
    'create_rlaif_config',
]

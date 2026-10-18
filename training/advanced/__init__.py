"""
Advanced Training Techniques for CapibaraGPT.

Includes:
- Expert Pruning: Post-training MoE optimization
- Scaling Laws: Inference-aware scaling configuration
"""

from .expert_pruning import (
    PruningConfig,
    PruningStrategy,
    ExpertStats,
    ExpertImportanceAnalyzer,
    ExpertClusterer,
    ExpertMerger,
    ExpertPruner,
    create_pruning_config,
)

from .scaling_laws import (
    ScalingConfig,
    ScalingRegime,
    ModelArchitecture,
    ScalingLawPredictor,
    ArchitectureAwareScaling,
    InferenceOptimizer,
    create_scaling_config,
    print_scaling_report,
)

__all__ = [
    # Expert Pruning
    'PruningConfig',
    'PruningStrategy',
    'ExpertStats',
    'ExpertImportanceAnalyzer',
    'ExpertClusterer',
    'ExpertMerger',
    'ExpertPruner',
    'create_pruning_config',
    # Scaling Laws
    'ScalingConfig',
    'ScalingRegime',
    'ModelArchitecture',
    'ScalingLawPredictor',
    'ArchitectureAwareScaling',
    'InferenceOptimizer',
    'create_scaling_config',
    'print_scaling_report',
]

"""
Advanced Training Techniques for CapibaraGPT.

Includes:
- Expert Pruning: Post-training MoE optimization
- Scaling Laws: Inference-aware scaling configuration
- Tensor Networks: Quantum-inspired compression
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

from .tensor_networks import (
    TensorNetworkConfig,
    DecompositionType,
    CompressionTarget,
    TensorTrainDecomposition,
    TensorTrainCore,
    TuckerDecomposition,
    LayerCompressor,
    ModelCompressor,
    create_tensor_network_config,
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
    # Tensor Networks
    'TensorNetworkConfig',
    'DecompositionType',
    'CompressionTarget',
    'TensorTrainDecomposition',
    'TensorTrainCore',
    'TuckerDecomposition',
    'LayerCompressor',
    'ModelCompressor',
    'create_tensor_network_config',
]

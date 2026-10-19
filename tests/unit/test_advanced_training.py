"""
Unit tests for advanced training modules.

Tests:
- Hybrid Mamba-Transformer blocks
- Ring Attention
- RLAIF Multi-objective
- Expert Pruning
- Scaling Laws
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import modules with error handling
MODULES_AVAILABLE = True
try:
    from capibara.hybrid_blocks import (
        HybridConfig,
        BlockType,
        MambaBlock,
        AttentionBlock,
        HybridBlock,
        HybridMambaTransformer,
        create_hybrid_config,
    )
    from capibara.attention import (
        RingAttentionConfig,
        RingAttention,
        CommunicationBackend,
        create_ring_attention_config,
    )
    from training.alignment import (
        RLAIFConfig,
        AlignmentObjective,
        ConstitutionalPrinciple,
        AIFeedbackModel,
        RLAIFTrainer,
        DEFAULT_CONSTITUTION,
        create_rlaif_config,
    )
    from training.advanced import (
        PruningConfig,
        PruningStrategy,
        ExpertPruner,
        create_pruning_config,
        ScalingConfig,
        ScalingRegime,
        ScalingLawPredictor,
        create_scaling_config,
        TensorNetworkConfig,
        DecompositionType,
        CompressionTarget,
        TensorTrainDecomposition,
        TuckerDecomposition,
        LayerCompressor,
        ModelCompressor,
        create_tensor_network_config,
    )
except ImportError as e:
    MODULES_AVAILABLE = False
    IMPORT_ERROR = str(e)


pytestmark = pytest.mark.skipif(
    not MODULES_AVAILABLE,
    reason=f"Advanced training modules not importable: {IMPORT_ERROR if not MODULES_AVAILABLE else ''}"
)


class TestHybridBlocks:
    """Tests for hybrid Mamba-Transformer architecture."""

    def test_hybrid_config_defaults(self):
        """Test default configuration values."""
        config = HybridConfig()

        assert config.hidden_size == 4096
        assert config.num_layers == 32
        assert config.hybrid_pattern == "alternating"
        assert config.use_moe == True

    def test_hybrid_config_layer_types(self):
        """Test layer type generation."""
        config = HybridConfig(num_layers=8, hybrid_pattern="alternating")
        layer_types = config.get_layer_types()

        assert len(layer_types) == 8
        # Alternating pattern
        assert layer_types[0] == BlockType.MAMBA or layer_types[0] == BlockType.MOE_MAMBA
        assert layer_types[1] == BlockType.ATTENTION or layer_types[1] == BlockType.MOE_ATTENTION

    def test_hybrid_config_front_mamba(self):
        """Test front_mamba pattern."""
        config = HybridConfig(num_layers=8, hybrid_pattern="front_mamba", use_moe=False)
        layer_types = config.get_layer_types()

        # First half should be Mamba
        assert all(t == BlockType.MAMBA for t in layer_types[:4])
        # Second half should be Attention
        assert all(t == BlockType.ATTENTION for t in layer_types[4:])

    def test_create_hybrid_config_factory(self):
        """Test factory function."""
        config = create_hybrid_config(
            model_size="7B",
            context_length=131072,
            hybrid_pattern="alternating",
            use_moe=True
        )

        assert config.hidden_size == 4096
        assert config.num_layers == 32
        assert config.max_position_embeddings == 131072

    def test_mamba_block_init(self):
        """Test Mamba block initialization."""
        config = HybridConfig()
        block = MambaBlock(config)

        assert block.config == config

    def test_attention_block_init(self):
        """Test Attention block initialization."""
        config = HybridConfig()
        block = AttentionBlock(config)

        assert block.config == config

    def test_hybrid_transformer_init(self):
        """Test full hybrid transformer initialization."""
        config = HybridConfig(num_layers=4)
        model = HybridMambaTransformer(config)

        assert len(model.layers) == 4
        summary = model.get_architecture_summary()
        assert summary["total_layers"] == 4


class TestRingAttention:
    """Tests for Ring Attention."""

    def test_ring_config_defaults(self):
        """Test default configuration."""
        config = RingAttentionConfig()

        assert config.max_sequence_length == 1048576  # 1M
        assert config.ring_size == 32
        assert config.backend == CommunicationBackend.NCCL

    def test_ring_config_global_sequence(self):
        """Test global sequence length calculation."""
        config = RingAttentionConfig(
            local_sequence_length=32768,
            ring_size=8
        )

        assert config.global_sequence_length == 32768 * 8

    def test_create_ring_config_factory(self):
        """Test factory function."""
        config = create_ring_attention_config(
            sequence_length=131072,
            num_gpus=8,
            backend="nccl"
        )

        assert config.max_sequence_length == 131072
        assert config.ring_size == 8
        assert config.local_sequence_length == 131072 // 8

    def test_ring_attention_init(self):
        """Test Ring Attention initialization."""
        config = RingAttentionConfig(ring_size=4)
        ring_attn = RingAttention(config, rank=0, world_size=4)

        assert ring_attn.ring_size == 4
        assert ring_attn.rank == 0


class TestRLAIF:
    """Tests for RLAIF multi-objective alignment."""

    def test_rlaif_config_defaults(self):
        """Test default configuration."""
        config = RLAIFConfig()

        assert AlignmentObjective.HELPFULNESS in config.objectives
        assert AlignmentObjective.HARMLESSNESS in config.objectives
        assert AlignmentObjective.HONESTY in config.objectives
        assert config.use_constitutional_ai == True

    def test_alignment_objectives(self):
        """Test alignment objectives enum."""
        assert AlignmentObjective.HELPFULNESS.value == "helpfulness"
        assert AlignmentObjective.HARMLESSNESS.value == "harmlessness"
        assert AlignmentObjective.FACTUALITY.value == "factuality"

    def test_constitutional_principle(self):
        """Test constitutional principle dataclass."""
        principle = ConstitutionalPrinciple(
            name="test_principle",
            description="Test description",
            objective=AlignmentObjective.HELPFULNESS,
            weight=1.5
        )

        assert principle.name == "test_principle"
        assert principle.weight == 1.5

        data = principle.to_dict()
        assert data["objective"] == "helpfulness"

    def test_default_constitution(self):
        """Test default constitution."""
        assert len(DEFAULT_CONSTITUTION) > 0

        # Check required objectives are covered
        objectives = {p.objective for p in DEFAULT_CONSTITUTION}
        assert AlignmentObjective.HELPFULNESS in objectives
        assert AlignmentObjective.HARMLESSNESS in objectives

    def test_ai_feedback_model(self):
        """Test AI feedback model."""
        model = AIFeedbackModel("path/to/model", DEFAULT_CONSTITUTION)

        scores = model.score_response(
            prompt="What is 2+2?",
            response="The answer is 4.",
            objectives=[AlignmentObjective.HELPFULNESS, AlignmentObjective.HONESTY]
        )

        assert "helpfulness" in scores
        assert "honesty" in scores
        assert all(0 <= s <= 1 for s in scores.values())

    def test_create_rlaif_config_factory(self):
        """Test factory function."""
        config = create_rlaif_config(
            objectives=["helpfulness", "harmlessness"],
            use_constitutional_ai=True
        )

        assert len(config.objectives) == 2
        assert config.use_constitutional_ai == True


class TestExpertPruning:
    """Tests for expert pruning."""

    def test_pruning_config_defaults(self):
        """Test default configuration."""
        config = PruningConfig()

        assert config.strategy == PruningStrategy.CLUSTER_DRIVEN
        assert config.target_expert_ratio == 0.5
        assert config.calibration_samples == 1000

    def test_pruning_strategies(self):
        """Test pruning strategy enum."""
        assert PruningStrategy.IMPORTANCE_BASED.value == "importance"
        assert PruningStrategy.CLUSTER_DRIVEN.value == "cluster"
        assert PruningStrategy.SIMILARITY_MERGE.value == "similarity"

    def test_expert_pruner_init(self):
        """Test expert pruner initialization."""
        config = PruningConfig()
        pruner = ExpertPruner(config)

        assert pruner.analyzer is not None
        assert pruner.clusterer is not None
        assert pruner.merger is not None

    def test_expert_pruner_calibrate(self):
        """Test calibration."""
        config = PruningConfig(calibration_samples=100)
        pruner = ExpertPruner(config)

        # Mock calibration
        stats = pruner.calibrate(None, None)

        assert "total_samples" in stats
        assert "experts_tracked" in stats

    def test_expert_pruner_prune(self):
        """Test pruning."""
        config = PruningConfig(
            strategy=PruningStrategy.IMPORTANCE_BASED,
            target_expert_ratio=0.5
        )
        pruner = ExpertPruner(config)

        # Run calibration first
        pruner.calibrate(None, None)

        # Run pruning
        results = pruner.prune(None)

        assert "original_experts" in results
        assert "pruned_experts" in results

    def test_create_pruning_config_factory(self):
        """Test factory function."""
        config = create_pruning_config(
            target_compression=0.6,
            strategy="cluster",
            max_quality_drop=0.03
        )

        assert config.target_expert_ratio == 0.6
        assert config.strategy == PruningStrategy.CLUSTER_DRIVEN
        assert config.max_quality_degradation == 0.03


class TestScalingLaws:
    """Tests for scaling laws."""

    def test_scaling_config_defaults(self):
        """Test default configuration."""
        config = ScalingConfig()

        assert config.regime == ScalingRegime.INFERENCE_OPTIMAL
        assert config.target_params == 7_000_000_000
        assert config.tokens_per_param == 150.0

    def test_scaling_regimes(self):
        """Test scaling regime enum."""
        assert ScalingRegime.CHINCHILLA.value == "chinchilla"
        assert ScalingRegime.INFERENCE_OPTIMAL.value == "inference"
        assert ScalingRegime.COMPUTE_OPTIMAL.value == "compute"

    def test_compute_optimal_training_tokens(self):
        """Test training token computation."""
        config = ScalingConfig(
            regime=ScalingRegime.CHINCHILLA,
            target_params=1_000_000_000,
            tokens_per_param=20.0
        )

        tokens = config.compute_optimal_training_tokens()
        assert tokens == 1_000_000_000 * 20

    def test_inference_optimal_tokens(self):
        """Test inference-optimal token computation."""
        config = ScalingConfig(
            regime=ScalingRegime.INFERENCE_OPTIMAL,
            target_params=7_000_000_000,
            tokens_per_param=150.0
        )

        tokens = config.compute_optimal_training_tokens()
        assert tokens == 7_000_000_000 * 150

    def test_scaling_law_predictor(self):
        """Test scaling law predictor."""
        predictor = ScalingLawPredictor()

        loss = predictor.predict_loss(7_000_000_000, 140_000_000_000)
        perplexity = predictor.predict_perplexity(7_000_000_000, 140_000_000_000)

        assert loss > 0
        assert perplexity > 1

    def test_compute_optimal_params(self):
        """Test compute optimal calculation."""
        predictor = ScalingLawPredictor()

        params, tokens = predictor.compute_optimal_params_for_compute(
            compute_budget=1e22,
            regime=ScalingRegime.CHINCHILLA
        )

        assert params > 0
        assert tokens > 0
        assert tokens / params < 30  # Should be around 20 for Chinchilla

    def test_create_scaling_config_factory(self):
        """Test factory function."""
        config = create_scaling_config(
            model_size="7B",
            regime="inference",
            architecture="hybrid_mamba"
        )

        assert config.target_params == 7_000_000_000
        assert config.regime == ScalingRegime.INFERENCE_OPTIMAL


class TestTensorNetworks:
    """Tests for tensor network compression."""

    def test_tensor_network_config_defaults(self):
        """Test default configuration."""
        config = TensorNetworkConfig()

        assert config.decomposition == DecompositionType.TENSOR_TRAIN
        assert config.target_compression_ratio == 0.5
        assert config.max_rank == 256
        assert CompressionTarget.FFN in config.target_layers

    def test_decomposition_types(self):
        """Test decomposition type enum."""
        assert DecompositionType.TENSOR_TRAIN.value == "tt"
        assert DecompositionType.TUCKER.value == "tucker"
        assert DecompositionType.LOW_RANK_SPARSE.value == "lrs"

    def test_compression_targets(self):
        """Test compression target enum."""
        assert CompressionTarget.ATTENTION.value == "attention"
        assert CompressionTarget.FFN.value == "ffn"
        assert CompressionTarget.MOE_EXPERTS.value == "moe"

    def test_tensor_train_init(self):
        """Test TensorTrain decomposition initialization."""
        config = TensorNetworkConfig(max_rank=128)
        tt = TensorTrainDecomposition(config)

        assert tt.config.max_rank == 128

    def test_tensor_train_mock_decompose(self):
        """Test TT decomposition with mock tensor."""
        config = TensorNetworkConfig(max_rank=64)
        tt = TensorTrainDecomposition(config)

        # Mock tensor (no numpy needed)
        class MockTensor:
            shape = (4096, 4096)

        cores = tt.decompose(MockTensor())

        assert len(cores) > 0
        assert tt.original_shape == (4096, 4096)

    def test_tucker_init(self):
        """Test Tucker decomposition initialization."""
        config = TensorNetworkConfig(decomposition=DecompositionType.TUCKER)
        tucker = TuckerDecomposition(config)

        assert tucker.config.decomposition == DecompositionType.TUCKER

    def test_layer_compressor_init(self):
        """Test layer compressor initialization."""
        config = TensorNetworkConfig()
        compressor = LayerCompressor(config)

        assert compressor.tt is not None
        assert compressor.tucker is not None

    def test_model_compressor_init(self):
        """Test model compressor initialization."""
        config = TensorNetworkConfig(
            target_compression_ratio=0.3,
            target_layers=[CompressionTarget.ALL]
        )
        compressor = ModelCompressor(config)

        assert compressor.config.target_compression_ratio == 0.3

    def test_create_tensor_network_config_factory(self):
        """Test factory function."""
        config = create_tensor_network_config(
            target_compression=0.4,
            decomposition="tucker",
            targets=["attention", "ffn"],
            max_rank=128
        )

        assert config.target_compression_ratio == 0.4
        assert config.decomposition == DecompositionType.TUCKER
        assert config.max_rank == 128
        assert CompressionTarget.ATTENTION in config.target_layers
        assert CompressionTarget.FFN in config.target_layers

    def test_tensor_train_compression_ratio(self):
        """Test compression ratio calculation."""
        config = TensorNetworkConfig(max_rank=32)
        tt = TensorTrainDecomposition(config)

        class MockTensor:
            shape = (4096, 4096)

        tt.decompose(MockTensor())
        ratio = tt.compression_ratio()

        # With max_rank=32, should achieve significant compression
        assert ratio < 1.0

    def test_tensor_train_stats(self):
        """Test getting decomposition stats."""
        config = TensorNetworkConfig()
        tt = TensorTrainDecomposition(config)

        class MockTensor:
            shape = (1024, 1024)

        tt.decompose(MockTensor())
        stats = tt.get_stats()

        assert "original_shape" in stats
        assert "tt_shape" in stats
        assert "ranks" in stats
        assert "compression_ratio" in stats


class TestIntegration:
    """Integration tests for advanced training modules."""

    def test_hybrid_with_ring_attention(self):
        """Test hybrid model with ring attention config."""
        hybrid_config = create_hybrid_config(
            model_size="7B",
            context_length=131072
        )

        ring_config = create_ring_attention_config(
            sequence_length=131072,
            num_gpus=8
        )

        # Configs should be compatible
        assert hybrid_config.max_position_embeddings == ring_config.max_sequence_length

    def test_scaling_with_pruning(self):
        """Test scaling config with pruning config."""
        scaling_config = create_scaling_config(model_size="7B")
        pruning_config = create_pruning_config(target_compression=0.5)

        # Both should work together
        assert scaling_config is not None
        assert pruning_config is not None

    def test_full_training_pipeline_configs(self):
        """Test creating configs for full training pipeline."""
        # Model architecture
        model_config = create_hybrid_config(
            model_size="7B",
            context_length=131072,
            use_moe=True
        )

        # Scaling
        scaling_config = create_scaling_config(
            model_size="7B",
            regime="inference"
        )

        # Alignment
        alignment_config = create_rlaif_config(
            objectives=["helpfulness", "harmlessness", "honesty"]
        )

        # All configs should be valid
        assert model_config.num_experts == 8
        assert scaling_config.tokens_per_param == 150.0
        assert len(alignment_config.objectives) == 3

    def test_tensor_networks_with_pruning(self):
        """Test tensor network compression with expert pruning."""
        # Both can be used together for maximum compression
        tn_config = create_tensor_network_config(
            target_compression=0.5,
            decomposition="tt",
            targets=["ffn", "moe"]
        )

        pruning_config = create_pruning_config(
            target_compression=0.5,
            strategy="cluster"
        )

        # Combined compression: 0.5 * 0.5 = 0.25 (75% reduction)
        combined_ratio = (
            tn_config.target_compression_ratio *
            pruning_config.target_expert_ratio
        )

        assert combined_ratio == 0.25
        assert tn_config.decomposition == DecompositionType.TENSOR_TRAIN
        assert pruning_config.strategy == PruningStrategy.CLUSTER_DRIVEN

    def test_full_compression_pipeline(self):
        """Test creating configs for full compression pipeline."""
        # Step 1: Expert pruning (reduce experts)
        pruning_config = create_pruning_config(
            target_compression=0.5,  # 8 experts -> 4 experts
            strategy="cluster"
        )

        # Step 2: Tensor network compression (compress weights)
        tn_config = create_tensor_network_config(
            target_compression=0.4,  # 60% weight reduction
            decomposition="tt",
            targets=["ffn", "attention"]
        )

        # Combined: model goes from 7B to ~1.4B effective params
        # 7B * 0.5 (pruning) * 0.4 (TN) = 1.4B
        effective_ratio = (
            pruning_config.target_expert_ratio *
            tn_config.target_compression_ratio
        )

        assert effective_ratio == 0.2  # 80% total compression
        assert CompressionTarget.FFN in tn_config.target_layers
        assert CompressionTarget.ATTENTION in tn_config.target_layers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

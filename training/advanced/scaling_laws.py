"""
Inference-Aware Scaling Laws Configuration

Implements compute-optimal training beyond Chinchilla:
- Inference cost optimization (100-200 tokens/param vs 20)
- Architecture-aware scaling
- Multi-objective optimization (training + inference)

References:
- Beyond Chinchilla-Optimal (Sardana et al., ICML 2024)
- Architecture-Aware Scaling (arXiv:2510.18245)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import math

logger = logging.getLogger(__name__)


class ScalingRegime(Enum):
    """Scaling law regimes."""
    CHINCHILLA = "chinchilla"           # Classic 20 tokens/param
    INFERENCE_OPTIMAL = "inference"      # 100-200 tokens/param
    COMPUTE_OPTIMAL = "compute"          # Balance training + inference
    SMALL_MODEL = "small"                # Heavily overtrained
    ARCHITECTURE_AWARE = "architecture"  # Model structure dependent


class ModelArchitecture(Enum):
    """Model architecture types for scaling."""
    DENSE_TRANSFORMER = "dense_transformer"
    MOE_TRANSFORMER = "moe_transformer"
    HYBRID_MAMBA = "hybrid_mamba"
    SSM_ONLY = "ssm_only"


@dataclass
class ScalingConfig:
    """Configuration for inference-aware scaling."""

    # Scaling regime
    regime: ScalingRegime = ScalingRegime.INFERENCE_OPTIMAL

    # Target model size (parameters)
    target_params: int = 7_000_000_000  # 7B

    # Tokens per parameter ratio
    tokens_per_param: float = 150.0  # 150 for inference-optimal

    # Architecture
    architecture: ModelArchitecture = ModelArchitecture.HYBRID_MAMBA

    # Inference assumptions
    expected_inference_requests: int = 1_000_000_000  # 1B requests
    avg_output_tokens_per_request: int = 200

    # Compute constraints
    training_budget_flops: float = 1e23  # ~10M GPU hours on A100
    inference_budget_daily_flops: float = 1e18

    # Architecture ratios
    mlp_to_attention_ratio: float = 2.5  # Typical ratio
    num_layers: int = 32

    # MoE settings (if applicable)
    num_experts: int = 8
    active_experts: int = 2

    # Quality targets
    target_perplexity: float = 8.0

    def compute_optimal_training_tokens(self) -> int:
        """Compute optimal number of training tokens."""
        if self.regime == ScalingRegime.CHINCHILLA:
            return int(self.target_params * 20)
        elif self.regime == ScalingRegime.INFERENCE_OPTIMAL:
            return int(self.target_params * self.tokens_per_param)
        elif self.regime == ScalingRegime.COMPUTE_OPTIMAL:
            return self._compute_optimal_with_inference()
        elif self.regime == ScalingRegime.SMALL_MODEL:
            return int(self.target_params * 500)  # Heavy overtraining
        else:
            return int(self.target_params * 100)

    def _compute_optimal_with_inference(self) -> int:
        """Compute optimal tokens considering inference costs."""
        # Total inference tokens over lifetime
        total_inference_tokens = (
            self.expected_inference_requests *
            self.avg_output_tokens_per_request
        )

        # Inference FLOPs per token
        inference_flops_per_token = 2 * self.target_params
        total_inference_flops = total_inference_tokens * inference_flops_per_token

        # Training FLOPs per token
        training_flops_per_token = 6 * self.target_params

        # Optimal balance: minimize total compute
        # Training cost: 6 * N * D (N=params, D=tokens)
        # Inference cost: 2 * N * I (I=inference tokens)
        # Optimal D ≈ sqrt(I/3) for equal marginal costs

        optimal_tokens = int(math.sqrt(total_inference_tokens / 3) * self.target_params * 0.1)

        # Clamp to reasonable range
        min_tokens = self.target_params * 20  # At least Chinchilla
        max_tokens = self.target_params * 500

        return max(min_tokens, min(max_tokens, optimal_tokens))


@dataclass
class ScalingLawPredictor:
    """
    Predicts model performance based on scaling laws.

    Uses empirical scaling laws to estimate:
    - Loss at given scale
    - Optimal hyperparameters
    - Compute requirements
    """

    # Chinchilla scaling law parameters
    # L = E + A/N^alpha + B/D^beta
    chinchilla_E: float = 1.69  # Irreducible entropy
    chinchilla_A: float = 406.4
    chinchilla_alpha: float = 0.34
    chinchilla_B: float = 410.7
    chinchilla_beta: float = 0.28

    def predict_loss(self, num_params: int, num_tokens: int) -> float:
        """Predict loss given scale."""
        N = num_params
        D = num_tokens

        loss = (
            self.chinchilla_E +
            self.chinchilla_A / (N ** self.chinchilla_alpha) +
            self.chinchilla_B / (D ** self.chinchilla_beta)
        )

        return loss

    def predict_perplexity(self, num_params: int, num_tokens: int) -> float:
        """Predict perplexity from loss."""
        loss = self.predict_loss(num_params, num_tokens)
        return math.exp(loss)

    def compute_optimal_params_for_compute(
        self,
        compute_budget: float,
        regime: ScalingRegime = ScalingRegime.CHINCHILLA
    ) -> Tuple[int, int]:
        """
        Compute optimal model size and tokens for compute budget.

        Args:
            compute_budget: FLOPs available
            regime: Scaling regime

        Returns:
            Tuple of (optimal_params, optimal_tokens)
        """
        if regime == ScalingRegime.CHINCHILLA:
            # C ≈ 6 * N * D, D ≈ 20 * N
            # C ≈ 120 * N^2
            optimal_params = int(math.sqrt(compute_budget / 120))
            optimal_tokens = int(optimal_params * 20)
        elif regime == ScalingRegime.INFERENCE_OPTIMAL:
            # Use more tokens, smaller model
            # C ≈ 6 * N * D, D ≈ 150 * N
            # C ≈ 900 * N^2
            optimal_params = int(math.sqrt(compute_budget / 900))
            optimal_tokens = int(optimal_params * 150)
        else:
            # Default to balanced
            optimal_params = int(math.sqrt(compute_budget / 300))
            optimal_tokens = int(optimal_params * 50)

        return optimal_params, optimal_tokens

    def estimate_training_compute(
        self,
        num_params: int,
        num_tokens: int
    ) -> float:
        """Estimate training compute in FLOPs."""
        # Approximately 6 FLOPs per parameter per token
        return 6 * num_params * num_tokens

    def estimate_training_time(
        self,
        num_params: int,
        num_tokens: int,
        gpu_flops: float = 312e12,  # A100 BF16
        num_gpus: int = 1,
        efficiency: float = 0.4
    ) -> float:
        """Estimate training time in hours."""
        total_flops = self.estimate_training_compute(num_params, num_tokens)
        effective_flops = gpu_flops * num_gpus * efficiency
        seconds = total_flops / effective_flops
        hours = seconds / 3600
        return hours


class ArchitectureAwareScaling:
    """
    Architecture-aware scaling that considers model structure.

    Different architectures scale differently:
    - Dense: Standard scaling
    - MoE: Parameter efficiency varies with expert utilization
    - Hybrid: SSM layers scale differently than attention
    """

    def __init__(self, config: ScalingConfig):
        self.config = config

        logger.info(f"ArchitectureAwareScaling initialized for {config.architecture.value}")

    def effective_params(self) -> int:
        """Compute effective parameters considering architecture."""
        total = self.config.target_params

        if self.config.architecture == ModelArchitecture.MOE_TRANSFORMER:
            # MoE has more total params but fewer active
            active_ratio = self.config.active_experts / self.config.num_experts
            # Effective params ≈ dense_params + shared_params
            # Assuming 2/3 of params are in experts
            expert_params = total * 0.67
            shared_params = total * 0.33
            effective = shared_params + expert_params * active_ratio
            return int(effective)

        elif self.config.architecture == ModelArchitecture.HYBRID_MAMBA:
            # Mamba layers have ~2x params but ~0.5x compute
            # Effective compute is reduced
            mamba_ratio = 0.5  # Assume 50% Mamba layers
            mamba_compute_factor = 0.5
            effective = total * (1 - mamba_ratio + mamba_ratio * mamba_compute_factor)
            return int(effective)

        return total

    def optimal_layer_ratio(self) -> Dict[str, float]:
        """Compute optimal ratio of different layer types."""
        if self.config.architecture == ModelArchitecture.HYBRID_MAMBA:
            # Based on recent research: slightly more attention for recall
            return {
                "mamba": 0.45,
                "attention": 0.45,
                "moe": 0.10
            }
        elif self.config.architecture == ModelArchitecture.MOE_TRANSFORMER:
            return {
                "attention": 0.50,
                "moe_ffn": 0.50
            }
        else:
            return {
                "attention": 0.33,
                "ffn": 0.67
            }

    def compute_efficiency_factor(self) -> float:
        """Compute expected training efficiency factor."""
        if self.config.architecture == ModelArchitecture.HYBRID_MAMBA:
            return 1.3  # 30% more efficient due to SSM
        elif self.config.architecture == ModelArchitecture.MOE_TRANSFORMER:
            return 1.0 / (self.config.active_experts / self.config.num_experts)
        else:
            return 1.0

    def recommend_hyperparameters(self) -> Dict[str, Any]:
        """Recommend hyperparameters based on architecture and scale."""
        params = self.config.target_params

        # Learning rate scales with sqrt(batch_size)
        base_lr = 3e-4

        # Batch size scales with model size
        if params < 1e9:
            batch_size = 256
        elif params < 10e9:
            batch_size = 512
        elif params < 100e9:
            batch_size = 1024
        else:
            batch_size = 2048

        # Learning rate adjustment
        lr = base_lr * math.sqrt(batch_size / 256)

        # Warmup steps
        warmup = min(2000, int(self.config.compute_optimal_training_tokens() / batch_size / 50))

        return {
            "learning_rate": lr,
            "batch_size": batch_size,
            "warmup_steps": warmup,
            "weight_decay": 0.1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "grad_clip": 1.0,
            "dropout": 0.0 if params > 1e9 else 0.1,  # No dropout for large models
        }


class InferenceOptimizer:
    """
    Optimizes model configuration for inference efficiency.

    Considers:
    - Latency requirements
    - Throughput targets
    - Memory constraints
    - Cost per query
    """

    def __init__(self, config: ScalingConfig):
        self.config = config

    def estimate_inference_latency(
        self,
        batch_size: int = 1,
        input_tokens: int = 1000,
        output_tokens: int = 200,
        gpu_type: str = "A100"
    ) -> float:
        """Estimate inference latency in milliseconds."""
        # Memory bandwidth bound for large batch
        # Compute bound for small batch

        params = self.config.target_params

        # Time per output token (simplified model)
        # Dominated by reading weights from memory

        # A100: 2TB/s bandwidth, 312 TFLOPS
        gpu_specs = {
            "A100": {"bandwidth_gb": 2000, "flops_t": 312},
            "H100": {"bandwidth_gb": 3350, "flops_t": 989},
            "MI250X": {"bandwidth_gb": 3200, "flops_t": 383},
        }

        specs = gpu_specs.get(gpu_type, gpu_specs["A100"])

        # Bytes to read (BF16 = 2 bytes per param)
        bytes_per_token = params * 2

        # Time to read weights
        memory_time_ms = (bytes_per_token / (specs["bandwidth_gb"] * 1e9)) * 1000

        # Time for compute
        flops_per_token = 2 * params
        compute_time_ms = (flops_per_token / (specs["flops_t"] * 1e12)) * 1000

        # Total time per token
        time_per_token = max(memory_time_ms, compute_time_ms) / batch_size

        # Total latency
        total_latency = time_per_token * output_tokens

        # Add prefill time (input processing)
        prefill_time = memory_time_ms * input_tokens / 100  # Faster due to batching

        return total_latency + prefill_time

    def estimate_throughput(
        self,
        batch_size: int = 32,
        gpu_type: str = "A100"
    ) -> float:
        """Estimate throughput in tokens per second."""
        latency_per_batch = self.estimate_inference_latency(
            batch_size=batch_size,
            gpu_type=gpu_type
        )

        tokens_per_batch = batch_size * 200  # Assuming 200 output tokens

        throughput = tokens_per_batch / (latency_per_batch / 1000)  # tokens/second

        return throughput

    def estimate_cost_per_query(
        self,
        gpu_type: str = "A100",
        gpu_cost_per_hour: float = 2.0
    ) -> float:
        """Estimate cost per inference query."""
        latency_ms = self.estimate_inference_latency(gpu_type=gpu_type)

        hours_per_query = latency_ms / (1000 * 3600)
        cost = hours_per_query * gpu_cost_per_hour

        return cost

    def recommend_serving_config(self) -> Dict[str, Any]:
        """Recommend serving configuration."""
        params = self.config.target_params

        # Determine if quantization is beneficial
        use_quantization = params > 7e9

        # Determine optimal batch size
        if params < 1e9:
            optimal_batch = 64
        elif params < 7e9:
            optimal_batch = 32
        elif params < 30e9:
            optimal_batch = 16
        else:
            optimal_batch = 8

        return {
            "batch_size": optimal_batch,
            "use_quantization": use_quantization,
            "quantization_bits": 8 if use_quantization else 16,
            "use_kv_cache": True,
            "use_speculative_decoding": params > 7e9,
            "use_continuous_batching": True,
            "max_concurrent_requests": optimal_batch * 2,
        }


def create_scaling_config(
    model_size: str = "7B",
    regime: str = "inference",
    architecture: str = "hybrid_mamba",
    inference_volume: int = 1_000_000_000
) -> ScalingConfig:
    """
    Factory function for scaling configuration.

    Args:
        model_size: Target model size (3B, 7B, 13B, 30B, 70B)
        regime: Scaling regime
        architecture: Model architecture
        inference_volume: Expected inference requests

    Returns:
        Configured ScalingConfig
    """
    size_map = {
        "3B": 3_000_000_000,
        "7B": 7_000_000_000,
        "13B": 13_000_000_000,
        "30B": 30_000_000_000,
        "70B": 70_000_000_000,
    }

    regime_map = {
        "chinchilla": (ScalingRegime.CHINCHILLA, 20.0),
        "inference": (ScalingRegime.INFERENCE_OPTIMAL, 150.0),
        "compute": (ScalingRegime.COMPUTE_OPTIMAL, 100.0),
        "small": (ScalingRegime.SMALL_MODEL, 500.0),
    }

    target_params = size_map.get(model_size, 7_000_000_000)
    scaling_regime, tokens_ratio = regime_map.get(regime, regime_map["inference"])

    return ScalingConfig(
        regime=scaling_regime,
        target_params=target_params,
        tokens_per_param=tokens_ratio,
        architecture=ModelArchitecture(architecture),
        expected_inference_requests=inference_volume,
    )


def print_scaling_report(config: ScalingConfig):
    """Print detailed scaling report."""
    predictor = ScalingLawPredictor()
    arch_scaling = ArchitectureAwareScaling(config)
    inference_opt = InferenceOptimizer(config)

    tokens = config.compute_optimal_training_tokens()
    loss = predictor.predict_loss(config.target_params, tokens)
    perplexity = predictor.predict_perplexity(config.target_params, tokens)
    compute = predictor.estimate_training_compute(config.target_params, tokens)

    print("=" * 60)
    print("SCALING CONFIGURATION REPORT")
    print("=" * 60)
    print(f"\nModel Configuration:")
    print(f"  Target Parameters: {config.target_params / 1e9:.1f}B")
    print(f"  Architecture: {config.architecture.value}")
    print(f"  Scaling Regime: {config.regime.value}")
    print(f"  Tokens/Parameter: {config.tokens_per_param:.0f}")

    print(f"\nTraining Requirements:")
    print(f"  Training Tokens: {tokens / 1e12:.2f}T")
    print(f"  Compute (FLOPs): {compute:.2e}")
    print(f"  Predicted Loss: {loss:.3f}")
    print(f"  Predicted Perplexity: {perplexity:.1f}")

    print(f"\nArchitecture-Aware Adjustments:")
    print(f"  Effective Parameters: {arch_scaling.effective_params() / 1e9:.1f}B")
    print(f"  Efficiency Factor: {arch_scaling.compute_efficiency_factor():.2f}x")
    print(f"  Layer Ratio: {arch_scaling.optimal_layer_ratio()}")

    hyperparams = arch_scaling.recommend_hyperparameters()
    print(f"\nRecommended Hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")

    print(f"\nInference Estimates:")
    print(f"  Latency (1 request): {inference_opt.estimate_inference_latency():.0f}ms")
    print(f"  Throughput: {inference_opt.estimate_throughput():.0f} tok/s")
    print(f"  Cost per Query: ${inference_opt.estimate_cost_per_query():.6f}")

    serving = inference_opt.recommend_serving_config()
    print(f"\nServing Configuration:")
    for key, value in serving.items():
        print(f"  {key}: {value}")

    print("=" * 60)


__all__ = [
    'ScalingConfig',
    'ScalingRegime',
    'ModelArchitecture',
    'ScalingLawPredictor',
    'ArchitectureAwareScaling',
    'InferenceOptimizer',
    'create_scaling_config',
    'print_scaling_report',
]

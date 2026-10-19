"""Metrics of evaluation for ViM-VQ"""

from typing import Dict, Tuple
import numpy as np
from jax import numpy as jnp

try:
    from core.decorators import profile_execution
except ImportError:
    profile_execution = None


def compute_mse(original: jnp.ndarray, reconstructed: jnp.ndarray) -> float:
    """Calculate mean squared error"""
    return float(jnp.mean((original - reconstructed) ** 2))

def compute_snr(original: jnp.ndarray, reconstructed: jnp.ndarray) -> float:
    """Calculate signal-to-noise ratio in dB"""
    mse = compute_mse(original, reconstructed)
    if mse == 0:
        return float('inf')
    power = jnp.mean(original ** 2)
    return float(10 * jnp.log10(power / mse))

def compute_correlation(original: jnp.ndarray, reconstructed: jnp.ndarray) -> float:
    """Calculate Pearson correlation"""
    orig_flat = original.reshape(-1)
    recon_flat = reconstructed.reshape(-1)

    orig_mean = jnp.mean(orig_flat)
    recon_mean = jnp.mean(recon_flat)

    numerator = jnp.sum((orig_flat - orig_mean) * (recon_flat - recon_mean))
    denominator = jnp.sqrt(
        jnp.sum((orig_flat - orig_mean) ** 2) *
        jnp.sum((recon_flat - recon_mean) ** 2)
    )

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return float(numerator / denominator)

def compute_compression_ratio(original_size: int, compressed_size: int) -> float:
    """Calculate compression ratio"""
    return original_size / compressed_size

def _evaluate_layer_quantization_impl(
    original: jnp.ndarray,
    reconstructed: jnp.ndarray,
    codebook_size: int,
    original_dtype
) -> Dict[str, float]:
    """Evaluate layer quantization quality"""

    # Sizes for compression ratio
    original_size = original.size * original_dtype.itemsize
    compressed_size = (
        codebook_size * original.shape[-1] * original_dtype.itemsize +  # codebook
        original.size * np.log2(codebook_size) / 8  # indices
    )

    return {
        "mse": compute_mse(original, reconstructed),
        "snr_db": compute_snr(original, reconstructed),
        "correlation": compute_correlation(original, reconstructed),
        "compression_ratio": compute_compression_ratio(original_size, int(compressed_size))
    }


if profile_execution is not None:
    evaluate_layer_quantization = profile_execution("evaluate_layer_quantization")(_evaluate_layer_quantization_impl)
else:
    evaluate_layer_quantization = _evaluate_layer_quantization_impl


def evaluate_model_quantization(
    results: Dict[str, Dict]
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Evaluate model quantization quality"""

    # Metrics by layer
    layer_metrics = {}

    # Aggregated metrics
    total_mse = 0
    total_snr = 0
    total_correlation = 0
    total_compression = 0
    n_layers = len(results)

    for layer_name, layer_results in results.items():
        metrics = evaluate_layer_quantization(
            layer_results["original"],
            layer_results["reconstructed"],
            layer_results["codebook"].shape[0],
            layer_results["original"].dtype
        )

        layer_metrics[layer_name] = metrics

        # Accumulate metrics
        total_mse += metrics["mse"]
        total_snr += metrics["snr_db"]
        total_correlation += metrics["correlation"]
        total_compression += metrics["compression_ratio"]

    # Averages
    avg_metrics = {
        "avg_mse": total_mse / n_layers,
        "avg_snr_db": total_snr / n_layers,
        "avg_correlation": total_correlation / n_layers,
        "avg_compression_ratio": total_compression / n_layers
    }

    return avg_metrics, layer_metrics

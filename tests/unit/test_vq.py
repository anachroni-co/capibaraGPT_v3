"""
VQ Tests - Unit tests for Vector Quantization layers.

This module provides tests for VQ-bit layers, validating quantization
operations, codebook management, and CPU fallback functionality.

Author: Skydesk International Dev Team.
"""

import numpy as np
import pytest


def _get_vqbit_layer():
    """Load VQbitLayer or skip if unavailable."""
    try:
        from capibara.vq.vqbit_layer import VQbitLayer, VQbitConfig
        return VQbitLayer, VQbitConfig
    except Exception:
        pytest.skip("VQbitLayer unavailable (missing dependencies or not installed)")


def test_vqbit_quantize_cpu():
    """Quick CPU VQ sanity check."""
    VQbitLayer, VQbitConfig = _get_vqbit_layer()
    config = VQbitConfig(codebook_size=8, embedding_dim=4)
    layer = VQbitLayer(config)

    x = np.ones((2, 4), dtype=np.float32) * 0.5
    quantized, metrics = layer.quantize(x)

    assert quantized.shape == x.shape
    assert isinstance(metrics, dict)

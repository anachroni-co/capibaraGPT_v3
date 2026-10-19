"""
Pytest Configuration and Shared Fixtures

This module provides common fixtures used across all tests.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
import subprocess

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ==================== Markers ====================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "tpu: Tests requiring TPU")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")


def _torch_import_works() -> bool:
    """
    Check if importing torch works without crashing the interpreter.

    We use a subprocess to avoid hard crashes (e.g., Windows DLL issues) that
    can terminate the pytest process during collection.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


# ==================== Optional Benchmark Fixture ====================

try:
    import pytest_benchmark  # noqa: F401
    _HAS_PYTEST_BENCHMARK = True
except Exception:
    _HAS_PYTEST_BENCHMARK = False

    @pytest.fixture
    def benchmark():
        """Skip benchmark tests if pytest-benchmark is not installed."""
        pytest.skip("pytest-benchmark not installed")


def pytest_collection_modifyitems(config, items):
    """
    Skip tests that require torch/transformers when torch cannot import,
    and benchmark tests if pytest-benchmark is not installed.
    """
    skip_marker = None
    if not _torch_import_works():
        skip_marker = pytest.mark.skip(
            reason="Skipping tests that import torch/transformers (torch import fails on this system)."
        )

    skip_bench = None
    if not _HAS_PYTEST_BENCHMARK:
        skip_bench = pytest.mark.skip(reason="pytest-benchmark not installed")

    for item in items:
        nodeid = item.nodeid.lower()
        if skip_marker and any(key in nodeid for key in ("tokenizer", "inference", "transformers", "hf_model")):
            item.add_marker(skip_marker)
        if skip_bench and item.get_closest_marker("benchmark") is not None:
            item.add_marker(skip_bench)


# ==================== Backend Fixtures ====================

@pytest.fixture(scope="session")
def cpu_backend():
    """Get CPU backend for testing."""
    from core.backends import get_backend, BackendType
    return get_backend(BackendType.CPU)


@pytest.fixture(scope="session")
def available_backend():
    """Get best available backend."""
    from core.backends import get_backend
    return get_backend()


@pytest.fixture(scope="session")
def best_backend():
    """
    Get best available backend with explicit fallback: GPU → TPU → CPU.

    This fixture always succeeds and returns the most capable backend available.
    """
    from core.backends import get_backend, BackendType

    # Try GPU first
    try:
        backend = get_backend(BackendType.GPU)
        if backend.name == "gpu":
            return backend
    except Exception:
        pass

    # Try TPU second
    try:
        backend = get_backend(BackendType.TPU)
        if backend.name == "tpu":
            return backend
    except Exception:
        pass

    # Fallback to CPU (always available)
    return get_backend(BackendType.CPU)


@pytest.fixture(scope="session")
def backend_info(best_backend):
    """Return info about which backend is being used."""
    return {
        "name": best_backend.name,
        "is_gpu": best_backend.name == "gpu",
        "is_tpu": best_backend.name == "tpu",
        "is_cpu": best_backend.name == "cpu",
        "is_accelerated": best_backend.name in ("gpu", "tpu"),
    }


@pytest.fixture
def backend(request):
    """
    Parameterized backend fixture.

    Use with: @pytest.mark.parametrize("backend", ["cpu", "gpu", "tpu"], indirect=True)
    """
    from core.backends import get_backend, BackendType

    backend_name = getattr(request, "param", "cpu")
    backend_map = {
        "cpu": BackendType.CPU,
        "gpu": BackendType.GPU,
        "tpu": BackendType.TPU,
    }

    backend_type = backend_map.get(backend_name, BackendType.CPU)

    try:
        return get_backend(backend_type)
    except Exception:
        if backend_name != "cpu":
            pytest.skip(f"{backend_name.upper()} not available")
        raise


# ==================== Model Fixtures ====================

@pytest.fixture
def model_config():
    """Default model configuration for testing."""
    return {
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "intermediate_size": 3072,
        "vocab_size": 32000,
        "max_position_embeddings": 2048,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
    }


@pytest.fixture
def small_model_config():
    """Small model configuration for fast testing."""
    return {
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "intermediate_size": 512,
        "vocab_size": 1000,
        "max_position_embeddings": 256,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
    }


# ==================== Data Fixtures ====================

@pytest.fixture
def random_batch():
    """Generate random batch data."""
    def _generate(
        batch_size: int = 4,
        seq_len: int = 128,
        hidden_size: int = 768,
        vocab_size: int = 32000,
    ) -> Dict[str, np.ndarray]:
        return {
            "input_ids": np.random.randint(0, vocab_size, (batch_size, seq_len)),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.float32),
            "hidden_states": np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32),
            "labels": np.random.randint(0, vocab_size, (batch_size, seq_len)),
        }
    return _generate


@pytest.fixture
def synthetic_text_batch():
    """Generate synthetic text batch."""
    def _generate(
        batch_size: int = 4,
        seq_len: int = 128,
    ) -> Dict[str, Any]:
        texts = [
            f"This is synthetic text sample {i} for testing purposes. "
            f"It contains various words and sentences to simulate real data."
            for i in range(batch_size)
        ]
        return {
            "texts": texts,
            "max_length": seq_len,
        }
    return _generate


@pytest.fixture
def attention_inputs():
    """Generate inputs for attention testing."""
    def _generate(
        batch_size: int = 2,
        num_heads: int = 4,
        seq_len: int = 64,
        head_dim: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        query = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
        key = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
        value = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
        return query, key, value
    return _generate


# ==================== Temporary Files ====================

@pytest.fixture
def temp_dir(tmp_path):
    """Provide temporary directory for tests."""
    return tmp_path


@pytest.fixture
def temp_checkpoint_path(tmp_path):
    """Provide temporary checkpoint path."""
    return tmp_path / "checkpoint.pt"


# ==================== Skip Conditions ====================

@pytest.fixture(scope="session")
def has_gpu():
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.fixture(scope="session")
def has_tpu():
    """Check if TPU is available."""
    try:
        import jax
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


# Skip decorators - check availability safely
def _check_gpu_available():
    """Check if GPU is available without crashing if torch is missing."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def _check_tpu_available():
    """Check if TPU is available without crashing if jax is missing."""
    try:
        import jax
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False

requires_gpu = pytest.mark.skipif(
    not _check_gpu_available(),
    reason="GPU not available"
)

requires_tpu = pytest.mark.skipif(
    not _check_tpu_available(),
    reason="TPU not available"
)


# ==================== Utility Functions ====================

def assert_tensors_close(
    a: np.ndarray,
    b: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """Assert two tensors are close."""
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def assert_tensor_shape(tensor: np.ndarray, expected_shape: Tuple[int, ...]) -> None:
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, f"Expected {expected_shape}, got {tensor.shape}"

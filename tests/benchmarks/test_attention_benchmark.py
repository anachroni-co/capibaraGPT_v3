"""
Attention Mechanism Benchmarks

Measures performance of:
- Scaled dot-product attention
- Flash Attention (when available)
- Multi-head attention
"""

import pytest
import numpy as np
from typing import Tuple

try:
    import pytest_benchmark  # noqa: F401
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not BENCHMARK_AVAILABLE,
        reason="pytest-benchmark not installed",
    ),
]


class TestAttentionBenchmarks:
    """Benchmark attention implementations."""

    @pytest.fixture
    def backend(self):
        from core.backends import get_backend
        return get_backend()

    def _create_attention_inputs(
        self,
        backend,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
    ) -> Tuple:
        """Create QKV inputs for attention."""
        query = backend.randn((batch_size, num_heads, seq_len, head_dim))
        key = backend.randn((batch_size, num_heads, seq_len, head_dim))
        value = backend.randn((batch_size, num_heads, seq_len, head_dim))
        return query, key, value

    @pytest.mark.benchmark(group="attention")
    def test_attention_small_seq(self, backend, benchmark):
        """Benchmark attention with small sequence length."""
        q, k, v = self._create_attention_inputs(
            backend,
            batch_size=4,
            num_heads=8,
            seq_len=128,
            head_dim=64,
        )

        def run_attention():
            return backend.scaled_dot_product_attention(q, k, v)

        result = benchmark(run_attention)
        assert result.shape == q.shape

    @pytest.mark.benchmark(group="attention")
    def test_attention_medium_seq(self, backend, benchmark):
        """Benchmark attention with medium sequence length."""
        q, k, v = self._create_attention_inputs(
            backend,
            batch_size=2,
            num_heads=8,
            seq_len=512,
            head_dim=64,
        )

        def run_attention():
            return backend.scaled_dot_product_attention(q, k, v)

        result = benchmark(run_attention)
        assert result.shape == q.shape

    @pytest.mark.benchmark(group="attention")
    @pytest.mark.slow
    def test_attention_long_seq(self, backend, benchmark):
        """Benchmark attention with long sequence length."""
        q, k, v = self._create_attention_inputs(
            backend,
            batch_size=1,
            num_heads=8,
            seq_len=2048,
            head_dim=64,
        )

        def run_attention():
            return backend.scaled_dot_product_attention(q, k, v)

        result = benchmark(run_attention)
        assert result.shape == q.shape

    @pytest.mark.benchmark(group="attention")
    def test_causal_attention(self, backend, benchmark):
        """Benchmark causal attention."""
        q, k, v = self._create_attention_inputs(
            backend,
            batch_size=4,
            num_heads=8,
            seq_len=256,
            head_dim=64,
        )

        def run_attention():
            return backend.scaled_dot_product_attention(q, k, v, is_causal=True)

        result = benchmark(run_attention)
        assert result.shape == q.shape


class TestMatmulBenchmarks:
    """Benchmark matrix multiplication."""

    @pytest.fixture
    def backend(self):
        from core.backends import get_backend
        return get_backend()

    @pytest.mark.benchmark(group="matmul")
    def test_matmul_small(self, backend, benchmark):
        """Benchmark small matrix multiplication."""
        a = backend.randn((64, 256, 256))
        b = backend.randn((64, 256, 256))

        result = benchmark(lambda: backend.matmul(a, b))
        assert result.shape == (64, 256, 256)

    @pytest.mark.benchmark(group="matmul")
    def test_matmul_large(self, backend, benchmark):
        """Benchmark large matrix multiplication."""
        a = backend.randn((4, 2048, 768))
        b = backend.randn((4, 768, 2048))

        result = benchmark(lambda: backend.matmul(a, b))
        assert result.shape == (4, 2048, 2048)


class TestActivationBenchmarks:
    """Benchmark activation functions."""

    @pytest.fixture
    def backend(self):
        from core.backends import get_backend
        return get_backend()

    @pytest.fixture
    def input_tensor(self, backend):
        return backend.randn((32, 512, 768))

    @pytest.mark.benchmark(group="activation")
    def test_gelu(self, backend, benchmark, input_tensor):
        """Benchmark GELU activation."""
        result = benchmark(lambda: backend.gelu(input_tensor))
        assert result.shape == input_tensor.shape

    @pytest.mark.benchmark(group="activation")
    def test_silu(self, backend, benchmark, input_tensor):
        """Benchmark SiLU activation."""
        result = benchmark(lambda: backend.silu(input_tensor))
        assert result.shape == input_tensor.shape

    @pytest.mark.benchmark(group="activation")
    def test_softmax(self, backend, benchmark, input_tensor):
        """Benchmark softmax."""
        result = benchmark(lambda: backend.softmax(input_tensor, axis=-1))
        assert result.shape == input_tensor.shape

    @pytest.mark.benchmark(group="normalization")
    def test_layer_norm(self, backend, benchmark, input_tensor):
        """Benchmark layer normalization."""
        result = benchmark(lambda: backend.layer_norm(input_tensor, (768,)))
        assert result.shape == input_tensor.shape

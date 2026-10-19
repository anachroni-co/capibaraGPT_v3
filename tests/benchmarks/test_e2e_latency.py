"""
End-to-End Latency Benchmarks for CapibaraGPT

Measures wall-clock latency of full pipeline operations:
- Forward pass through core layers
- MoE routing + expert dispatch
- SSM sequence processing
- Sparse attention end-to-end
- VQ encoding/decoding
- Full inference pipeline simulation

Run with:
    pytest tests/benchmarks/test_e2e_latency.py -s --benchmark-enable
    pytest tests/benchmarks/test_e2e_latency.py::TestManualLatencyReport -s
"""

import os
import time
import statistics
import pytest
import numpy as np
from typing import Dict, List

import logging
logger = logging.getLogger(__name__)

pytestmark = pytest.mark.slow
_RUN_LATENCY_REPORTS = os.getenv("CAPIBARA_RUN_LATENCY_REPORTS") == "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _measure_latency(fn, warmup: int = 3, repeats: int = 20) -> Dict[str, float]:
    """Run *fn* multiple times and return latency statistics (in ms)."""
    for _ in range(warmup):
        fn()

    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "p95_ms": sorted(times)[int(len(times) * 0.95)],
        "p99_ms": sorted(times)[int(len(times) * 0.99)],
        "samples": len(times),
    }


def _print_latency(name: str, stats: Dict[str, float]):
    logger.info(
        f"\n  {name}:"
        f"\n    mean={stats['mean_ms']:.3f}ms  median={stats['median_ms']:.3f}ms"
        f"  min={stats['min_ms']:.3f}ms  max={stats['max_ms']:.3f}ms"
        f"  p95={stats['p95_ms']:.3f}ms  p99={stats['p99_ms']:.3f}ms"
    )


# ---------------------------------------------------------------------------
# Backend fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def backend():
    from core.backends import get_backend
    return get_backend()


# ===========================================================================
# 1. Attention latency
# ===========================================================================

class TestAttentionLatency:
    """End-to-end latency for attention mechanisms."""

    @pytest.mark.benchmark(group="e2e-attention")
    def test_attention_small(self, backend, benchmark):
        """MHA: batch=4, heads=8, seq=512, dim=64."""
        q = backend.randn((4, 8, 512, 64))
        k = backend.randn((4, 8, 512, 64))
        v = backend.randn((4, 8, 512, 64))
        result = benchmark(backend.scaled_dot_product_attention, q, k, v)
        assert result is not None

    @pytest.mark.benchmark(group="e2e-attention")
    def test_attention_long_seq(self, backend, benchmark):
        """MHA: batch=2, heads=8, seq=2048, dim=64."""
        q = backend.randn((2, 8, 2048, 64))
        k = backend.randn((2, 8, 2048, 64))
        v = backend.randn((2, 8, 2048, 64))
        result = benchmark(backend.scaled_dot_product_attention, q, k, v)
        assert result is not None

    @pytest.mark.benchmark(group="e2e-attention")
    def test_attention_causal(self, backend, benchmark):
        """Causal MHA: batch=4, heads=8, seq=512."""
        q = backend.randn((4, 8, 512, 64))
        k = backend.randn((4, 8, 512, 64))
        v = backend.randn((4, 8, 512, 64))

        def run():
            return backend.scaled_dot_product_attention(q, k, v, is_causal=True)

        result = benchmark(run)
        assert result is not None


# ===========================================================================
# 2. MoE routing + dispatch
# ===========================================================================

class TestMoELatency:
    """End-to-end latency for Mixture-of-Experts."""

    def _get_moe_imports(self):
        """Import MoE modules, skip test if unavailable."""
        try:
            from core.moe.dynamic_moe import MoEConfig, DynamicRouter, DynamicMoELayer
            from jax import numpy as jnp
            from layers.jax_compat import JAX_AVAILABLE
            if not JAX_AVAILABLE:
                pytest.skip("MoE modules require JAX/Flax")
            return MoEConfig, DynamicRouter, DynamicMoELayer, jnp
        except ImportError:
            pytest.skip("MoE modules require JAX (circular import in this env)")

    @pytest.mark.benchmark(group="e2e-moe")
    def test_moe_routing(self, backend, benchmark):
        """MoE routing: 32 experts, batch=4, seq=128."""
        MoEConfig, DynamicRouter, _, jnp = self._get_moe_imports()
        config = MoEConfig(num_experts=32, num_active_experts=4, hidden_size=768)
        router = DynamicRouter(config)
        x = jnp.ones((4, 128, 768)) * 0.01
        result = benchmark(router, x)
        assert result is not None

    @pytest.mark.benchmark(group="e2e-moe")
    def test_moe_full_layer(self, backend, benchmark):
        """Full MoE layer: 8 experts, batch=2, seq=64."""
        MoEConfig, _, DynamicMoELayer, jnp = self._get_moe_imports()
        config = MoEConfig(
            num_experts=8, num_active_experts=2,
            hidden_size=256, expert_hidden_size=512
        )
        layer = DynamicMoELayer(config, layer_id=0)
        x = jnp.ones((2, 64, 256)) * 0.01

        def run():
            return layer(x, training=False)

        result = benchmark(run)
        assert result is not None


# ===========================================================================
# 3. SSM latency
# ===========================================================================

class TestSSMLatency:
    """End-to-end latency for State Space Models."""

    def _get_ssm_imports(self):
        try:
            import jax
            import jax.random
            from capibara.ssm.spike_ssm import SpikeSSM, AdaptiveSpikeSSM
            return jax, SpikeSSM, AdaptiveSpikeSSM
        except ImportError:
            pytest.skip("SSM modules require JAX")

    @pytest.mark.benchmark(group="e2e-ssm")
    def test_spike_ssm(self, backend, benchmark):
        """SpikeSSM: batch=4, seq=256, hidden=128."""
        jax, SpikeSSM, _ = self._get_ssm_imports()
        key = jax.random.PRNGKey(42)
        model = SpikeSSM(hidden_size=128, state_dim=64)
        x = jax.random.normal(key, (4, 256, 128))
        params = model.init(key, x)

        def run():
            return model.apply(params, x)

        result = benchmark(run)
        assert result is not None

    @pytest.mark.benchmark(group="e2e-ssm")
    def test_adaptive_spike_ssm(self, backend, benchmark):
        """AdaptiveSpikeSSM: batch=4, seq=256, hidden=128."""
        jax, _, AdaptiveSpikeSSM = self._get_ssm_imports()
        key = jax.random.PRNGKey(42)
        model = AdaptiveSpikeSSM(hidden_size=128, state_dim=64, num_timescales=3)
        x = jax.random.normal(key, (4, 256, 128))
        params = model.init(key, x)

        def run():
            return model.apply(params, x)

        result = benchmark(run)
        assert result is not None


# ===========================================================================
# 4. Sparse attention
# ===========================================================================

class TestSparseAttentionLatency:
    """End-to-end latency for sparse attention."""

    def _get_sparse_imports(self):
        try:
            import jax
            from layers.jax_compat import JAX_AVAILABLE
            from layers.sparsity.sparse_capibara import SparseCapibara
            if not JAX_AVAILABLE:
                pytest.skip("SparseCapibara requires JAX/Flax")
            return jax, SparseCapibara
        except ImportError:
            pytest.skip("SparseCapibara requires JAX/Flax")

    @pytest.mark.benchmark(group="e2e-sparse")
    def test_sparse_inference(self, backend, benchmark):
        """SparseCapibara inference: batch=2, seq=256, features=512."""
        jax, SparseCapibara = self._get_sparse_imports()
        key = jax.random.PRNGKey(0)
        model = SparseCapibara(features=512, num_heads=8, sparsity_ratio=0.9)
        x = jax.random.normal(key, (2, 256, 512))
        params = model.init(key, x)

        def run():
            return model.apply(params, x, training=False)

        result = benchmark(run)
        assert result is not None

    @pytest.mark.benchmark(group="e2e-sparse")
    def test_sparse_training(self, backend, benchmark):
        """SparseCapibara training: batch=2, seq=256, features=512."""
        jax, SparseCapibara = self._get_sparse_imports()
        key = jax.random.PRNGKey(0)
        model = SparseCapibara(features=512, num_heads=8, sparsity_ratio=0.9)
        x = jax.random.normal(key, (2, 256, 512))
        params = model.init(key, x)

        def run():
            return model.apply(params, x, training=True)

        result = benchmark(run)
        assert result is not None


# ===========================================================================
# 5. VQ encoding
# ===========================================================================

class TestVQLatency:
    """End-to-end latency for Vector Quantization."""

    @pytest.mark.benchmark(group="e2e-vq")
    def test_vqbit_encode(self, backend, benchmark):
        """VQbit encode: batch=32, dim=64."""
        try:
            from jax import numpy as jnp
            from capibara.vq.vqbit_layer import VQbitLayer, VQbitConfig
        except ImportError:
            pytest.skip("VQ modules not available")

        config = VQbitConfig(codebook_size=512, embedding_dim=64)
        layer = VQbitLayer(config)
        data = jnp.ones((32, 64)) * 0.5

        def run():
            return layer.compress(data)

        result = benchmark(run)
        assert result is not None


# ===========================================================================
# 6. Full inference pipeline
# ===========================================================================

class TestInferencePipelineLatency:
    """End-to-end latency simulating a full transformer block."""

    def _transformer_block(self, backend, x, wq, wk, wv, wo, w1, w2,
                           batch_size, seq_len, hidden_dim, num_heads, head_dim):
        """Single transformer block: attention + FFN."""
        # Attention projections
        q = backend.matmul(x, wq).reshape(batch_size, seq_len, num_heads, head_dim)
        k = backend.matmul(x, wk).reshape(batch_size, seq_len, num_heads, head_dim)
        v = backend.matmul(x, wv).reshape(batch_size, seq_len, num_heads, head_dim)

        # Transpose to (batch, heads, seq, dim)
        q = np.transpose(q, (0, 2, 1, 3))
        k = np.transpose(k, (0, 2, 1, 3))
        v = np.transpose(v, (0, 2, 1, 3))

        # Attention
        attn_out = backend.scaled_dot_product_attention(q, k, v)
        attn_out = np.transpose(attn_out, (0, 2, 1, 3)).reshape(
            batch_size, seq_len, hidden_dim
        )
        attn_out = backend.matmul(attn_out, wo)

        # Residual
        h = x + attn_out

        # FFN with GELU
        ffn = backend.matmul(h, w1)
        ffn = backend.gelu(ffn)
        ffn = backend.matmul(ffn, w2)

        return h + ffn

    def _make_weights(self, backend, hidden_dim):
        wq = backend.randn((hidden_dim, hidden_dim))
        wk = backend.randn((hidden_dim, hidden_dim))
        wv = backend.randn((hidden_dim, hidden_dim))
        wo = backend.randn((hidden_dim, hidden_dim))
        w1 = backend.randn((hidden_dim, hidden_dim * 4))
        w2 = backend.randn((hidden_dim * 4, hidden_dim))
        return wq, wk, wv, wo, w1, w2

    @pytest.mark.benchmark(group="e2e-pipeline")
    def test_single_token_forward(self, backend, benchmark):
        """Single token forward: batch=1, seq=1, hidden=256."""
        bs, sl, hd, nh = 1, 1, 256, 8
        head_dim = hd // nh
        x = backend.randn((bs, sl, hd))
        wq, wk, wv, wo, w1, w2 = self._make_weights(backend, hd)

        def run():
            return self._transformer_block(
                backend, x, wq, wk, wv, wo, w1, w2, bs, sl, hd, nh, head_dim
            )

        result = benchmark(run)
        assert result is not None

    @pytest.mark.benchmark(group="e2e-pipeline")
    def test_batch_forward(self, backend, benchmark):
        """Batch forward: batch=8, seq=128, hidden=256."""
        bs, sl, hd, nh = 8, 128, 256, 8
        head_dim = hd // nh
        x = backend.randn((bs, sl, hd))
        wq, wk, wv, wo, w1, w2 = self._make_weights(backend, hd)

        def run():
            return self._transformer_block(
                backend, x, wq, wk, wv, wo, w1, w2, bs, sl, hd, nh, head_dim
            )

        result = benchmark(run)
        assert result is not None

    @pytest.mark.benchmark(group="e2e-pipeline")
    def test_multi_layer_forward(self, backend, benchmark):
        """Multi-layer forward: 4 layers, batch=2, seq=64, hidden=256."""
        bs, sl, hd, nh = 2, 64, 256, 8
        head_dim = hd // nh
        x = backend.randn((bs, sl, hd))
        layer_weights = [self._make_weights(backend, hd) for _ in range(4)]

        def run():
            h = x
            for wq, wk, wv, wo, w1, w2 in layer_weights:
                h = self._transformer_block(
                    backend, h, wq, wk, wv, wo, w1, w2, bs, sl, hd, nh, head_dim
                )
            return h

        result = benchmark(run)
        assert result is not None


# ===========================================================================
# 7. Scaling tests
# ===========================================================================

class TestScalingLatency:
    """Measure how latency scales with input size."""

    @pytest.mark.benchmark(group="e2e-scaling")
    @pytest.mark.parametrize("seq_len", [64, 256, 512, 1024])
    def test_attention_vs_seq_len(self, backend, benchmark, seq_len):
        """Attention latency vs sequence length."""
        q = backend.randn((2, 8, seq_len, 64))
        k = backend.randn((2, 8, seq_len, 64))
        v = backend.randn((2, 8, seq_len, 64))
        result = benchmark(backend.scaled_dot_product_attention, q, k, v)
        assert result is not None

    @pytest.mark.benchmark(group="e2e-scaling")
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_ffn_vs_batch_size(self, backend, benchmark, batch_size):
        """FFN latency vs batch size."""
        x = backend.randn((batch_size, 128, 256))
        w1 = backend.randn((256, 1024))
        w2 = backend.randn((1024, 256))

        def run():
            h = backend.gelu(backend.matmul(x, w1))
            return backend.matmul(h, w2)

        result = benchmark(run)
        assert result is not None


# ===========================================================================
# 8. Manual latency report (no pytest-benchmark needed)
# ===========================================================================

class TestManualLatencyReport:
    """
    Print detailed latency reports.
    Run: pytest -s tests/benchmarks/test_e2e_latency.py::TestManualLatencyReport
    """

    @pytest.mark.skipif(
        not _RUN_LATENCY_REPORTS,
        reason="Set CAPIBARA_RUN_LATENCY_REPORTS=1 to run latency reports.",
    )
    def test_attention_latency_report(self, backend):
        """Detailed attention latency report."""
        configs = [
            ("small",  2, 8,  128, 64),
            ("medium", 4, 8,  512, 64),
            ("large",  2, 12, 1024, 64),
            ("xlarge", 1, 16, 2048, 64),
        ]

        logger.info("\n" + "=" * 60)
        logger.warning("ATTENTION LATENCY REPORT")
        logger.info("=" * 60)

        for name, bs, nh, sl, hd in configs:
            q = backend.randn((bs, nh, sl, hd))
            k = backend.randn((bs, nh, sl, hd))
            v = backend.randn((bs, nh, sl, hd))

            stats = _measure_latency(
                lambda q=q, k=k, v=v: backend.scaled_dot_product_attention(q, k, v)
            )
            _print_latency(f"{name} (b={bs}, h={nh}, s={sl}, d={hd})", stats)

    @pytest.mark.skipif(
        not _RUN_LATENCY_REPORTS,
        reason="Set CAPIBARA_RUN_LATENCY_REPORTS=1 to run latency reports.",
    )
    def test_pipeline_latency_report(self, backend):
        """Detailed inference pipeline latency report."""
        logger.info("\n" + "=" * 60)
        logger.info("INFERENCE PIPELINE LATENCY REPORT")
        logger.info("=" * 60)

        configs = [
            ("single-token", 1, 1, 256, 8),
            ("short-seq",    1, 32, 256, 8),
            ("batch-short",  8, 32, 256, 8),
            ("batch-long",   4, 256, 256, 8),
        ]

        for name, bs, sl, hd, nh in configs:
            head_dim = hd // nh
            x = backend.randn((bs, sl, hd))
            wq = backend.randn((hd, hd))
            wk = backend.randn((hd, hd))
            wv = backend.randn((hd, hd))
            wo = backend.randn((hd, hd))
            w1 = backend.randn((hd, hd * 4))
            w2 = backend.randn((hd * 4, hd))

            def run(x=x, wq=wq, wk=wk, wv=wv, wo=wo, w1=w1, w2=w2,
                    bs=bs, sl=sl, hd=hd, nh=nh, head_dim=head_dim):
                q = backend.matmul(x, wq).reshape(bs, sl, nh, head_dim)
                k = backend.matmul(x, wk).reshape(bs, sl, nh, head_dim)
                v = backend.matmul(x, wv).reshape(bs, sl, nh, head_dim)
                q = np.transpose(q, (0, 2, 1, 3))
                k = np.transpose(k, (0, 2, 1, 3))
                v = np.transpose(v, (0, 2, 1, 3))
                attn = backend.scaled_dot_product_attention(q, k, v)
                attn = np.transpose(attn, (0, 2, 1, 3)).reshape(bs, sl, hd)
                attn = backend.matmul(attn, wo)
                h = x + attn
                ffn = backend.gelu(backend.matmul(h, w1))
                ffn = backend.matmul(ffn, w2)
                return h + ffn

            stats = _measure_latency(run)
            _print_latency(f"{name} (b={bs}, s={sl}, h={hd})", stats)

    @pytest.mark.skipif(
        not _RUN_LATENCY_REPORTS,
        reason="Set CAPIBARA_RUN_LATENCY_REPORTS=1 to run latency reports.",
    )
    def test_scaling_latency_report(self, backend):
        """Detailed scaling report: attention vs sequence length."""
        logger.info("\n" + "=" * 60)
        logger.warning("ATTENTION SCALING REPORT (seq_len)")
        logger.info("=" * 60)

        for sl in [64, 128, 256, 512, 1024, 2048]:
            q = backend.randn((2, 8, sl, 64))
            k = backend.randn((2, 8, sl, 64))
            v = backend.randn((2, 8, sl, 64))

            stats = _measure_latency(
                lambda q=q, k=k, v=v: backend.scaled_dot_product_attention(q, k, v)
            )
            _print_latency(f"seq_len={sl}", stats)

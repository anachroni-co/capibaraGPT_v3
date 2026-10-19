"""Integration tests for ISSUE-001: remove TPU consensus mocks.

Verifies that :mod:`training.tpu.tpu_v6_consensus_optimizer` no longer relies
on randomness in its main flow:

- Query and expert embeddings are deterministic and depend only on text /
  caller-supplied data. The prior implementation called
  ``jax.random.normal`` seeded by Python's salted ``hash``, producing values
  that changed every process.
- ``_update_tpu_metrics`` produces the same numbers for two identical runs
  and leaves hardware counters at ``0.0`` when no profiler is wired in. The
  prior implementation populated them with ``np.random.uniform`` samples.
- The fallback path keeps working when JAX is unavailable or the mesh
  cannot be initialized (so CPU-only CI nodes still exercise the code).

The tests do **not** exercise the ``@jit``-decorated mesh kernels because
those require a real multi-core TPU substrate; that is out of scope for
ISSUE-001. The JIT paths remain in place unchanged and will be covered by
a follow-up issue that wires them to an actual TPU runner.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

pytest.importorskip(
    "jax",
    reason="training.tpu.tpu_v6_consensus_optimizer requires JAX to exercise "
    "the code paths under test (embedding / metrics plumbing).",
)

import jax.numpy as jnp  # noqa: E402  imported after importorskip

from training.tpu.tpu_v6_consensus_optimizer import (  # noqa: E402
    TPUConsensusMode,
    TPUv6ConsensusConfig,
    TPUv6ConsensusOptimizer,
    _deterministic_embedding,
)


pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _run(coro):
    """Run an ``async`` coroutine inside a fresh event loop.

    We build a dedicated loop per test instead of using ``asyncio.run`` so
    the tests work regardless of whether ``pytest-asyncio`` is configured
    in ``"auto"`` mode on the host project.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_optimizer(tpu_cores: int = 1, embedding_dim: int = 64):
    """Build a minimal optimizer and force the mesh off.

    Forcing ``mesh = None`` steers :meth:`optimize_consensus_for_tpu_v6`
    into its fallback path, which is enough for exercising the embedding /
    metric plumbing without compiling the ``@jit`` kernels that expect a
    real TPU topology.
    """
    config = TPUv6ConsensusConfig(
        tpu_cores=tpu_cores,
        mesh_shape=(1, 1),
        consensus_mode=TPUConsensusMode.MESH_DISTRIBUTED,
        embedding_dim=embedding_dim,
    )
    optimizer = TPUv6ConsensusOptimizer(config)
    optimizer.mesh = None  # force fallback irrespective of local JAX backend
    return optimizer


# --------------------------------------------------------------------------- #
# Deterministic embedding
# --------------------------------------------------------------------------- #


class TestDeterministicEmbedding:
    """``_deterministic_embedding`` replaces the previous random mock."""

    def test_reproducible_across_calls(self):
        v1 = _deterministic_embedding("hello world", dim=64)
        v2 = _deterministic_embedding("hello world", dim=64)
        np.testing.assert_allclose(np.asarray(v1), np.asarray(v2))

    def test_different_text_gives_different_vector(self):
        v1 = np.asarray(_deterministic_embedding("alpha", dim=64))
        v2 = np.asarray(_deterministic_embedding("beta", dim=64))
        assert not np.allclose(v1, v2)

    def test_output_is_unit_norm_and_correct_shape(self):
        vec = np.asarray(_deterministic_embedding("unit-norm-check", dim=64))
        assert vec.shape == (64,)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5

    def test_respects_requested_dimensionality(self):
        for dim in (16, 64, 128, 512):
            vec = np.asarray(_deterministic_embedding("dim-check", dim=dim))
            assert vec.shape == (dim,), f"shape mismatch for dim={dim}"


# --------------------------------------------------------------------------- #
# Query embedding resolution
# --------------------------------------------------------------------------- #


class TestQueryEmbedding:
    """``_generate_tpu_query_embedding`` honours the injected embedder or falls back."""

    def test_fallback_is_deterministic(self):
        optimizer = _make_optimizer()
        e1 = _run(optimizer._generate_tpu_query_embedding("same"))
        e2 = _run(optimizer._generate_tpu_query_embedding("same"))
        np.testing.assert_allclose(np.asarray(e1), np.asarray(e2))

    def test_custom_embedder_is_respected(self):
        calls = []

        def fake_embedder(text: str):
            calls.append(text)
            # Return shape must match the configured embedding_dim.
            return jnp.ones((64,))

        config = TPUv6ConsensusConfig(
            tpu_cores=1,
            mesh_shape=(1, 1),
            embedding_dim=64,
            query_embedder=fake_embedder,
        )
        optimizer = TPUv6ConsensusOptimizer(config)
        optimizer.mesh = None

        vec = _run(optimizer._generate_tpu_query_embedding("q"))

        assert calls == ["q"]
        arr = np.asarray(vec)
        assert arr.shape == (64,)
        # Uniform vectors stay uniform after L2 normalization.
        assert abs(float(np.linalg.norm(arr)) - 1.0) < 1e-5


# --------------------------------------------------------------------------- #
# Expert preparation
# --------------------------------------------------------------------------- #


class TestExpertPreparation:
    """Expert embeddings come from the pool, the embedder, or the fallback."""

    def test_precomputed_embedding_is_used_verbatim(self):
        optimizer = _make_optimizer(tpu_cores=1, embedding_dim=64)
        preset = jnp.arange(64, dtype=jnp.float32)
        pool = [
            {
                "name": "expert-a",
                "embedding": preset,
                "weight": 2.0,
                "quality_score": 9.0,
                "cost_per_1k_tokens": 0.005,
            }
        ]

        data = _run(optimizer._prepare_expert_data_for_tpu(pool))

        emb = np.asarray(data["embeddings"][0])
        expected = np.arange(64, dtype=np.float32)
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(emb, expected, atol=1e-6)
        assert float(data["weights"][0]) == pytest.approx(2.0)
        assert float(data["quality_scores"][0]) == pytest.approx(9.0)
        assert float(data["costs"][0]) == pytest.approx(0.005)

    def test_mismatched_embedding_dim_raises(self):
        optimizer = _make_optimizer(tpu_cores=1, embedding_dim=64)
        pool = [{"name": "bad", "embedding": jnp.ones((32,))}]  # dim mismatch

        with pytest.raises(ValueError, match="dim mismatch"):
            _run(optimizer._prepare_expert_data_for_tpu(pool))

    def test_padding_to_tpu_cores(self):
        optimizer = _make_optimizer(tpu_cores=4, embedding_dim=64)
        pool = [{"name": f"e{i}"} for i in range(3)]  # 3 experts, tpu_cores=4

        data = _run(optimizer._prepare_expert_data_for_tpu(pool))

        embeddings = np.asarray(data["embeddings"])
        assert embeddings.shape == (4, 64)
        np.testing.assert_allclose(embeddings[3], np.zeros(64))
        assert float(data["weights"][3]) == 0.0
        assert float(data["quality_scores"][3]) == 0.0
        assert float(data["costs"][3]) == 0.0

    def test_custom_embedder_used_when_pool_lacks_embedding(self):
        seen = []

        def fake_embedder(expert):
            seen.append(expert["name"])
            return jnp.ones((64,))

        config = TPUv6ConsensusConfig(
            tpu_cores=1,
            mesh_shape=(1, 1),
            embedding_dim=64,
            expert_embedder=fake_embedder,
        )
        optimizer = TPUv6ConsensusOptimizer(config)
        optimizer.mesh = None
        pool = [{"name": "x"}]

        data = _run(optimizer._prepare_expert_data_for_tpu(pool))

        assert seen == ["x"]
        assert np.asarray(data["embeddings"]).shape == (1, 64)


# --------------------------------------------------------------------------- #
# Metrics are real, not random
# --------------------------------------------------------------------------- #


class TestTpuMetrics:
    """``_update_tpu_metrics`` no longer uses random draws."""

    def test_two_identical_runs_produce_identical_snapshots(self):
        optimizer = _make_optimizer(tpu_cores=2)
        fake_result = {
            "consensus_confidence": 0.9,
            "selected_similarities": [0.8, 0.82, 0.85],
        }

        _run(optimizer._update_tpu_metrics(fake_result, 0.1))
        snap_a = optimizer._get_tpu_metrics_snapshot()
        _run(optimizer._update_tpu_metrics(fake_result, 0.1))
        snap_b = optimizer._get_tpu_metrics_snapshot()

        assert snap_a == snap_b

    def test_hardware_counters_are_zero_without_profiler(self):
        optimizer = _make_optimizer()

        _run(optimizer._update_tpu_metrics({"consensus_confidence": 0.5}, 0.01))

        # These require real TPU hardware counters — we decided not to fake them.
        assert optimizer.metrics.flops_utilization == 0.0
        assert optimizer.metrics.memory_bandwidth_utilization == 0.0
        assert optimizer.metrics.network_bandwidth_utilization == 0.0
        # compilation_time_ms is also unmeasurable here and must not drift.
        assert optimizer.metrics.compilation_time_ms == 0.0

    def test_cross_core_agreement_reflects_similarity_spread(self):
        optimizer = _make_optimizer()

        # Perfect agreement -> 1.0
        _run(
            optimizer._update_tpu_metrics(
                {
                    "consensus_confidence": 0.7,
                    "selected_similarities": [0.9, 0.9, 0.9],
                },
                0.01,
            )
        )
        assert optimizer.metrics.cross_core_agreement == pytest.approx(1.0, abs=1e-6)

        # Wide spread -> much lower agreement.
        _run(
            optimizer._update_tpu_metrics(
                {
                    "consensus_confidence": 0.7,
                    "selected_similarities": [0.1, 0.9],
                },
                0.01,
            )
        )
        assert optimizer.metrics.cross_core_agreement < 0.5

    def test_core_utilization_length_matches_tpu_cores(self):
        optimizer = _make_optimizer(tpu_cores=5)

        _run(optimizer._update_tpu_metrics({"consensus_confidence": 0.5}, 0.01))

        assert len(optimizer.metrics.core_utilization) == 5
        assert len(optimizer.metrics.memory_utilization_per_core) == 5

    def test_mesh_efficiency_tracks_mesh_state(self):
        optimizer = _make_optimizer()  # mesh forced to None
        _run(optimizer._update_tpu_metrics({"consensus_confidence": 0.5}, 0.01))
        assert optimizer.metrics.mesh_synchronization_efficiency == 0.0


# --------------------------------------------------------------------------- #
# End-to-end (fallback) path
# --------------------------------------------------------------------------- #


class TestEndToEndFallback:
    """The public entry point is reproducible on CPU-only hosts."""

    def test_optimize_returns_stable_fallback_result(self):
        optimizer = _make_optimizer()
        pool = [
            {
                "name": f"e{i}",
                "quality_score": 7.5 + i * 0.1,
                "cost_per_1k_tokens": 0.002,
            }
            for i in range(4)
        ]

        r1 = _run(optimizer.optimize_consensus_for_tpu_v6("test", pool))
        r2 = _run(optimizer.optimize_consensus_for_tpu_v6("test", pool))

        assert r1["tpu_optimized"] is False
        assert r1["processing_mode"] == "fallback"
        # The fallback must be deterministic.
        assert r1["selected_expert_indices"] == r2["selected_expert_indices"]
        assert r1["total_cost"] == pytest.approx(r2["total_cost"])

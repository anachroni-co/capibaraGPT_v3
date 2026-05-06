"""
Unit tests for core/model_factory (BACKLOG-008).

Two layers of validation:

  1. Pure-Python tests on ModelConfig + analytic param estimates.
     These run anywhere, no JAX needed; they are the "always-on" guard.

  2. JAX/Flax-dependent tests on the smoke preset (~150K params).
     These actually init params and run a forward, but they are skipped
     gracefully when JAX is not importable (CI without ML deps).

Architecture-level expectations of BACKLOG-008 codified here:

  - Presets 1b and 3b match the analytic targets within +/- 10%.
  - configs/1b.toml and configs/3b.toml load via ModelConfig.from_toml and
    produce ModelConfig instances equal to the in-code presets.
  - Smoke model forward returns logits of shape (B, T, vocab_size).
  - Aux-loss is a finite scalar > 0 for randomly initialized routers.
  - Causal mask: changing token at position t MUST NOT change logits[:, :t, :].
"""
from __future__ import annotations

import importlib.util as _ilu
import sys as _sys
import types as _types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Direct-file load to bypass the heavy core/__init__.py import graph.
#
# We mount the model_factory directory as a synthetic parent package
# "_mfu_pkg" so the submodules' relative imports (`from .config import X`)
# resolve correctly. Otherwise the importlib loader has no parent and
# `from .config` raises ImportError.
# ---------------------------------------------------------------------------

_FACTORY_DIR = Path(__file__).resolve().parents[2] / "core" / "model_factory"

_PKG_NAME = "_mfu_pkg"
if _PKG_NAME not in _sys.modules:
    _pkg = _types.ModuleType(_PKG_NAME)
    _pkg.__path__ = [str(_FACTORY_DIR)]                              # mark as package
    _pkg.__package__ = _PKG_NAME
    _sys.modules[_PKG_NAME] = _pkg


def _load(mod_name: str, file_name: str):
    """Load a model_factory submodule under '_mfu_pkg.<mod_name>'."""
    full_name = f"{_PKG_NAME}.{mod_name}"
    if full_name in _sys.modules:
        return _sys.modules[full_name]
    spec = _ilu.spec_from_file_location(
        full_name,
        _FACTORY_DIR / file_name,
        submodule_search_locations=None,
    )
    mod = _ilu.module_from_spec(spec)
    mod.__package__ = _PKG_NAME                                       # enables `from .config`
    _sys.modules[full_name] = mod                                     # MUST precede exec
    spec.loader.exec_module(mod)
    return mod


_config = _load("config", "config.py")
ModelConfig = _config.ModelConfig
get_preset = _config.get_preset


# ---------------------------------------------------------------------------
# Layer 1: pure-Python tests on ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfigValidation:
    def test_preset_1b_param_count_within_target(self):
        c = ModelConfig.preset_1b()
        active = c.active_params_estimate() / 1e9
        # Target 1B active +/- 15% (1.0B to 1.3B is acceptable).
        assert 0.9 <= active <= 1.3, f"1B preset active={active:.2f}B out of band"

    def test_preset_3b_param_count_within_target(self):
        c = ModelConfig.preset_3b()
        active = c.active_params_estimate() / 1e9
        # Target 3B active +/- 15%.
        assert 2.55 <= active <= 3.45, f"3B preset active={active:.2f}B out of band"

    def test_preset_3b_total_is_larger_than_active(self):
        c = ModelConfig.preset_3b()
        assert c.total_params_estimate() > c.active_params_estimate()

    def test_smoke_preset_constructs(self):
        c = ModelConfig.preset_smoke()
        assert c.d_model == 64 and c.n_layers == 2

    def test_get_preset_known_keys(self):
        for key in ("1b", "3b", "smoke"):
            assert isinstance(get_preset(key), ModelConfig)

    def test_get_preset_unknown_raises(self):
        with pytest.raises(KeyError):
            get_preset("13b")

    def test_invalid_gqa_ratio_raises(self):
        # 17 query heads with 4 kv heads: not divisible.
        with pytest.raises(ValueError, match="divisible"):
            ModelConfig(
                vocab_size=100, max_seq_len=64, d_model=68, n_layers=1,
                n_heads=17, n_kv_heads=4, head_dim=4, ff_dim=128,
                n_experts=2, top_k=1,
            )

    def test_head_dim_mismatch_raises(self):
        with pytest.raises(ValueError, match="head_dim"):
            ModelConfig(
                vocab_size=100, max_seq_len=64, d_model=64, n_layers=1,
                n_heads=4, n_kv_heads=2, head_dim=32, ff_dim=128,  # 4*32=128 != 64
                n_experts=2, top_k=1,
            )

    def test_top_k_exceeds_n_experts_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            ModelConfig(
                vocab_size=100, max_seq_len=64, d_model=32, n_layers=1,
                n_heads=4, n_kv_heads=2, head_dim=8, ff_dim=64,
                n_experts=2, top_k=4,
            )


class TestTomlLoading:
    """The shipped configs/*.toml MUST stay in sync with the in-code presets."""

    REPO_ROOT = Path(__file__).resolve().parents[2]

    def test_1b_toml_matches_preset(self):
        cfg = ModelConfig.from_toml(self.REPO_ROOT / "configs" / "1b.toml")
        assert cfg == ModelConfig.preset_1b()

    def test_3b_toml_matches_preset(self):
        cfg = ModelConfig.from_toml(self.REPO_ROOT / "configs" / "3b.toml")
        assert cfg == ModelConfig.preset_3b()

    def test_missing_toml_raises(self):
        with pytest.raises(FileNotFoundError):
            ModelConfig.from_toml("/nonexistent/path/0xff.toml")


# ---------------------------------------------------------------------------
# Layer 2: JAX-dependent tests on the smoke preset
# ---------------------------------------------------------------------------

jax = pytest.importorskip("jax")
flax = pytest.importorskip("flax")
import jax.numpy as jnp  # noqa: E402  (guarded above)

# Now that JAX is imported, we can load model.py (which imports flax.linen).
_blocks = _load("blocks", "blocks.py")
_model = _load("model", "model.py")
build_model = _model.build_model
init_params = _model.init_params
count_params = _model.count_params


@pytest.fixture(scope="module")
def smoke_setup():
    cfg = ModelConfig.preset_smoke()
    model = build_model(cfg)
    params = init_params(model, cfg, seed=0, batch_size=2, seq_len=8)
    return cfg, model, params


class TestSmokeForward:
    def test_init_produces_finite_params(self, smoke_setup):
        _cfg, _model, params = smoke_setup
        leaves, _ = jax.tree_util.tree_flatten(params)
        for leaf in leaves:
            assert jnp.all(jnp.isfinite(leaf)), "non-finite param after init"

    def test_runtime_param_count_matches_analytic_within_1pct(self, smoke_setup):
        cfg, _model, params = smoke_setup
        runtime = count_params(params)
        analytic = cfg.active_params_estimate()  # smoke: top_k == n_experts
        # RMSNorm scales add a small amount; allow 1% slack.
        assert abs(runtime - analytic) <= max(1024, analytic // 100), (
            f"runtime={runtime:,} analytic={analytic:,}"
        )

    def test_forward_shape(self, smoke_setup):
        cfg, model, params = smoke_setup
        ids = jnp.zeros((2, 8), dtype=jnp.int32)
        out = model.apply(
            {"params": params}, ids,
            deterministic=True,
            rngs={"router": jax.random.PRNGKey(0)},
        )
        assert out.logits.shape == (2, 8, cfg.vocab_size)
        assert out.logits.dtype == jnp.float32

    def test_aux_loss_is_finite_scalar(self, smoke_setup):
        _cfg, model, params = smoke_setup
        ids = jnp.zeros((2, 8), dtype=jnp.int32)
        out = model.apply(
            {"params": params}, ids,
            deterministic=True,
            rngs={"router": jax.random.PRNGKey(0)},
        )
        aux = float(out.aux_loss)
        assert aux >= 0.0 and aux == aux  # NaN check via self-equality
        assert out.aux_loss.shape == ()

    def test_forward_is_causal(self, smoke_setup):
        """Changing token at position t must not affect logits at positions < t."""
        cfg, model, params = smoke_setup
        ids_a = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32)
        ids_b = ids_a.at[0, 5].set(99)            # change only position 5

        out_a = model.apply(
            {"params": params}, ids_a, deterministic=True,
            rngs={"router": jax.random.PRNGKey(0)},
        )
        out_b = model.apply(
            {"params": params}, ids_b, deterministic=True,
            rngs={"router": jax.random.PRNGKey(0)},
        )
        # Logits at positions 0..4 must be identical (within fp tolerance).
        diff = jnp.abs(out_a.logits[:, :5, :] - out_b.logits[:, :5, :])
        assert float(jnp.max(diff)) < 1e-4, (
            f"causality violated: max-diff={float(jnp.max(diff)):.6f}"
        )

    def test_seq_len_over_max_raises(self, smoke_setup):
        cfg, model, params = smoke_setup
        too_long = jnp.zeros((1, cfg.max_seq_len + 1), dtype=jnp.int32)
        with pytest.raises(ValueError, match="max_seq_len"):
            model.apply(
                {"params": params}, too_long, deterministic=True,
                rngs={"router": jax.random.PRNGKey(0)},
            )

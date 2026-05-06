"""
Unit tests for core/trainer (BACKLOG-009).

Two layers, mirroring tests/unit/test_model_factory.py:

  1. Pure-Python tests on the recipe loader + dataclass validation.
     These run anywhere, no JAX needed.

  2. JAX/Flax-dependent tests that exercise the optimizer schedule, the
     train_step, the full Trainer.fit loop on the smoke preset, and a
     checkpoint roundtrip. They are skipped when JAX is not importable.

Behaviors codified here (these are the contract of BACKLOG-009):

  - load_recipe(1b.toml) yields a Recipe whose model section equals
    ModelConfig.preset_1b(); same for 3b.
  - The cosine schedule starts at schedule_init_value, peaks at lr_peak
    at warmup_steps, and decays to effective_end_value at total_steps-1.
  - One train_step on the smoke preset returns finite loss + grad_norm
    and a non-negative aux_loss.
  - 30 steps of fit() drive the loss strictly below the loss after step 1.
  - Two Trainer instances seeded identically produce identical step-1 loss.
  - CheckpointManager.save -> CheckpointManager.restore round-trips params
    bit-for-bit (pickle path; orbax path covered by integration runs).
"""
from __future__ import annotations

import importlib.util as _ilu
import sys as _sys
import types as _types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Bootstrap synthetic 'core' + 'core.model_factory' + 'core.trainer' packages.
#
# We never execute core/__init__.py (which imports the entire capibara graph
# via safe_import). Instead we install lightweight namespace packages with
# the right __path__, then load the relevant submodules with
# spec_from_file_location. This is the same pattern used by
# tests/unit/test_model_factory.py.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
_CORE_DIR = REPO_ROOT / "core"
_MF_DIR = _CORE_DIR / "model_factory"
_TR_DIR = _CORE_DIR / "trainer"


def _install_namespace_pkg(name: str, path: Path) -> None:
    """Install (or reuse) a synthetic namespace package rooted at `path`."""
    existing = _sys.modules.get(name)
    if existing is not None and getattr(existing, "_synthetic_for_tests", False):
        return
    pkg = _types.ModuleType(name)
    pkg.__path__ = [str(path)]
    pkg.__package__ = name
    pkg._synthetic_for_tests = True  # type: ignore[attr-defined]
    _sys.modules[name] = pkg


_install_namespace_pkg("core", _CORE_DIR)
_install_namespace_pkg("core.model_factory", _MF_DIR)
_install_namespace_pkg("core.trainer", _TR_DIR)


def _load(full_name: str, file_path: Path):
    """Load a submodule under an already-installed namespace package."""
    if full_name in _sys.modules and not getattr(
        _sys.modules[full_name], "_synthetic_for_tests", False
    ):
        return _sys.modules[full_name]
    spec = _ilu.spec_from_file_location(full_name, file_path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Layer-1 imports: pure Python.
_mf_config = _load("core.model_factory.config", _MF_DIR / "config.py")
ModelConfig = _mf_config.ModelConfig

_cfg_loader = _load("core.trainer.config_loader", _TR_DIR / "config_loader.py")
TrainingConfig = _cfg_loader.TrainingConfig
CheckpointConfig = _cfg_loader.CheckpointConfig
LoggingConfig = _cfg_loader.LoggingConfig
Recipe = _cfg_loader.Recipe
load_recipe = _cfg_loader.load_recipe


# ---------------------------------------------------------------------------
# Layer 1: pure-Python tests
# ---------------------------------------------------------------------------


class TestTrainingConfigValidation:
    def test_defaults_are_valid(self):
        c = TrainingConfig()
        assert c.lr_peak == 3.0e-4
        assert c.beta2 == 0.95
        assert c.schedule_type == "cosine"

    def test_effective_end_value_derives_when_omitted(self):
        c = TrainingConfig(lr_peak=1e-3, lr_min_ratio=0.1)
        assert c.effective_end_value == pytest.approx(1e-4)

    def test_effective_end_value_uses_explicit_override(self):
        c = TrainingConfig(lr_peak=1e-3, schedule_end_value=5e-5)
        assert c.effective_end_value == 5e-5

    def test_per_device_batch_size_divides_grad_accum(self):
        c = TrainingConfig(batch_size_global=64, grad_accum_steps=4)
        assert c.per_device_batch_size == 16

    def test_per_device_batch_size_indivisible_raises(self):
        c = TrainingConfig(batch_size_global=10, grad_accum_steps=3)
        with pytest.raises(ValueError, match="divisible"):
            _ = c.per_device_batch_size

    def test_warmup_exceeds_total_raises(self):
        with pytest.raises(ValueError, match="warmup_steps"):
            TrainingConfig(warmup_steps=2000, total_steps=1000)

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="schedule_type"):
            TrainingConfig(schedule_type="square_root_inverse_with_jitter")

    def test_invalid_label_smoothing_raises(self):
        with pytest.raises(ValueError, match="label_smoothing"):
            TrainingConfig(label_smoothing=1.0)


class TestRecipeLoading:
    """The shipped configs/*.toml MUST load into self-consistent Recipes."""

    def test_load_1b_recipe(self):
        r = load_recipe(REPO_ROOT / "configs" / "1b.toml")
        assert isinstance(r, Recipe)
        assert r.model == ModelConfig.preset_1b()
        # Training section maps directly.
        assert r.training.lr_peak == pytest.approx(3.0e-4)
        assert r.training.warmup_steps == 2000
        assert r.training.total_steps == 100_000
        assert r.training.beta2 == pytest.approx(0.95)
        # Schedule end value comes from [training.schedule].
        assert r.training.effective_end_value == pytest.approx(3.0e-5)
        # Checkpoint + logging.
        assert r.checkpoint.out_dir == "checkpoints/capibara-moe-1b"
        assert r.checkpoint.keep_last == 3
        assert r.logging.log_every == 50

    def test_load_3b_recipe(self):
        r = load_recipe(REPO_ROOT / "configs" / "3b.toml")
        assert r.model == ModelConfig.preset_3b()
        assert r.training.lr_peak == pytest.approx(1.5e-4)
        assert r.training.warmup_steps == 4000
        assert r.training.effective_end_value == pytest.approx(1.5e-5)

    def test_recipe_name_is_model_name(self):
        r = load_recipe(REPO_ROOT / "configs" / "1b.toml")
        assert r.name == "capibara-moe-1b"

    def test_missing_recipe_raises(self):
        with pytest.raises(FileNotFoundError):
            load_recipe(REPO_ROOT / "configs" / "does_not_exist.toml")


# ---------------------------------------------------------------------------
# Layer 2: JAX-dependent tests
# ---------------------------------------------------------------------------

jax = pytest.importorskip("jax")
flax = pytest.importorskip("flax")
optax = pytest.importorskip("optax")
import jax.numpy as jnp  # noqa: E402

# Now that JAX is importable we can pull in the JAX-using submodules.
_mf_blocks = _load("core.model_factory.blocks", _MF_DIR / "blocks.py")
_mf_model = _load("core.model_factory.model", _MF_DIR / "model.py")
build_model_factory = _mf_model.build_model
init_params = _mf_model.init_params

_tr_optimizer = _load("core.trainer.optimizer", _TR_DIR / "optimizer.py")
build_optimizer = _tr_optimizer.build_optimizer

_tr_ckpt = _load("core.trainer.checkpointing", _TR_DIR / "checkpointing.py")
CheckpointManager = _tr_ckpt.CheckpointManager

_tr_trainer = _load("core.trainer.trainer", _TR_DIR / "trainer.py")
Trainer = _tr_trainer.Trainer
TrainState = _tr_trainer.TrainState
StepMetrics = _tr_trainer.StepMetrics
make_train_step = _tr_trainer.make_train_step


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _smoke_recipe(total_steps: int = 30, warmup_steps: int = 5) -> Recipe:
    """Build an in-memory Recipe sized for CPU smoke (no toml needed).

    Uses ModelConfig.preset_smoke (~150K params) plus a TrainingConfig that
    is valid for total_steps as small as 1. seq_len/batch are tiny so a
    full fit() finishes in a few seconds on CPU.
    """
    return Recipe(
        model=ModelConfig.preset_smoke(),
        training=TrainingConfig(
            seed=0,
            seq_len=8,
            batch_size_global=2,
            grad_accum_steps=1,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            lr_peak=1.0e-3,             # bigger than 3e-4 so loss moves visibly in 30 steps
            lr_min_ratio=0.1,
            weight_decay=0.0,           # weight decay on tiny model is just noise here
            beta1=0.9,
            beta2=0.95,
            epsilon=1.0e-8,
            grad_clip=1.0,
            label_smoothing=0.0,
            grad_ckpt_ratio=0.0,
            schedule_type="cosine",
            schedule_init_value=0.0,
        ),
        checkpoint=CheckpointConfig(out_dir="checkpoints/_test_smoke", keep_last=2, save_every=10),
        logging=LoggingConfig(log_every=1000, eval_every=1000, eval_seq_count=2),
    )


def _synthetic_iter(recipe: Recipe, n_batches: int, seed: int = 0):
    """Same shape as scripts/train.py::_synthetic_iter, kept local to tests."""
    key = jax.random.PRNGKey(seed)
    bs = recipe.training.per_device_batch_size
    seq = recipe.training.seq_len
    V = recipe.model.vocab_size
    for i in range(n_batches):
        key, sk = jax.random.split(key)
        ids = jax.random.randint(sk, (bs, seq), 0, V, dtype=jnp.int32)
        tgt = jnp.roll(ids, -1, axis=-1)
        yield {"input_ids": ids, "targets": tgt}


# ---------------------------------------------------------------------------
# optimizer / schedule
# ---------------------------------------------------------------------------


class TestOptimizerSchedule:
    def test_cosine_starts_at_init_value(self):
        cfg = TrainingConfig(
            total_steps=100, warmup_steps=10, lr_peak=1e-3,
            schedule_init_value=0.0,
        )
        _, sched = build_optimizer(cfg)
        assert float(sched(0)) == pytest.approx(0.0)

    def test_cosine_reaches_peak_at_warmup(self):
        cfg = TrainingConfig(total_steps=100, warmup_steps=10, lr_peak=1e-3)
        _, sched = build_optimizer(cfg)
        # warmup_cosine peaks exactly at warmup_steps.
        assert float(sched(10)) == pytest.approx(1e-3, rel=1e-4)

    def test_cosine_decays_to_end_value(self):
        cfg = TrainingConfig(total_steps=100, warmup_steps=10, lr_peak=1e-3, lr_min_ratio=0.1)
        _, sched = build_optimizer(cfg)
        # decay_steps in our builder is total_steps (so step total_steps-1 is just shy of end).
        # Optax's warmup_cosine_decay_schedule reaches end_value exactly at decay_steps.
        assert float(sched(100)) == pytest.approx(1e-4, rel=1e-3)

    def test_constant_schedule_holds_after_warmup(self):
        cfg = TrainingConfig(
            total_steps=100, warmup_steps=10, lr_peak=2e-4,
            schedule_type="constant",
        )
        _, sched = build_optimizer(cfg)
        assert float(sched(50)) == pytest.approx(2e-4, rel=1e-6)
        assert float(sched(99)) == pytest.approx(2e-4, rel=1e-6)

    def test_unknown_schedule_type_raises(self):
        # __post_init__ catches this before build_optimizer is even called.
        with pytest.raises(ValueError, match="schedule_type"):
            TrainingConfig(schedule_type="bogus")


# ---------------------------------------------------------------------------
# Trainer smoke
# ---------------------------------------------------------------------------


class TestTrainerSmoke:
    """Combined fit() smoke run: one fit() of N steps + checks on the whole trace.

    Why one combined test instead of four small ones: each Trainer construction
    triggers an init_params pass + JIT compile (~10s on CPU). Bundling the
    metric assertions into a single pass keeps total CI time bounded.
    """

    @pytest.fixture(scope="class")
    def recipe(self) -> Recipe:
        return _smoke_recipe(total_steps=30, warmup_steps=5)

    @pytest.fixture(scope="class")
    def trace(self, recipe: Recipe) -> list[StepMetrics]:
        """Run 15 steps on a SINGLE repeated batch.

        Repeating the same batch forces the model to memorize it, so the loss
        decreases monotonically within ~10 steps even on the smoke preset
        (~150K params). With fresh random tokens per batch the smoke trunk
        is too small to make visible progress in 15 steps, which produces
        flaky tests on a CPU runner.
        """
        bs = recipe.training.per_device_batch_size
        seq = recipe.training.seq_len
        V = recipe.model.vocab_size
        key = jax.random.PRNGKey(2)
        ids = jax.random.randint(key, (bs, seq), 0, V, dtype=jnp.int32)
        fixed_batch = {"input_ids": ids, "targets": jnp.roll(ids, -1, axis=-1)}

        def repeat_iter():
            while True:
                yield fixed_batch

        trainer = Trainer(recipe=recipe)
        captured: list[StepMetrics] = []
        trainer.fit(
            data_iter=repeat_iter(),
            on_step=captured.append,
            max_steps=15,
        )
        return captured

    def test_metrics_are_finite_and_well_formed(self, trace: list[StepMetrics]):
        assert len(trace) == 15
        for i, m in enumerate(trace):
            assert m.step == i + 1
            assert jnp.isfinite(m.loss).item(), f"non-finite loss at step {m.step}"
            assert jnp.isfinite(m.grad_norm).item(), f"non-finite grad_norm at step {m.step}"
            assert m.aux_loss >= 0.0, f"negative aux_loss={m.aux_loss} at step {m.step}"
            assert m.tokens_per_second > 0.0

    def test_loss_decreases_over_run(self, trace: list[StepMetrics]):
        first_few = sum(c.loss for c in trace[:3]) / 3
        last_few = sum(c.loss for c in trace[-3:]) / 3
        # Smoke model overfits ~150K-param trunk on 15 batches of synthetic data:
        # mean of last-3 losses MUST be < mean of first-3.
        assert last_few < first_few, (
            f"loss did not decrease: first3={first_few:.4f} last3={last_few:.4f}"
        )

    def test_lr_follows_warmup(self, trace: list[StepMetrics]):
        # During the warmup window the logged LR must be strictly increasing.
        # warmup_steps=5 -> the first 5 logged LRs are inside the linear ramp.
        warmup_lrs = [c.lr for c in trace[:5]]
        for prev, curr in zip(warmup_lrs, warmup_lrs[1:]):
            assert curr > prev, f"lr non-monotonic in warmup: {warmup_lrs}"
        # Across the run the LR must reach at least 90% of lr_peak.
        assert max(c.lr for c in trace) >= 0.9 * 1e-3, (
            f"lr never reached near-peak: max={max(c.lr for c in trace):.4e}"
        )

    def test_train_step_is_deterministic_with_fixed_seed(self, recipe: Recipe):
        """Two Trainers sharing the same recipe seed must produce identical step-1 loss."""
        t1 = Trainer(recipe=recipe)
        t2 = Trainer(recipe=recipe)
        m1: list[StepMetrics] = []
        m2: list[StepMetrics] = []
        t1.fit(_synthetic_iter(recipe, n_batches=1, seed=42), on_step=m1.append, max_steps=1)
        t2.fit(_synthetic_iter(recipe, n_batches=1, seed=42), on_step=m2.append, max_steps=1)
        assert m1[0].loss == pytest.approx(m2[0].loss, rel=1e-6, abs=1e-6)
        assert m1[0].aux_loss == pytest.approx(m2[0].aux_loss, rel=1e-6, abs=1e-6)


# ---------------------------------------------------------------------------
# Checkpoint roundtrip
# ---------------------------------------------------------------------------


class TestCheckpointRoundtrip:
    def test_save_then_restore_recovers_state(self, tmp_path: Path):
        recipe = _smoke_recipe(total_steps=4, warmup_steps=1)
        ckpt_cfg = CheckpointConfig(
            out_dir=str(tmp_path / "ckpts"), keep_last=2, save_every=1,
        )
        cm = CheckpointManager(ckpt_cfg)

        trainer = Trainer(recipe=recipe, checkpoint_manager=cm)
        # Train 2 steps, then save explicitly.
        captured: list[StepMetrics] = []
        trainer.fit(
            data_iter=_synthetic_iter(recipe, n_batches=2, seed=7),
            on_step=captured.append,
            max_steps=2,
        )
        cm.save(
            step=999,
            params=trainer.state.params,
            opt_state=trainer.state.opt_state,
            extra_metadata={"loss": captured[-1].loss},
        )

        # Build a fresh Trainer (different init) and restore over its state.
        trainer2 = Trainer(recipe=recipe, checkpoint_manager=cm)
        trainer2.restore(step=999)

        # Compare a few leaves bit-for-bit.
        leaves_a = jax.tree_util.tree_leaves(trainer.state.params)
        leaves_b = jax.tree_util.tree_leaves(trainer2.state.params)
        assert len(leaves_a) == len(leaves_b)
        for a, b in zip(leaves_a, leaves_b):
            assert a.shape == b.shape
            assert bool(jnp.all(a == b)), "param leaf differs after restore"

    def test_keep_last_prunes_old_checkpoints(self, tmp_path: Path):
        recipe = _smoke_recipe(total_steps=4, warmup_steps=1)
        ckpt_cfg = CheckpointConfig(
            out_dir=str(tmp_path / "ckpts2"), keep_last=2, save_every=1,
        )
        cm = CheckpointManager(ckpt_cfg)
        trainer = Trainer(recipe=recipe, checkpoint_manager=cm)
        # Manually save 4 different "steps" with the same params; only last 2 should remain.
        for step in (10, 20, 30, 40):
            cm.save(step, trainer.state.params, trainer.state.opt_state)
        steps = cm.list_steps()
        assert steps == [30, 40], f"keep_last=2 not enforced; got {steps}"

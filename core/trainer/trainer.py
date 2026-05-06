"""
Canonical pretraining trainer for `core.model_factory.CapibaraMoEModel`.

Public API:
    trainer = Trainer(recipe, model=None, params=None)
    trainer.fit(data_iter, eval_iter=None, on_step=None, max_steps=None)

Where `recipe` is a `core.trainer.config_loader.Recipe` and `data_iter`
yields `{"input_ids": (B, T) int32, "targets": (B, T) int32}` batches.
A `synthetic_data_iter(...)` helper lives in `scripts/train.py` for smoke
runs without a real corpus.

Design choices and why they matter for MoE training:

* The forward pass returns `ModelOutput(logits, aux_loss)`. `aux_loss`
  is ALREADY pre-multiplied by `cfg.load_balance_weight` inside the
  model (see core/model_factory/model.py). The trainer therefore adds
  `out.aux_loss` directly to the cross-entropy loss WITHOUT a second
  multiplication. Forgetting this is the BACKLOG-008 -> BACKLOG-009
  hand-off bug we explicitly avoid here.

* `loss_fn` returns `(total, (ce, aux))` with `has_aux=True` so the
  jitted forward pass is reused exactly once via `value_and_grad` -
  no second pass to compute logging metrics.

* Each step splits the PRNG with `jax.random.fold_in(base_key, step)`.
  The router consumes `rngs={"router": ...}`; we also pass a `dropout`
  key so future dropout layers do not need re-jitting.

* `grad_norm` is computed from the un-clipped gradient tree before
  optax applies its global-norm clip - that way the metric reflects
  the raw signal, not the post-clip residual.

* The trainer is fully synchronous and side-effect-free except for
  optional checkpoint hooks and the user-supplied `on_step` callback.
  `wandb` is NOT a dependency of this file; the launcher in
  scripts/train.py wraps it as an optional `on_step` callback.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, Optional

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from core.model_factory.model import build_model, init_params, CapibaraMoEModel
from core.model_factory.config import ModelConfig

from .config_loader import Recipe, TrainingConfig
from .optimizer import build_optimizer
from .checkpointing import CheckpointManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# state + metrics
# ---------------------------------------------------------------------------


class TrainState(train_state.TrainState):
    """Standard Flax TrainState - kept as a named subclass for forward extension."""


@dataclass
class StepMetrics:
    """Metrics emitted by a single train_step. All values are python floats."""
    step: int
    loss: float
    ce_loss: float
    aux_loss: float
    grad_norm: float
    lr: float
    seconds_per_step: float
    tokens_per_second: float


# ---------------------------------------------------------------------------
# loss + train_step
# ---------------------------------------------------------------------------


def _ce_loss_with_smoothing(
    logits: jnp.ndarray,           # (B, T, V) fp32
    targets: jnp.ndarray,          # (B, T) int32
    label_smoothing: float,
) -> jnp.ndarray:
    """Per-sequence-mean cross entropy, optionally with label smoothing."""
    if label_smoothing == 0.0:
        per_token = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    else:
        # Smooth labels: confidence on true class = 1 - eps, else eps / (V-1).
        V = logits.shape[-1]
        eps = label_smoothing
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        # nll for the true class
        nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
        smoothed = -log_probs.mean(axis=-1)
        per_token = (1.0 - eps) * nll + eps * smoothed
    return per_token.mean()


def make_train_step(
    apply_fn: Callable,
    label_smoothing: float,
):
    """Build a jitted train_step bound to a model.apply function."""

    def loss_fn(params, batch, rng):
        out = apply_fn(
            {"params": params},
            batch["input_ids"],
            deterministic=False,
            rngs={"router": rng, "dropout": rng},
        )
        ce = _ce_loss_with_smoothing(out.logits, batch["targets"], label_smoothing)
        # aux_loss is already weighted by cfg.load_balance_weight inside the model.
        total = ce + out.aux_loss
        return total, (ce, out.aux_loss)

    @jax.jit
    def train_step(state: TrainState, batch: Dict[str, jnp.ndarray], rng: jnp.ndarray):
        (total, (ce, aux)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, batch, rng,
        )
        grad_norm = optax.global_norm(grads)
        new_state = state.apply_gradients(grads=grads)
        metrics = {
            "loss": total,
            "ce_loss": ce,
            "aux_loss": aux,
            "grad_norm": grad_norm,
        }
        return new_state, metrics

    return train_step


def make_eval_step(apply_fn: Callable, label_smoothing: float):
    def eval_loss(params, batch, rng):
        out = apply_fn(
            {"params": params}, batch["input_ids"],
            deterministic=True, rngs={"router": rng},
        )
        ce = _ce_loss_with_smoothing(out.logits, batch["targets"], label_smoothing)
        return ce, out.aux_loss

    @jax.jit
    def eval_step(state: TrainState, batch: Dict[str, jnp.ndarray], rng: jnp.ndarray):
        ce, aux = eval_loss(state.params, batch, rng)
        return {"eval_ce": ce, "eval_aux": aux, "eval_ppl": jnp.exp(ce)}

    return eval_step


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """End-to-end training loop driver. Keeps Recipe + state + checkpointing."""

    def __init__(
        self,
        recipe: Recipe,
        model: Optional[CapibaraMoEModel] = None,
        params: Optional[Any] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        self.recipe = recipe
        self.cfg_model: ModelConfig = recipe.model
        self.cfg_train: TrainingConfig = recipe.training

        self.model = model if model is not None else build_model(self.cfg_model)
        self.tx, self.lr_schedule = build_optimizer(self.cfg_train)

        if params is None:
            params = init_params(
                self.model, self.cfg_model,
                seed=self.cfg_train.seed,
                batch_size=2,
                seq_len=min(8, self.cfg_model.max_seq_len),
            )
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.tx,
        )

        self.ckpt_mgr = checkpoint_manager  # may be None for tests
        self._train_step = make_train_step(
            self.model.apply, self.cfg_train.label_smoothing,
        )
        self._eval_step = make_eval_step(
            self.model.apply, self.cfg_train.label_smoothing,
        )
        self._base_rng = jax.random.PRNGKey(self.cfg_train.seed)

    # ------------------------------------------------------------------
    # public driver
    # ------------------------------------------------------------------

    def fit(
        self,
        data_iter: Iterable[Dict[str, jnp.ndarray]],
        eval_iter: Optional[Iterable[Dict[str, jnp.ndarray]]] = None,
        on_step: Optional[Callable[[StepMetrics], None]] = None,
        max_steps: Optional[int] = None,
    ) -> TrainState:
        """Run the training loop.

        Args:
            data_iter: yields batches with keys 'input_ids' and 'targets'.
            eval_iter: optional held-out iterator; runs every recipe.logging.eval_every.
            on_step: callback invoked with StepMetrics after each train_step.
                     Use this from scripts/train.py to wire wandb / progress bars.
            max_steps: hard cap; if None, uses recipe.training.total_steps.

        Returns the final TrainState.
        """
        steps_target = max_steps if max_steps is not None else self.cfg_train.total_steps
        log_every = self.recipe.logging.log_every
        eval_every = self.recipe.logging.eval_every
        save_every = self.recipe.checkpoint.save_every

        it = iter(data_iter)
        for raw_step in range(steps_target):
            try:
                batch = next(it)
            except StopIteration:
                logger.info("data iterator exhausted at step %d", raw_step)
                break

            t0 = time.time()
            step_rng = jax.random.fold_in(self._base_rng, raw_step)
            self.state, raw_metrics = self._train_step(self.state, batch, step_rng)
            # Block_until_ready on the loss to get accurate wall-clock per step.
            _ = float(raw_metrics["loss"])
            dt = time.time() - t0

            current_step = int(self.state.step)
            B, T = batch["input_ids"].shape
            metrics = StepMetrics(
                step=current_step,
                loss=float(raw_metrics["loss"]),
                ce_loss=float(raw_metrics["ce_loss"]),
                aux_loss=float(raw_metrics["aux_loss"]),
                grad_norm=float(raw_metrics["grad_norm"]),
                lr=float(self.lr_schedule(current_step - 1)),
                seconds_per_step=dt,
                tokens_per_second=(B * T) / dt if dt > 0 else 0.0,
            )

            if on_step is not None:
                on_step(metrics)

            if current_step % log_every == 0:
                logger.info(
                    "step=%d loss=%.4f ce=%.4f aux=%.4f gn=%.3f lr=%.2e tok/s=%.0f",
                    metrics.step, metrics.loss, metrics.ce_loss, metrics.aux_loss,
                    metrics.grad_norm, metrics.lr, metrics.tokens_per_second,
                )

            if eval_iter is not None and current_step % eval_every == 0 and current_step > 0:
                self._run_eval(eval_iter, current_step)

            if (
                self.ckpt_mgr is not None
                and current_step % save_every == 0
                and current_step > 0
            ):
                self.ckpt_mgr.save(
                    current_step,
                    params=self.state.params,
                    opt_state=self.state.opt_state,
                    extra_metadata={"loss": metrics.loss, "lr": metrics.lr},
                )

        return self.state

    # ------------------------------------------------------------------
    # eval + checkpointing helpers
    # ------------------------------------------------------------------

    def _run_eval(
        self,
        eval_iter: Iterable[Dict[str, jnp.ndarray]],
        step: int,
    ) -> Dict[str, float]:
        """Run a single pass over `eval_iter` and log mean metrics."""
        ce_acc, aux_acc, n = 0.0, 0.0, 0
        for batch in eval_iter:
            rng = jax.random.fold_in(self._base_rng, 1_000_000_000 + step + n)
            m = self._eval_step(self.state, batch, rng)
            ce_acc += float(m["eval_ce"])
            aux_acc += float(m["eval_aux"])
            n += 1
        if n == 0:
            return {}
        ce_mean = ce_acc / n
        out = {"eval_ce": ce_mean, "eval_aux": aux_acc / n, "eval_ppl": float(jnp.exp(ce_mean))}
        logger.info(
            "[eval] step=%d ce=%.4f ppl=%.2f aux=%.4f (n=%d)",
            step, out["eval_ce"], out["eval_ppl"], out["eval_aux"], n,
        )
        return out

    def restore(self, step: Optional[int] = None) -> None:
        """Replace state with the params/opt_state from a saved checkpoint."""
        if self.ckpt_mgr is None:
            raise RuntimeError("Trainer has no CheckpointManager; pass one to __init__")
        params, opt_state, _meta = self.ckpt_mgr.restore(step=step)
        self.state = self.state.replace(params=params, opt_state=opt_state)


__all__ = [
    "Trainer",
    "TrainState",
    "StepMetrics",
    "make_train_step",
    "make_eval_step",
]

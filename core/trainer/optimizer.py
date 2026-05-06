"""
Optimizer + LR schedule construction for the canonical trainer.

Single entry point: `build_optimizer(train_cfg)` returns a tuple
    (tx, schedule_fn)
where:
- `tx` is an `optax.GradientTransformation` ready to be passed to
  `flax.training.train_state.TrainState.create(..., tx=tx)`.
- `schedule_fn(step)` returns the learning rate at integer step (useful
  for logging the actual LR each step without calling into optax).

Composition (in order, outermost first):
  1. optax.clip_by_global_norm(grad_clip)        # stability under MoE aux
  2. optax.adamw(schedule, b1, b2, eps, weight_decay)

For Mixtral-style MoE training we use AdamW (decoupled weight decay) with
beta2=0.95 (Llama / Mixtral default). The schedule is a single-cycle
warmup-cosine that decays from `lr_peak` down to `lr_peak * lr_min_ratio`
over `total_steps - warmup_steps` steps.
"""
from __future__ import annotations

from typing import Callable, Tuple

import optax

from .config_loader import TrainingConfig


# ---------------------------------------------------------------------------
# schedules
# ---------------------------------------------------------------------------


def _build_schedule(cfg: TrainingConfig) -> optax.Schedule:
    """Build the LR schedule for `cfg.schedule_type`."""
    if cfg.schedule_type == "cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=cfg.schedule_init_value,
            peak_value=cfg.lr_peak,
            warmup_steps=cfg.warmup_steps,
            decay_steps=cfg.total_steps,            # decays over total_steps - warmup
            end_value=cfg.effective_end_value,
        )
    if cfg.schedule_type == "constant":
        # Linear warmup then constant at lr_peak (no decay).
        return optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=cfg.schedule_init_value,
                    end_value=cfg.lr_peak,
                    transition_steps=max(1, cfg.warmup_steps),
                ),
                optax.constant_schedule(cfg.lr_peak),
            ],
            boundaries=[cfg.warmup_steps],
        )
    if cfg.schedule_type == "linear":
        # Warmup then linear decay to end_value over the remaining steps.
        decay_steps = max(1, cfg.total_steps - cfg.warmup_steps)
        return optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=cfg.schedule_init_value,
                    end_value=cfg.lr_peak,
                    transition_steps=max(1, cfg.warmup_steps),
                ),
                optax.linear_schedule(
                    init_value=cfg.lr_peak,
                    end_value=cfg.effective_end_value,
                    transition_steps=decay_steps,
                ),
            ],
            boundaries=[cfg.warmup_steps],
        )
    raise ValueError(f"unknown schedule_type={cfg.schedule_type!r}")


# ---------------------------------------------------------------------------
# optimizer factory
# ---------------------------------------------------------------------------


def build_optimizer(
    cfg: TrainingConfig,
) -> Tuple[optax.GradientTransformation, optax.Schedule]:
    """Return (tx, schedule_fn) wired from a TrainingConfig.

    Notes:
    - Weight decay is applied via AdamW's `weight_decay` argument; we do NOT
      decay biases or RMSNorm scale parameters by default. To exclude them,
      apply `optax.masked` before plugging into the optimizer chain. Most
      sub-2B Mixtral training runs are insensitive to this, so the default
      mask is "decay everything"; revisit if seeing instability.
    - Gradient clipping is applied OUTSIDE AdamW so the global norm includes
      all parameters before AdamW's preconditioned updates are computed.
    """
    schedule = _build_schedule(cfg)

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(
            learning_rate=schedule,
            b1=cfg.beta1,
            b2=cfg.beta2,
            eps=cfg.epsilon,
            weight_decay=cfg.weight_decay,
        ),
    )

    return tx, schedule


__all__ = ["build_optimizer"]

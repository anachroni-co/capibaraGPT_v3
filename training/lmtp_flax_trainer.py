"""training/lmtp_flax_trainer.py

Two-stage L-MTP trainer for Capibara Slim using JAX + Flax + Optax.

Stage 1 — head warm-up
    Backbone parameters frozen; only LMTPHeads trained.
    Helps heads adapt to the backbone's representation before joint tuning.

Stage 2 — full tuning
    All parameters (backbone + heads) trained jointly with a combined loss:
      L_total = L_ntp + λ × Σ_i L_head_i

Usage
-----
    from models.lmtp_flax import create_lmtp_model
    from training.lmtp_flax_trainer import LMTPTrainConfig, train_lmtp

    model = create_lmtp_model("1.5b", n_head=4, leap_k=2)
    cfg   = LMTPTrainConfig(warmup_steps=2000, full_steps=8000, batch_size=32)
    state = train_lmtp(model, train_dataset, cfg)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    _JAX = True
except ImportError:
    _JAX = False

try:
    import optax
    _OPTAX = True
except ImportError:
    _OPTAX = False

try:
    import flax
    from flax.training import train_state
    _FLAX = True
except ImportError:
    _FLAX = False

_AVAILABLE = _JAX and _OPTAX and _FLAX


def _require(name: str = "train_lmtp") -> None:
    if not _AVAILABLE:
        missing = [p for p, ok in [("jax", _JAX), ("optax", _OPTAX), ("flax", _FLAX)] if not ok]
        raise ImportError(
            f"{name} requires {', '.join(missing)}. "
            f"Install: pip install jax[cpu] optax flax"
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LMTPTrainConfig:
    """Training hyper-parameters for two-stage L-MTP.

    Attributes:
        warmup_steps:    Steps for head-only warm-up (stage 1).
        full_steps:      Steps for full joint training (stage 2).
        batch_size:      Sequences per batch.
        seq_len:         Tokens per sequence.
        learning_rate:   Peak learning rate (Adam).
        weight_decay:    AdamW weight decay.
        grad_clip:       Global gradient norm clip.
        lmtp_weight:     λ — weight of L-MTP auxiliary loss relative to NTP.
        warmup_lr_steps: Linear LR warmup inside each stage (steps).
        log_interval:    Log every N steps.
        eval_interval:   Evaluate every N steps (0 = disabled).
        seed:            JAX PRNG seed.
    """
    warmup_steps: int = 2_000
    full_steps: int = 8_000
    batch_size: int = 32
    seq_len: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    lmtp_weight: float = 1.0
    warmup_lr_steps: int = 200
    log_interval: int = 50
    eval_interval: int = 500
    seed: int = 0


# ---------------------------------------------------------------------------
# TrainState
# ---------------------------------------------------------------------------

if _AVAILABLE:
    class LMTPTrainState(train_state.TrainState):
        """Extended TrainState that carries the backbone's frozen param mask."""
        backbone_frozen: bool = False


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

if _AVAILABLE:

    def _cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray,
                       mask: jnp.ndarray) -> jnp.ndarray:
        """Masked cross-entropy. logits (B,T,V), targets (B,T), mask (B,T)."""
        B, T, V = logits.shape
        log_probs = jax.nn.log_softmax(logits, axis=-1)   # (B, T, V)
        label_lp = log_probs[
            jnp.arange(B)[:, None],
            jnp.arange(T)[None, :],
            targets,
        ]                                                  # (B, T)
        n_valid = jnp.maximum(mask.sum(), 1.0)
        return -(label_lp * mask).sum() / n_valid

    def compute_lmtp_loss(
        ntp_logits: jnp.ndarray,           # (B, T, V)
        lmtp_logits: list,                 # list of n_head (B, T, V)
        token_ids_ext: jnp.ndarray,        # (B, T + max_offset) extended ids
        mask: jnp.ndarray,                 # (B, T)
        leap_k: int,
        lmtp_weight: float = 1.0,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Combined NTP + L-MTP loss.

        Returns:
            total_loss, {"ntp": ..., "lmtp": ..., "per_head": [...]}
        """
        T = mask.shape[1]
        # NTP: target = next token
        ntp_targets = token_ids_ext[:, 1: T + 1]
        ntp_loss = _cross_entropy(ntp_logits, ntp_targets, mask)

        head_losses = []
        for i, logits in enumerate(lmtp_logits):
            offset = (i + 1) * leap_k
            if offset >= token_ids_ext.shape[1] - T + 1:
                continue
            targets_i = jnp.clip(
                token_ids_ext[:, offset: offset + T],
                0, logits.shape[-1] - 1,
            )
            head_losses.append(_cross_entropy(logits, targets_i, mask))

        lmtp_loss = sum(head_losses) if head_losses else jnp.zeros(())
        total = ntp_loss + lmtp_weight * lmtp_loss

        metrics = {
            "ntp": ntp_loss,
            "lmtp": lmtp_loss,
            "total": total,
            "per_head": head_losses,
        }
        return total, metrics


# ---------------------------------------------------------------------------
# Train step builders
# ---------------------------------------------------------------------------

if _AVAILABLE:

    def _make_optimizer(cfg: LMTPTrainConfig, total_steps: int) -> optax.GradientTransformation:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.learning_rate,
            warmup_steps=cfg.warmup_lr_steps,
            decay_steps=total_steps,
            end_value=cfg.learning_rate * 0.1,
        )
        return optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.adamw(schedule, weight_decay=cfg.weight_decay),
        )

    def _heads_only_mask(params: Dict) -> Dict:
        """Return a param mask that is True only for head parameters."""
        def _mask(path, _):
            return "heads" in "/".join(str(k) for k in path)
        return jax.tree_util.tree_map_with_path(_mask, params)

    def make_warmup_train_step(model, cfg: LMTPTrainConfig):
        """Stage 1: train only heads (backbone gradients zeroed out)."""

        @partial(jax.jit)
        def step(state, batch):
            ids_ext, mask = batch

            def loss_fn(params):
                ntp_logits, lmtp_logits, _ = model.apply(
                    params, ids_ext[:, :cfg.seq_len], deterministic=False,
                    rngs={"dropout": jax.random.PRNGKey(0)},
                )
                loss, metrics = compute_lmtp_loss(
                    ntp_logits, lmtp_logits, ids_ext, mask,
                    model.cfg.leap_k, cfg.lmtp_weight,
                )
                return loss, metrics

            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

            # Zero backbone grads — only update heads
            def zero_backbone(path, g):
                return g if "heads" in "/".join(str(k) for k in path) else jnp.zeros_like(g)

            grads = jax.tree_util.tree_map_with_path(zero_backbone, grads)
            new_state = state.apply_gradients(grads=grads)
            return new_state, metrics

        return step

    def make_full_train_step(model, cfg: LMTPTrainConfig):
        """Stage 2: train all parameters jointly."""

        @jax.jit
        def step(state, batch):
            ids_ext, mask = batch

            def loss_fn(params):
                ntp_logits, lmtp_logits, _ = model.apply(
                    params, ids_ext[:, :cfg.seq_len], deterministic=False,
                    rngs={"dropout": jax.random.PRNGKey(0)},
                )
                loss, metrics = compute_lmtp_loss(
                    ntp_logits, lmtp_logits, ids_ext, mask,
                    model.cfg.leap_k, cfg.lmtp_weight,
                )
                return loss, metrics

            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state, metrics

        return step


# ---------------------------------------------------------------------------
# Public training entry point
# ---------------------------------------------------------------------------

def train_lmtp(
    model: "LMTPWrapper",
    data_iter: Iterator,
    cfg: LMTPTrainConfig,
    init_params: Optional[Dict] = None,
) -> "LMTPTrainState":
    """Run two-stage L-MTP training.

    Args:
        model:       LMTPWrapper (Flax module).
        data_iter:   Iterator yielding (ids_ext, mask) batches.
                     ids_ext shape: (B, seq_len + n_head*leap_k)
                     mask shape:    (B, seq_len)
        cfg:         LMTPTrainConfig.
        init_params: Pre-initialised params dict (optional); if None, params
                     are initialised from a dummy batch.

    Returns:
        Final LMTPTrainState after both stages.
    """
    _require("train_lmtp")

    key = jax.random.PRNGKey(cfg.seed)

    # ── Initialise params ──────────────────────────────────────────────────
    if init_params is None:
        dummy = jnp.ones((1, cfg.seq_len), dtype=jnp.int32)
        key, sub = jax.random.split(key)
        init_params = model.init(sub, dummy)
        logger.info("Params initialised: %d", sum(
            x.size for x in jax.tree_util.tree_leaves(init_params)
        ))

    # ── Stage 1 — head warm-up ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 1 — L-MTP head warm-up  (%d steps)", cfg.warmup_steps)
    logger.info("=" * 60)

    opt1 = _make_optimizer(cfg, cfg.warmup_steps)
    state = LMTPTrainState.create(
        apply_fn=model.apply,
        params=init_params,
        tx=opt1,
    )
    warmup_step_fn = make_warmup_train_step(model, cfg)

    t0 = time.perf_counter()
    for step in range(1, cfg.warmup_steps + 1):
        batch = next(data_iter)
        state, metrics = warmup_step_fn(state, batch)
        if step % cfg.log_interval == 0 or step == 1:
            dt = time.perf_counter() - t0
            tps = step * cfg.batch_size * cfg.seq_len / dt
            logger.info(
                "warm-up %5d/%d | ntp=%.4f lmtp=%.4f | %.0f tok/s",
                step, cfg.warmup_steps,
                float(metrics["ntp"]), float(metrics["lmtp"]), tps,
            )

    # ── Stage 2 — full tuning ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 2 — Full tuning  (%d steps)", cfg.full_steps)
    logger.info("=" * 60)

    opt2 = _make_optimizer(cfg, cfg.full_steps)
    state = LMTPTrainState.create(
        apply_fn=model.apply,
        params=state.params,
        tx=opt2,
    )
    full_step_fn = make_full_train_step(model, cfg)

    t0 = time.perf_counter()
    for step in range(1, cfg.full_steps + 1):
        batch = next(data_iter)
        state, metrics = full_step_fn(state, batch)
        if step % cfg.log_interval == 0 or step == 1:
            dt = time.perf_counter() - t0
            tps = step * cfg.batch_size * cfg.seq_len / dt
            logger.info(
                "full  %5d/%d | ntp=%.4f lmtp=%.4f total=%.4f | %.0f tok/s",
                step, cfg.full_steps,
                float(metrics["ntp"]), float(metrics["lmtp"]),
                float(metrics["total"]), tps,
            )

    logger.info("Training complete.")
    return state


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: "LMTPTrainState", path: str) -> None:
    """Serialise train state to disk (requires orbax-checkpoint)."""
    try:
        import orbax.checkpoint as ocp
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(path, state)
        logger.info("Checkpoint saved → %s", path)
    except ImportError:
        import pickle
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Checkpoint saved (pickle) → %s", path)


def load_checkpoint(path: str) -> Dict:
    """Load params dict from a checkpoint."""
    try:
        import orbax.checkpoint as ocp
        checkpointer = ocp.PyTreeCheckpointer()
        return checkpointer.restore(path)
    except ImportError:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)  # nosec B301 - trusted local checkpoint, not user input


__all__ = [
    "LMTPTrainConfig",
    "train_lmtp",
    "save_checkpoint",
    "load_checkpoint",
    "compute_lmtp_loss",
    "_AVAILABLE",
]

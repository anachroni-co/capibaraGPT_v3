#!/usr/bin/env python3
"""
Canonical CapibaraGPT pretraining launcher (BACKLOG-009).

Usage:
    python scripts/train.py --config configs/1b.toml
    python scripts/train.py --config configs/3b.toml --steps 1000
    python scripts/train.py --config configs/1b.toml --resume checkpoints/capibara-moe-1b
    python scripts/train.py --config configs/1b.toml --synthetic-data --steps 50

This is the launcher that wires:
    Recipe (TOML) -> CapibaraMoEModel -> Trainer -> data iter -> checkpoints

For the data path: until BACKLOG-010 lands the real tokenized data loader,
this launcher accepts `--synthetic-data` to drive smoke runs with random
token ids; the real loader will replace `_synthetic_iter` and read from
the streaming dataset configured in [data] of the TOML.

W&B is optional and disabled unless `--enable-wandb` is passed.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterator, Dict, Any, Optional

# Make the repo root importable regardless of where this is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp

from core.model_factory import build_model
from core.trainer import (
    load_recipe, Recipe, Trainer, CheckpointManager, StepMetrics,
)


logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# data iterators
# ---------------------------------------------------------------------------


def _synthetic_iter(
    recipe: Recipe,
    num_batches: Optional[int] = None,
    seed: int = 0,
) -> Iterator[Dict[str, Any]]:
    """Random token ids in [0, vocab_size). Targets are inputs shifted by 1.

    For BACKLOG-009 smoke. Real tokenized streaming data lands in BACKLOG-010.
    """
    key = jax.random.PRNGKey(seed)
    bs = recipe.training.per_device_batch_size
    seq = recipe.training.seq_len
    V = recipe.model.vocab_size
    i = 0
    while num_batches is None or i < num_batches:
        key, sk = jax.random.split(key)
        ids = jax.random.randint(sk, (bs, seq), 0, V, dtype=jnp.int32)
        tgt = jnp.roll(ids, -1, axis=-1)
        yield {"input_ids": ids, "targets": tgt}
        i += 1


# ---------------------------------------------------------------------------
# wandb hook (optional)
# ---------------------------------------------------------------------------


def _make_wandb_hook(recipe: Recipe):
    """Return an on_step callback that logs to wandb, or None on import failure."""
    try:
        import wandb                              # noqa: WPS433
    except ImportError:
        logger.warning("wandb not installed; --enable-wandb has no effect")
        return None

    log_cfg = recipe.logging
    run = wandb.init(
        project=log_cfg.wandb_project or "capibara-moe",
        name=log_cfg.wandb_run_name or recipe.model.name,
        config={
            "model": recipe.model.__dict__,
            "training": recipe.training.__dict__,
        },
    )

    def hook(m: StepMetrics) -> None:
        run.log({
            "loss": m.loss,
            "ce_loss": m.ce_loss,
            "aux_loss": m.aux_loss,
            "grad_norm": m.grad_norm,
            "lr": m.lr,
            "tokens_per_second": m.tokens_per_second,
            "seconds_per_step": m.seconds_per_step,
        }, step=m.step)

    return hook


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CapibaraGPT pretraining launcher")
    parser.add_argument("--config", required=True, type=Path,
                        help="path to a recipe TOML, e.g. configs/1b.toml")
    parser.add_argument("--steps", type=int, default=None,
                        help="override [training].total_steps")
    parser.add_argument("--resume", type=Path, default=None,
                        help="path to a checkpoint directory; load latest step")
    parser.add_argument("--synthetic-data", action="store_true",
                        help="use random tokens instead of the (not-yet-wired) real loader")
    parser.add_argument("--enable-wandb", action="store_true",
                        help="log metrics to Weights & Biases (default: off)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if not args.config.exists():
        logger.error("recipe not found: %s", args.config)
        return 2

    recipe = load_recipe(args.config)
    logger.info("loaded recipe %s (model.d_model=%d, total_steps=%d)",
                recipe.name, recipe.model.d_model, recipe.training.total_steps)

    # Checkpoint manager (always; the trainer uses it for save/restore).
    if args.resume is not None:
        ckpt_dir = args.resume
    else:
        ckpt_dir = Path(recipe.checkpoint.out_dir)
    ckpt_cfg = type(recipe.checkpoint)(   # rebuild with possibly-overridden out_dir
        out_dir=str(ckpt_dir),
        keep_last=recipe.checkpoint.keep_last,
        save_every=recipe.checkpoint.save_every,
    )
    ckpt_mgr = CheckpointManager(ckpt_cfg)

    model = build_model(recipe.model)
    trainer = Trainer(recipe=recipe, model=model, checkpoint_manager=ckpt_mgr)

    if args.resume is not None and ckpt_mgr.latest_step() is not None:
        trainer.restore()
        logger.info("resumed from step %d under %s", ckpt_mgr.latest_step(), ckpt_dir)

    on_step = _make_wandb_hook(recipe) if args.enable_wandb else None

    # Data iterator. In synthetic mode we stop at total_steps (or --steps).
    n_batches = args.steps if args.steps is not None else recipe.training.total_steps
    if args.synthetic_data:
        data_iter = _synthetic_iter(recipe, num_batches=n_batches, seed=recipe.training.seed)
    else:
        logger.error(
            "real data loader not wired yet (BACKLOG-010). "
            "Pass --synthetic-data to run a smoke pass with random tokens."
        )
        return 3

    trainer.fit(
        data_iter=data_iter,
        on_step=on_step,
        max_steps=args.steps,
    )
    logger.info("training complete; final step=%d", int(trainer.state.step))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

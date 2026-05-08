#!/usr/bin/env python3
"""Model Soup — average weights of N checkpoints to reduce variance.

Based on: Wortsman et al. "Model soups: averaging weights of multiple
fine-tuned models improves accuracy without increasing inference time"
(ICML 2022). https://arxiv.org/abs/2203.05482

Three modes:
  uniform  — average the last N checkpoints unconditionally (default)
  greedy   — add each checkpoint only if it lowers validation loss
  slerp    — spherical interpolation between exactly 2 checkpoints

The soup is saved as a standard checkpoint dict and is compatible with
launch_axion_training.py / launch_tpu_training.py checkpoint format.

Usage:
    # Uniform soup of last 3 checkpoints (recommended after each run)
    python scripts/soup_checkpoints.py checkpoints/axion_mixed/

    # Use last 5 checkpoints
    python scripts/soup_checkpoints.py checkpoints/axion_mixed/ --n 5

    # Greedy soup (needs a validation shard)
    python scripts/soup_checkpoints.py checkpoints/axion_mixed/ \\
        --mode greedy --val-dir data/tokenized/gl/

    # SLERP between two specific checkpoints (t=0.5 = midpoint)
    python scripts/soup_checkpoints.py checkpoints/axion_mixed/ \\
        --mode slerp --t 0.5 \\
        --slerp-a checkpoints/axion_mixed/ckpt_step_0004000.pkl \\
        --slerp-b checkpoints/axion_mixed/ckpt_step_0005000.pkl
"""
from __future__ import annotations

import argparse
import logging
import math
import pickle
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("soup")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _save(path: Path, params, step: int, meta: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump({"step": step, "params": params, **meta}, f)
    logger.info("Saved → %s", path)


def _tree_mean(*params_list):
    import jax
    n = len(params_list)
    return jax.tree_util.tree_map(lambda *xs: sum(xs) / n, *params_list)


def _val_loss(params, val_dir: str, seq_len: int, batch_size: int) -> float:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.data_loader import ShardDataLoader, DataLoaderConfig

    loader = ShardDataLoader(DataLoaderConfig(
        data_dir=val_dir, batch_size=batch_size,
        seq_len=seq_len, shuffle_shards=False,
    ))
    losses = []
    for i, batch in enumerate(loader):
        if i >= 50:
            break
        logits = params["apply_fn"](params["params"], batch["input_ids"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch["labels"]
        ).mean()
        losses.append(float(loss))
    return float(sum(losses) / len(losses)) if losses else float("inf")


# ── Soup modes ────────────────────────────────────────────────────────────────

def uniform_soup(ckpt_paths: list[Path]) -> tuple:
    logger.info("Uniform soup over %d checkpoints:", len(ckpt_paths))
    params_list, last_step = [], 0
    for p in ckpt_paths:
        d = _load(p)
        params_list.append(d["params"])
        last_step = max(last_step, d.get("step", 0))
        logger.info("  %s  (step %d)", p.name, d.get("step", 0))
    soup = _tree_mean(*params_list)
    return soup, last_step


def greedy_soup(ckpt_paths: list[Path], val_dir: str,
                seq_len: int = 512, batch_size: int = 8) -> tuple:
    import jax
    logger.info("Greedy soup — evaluating %d candidates on %s", len(ckpt_paths), val_dir)

    best_params = None
    best_loss = float("inf")
    last_step = 0

    for p in ckpt_paths:
        d = _load(p)
        candidate = d["params"]
        step = d.get("step", 0)

        if best_params is None:
            current_soup = candidate
        else:
            current_soup = _tree_mean(best_params, candidate)

        loss = _val_loss({"params": current_soup, "apply_fn": None}, val_dir, seq_len, batch_size)
        logger.info("  %s step=%d | val_loss=%.4f %s",
                    p.name, step, loss,
                    "✓ added" if loss < best_loss else "✗ skipped")

        if loss < best_loss:
            best_loss = loss
            best_params = current_soup
            last_step = step

    logger.info("Greedy soup final val_loss=%.4f", best_loss)
    return best_params, last_step


def slerp_soup(path_a: Path, path_b: Path, t: float) -> tuple:
    import jax
    import jax.numpy as jnp

    logger.info("SLERP t=%.3f between:", t)
    logger.info("  A: %s", path_a.name)
    logger.info("  B: %s", path_b.name)

    da, db = _load(path_a), _load(path_b)
    pa, pb = da["params"], db["params"]
    last_step = max(da.get("step", 0), db.get("step", 0))

    def _slerp_leaf(a, b):
        a_f = a.astype(jnp.float32).ravel()
        b_f = b.astype(jnp.float32).ravel()
        dot = jnp.clip(jnp.dot(a_f, b_f) /
                       (jnp.linalg.norm(a_f) * jnp.linalg.norm(b_f) + 1e-8),
                       -1.0, 1.0)
        omega = jnp.arccos(dot)
        # Fall back to lerp when angle is near 0 (parallel vectors)
        slerp = jnp.where(
            jnp.abs(omega) < 1e-6,
            (1 - t) * a_f + t * b_f,
            (jnp.sin((1 - t) * omega) * a_f + jnp.sin(t * omega) * b_f)
            / (jnp.sin(omega) + 1e-8),
        )
        return slerp.reshape(a.shape).astype(a.dtype)

    soup = jax.tree_util.tree_map(_slerp_leaf, pa, pb)
    return soup, last_step


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("ckpt_dir", help="Directory containing ckpt_step_*.pkl files")
    parser.add_argument("--mode", choices=["uniform", "greedy", "slerp"],
                        default="uniform", help="Soup mode (default: uniform)")
    parser.add_argument("--n", type=int, default=3,
                        help="Number of last checkpoints to use (uniform/greedy, default: 3)")
    parser.add_argument("--out", default=None,
                        help="Output path (default: <ckpt_dir>/soup_<mode>.pkl)")

    # Greedy options
    parser.add_argument("--val-dir", default=None,
                        help="Validation shard directory (required for greedy mode)")
    parser.add_argument("--val-seq-len",   type=int, default=512)
    parser.add_argument("--val-batch-size", type=int, default=8)

    # SLERP options
    parser.add_argument("--slerp-a", default=None, help="First checkpoint for SLERP")
    parser.add_argument("--slerp-b", default=None, help="Second checkpoint for SLERP")
    parser.add_argument("--t", type=float, default=0.5,
                        help="Interpolation factor for SLERP (0=A, 1=B, default: 0.5)")

    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        logger.error("Directory not found: %s", ckpt_dir)
        sys.exit(1)

    out_path = Path(args.out) if args.out else ckpt_dir / f"soup_{args.mode}.pkl"

    if args.mode == "uniform":
        ckpts = sorted(ckpt_dir.glob("ckpt_step_*.pkl"))[-args.n:]
        if not ckpts:
            logger.error("No checkpoints found in %s", ckpt_dir)
            sys.exit(1)
        soup, step = uniform_soup(ckpts)
        _save(out_path, soup, step, {"type": "soup_uniform", "n": len(ckpts)})

    elif args.mode == "greedy":
        if not args.val_dir:
            logger.error("--val-dir required for greedy mode")
            sys.exit(1)
        ckpts = sorted(ckpt_dir.glob("ckpt_step_*.pkl"))[-args.n:]
        soup, step = greedy_soup(ckpts, args.val_dir, args.val_seq_len, args.val_batch_size)
        _save(out_path, soup, step, {"type": "soup_greedy", "val_dir": args.val_dir})

    elif args.mode == "slerp":
        if not args.slerp_a or not args.slerp_b:
            logger.error("--slerp-a and --slerp-b required for slerp mode")
            sys.exit(1)
        soup, step = slerp_soup(Path(args.slerp_a), Path(args.slerp_b), args.t)
        _save(out_path, soup, step, {"type": "soup_slerp", "t": args.t})

    logger.info("Done.")


if __name__ == "__main__":
    main()

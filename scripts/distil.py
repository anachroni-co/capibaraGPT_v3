#!/usr/bin/env python3
"""Knowledge distillation — large teacher → smaller student.

Implements temperature-scaled KL + cross-entropy distillation
(Hinton et al. 2015). The teacher runs frozen; only student params
are updated. Compatible with all capibara-slim checkpoints.

Loss:
    L = α · T² · KL(softmax(z_t/T) ‖ log_softmax(z_s/T))
      + (1-α) · CE(z_s, labels)

where z_t = teacher logits, z_s = student logits, T = temperature.

Recommended settings:
  T=4  α=0.7  → strong knowledge transfer (use when teacher >> student)
  T=2  α=0.5  → balanced (use when models are close in size)

Usage:
    # Distil large legal (474M) → medium (114M)
    python scripts/distil.py \\
        --teacher   checkpoints/axion_large_legal/soup_uniform.pkl \\
        --student-preset medium \\
        --student-resume checkpoints/axion_medium_bf16/soup_uniform.pkl \\
        --data-dir  data/tokenized/legal/ \\
        --output    checkpoints/distil_medium_legal/ \\
        --steps 10000

    # Distil medium (114M) → small (34M)
    python scripts/distil.py \\
        --teacher   checkpoints/distil_medium_legal/soup_uniform.pkl \\
        --student-preset small \\
        --student-resume checkpoints/axion_small_phase2/soup_uniform.pkl \\
        --data-dir  data/tokenized/legal/ \\
        --output    checkpoints/distil_small_legal/ \\
        --steps 10000
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("distil")

sys.path.insert(0, str(Path(__file__).parent.parent))

PRESETS = {
    "smoke":  dict(hidden_size=256,  num_layers=4,  num_heads=4,  seq_len=256),
    "small":  dict(hidden_size=512,  num_layers=8,  num_heads=8,  seq_len=512),
    "medium": dict(hidden_size=768,  num_layers=12, num_heads=12, seq_len=1024),
    "full":   dict(hidden_size=1024, num_layers=12, num_heads=16, seq_len=2048),
    "large":  dict(hidden_size=1280, num_layers=24, num_heads=20, seq_len=1024),
}


def _load_ckpt(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_ckpt(path: Path, params, step: int, loss: float, meta: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump({"step": step, "loss": loss, "params": params, **meta}, f)
    logger.info("Saved → %s (loss=%.4f)", path.name, loss)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Teacher
    parser.add_argument("--teacher", required=True,
                        help="Teacher checkpoint .pkl (large model, frozen)")
    parser.add_argument("--teacher-preset", default="large",
                        choices=list(PRESETS),
                        help="Teacher architecture preset (default: large)")

    # Student
    parser.add_argument("--student-preset", required=True,
                        choices=list(PRESETS),
                        help="Student architecture preset")
    parser.add_argument("--student-resume", default=None,
                        help="Student checkpoint to start from (recommended: its soup)")

    # Data
    parser.add_argument("--data-dir", required=True,
                        help="Tokenized shard directory")

    # Distillation hyper-parameters
    parser.add_argument("--temperature", "-T", type=float, default=4.0,
                        help="Softmax temperature for soft targets (default: 4.0)")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weight of KL loss vs CE loss (default: 0.7). "
                             "0=pure CE, 1=pure KL")

    # Training
    parser.add_argument("--steps",      type=int,   default=10000)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--grad-accum", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--dtype",      choices=["float32", "bf16"], default="bf16")
    parser.add_argument("--threads",    type=int,   default=32)
    parser.add_argument("--seed",       type=int,   default=42)

    # Output
    parser.add_argument("--output",           default="checkpoints/distil")
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--log-steps",        type=int, default=10)

    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────

    import os
    threads = args.threads
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)
    os.environ["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen=true "
        f"intra_op_parallelism_threads={threads} "
        f"--xla_cpu_enable_fast_math=true "
        f"--xla_force_host_platform_device_count=1"
    )
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("OMP_PROC_BIND", "spread")
    os.environ.setdefault("OMP_PLACES",    "cores")

    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state
    from models.slim_200m import Slim200M, ModelConfig, count_params
    from training.data_loader import ShardDataLoader, DataLoaderConfig

    use_bf16 = args.dtype == "bf16"
    t_preset = PRESETS[args.teacher_preset]
    s_preset = PRESETS[args.student_preset]

    # ── Teacher (frozen) ──────────────────────────────────────────────────────

    logger.info("Loading teacher: %s (%s)", args.teacher, args.teacher_preset)
    t_ckpt = _load_ckpt(args.teacher)
    t_cfg = ModelConfig(
        vocab_size=512,
        hidden_size=t_preset["hidden_size"],
        num_layers=t_preset["num_layers"],
        num_heads=t_preset["num_heads"],
        max_seq_len=t_preset["seq_len"],
        dropout_rate=0.0,
    )
    teacher = Slim200M(t_cfg)
    t_params = t_ckpt["params"]
    logger.info("Teacher params: %.1fM", count_params(t_params) / 1e6)

    # ── Student ───────────────────────────────────────────────────────────────

    logger.info("Building student: %s", args.student_preset)
    s_cfg = ModelConfig(
        vocab_size=512,
        hidden_size=s_preset["hidden_size"],
        num_layers=s_preset["num_layers"],
        num_heads=s_preset["num_heads"],
        max_seq_len=s_preset["seq_len"],
        dropout_rate=0.0,
    )
    student = Slim200M(s_cfg)
    rng = jax.random.PRNGKey(args.seed)
    dummy = jnp.ones((1, min(32, s_preset["seq_len"])), dtype=jnp.int32)
    s_params = jax.jit(student.init)(rng, dummy)
    logger.info("Student params: %.1fM", count_params(s_params) / 1e6)

    if args.student_resume:
        logger.info("Resuming student from %s", args.student_resume)
        s_ckpt = _load_ckpt(args.student_resume)
        s_params = s_ckpt["params"]

    # ── Optimizer ─────────────────────────────────────────────────────────────

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=args.lr,
        warmup_steps=args.warmup_steps,
        decay_steps=args.steps,
        end_value=args.lr * 0.05,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=0.01),
    )
    if args.grad_accum > 1:
        optimizer = optax.MultiSteps(optimizer, every_k_schedule=args.grad_accum)

    state = train_state.TrainState.create(
        apply_fn=student.apply,
        params=s_params,
        tx=optimizer,
    )

    # ── Data ──────────────────────────────────────────────────────────────────

    seq_len = min(s_preset["seq_len"], t_preset["seq_len"])
    loader = ShardDataLoader(DataLoaderConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seq_len=seq_len,
        shuffle_shards=True,
        seed=args.seed,
    ))
    data_iter = iter(loader)

    # ── JIT steps ─────────────────────────────────────────────────────────────

    T = args.temperature
    alpha = args.alpha

    @jax.jit
    def teacher_forward(t_params, input_ids):
        """Teacher inference — no gradient."""
        fwd = t_params
        if use_bf16:
            fwd = jax.tree_util.tree_map(
                lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x,
                t_params,
            )
        logits = teacher.apply(fwd, input_ids)
        return logits.astype(jnp.float32)

    @jax.jit
    def train_step(state, batch, t_logits):
        def loss_fn(params):
            fwd = params
            if use_bf16:
                fwd = jax.tree_util.tree_map(
                    lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x,
                    params,
                )
            s_logits = state.apply_fn(fwd, batch["input_ids"]).astype(jnp.float32)

            # KL divergence with temperature scaling
            t_probs    = jax.nn.softmax(t_logits / T)
            s_log_probs = jax.nn.log_softmax(s_logits / T)
            kl_loss = -(t_probs * s_log_probs).sum(-1).mean() * (T ** 2)

            # Standard cross-entropy on hard labels
            ce_loss = optax.softmax_cross_entropy_with_integer_labels(
                s_logits, batch["labels"]
            ).mean()

            return alpha * kl_loss + (1.0 - alpha) * ce_loss, (kl_loss, ce_loss)

        (loss, (kl, ce)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, kl, ce

    # ── Output + checkpoints ──────────────────────────────────────────────────

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(step: int, loss: float) -> None:
        path = output_dir / f"ckpt_step_{step:07d}.pkl"
        _save_ckpt(path, state.params, step, loss, {
            "type": "distil",
            "teacher_preset": args.teacher_preset,
            "student_preset": args.student_preset,
            "temperature": T,
            "alpha": alpha,
        })
        for old in sorted(output_dir.glob("ckpt_step_*.pkl"))[:-args.keep_checkpoints]:
            old.unlink()

    # ── Training loop ─────────────────────────────────────────────────────────

    logger.info("=" * 60)
    logger.info("Distillation: %s (%.1fM) → %s (%.1fM)",
                args.teacher_preset, count_params(t_params) / 1e6,
                args.student_preset, count_params(s_params) / 1e6)
    logger.info("T=%.1f  α=%.2f  steps=%d  lr=%g  dtype=%s",
                T, alpha, args.steps, args.lr, args.dtype)
    logger.info("=" * 60)

    t0 = time.perf_counter()
    t_log = t0
    recent_losses: list[float] = []

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Teacher forward (no grad — stop_gradient implicit via separate jit)
        t_logits = teacher_forward(t_params, batch["input_ids"])

        state, loss, kl, ce = train_step(state, batch, t_logits)
        loss_val = float(loss)
        recent_losses.append(loss_val)

        if step % args.log_steps == 0:
            now = time.perf_counter()
            step_time = (now - t_log) / args.log_steps
            tok_s = (args.batch_size * seq_len) / step_time
            avg = sum(recent_losses[-args.log_steps:]) / min(len(recent_losses), args.log_steps)
            eta_h = step_time * (args.steps - step) / 3600
            t_log = now
            logger.info(
                "step %6d/%d | loss %.4f (kl %.4f ce %.4f) | %.0f tok/s | ETA %.1fh",
                step, args.steps, avg, float(kl), float(ce), tok_s, eta_h,
            )

        if step % args.checkpoint_steps == 0 or step == args.steps:
            save_checkpoint(step, loss_val)

    total = (time.perf_counter() - t0) / 60
    logger.info("Distillation complete in %.1f min | final loss %.4f",
                total, recent_losses[-1])
    logger.info("Next: python scripts/soup_checkpoints.py %s --n 3", output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Launch Capibara Slim training on Google Axion ARM64 (c4a-standard-32).

Optimised for:  32 vCPU · 128 GB RAM · Neoverse V2 · JAX CPU backend

Presets (--preset):
  smoke   ~3M  params, seq=256  — full loop in seconds, CI/sanity
  small   ~50M params, seq=512  — smoke test overnight
  medium  ~125M params, seq=1024 — multi-day run
  full    ~202M params, seq=2048 — full 200M run (slow on CPU)

Throughput estimates on c4a-standard-32:
  smoke  → ~2 000 tok/s → 5k steps in ~1 min
  small  → ~400  tok/s → 5k steps in ~45 min
  medium → ~120  tok/s → 5k steps in ~6 h
  full   → ~30   tok/s → 5k steps in ~24 h

Usage:
    # Smoke test — verify full loop in ~1 min
    python scripts/launch_axion_training.py \\
        --data-dir data/tokenized/ \\
        --preset smoke --steps 200

    # Small model — overnight run
    python scripts/launch_axion_training.py \\
        --data-dir gs://my-bucket/tokenized/ \\
        --output   gs://my-bucket/checkpoints/axion/ \\
        --preset small --steps 50000

    # Custom config
    python scripts/launch_axion_training.py \\
        --data-dir data/tokenized/ \\
        --hidden-size 512 --num-layers 8 --num-heads 8 \\
        --seq-len 512 --batch-size 32 --steps 5000
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# ── repo root on path ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("axion")

# ── Model presets ─────────────────────────────────────────────────────────────
PRESETS = {
    "smoke":  dict(hidden_size=256, num_layers=4,  num_heads=4,  seq_len=256),
    "small":  dict(hidden_size=512, num_layers=8,  num_heads=8,  seq_len=512),
    "medium": dict(hidden_size=768, num_layers=12, num_heads=12, seq_len=1024),
    "full":   dict(hidden_size=1024, num_layers=12, num_heads=16, seq_len=2048),
}


# ── ARM / Axion thread configuration ─────────────────────────────────────────

def _configure_arm_threads(n_threads: int) -> None:
    """Set environment variables that control CPU parallelism before JAX init."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = str(n_threads)

    # XLA CPU flags for Neoverse V2 / SVE
    xla_flags = (
        "--xla_cpu_multi_thread_eigen=true "
        f"intra_op_parallelism_threads={n_threads} "
        "--xla_cpu_enable_fast_math=true "
        "--xla_force_host_platform_device_count=1"
    )
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["JAX_PLATFORMS"] = "cpu"

    logger.info("ARM threads: %d | XLA_FLAGS set", n_threads)


def _log_system_info() -> None:
    import platform
    import psutil  # type: ignore[import]

    logger.info("Platform  : %s %s", platform.system(), platform.machine())
    logger.info("CPU cores : %d logical / %d physical",
                psutil.cpu_count(logical=True),
                psutil.cpu_count(logical=False) or "?")
    logger.info("RAM       : %.1f GB total / %.1f GB available",
                psutil.virtual_memory().total / 1e9,
                psutil.virtual_memory().available / 1e9)

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line or "Hardware" in line:
                    logger.info("CPU info  : %s", line.strip().split(":", 1)[-1].strip())
                    break
    except Exception:
        pass

    try:
        import jax
        logger.info("JAX       : %s | backend: %s | devices: %d",
                    jax.__version__, jax.default_backend(), len(jax.devices()))
    except Exception as exc:
        logger.warning("JAX not available: %s", exc)


# ── Training loop ─────────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    # 1. Configure threads BEFORE importing JAX
    _configure_arm_threads(args.threads)

    try:
        import jax
        import jax.numpy as jnp
        import optax
        from flax.training import train_state
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        logger.error("Install: pip install jax flax optax")
        sys.exit(1)

    _log_system_info()

    assert jax.default_backend() == "cpu", (
        f"Expected CPU backend, got {jax.default_backend()}. "
        "Set JAX_PLATFORMS=cpu"
    )

    # 2. Model
    from models.slim_200m import Slim200M, ModelConfig, count_params

    model_cfg = ModelConfig(
        vocab_size=512,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
        dropout_rate=0.0,
    )
    model = Slim200M(model_cfg)

    rng = jax.random.PRNGKey(args.seed)
    dummy = jnp.ones((1, min(32, args.seq_len)), dtype=jnp.int32)
    params = jax.jit(model.init)(rng, dummy)
    n_params = count_params(params)
    logger.info("Model: vocab=%d d=%d L=%d H=%d | %.1fM params",
                model_cfg.vocab_size, model_cfg.hidden_size,
                model_cfg.num_layers, model_cfg.num_heads, n_params / 1e6)

    # 3. Optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=args.lr,
        warmup_steps=args.warmup_steps,
        decay_steps=args.steps,
        end_value=args.lr * 0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=0.1),
    )
    if args.grad_accum > 1:
        optimizer = optax.MultiSteps(optimizer, every_k_schedule=args.grad_accum)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    # 4. Data
    from training.data_loader import ShardDataLoader, DataLoaderConfig

    loader = ShardDataLoader(DataLoaderConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        shuffle_shards=True,
        seed=args.seed,
    ))
    data_iter = iter(loader)

    # 5. JIT-compiled train step
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            logits = state.apply_fn(params, batch["input_ids"])  # (B,T,V)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, batch["labels"]
            ).mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    # 6. Checkpoint helpers
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(step: int, loss: float) -> None:
        import pickle
        ckpt = {"step": step, "loss": loss, "params": state.params}
        path = output_dir / f"ckpt_step_{step:07d}.pkl"
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)
        logger.info("Checkpoint saved: %s (loss=%.4f)", path.name, loss)

        # Keep only last N
        all_ckpts = sorted(output_dir.glob("ckpt_step_*.pkl"))
        for old in all_ckpts[:-args.keep_checkpoints]:
            old.unlink()

    # 7. Training loop
    logger.info("=" * 60)
    logger.info("Starting Axion training | %d steps | batch=%d seq=%d accum=%d",
                args.steps, args.batch_size, args.seq_len, args.grad_accum)
    logger.info("Effective batch tokens: %d",
                args.batch_size * args.seq_len * args.grad_accum)
    logger.info("Output: %s", output_dir)
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

        state, loss = train_step(state, batch)
        loss_val = float(loss)
        recent_losses.append(loss_val)

        if step % args.log_steps == 0:
            now = time.perf_counter()
            elapsed = now - t0
            step_time = (now - t_log) / args.log_steps
            tokens_per_sec = (args.batch_size * args.seq_len) / step_time
            avg_loss = sum(recent_losses[-args.log_steps:]) / min(len(recent_losses), args.log_steps)
            t_log = now

            eta_s = step_time * (args.steps - step)
            eta_h = eta_s / 3600

            logger.info(
                "step %6d/%d | loss %.4f | %.0f tok/s | step %.2fs | ETA %.1fh",
                step, args.steps, avg_loss, tokens_per_sec, step_time, eta_h,
            )

        if step % args.checkpoint_steps == 0 or step == args.steps:
            save_checkpoint(step, loss_val)

    total_time = time.perf_counter() - t0
    logger.info("Training complete in %.1f min", total_time / 60)
    logger.info("Final loss: %.4f", recent_losses[-1])


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Preset (overrides model args)
    parser.add_argument("--preset", choices=list(PRESETS), default=None,
                        help="Model size preset (overrides individual model args)")

    # Data
    parser.add_argument("--data-dir", required=True,
                        help="Tokenized shard directory (local or gs://)")
    parser.add_argument("--output", default="checkpoints/axion",
                        help="Checkpoint directory")

    # Model (ignored if --preset is set)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers",  type=int, default=8)
    parser.add_argument("--num-heads",   type=int, default=8)
    parser.add_argument("--seq-len",     type=int, default=512)

    # Training
    parser.add_argument("--steps",            type=int,   default=5000)
    parser.add_argument("--batch-size",       type=int,   default=32)
    parser.add_argument("--grad-accum",       type=int,   default=8)
    parser.add_argument("--lr",               type=float, default=3e-4)
    parser.add_argument("--warmup-steps",     type=int,   default=200)
    parser.add_argument("--seed",             type=int,   default=42)

    # Checkpointing / logging
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--log-steps",        type=int, default=10)

    # Hardware
    parser.add_argument("--threads", type=int, default=32,
                        help="CPU threads (default: 32 for c4a-standard-32)")

    args = parser.parse_args()

    # Apply preset
    if args.preset:
        p = PRESETS[args.preset]
        args.hidden_size = p["hidden_size"]
        args.num_layers  = p["num_layers"]
        args.num_heads   = p["num_heads"]
        args.seq_len     = p["seq_len"]
        logger.info("Preset '%s': d=%d L=%d H=%d seq=%d",
                    args.preset, args.hidden_size, args.num_layers,
                    args.num_heads, args.seq_len)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()

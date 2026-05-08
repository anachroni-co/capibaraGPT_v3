#!/usr/bin/env python3
"""Launch Capibara Slim training on Google Axion ARM64 (c4a-standard-32).

Optimised for:  32 vCPU · 128 GB RAM · Neoverse V2 · JAX CPU backend

Presets (--preset):
  smoke   ~4M   params, seq=256  — full loop in seconds, CI/sanity
  small   ~34M  params, seq=512  — overnight smoke test
  medium  ~114M params, seq=1024 — multi-day run (~29 h bf16)
  full    ~202M params, seq=2048 — full 200M run (~4 d bf16)
  large   ~474M params, seq=1024 — legal specialization (~10 d bf16)

Measured throughput on c4a-standard-32 (bfloat16, single process):
  small  → ~3 400 tok/s  →  5k steps ~6 h
  medium → ~1 500 tok/s  → 10k steps ~29 h
  large  →   ~350 tok/s  → 35k steps ~10 d

Large model — 2-phase training plan:
  Phase 1 (general):  35k steps on mixed corpus (gl+es+pt+c4+code)
  Phase 2 (legal DAPT): 10k steps on legal corpus, lr=5e-5, --resume phase1/

Usage:
    # Smoke test — verify full loop in ~1 min
    python scripts/launch_axion_training.py \\
        --data-dir data/tokenized/ \\
        --preset smoke --steps 200

    # Medium model — bfloat16, ~29 h
    python scripts/launch_axion_training.py \\
        --data-dir data/tokenized/ \\
        --preset medium --batch-size 16 --grad-accum 8 --dtype bf16 --steps 10000

    # Large 474M — Phase 1, bfloat16, ~10 days
    python scripts/launch_axion_training.py \\
        --data-dir data/tokenized/ \\
        --preset large --batch-size 8 --grad-accum 16 --dtype bf16 --steps 35000 \\
        --output checkpoints/axion_large_phase1/

    # Large 474M — Phase 2 (legal DAPT), resume from phase 1 soup
    python scripts/launch_axion_training.py \\
        --data-dir data/tokenized/legal/ \\
        --preset large --batch-size 8 --grad-accum 16 --dtype bf16 --steps 10000 \\
        --lr 5e-5 --output checkpoints/axion_large_legal/ \\
        --resume checkpoints/axion_large_phase1/soup_uniform.pkl
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
    "smoke":  dict(hidden_size=256,  num_layers=4,  num_heads=4,  seq_len=256),
    "small":  dict(hidden_size=512,  num_layers=8,  num_heads=8,  seq_len=512),
    "medium": dict(hidden_size=768,  num_layers=12, num_heads=12, seq_len=1024),
    "full":   dict(hidden_size=1024, num_layers=12, num_heads=16, seq_len=2048),
    # ~474M params — d=1280, L=24, H=20, seq=1024
    # Recommended: --batch-size 8 --grad-accum 16 --dtype bf16
    # Memory: ~30 GB peak | Axion throughput: ~350 tok/s → 10 days @ 35k steps
    "large":  dict(hidden_size=1280, num_layers=24, num_heads=20, seq_len=1024),
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

    use_bf16 = args.dtype == "bf16"
    use_grad_ckpt = args.grad_checkpoint
    logger.info("dtype    : %s (master weights float32, forward %s)",
                args.dtype, "bfloat16" if use_bf16 else "float32")
    logger.info("grad_ckpt: %s%s", use_grad_ckpt,
                " (saves ~60% activation memory, ~20% slower compute)" if use_grad_ckpt else "")

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
    warmup = min(args.warmup_steps, max(1, args.steps // 10))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=args.lr,
        warmup_steps=warmup,
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
    # Master weights stay float32; forward pass optionally runs in bfloat16.
    # Gradient checkpointing (--grad-checkpoint): instead of storing all
    # intermediate activations for backprop, recomputes them on demand.
    # Trades ~20% extra compute for ~60% less activation memory, allowing
    # 2x larger batch on the same RAM → net throughput gain on large models.
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            fwd_params = params
            if use_bf16:
                fwd_params = jax.tree_util.tree_map(
                    lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x,
                    params,
                )
            apply = (jax.checkpoint(state.apply_fn)
                     if use_grad_ckpt else state.apply_fn)
            logits = apply(fwd_params, batch["input_ids"])  # (B,T,V)
            logits = logits.astype(jnp.float32)  # stable softmax cross-entropy
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

    start_step = 0

    # Resume from checkpoint if requested
    if args.resume:
        import pickle
        resume_path = Path(args.resume)
        if not resume_path.exists():
            # Try to auto-find the latest checkpoint in the directory
            candidates = sorted(output_dir.glob("ckpt_step_*.pkl"))
            if not candidates:
                logger.error("No checkpoint found to resume from in %s", output_dir)
                sys.exit(1)
            resume_path = candidates[-1]
            logger.info("Auto-selected latest checkpoint: %s", resume_path.name)

        logger.info("Resuming from %s …", resume_path)
        with open(resume_path, "rb") as f:
            ckpt = pickle.load(f)
        # Restore params into train state (optimizer state is lost — warm restart)
        state = state.replace(params=ckpt["params"])
        start_step = ckpt.get("step", 0)
        logger.info("Resumed at step %d (loss was %.4f)", start_step, ckpt.get("loss", float("nan")))

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
    if start_step:
        logger.info("Continuing from step %d → target step %d",
                    start_step, start_step + args.steps)
    logger.info("Effective batch tokens: %d",
                args.batch_size * args.seq_len * args.grad_accum)
    logger.info("Output: %s", output_dir)
    logger.info("=" * 60)

    t0 = time.perf_counter()
    t_log = t0
    recent_losses: list[float] = []

    for step in range(start_step + 1, start_step + args.steps + 1):
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
            step_time = (now - t_log) / args.log_steps
            tokens_per_sec = (args.batch_size * args.seq_len) / step_time
            avg_loss = sum(recent_losses[-args.log_steps:]) / min(len(recent_losses), args.log_steps)
            t_log = now

            target_step = start_step + args.steps
            remaining = target_step - step
            eta_h = step_time * remaining / 3600

            logger.info(
                "step %6d/%d | loss %.4f | %.0f tok/s | step %.2fs | ETA %.1fh",
                step, target_step, avg_loss, tokens_per_sec, step_time, eta_h,
            )

        target_step = start_step + args.steps
        if step % args.checkpoint_steps == 0 or step == target_step:
            save_checkpoint(step, loss_val)

    total_time = time.perf_counter() - t0
    logger.info("Training complete in %.1f min", total_time / 60)
    logger.info("Steps: %d → %d", start_step, start_step + args.steps)
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
    parser.add_argument("--resume", default=None,
                        help="Path to .pkl checkpoint to resume from "
                             "(or omit path to auto-pick latest in --output dir). "
                             "Restores params; optimizer resets (warm restart).")

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
    parser.add_argument("--dtype", choices=["float32", "bf16"], default="float32",
                        help="Training dtype: float32 (safe) or bf16 (faster, Neoverse V2)")
    parser.add_argument("--grad-checkpoint", action="store_true", default=False,
                        help="Gradient checkpointing: recompute activations during "
                             "backprop instead of storing them. Saves ~60%% activation "
                             "memory at ~20%% compute cost. Recommended for large/full "
                             "presets or when batch size is memory-limited.")

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

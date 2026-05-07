#!/usr/bin/env python3
"""Launch Capibara Slim 200M training on a single TPU VM.

Connects:
  - models/slim_200m.py    → Flax model (~200M params, byte-level)
  - training/data_loader.py → ShardDataLoader over .npy shards
  - training/tpu/tpu_v6e_trainer.py → TPUv6eRobustTrainer (mesh adapted for v5)

Tested on:
  - TPU v5e-4  (1 host, 4 chips, 2×2 mesh)   ← dev / smoke test
  - TPU v5e-8  (1 host, 8 chips, 2×4 mesh)
  - TPU v6e-64 (8 hosts, 64 chips, 8×8 mesh)  ← production

Usage:
    # Smoke test — CPU / 2-chip sim (no real TPU needed)
    python scripts/launch_tpu_training.py \\
        --data-dir  data/tokenized/ \\
        --output    checkpoints/ \\
        --steps     500 \\
        --mesh-rows 1 --mesh-cols 1

    # TPU v5e-4
    python scripts/launch_tpu_training.py \\
        --data-dir  gs://my-bucket/tokenized/ \\
        --output    gs://my-bucket/checkpoints/ \\
        --steps     100000 \\
        --mesh-rows 2 --mesh-cols 2

    # TPU v6e-64
    python scripts/launch_tpu_training.py \\
        --data-dir  gs://my-bucket/tokenized/ \\
        --output    gs://my-bucket/checkpoints/ \\
        --steps     500000 \\
        --mesh-rows 8 --mesh-cols 8
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("launch")

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_deps() -> None:
    missing = []
    try:
        import jax
    except ImportError:
        missing.append("jax[tpu]")
    try:
        import flax
    except ImportError:
        missing.append("flax")
    try:
        import optax
    except ImportError:
        missing.append("optax")
    if missing:
        logger.error("Missing dependencies: %s", ", ".join(missing))
        logger.error("Install with: pip install %s", " ".join(missing))
        sys.exit(1)


def _log_system_info() -> None:
    import jax
    devices = jax.devices()
    logger.info("JAX version  : %s", jax.__version__)
    logger.info("Backend      : %s", jax.default_backend())
    logger.info("Devices      : %d × %s", len(devices), devices[0].device_kind if devices else "?")


def _build_mesh(mesh_rows: int, mesh_cols: int):
    """Build a JAX device mesh for the given topology."""
    import jax
    import numpy as np

    devices = jax.devices()
    n_needed = mesh_rows * mesh_cols
    if len(devices) < n_needed:
        logger.warning(
            "Only %d devices available, need %d. Using all %d devices in 1×%d mesh.",
            len(devices), n_needed, len(devices), len(devices),
        )
        mesh_rows = 1
        mesh_cols = len(devices)
        n_needed = mesh_cols

    devices_array = np.array(devices[:n_needed]).reshape(mesh_rows, mesh_cols)
    mesh = jax.sharding.Mesh(devices_array, ("data", "model"))
    logger.info("Mesh: %d×%d (%d chips)", mesh_rows, mesh_cols, n_needed)
    return mesh


def _patch_trainer_config(trainer, mesh_rows: int, mesh_cols: int) -> None:
    """Override the mesh topology in the trainer config to match our VM."""
    trainer.config.mesh_rows = mesh_rows
    trainer.config.mesh_cols = mesh_cols
    trainer.config.total_chips = mesh_rows * mesh_cols
    trainer.mesh = _build_mesh(mesh_rows, mesh_cols)


# ---------------------------------------------------------------------------
# Main async entrypoint
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    _check_deps()
    _log_system_info()

    import jax
    import jax.numpy as jnp

    # --- Model ---
    from models.slim_200m import Slim200M, ModelConfig, count_params

    model_cfg = ModelConfig(
        vocab_size=512,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
        dropout_rate=0.1,
    )
    model = Slim200M(model_cfg)

    # Count params via a dummy init (on CPU, no sharding)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, min(128, args.seq_len)), dtype=jnp.int32)
    dummy_params = jax.jit(model.init)(rng, dummy)
    n_params = count_params(dummy_params)
    logger.info(
        "Model: Slim200M | vocab=%d d=%d L=%d H=%d | params=%.1fM",
        model_cfg.vocab_size, model_cfg.hidden_size,
        model_cfg.num_layers, model_cfg.num_heads,
        n_params / 1e6,
    )
    del dummy_params  # free before training

    # --- Data ---
    from training.data_loader import ShardDataLoader, DataLoaderConfig

    train_loader = ShardDataLoader(DataLoaderConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        shuffle_shards=True,
        seed=args.seed,
    ))

    val_loader = None
    if args.val_dir:
        val_loader = ShardDataLoader(DataLoaderConfig(
            data_dir=args.val_dir,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            shuffle_shards=False,
        ))

    est = train_loader.steps_per_epoch_estimate()
    if est:
        logger.info("Est. steps/epoch: %d", est)

    # --- Trainer ---
    from training.tpu.tpu_v6e_trainer import TPUv6eRobustTrainer, TPUv6eConfig

    trainer_cfg = TPUv6eConfig(
        mesh_rows=args.mesh_rows,
        mesh_cols=args.mesh_cols,
        total_chips=args.mesh_rows * args.mesh_cols,
        use_bf16=True,
        checkpoint_every_steps=args.checkpoint_steps,
        keep_last_n_checkpoints=3,
        emergency_checkpoint_interval=args.checkpoint_steps // 2,
        use_moe_aux_loss=False,
    )

    trainer = TPUv6eRobustTrainer(
        model_scale="200M",
        base_output_dir=args.output,
        use_wandb=args.wandb,
        config=trainer_cfg,
    )

    # Patch mesh to match actual device count
    _patch_trainer_config(trainer, args.mesh_rows, args.mesh_cols)

    # --- Train ---
    logger.info("Starting training for %d steps", args.steps)
    logger.info("Checkpoint dir : %s", args.output)
    logger.info("W&B            : %s", "enabled" if args.wandb else "disabled")

    await trainer.train(
        model=model,
        train_dataset=iter(train_loader),
        val_dataset=iter(val_loader) if val_loader else None,
        max_steps=args.steps,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument("--data-dir", required=True,
                        help="Tokenized shard directory (local or gs://)")
    parser.add_argument("--val-dir", default=None,
                        help="Validation shard directory (optional)")
    parser.add_argument("--output", default="checkpoints",
                        help="Checkpoint output directory (default: checkpoints)")

    # Model
    parser.add_argument("--hidden-size",  type=int, default=1024)
    parser.add_argument("--num-layers",   type=int, default=18)
    parser.add_argument("--num-heads",    type=int, default=16)
    parser.add_argument("--seq-len",      type=int, default=2048)

    # Training
    parser.add_argument("--steps",            type=int, default=100_000)
    parser.add_argument("--batch-size",       type=int, default=16,
                        help="Per-device batch size")
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--seed",             type=int, default=42)

    # Hardware
    parser.add_argument("--mesh-rows", type=int, default=2,
                        help="TPU mesh rows (default: 2 for v5e-4)")
    parser.add_argument("--mesh-cols", type=int, default=2,
                        help="TPU mesh cols (default: 2 for v5e-4)")

    # Monitoring
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

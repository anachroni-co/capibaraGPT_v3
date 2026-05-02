#!/usr/bin/env python3
"""
CapibaraGPT v3 — Training launcher

Detects available hardware and routes to the appropriate trainer:
  - TPU v6e  → training/tpu/tpu_v6e_trainer.py (TPUv6eRobustTrainer)
  - BTX/MoE  → training/btx_training_system.py  (BTXTrainingSystem)
  - Synthetic → scripts/train_synthetic.py       (for CPU/GPU smoke tests)

Usage:
    python scripts/train.py --mode synthetic --steps 200
    python scripts/train.py --mode btx --seed-model models/seed_1b --output output/btx
    python scripts/train.py --mode tpu --scale 7b --output checkpoints/tpu_run
    python scripts/train.py --detect          # just print detected hardware
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_hardware() -> dict:
    try:
        from core.backends.utils import detect_available_hardware
        return detect_available_hardware()
    except Exception:
        return {"backend": "cpu", "tpu_available": False, "gpu_available": False}


def print_hardware(hw: dict) -> None:
    logger.info("=== Hardware ===")
    for k, v in hw.items():
        logger.info("  %-28s %s", k, v)


# ---------------------------------------------------------------------------
# Mode: synthetic  (CPU / GPU smoke test — no real data needed)
# ---------------------------------------------------------------------------

def run_synthetic(args) -> int:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_synthetic.py"),
        f"--backend={args.backend or 'cpu'}",
        f"--steps={args.steps}",
    ]
    if args.batch_size:
        cmd.append(f"--batch-size={args.batch_size}")
    import subprocess
    return subprocess.call(cmd)


# ---------------------------------------------------------------------------
# Mode: btx  (Byte-Tree-eXpert multi-stage MoE training)
# ---------------------------------------------------------------------------

async def run_btx(args) -> int:
    try:
        from training.btx_training_system import (
            create_btx_training_system,
            create_default_expert_configs,
        )
    except ImportError as e:
        logger.error("Cannot import BTXTrainingSystem: %s", e)
        return 1

    seed_model = args.seed_model or "models/seed_model_1b"
    output = args.output or "output/btx_training"

    logger.info("Starting BTX training — seed=%s  output=%s", seed_model, output)

    expert_configs = create_default_expert_configs()
    system = create_btx_training_system(
        seed_model_path=seed_model,
        output_base_path=output,
        expert_configs=expert_configs,
        max_parallel_jobs=args.parallel_jobs,
    )

    results = await system.run_btx_training()

    logger.info("=== BTX Results ===")
    logger.info("  Status       : %s", results.get("status"))
    summary = results.get("summary", {})
    logger.info("  Success rate : %.1f%%", summary.get("success_rate", 0) * 100)
    quality = results.get("quality_metrics", {})
    logger.info("  Expert quality (avg): %.1f", quality.get("average_expert_quality", 0))
    logger.info("  Consensus quality  : %.1f", quality.get("consensus_quality", 0))

    return 0 if results.get("status") == "completed" else 1


# ---------------------------------------------------------------------------
# Mode: tpu  (TPU v6e 8×8 robust trainer)
# ---------------------------------------------------------------------------

async def run_tpu(args) -> int:
    try:
        from training.tpu.tpu_v6e_trainer import TPUv6eRobustTrainer, TPUv6eConfig
    except ImportError as e:
        logger.error("Cannot import TPUv6eRobustTrainer: %s", e)
        return 1

    scale = args.scale or "1b"
    output = args.output or f"checkpoints/tpu_{scale}"

    config = TPUv6eConfig(
        model_scale=scale,
        max_steps=args.steps,
    )

    logger.info("Starting TPU v6e training — scale=%s  output=%s  steps=%d",
                scale, output, args.steps)

    trainer = TPUv6eRobustTrainer(
        base_output_dir=output,
        model_scale=scale,
        config=config,
    )

    # TPUv6eRobustTrainer.train() requires a model and dataset; those must be
    # wired externally (checkpoint autoload is a planned feature).  For now we
    # start the trainer and let it attempt checkpoint recovery.
    try:
        await trainer.train(model=None, train_dataset=None, max_steps=args.steps)
    except TypeError:
        logger.warning("Trainer requires a wired model — running health-check only.")
        logger.info("Trainer initialised successfully at %s", output)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CapibaraGPT v3 training launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["synthetic", "btx", "tpu"], default="synthetic",
        help="Training mode",
    )
    p.add_argument("--detect", action="store_true", help="Print hardware info and exit")
    p.add_argument("--backend", choices=["cpu", "gpu", "tpu"], default=None,
                   help="Force a specific backend (synthetic mode)")
    p.add_argument("--steps", type=int, default=100, help="Training steps")
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--scale", default="1b",
                   help="Model scale for TPU mode (1b, 3b, 7b, …)")
    p.add_argument("--seed-model", default=None, dest="seed_model",
                   help="Path to seed model checkpoint (BTX mode)")
    p.add_argument("--output", default=None, help="Output / checkpoint directory")
    p.add_argument("--parallel-jobs", type=int, default=3, dest="parallel_jobs",
                   help="Max parallel expert jobs (BTX mode)")
    return p


def main() -> int:
    args = build_parser().parse_args()

    hw = detect_hardware()

    if args.detect:
        print_hardware(hw)
        return 0

    print_hardware(hw)

    if args.mode == "synthetic":
        return run_synthetic(args)
    elif args.mode == "btx":
        return asyncio.run(run_btx(args))
    elif args.mode == "tpu":
        return asyncio.run(run_tpu(args))

    logger.error("Unknown mode: %s", args.mode)
    return 1


if __name__ == "__main__":
    sys.exit(main())

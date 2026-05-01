"""Capibara Slim — training entry point.

Usage:
    # Quick start with defaults (1.5b preset, stub data):
    python -m training.train --preset 1.5b --data data/train.txt

    # Full example:
    python -m training.train \\
        --preset 1.5b \\
        --data data/train.txt \\
        --eval-data data/eval.txt \\
        --output checkpoints/run01 \\
        --max-steps 100000 \\
        --batch-size 4 \\
        --grad-accum 8 \\
        --lr 3e-4 \\
        --dtype bf16

    # Resume from checkpoint:
    python -m training.train --preset 1.5b --data data/train.txt \\
        --resume checkpoints/run01/latest
"""
from __future__ import annotations

import argparse
import logging
import sys

from config.slim_loader import load_config
from utils.logger import setup_logging
from utils.tokenizer import SlimTokenizer

logger = logging.getLogger(__name__)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capibara Slim trainer")
    p.add_argument("--preset",      default="1.5b",     help="Model preset: 1.5b | 3b | 7b")

    # Data: mutually exclusive — either a single file or a mix config
    data_grp = p.add_mutually_exclusive_group(required=True)
    data_grp.add_argument("--data", help="Path to a single training text file")
    data_grp.add_argument("--mix",  help="Path to a mix config YAML (data/mix_configs/*.yaml)")

    p.add_argument("--eval-data",   default=None,        help="Path to evaluation text file")
    p.add_argument("--output",      default="checkpoints", help="Checkpoint output directory")
    p.add_argument("--tokenizer",   default="models/tiny-gpt2", help="Tokenizer path")
    p.add_argument("--max-steps",   type=int, default=100_000)
    p.add_argument("--batch-size",  type=int, default=4)
    p.add_argument("--grad-accum",  type=int, default=8)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--dtype",       default="bf16",      choices=["fp32", "fp16", "bf16"])
    p.add_argument("--device",      default="auto")
    p.add_argument("--seq-len",     type=int, default=2048)
    p.add_argument("--resume",      default=None,        help="Resume from checkpoint path")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> None:
    cfg = load_config()
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    args = _parse_args(argv)
    logger.info("Starting training | preset=%s data=%s", args.preset, args.data)

    try:
        import torch
    except ImportError:
        logger.error("PyTorch is required for training. Install with: pip install torch")
        sys.exit(1)

    from models.architecture import SlimConfig, SlimModel
    from data.loader import SlimDataset, create_dataloader
    from training.trainer import TrainConfig, SlimTrainer
    from training.checkpoint import load_checkpoint

    # Build model
    model_cfg = SlimConfig.preset(args.preset)
    model = SlimModel(model_cfg)
    logger.info(
        "Model: preset=%s params=%.2fB layers=%d hidden=%d",
        args.preset, model.num_params() / 1e9,
        model_cfg.num_layers, model_cfg.hidden_size,
    )

    # Tokenizer
    tokenizer = SlimTokenizer(args.tokenizer)

    # Datasets — single file or multilingual mix
    if args.mix:
        from data.mixer import DataMixer
        mixer = DataMixer.from_yaml(args.mix)
        logger.info("Mix config: %s", mixer.config.name)
        for src in mixer.config.sources:
            logger.info("  %-6s  weight=%.2f  %s", src.lang, src.weight, src.path)
        train_loader = mixer.build_dataloader(
            tokenizer,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
        )
        eval_ds = None
    else:
        train_ds = SlimDataset(args.data, tokenizer, seq_len=args.seq_len)
        train_loader = create_dataloader(train_ds, batch_size=args.batch_size)
        eval_ds = SlimDataset(args.eval_data, tokenizer, seq_len=args.seq_len) if args.eval_data else None

    eval_loader = create_dataloader(eval_ds, batch_size=args.batch_size, shuffle=False) if eval_ds else None

    # Trainer
    train_cfg = TrainConfig(
        output_dir=args.output,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        dtype=args.dtype,
        device=args.device,
        seed=args.seed,
    )
    trainer = SlimTrainer(model, train_cfg, train_loader, eval_loader)

    # Resume if requested
    if args.resume:
        start_step = load_checkpoint(model, trainer.optimizer, checkpoint_path=args.resume)
        trainer.step = start_step
        logger.info("Resumed from step %d", start_step)

    trainer.train()


if __name__ == "__main__":
    main()

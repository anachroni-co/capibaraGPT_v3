#!/usr/bin/env python3
"""
EuroHPC Training Script for CapibaraGPT

Usage:
    # Generate SLURM script for Leonardo (NVIDIA A100)
    python scripts/train_eurohpc.py --system leonardo --nodes 4 --generate-slurm

    # Run training directly (inside SLURM job)
    srun python scripts/train_eurohpc.py --system leonardo --nodes 4 --run

    # Auto-detect system and run
    python scripts/train_eurohpc.py --run

Example SLURM submission:
    python scripts/train_eurohpc.py --system juwels --nodes 8 --model-size 13B --generate-slurm
    sbatch slurm_jobs/capibara_train.sh
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.hpc import (
    EuroHPCConfig,
    EuroHPCSystem,
    EuroHPCTrainer,
    DistributedBackend,
    create_eurohpc_trainer,
    detect_eurohpc_system,
    detect_slurm_environment,
    get_slurm_job_info,
    init_distributed,
    get_rank,
    get_world_size,
    is_main_process,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="CapibaraGPT EuroHPC Training")

    # System configuration
    parser.add_argument(
        "--system",
        type=str,
        default="auto",
        choices=["auto", "lumi", "leonardo", "marenostrum5", "juwels", "karolina", "generic"],
        help="EuroHPC system (auto-detect if not specified)"
    )
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--partition", type=str, default="gpu", help="SLURM partition")
    parser.add_argument("--account", type=str, default=None, help="SLURM account")
    parser.add_argument("--time", type=str, default="24:00:00", help="Time limit")

    # Model configuration
    parser.add_argument(
        "--model-size",
        type=str,
        default="7B",
        choices=["3B", "7B", "13B", "30B", "70B"],
        help="Model size"
    )

    # Training configuration
    parser.add_argument("--batch-size", type=int, default=4, help="Micro batch size per GPU")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=100000, help="Maximum training steps")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Warmup steps")

    # Distributed backend
    parser.add_argument(
        "--backend",
        type=str,
        default="deepspeed",
        choices=["deepspeed", "fsdp", "ddp"],
        help="Distributed backend"
    )
    parser.add_argument("--zero-stage", type=int, default=2, choices=[1, 2, 3], help="ZeRO stage")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offloading")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--save-steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # Output
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name")

    # Actions
    parser.add_argument("--generate-slurm", action="store_true", help="Generate SLURM script only")
    parser.add_argument("--run", action="store_true", help="Run training")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration and exit")

    return parser.parse_args()


def create_config(args) -> EuroHPCConfig:
    """Create EuroHPC configuration from arguments."""

    # Auto-detect system if needed
    if args.system == "auto":
        system = detect_eurohpc_system()
        logger.info(f"Auto-detected system: {system.value}")
    else:
        system = EuroHPCSystem(args.system)

    config = EuroHPCConfig(
        system=system,
        job_name=f"capibara_{args.model_size.lower()}",
        nodes=args.nodes,
        time_limit=args.time,
        partition=args.partition,
        account=args.account,
        model_size=args.model_size,
        micro_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        backend=DistributedBackend(args.backend),
        zero_stage=args.zero_stage,
        cpu_offload=args.cpu_offload,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_steps=args.save_steps,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
    )

    return config


def generate_slurm_script(config: EuroHPCConfig, args):
    """Generate SLURM submission script."""

    # Create output directory
    slurm_dir = Path("slurm_jobs")
    slurm_dir.mkdir(exist_ok=True)

    # Generate script
    trainer = EuroHPCTrainer(config)

    script_path = slurm_dir / f"{config.job_name}.sh"
    training_args = [
        f"--system {config.system.value}",
        f"--nodes {config.nodes}",
        f"--model-size {config.model_size}",
        f"--backend {config.backend.value}",
        f"--batch-size {config.micro_batch_size}",
        f"--grad-accum {config.gradient_accumulation_steps}",
        f"--lr {config.learning_rate}",
        f"--max-steps {config.max_steps}",
        "--run",
    ]

    trainer.generate_slurm_script(
        output_path=str(script_path),
        training_script="scripts/train_eurohpc.py",
        training_args=training_args,
    )

    logger.info(f"SLURM script generated: {script_path}")
    logger.info(f"Submit with: sbatch {script_path}")

    # Also save DeepSpeed config if using DeepSpeed
    if config.backend == DistributedBackend.DEEPSPEED:
        ds_config = config.get_deepspeed_config()
        ds_config_path = Path(config.output_dir) / "deepspeed_config.json"
        ds_config_path.parent.mkdir(parents=True, exist_ok=True)
        ds_config.save(ds_config_path)
        logger.info(f"DeepSpeed config saved: {ds_config_path}")


def run_training(config: EuroHPCConfig, args):
    """Run distributed training."""

    # Print job info if in SLURM
    if detect_slurm_environment():
        job_info = get_slurm_job_info()
        logger.info(f"Running in SLURM job: {job_info}")

    # Initialize trainer
    trainer = EuroHPCTrainer(config)

    if is_main_process():
        logger.info("=" * 60)
        logger.info("CapibaraGPT EuroHPC Training")
        logger.info("=" * 60)
        logger.info(f"System: {config.system.value}")
        logger.info(f"Nodes: {config.nodes}")
        logger.info(f"Total GPUs: {config.total_gpus}")
        logger.info(f"Model size: {config.model_size}")
        logger.info(f"Backend: {config.backend.value}")
        logger.info(f"Global batch size: {config.global_batch_size}")
        logger.info(f"Learning rate: {config.learning_rate}")
        logger.info("=" * 60)

    # Load checkpoint if resuming
    checkpoint = None
    if args.resume:
        checkpoint = trainer.load_checkpoint(args.resume)
        if checkpoint:
            logger.info(f"Resumed from checkpoint: step {checkpoint.get('step', 'unknown')}")
    elif trainer.checkpoint_manager.get_latest_checkpoint():
        checkpoint = trainer.load_checkpoint()
        if checkpoint:
            logger.info(f"Auto-resumed from latest checkpoint: step {checkpoint.get('step', 'unknown')}")

    # TODO: Initialize model, optimizer, dataloader
    # This is where you would add the actual training loop
    # For now, we just demonstrate the infrastructure

    logger.info("Training infrastructure initialized successfully!")
    logger.info("To complete training, implement the training loop with your model.")

    # Example training loop structure:
    """
    model = create_model(config.model_size)
    model = trainer.wrap_model(model)
    optimizer = create_optimizer(model, config.learning_rate)
    dataloader = create_dataloader(config.micro_batch_size)

    start_step = checkpoint.get('step', 0) if checkpoint else 0

    for step, batch in enumerate(dataloader, start=start_step):
        if step >= config.max_steps:
            break

        loss = train_step(model, batch, optimizer)

        if trainer.checkpoint_manager.should_save(step):
            trainer.save_checkpoint(
                step=step,
                epoch=0,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                metrics={'loss': loss},
            )
    """


def main():
    args = parse_args()

    # Create configuration
    config = create_config(args)

    # Dry run - just print config
    if args.dry_run:
        logger.info("Configuration:")
        logger.info(f"  System: {config.system.value}")
        logger.info(f"  Specs: {config.specs}")
        logger.info(f"  Nodes: {config.nodes}")
        logger.info(f"  Total GPUs: {config.total_gpus}")
        logger.info(f"  Model size: {config.model_size}")
        logger.info(f"  Backend: {config.backend.value}")
        logger.info(f"  Global batch size: {config.global_batch_size}")
        return

    # Generate SLURM script
    if args.generate_slurm:
        generate_slurm_script(config, args)
        return

    # Run training
    if args.run:
        run_training(config, args)
        return

    # Default: show help
    logger.info("Use --generate-slurm to create SLURM script, or --run to start training")
    logger.info("Use --dry-run to see configuration without running")


if __name__ == "__main__":
    main()

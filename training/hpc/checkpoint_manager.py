"""
Distributed Checkpoint Manager for HPC Training

Provides robust checkpoint/restart capabilities for:
- Long-running jobs with time limits
- Preemption handling
- Distributed checkpoint saving
- Automatic resume from latest checkpoint
"""

import os
import json
import shutil
import logging
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Union, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# Lazy imports
torch = None
dist = None
TORCH_AVAILABLE = True


def _ensure_torch():
    global torch, dist, TORCH_AVAILABLE
    if torch is None:
        try:
            import torch as _torch
            import torch.distributed as _dist
            torch = _torch
            dist = _dist
        except ImportError:
            TORCH_AVAILABLE = False
            logger.warning("PyTorch not available - checkpoint manager in mock mode")


class CheckpointFormat(Enum):
    """Checkpoint storage formats."""
    PYTORCH = "pytorch"          # Standard torch.save
    SAFETENSORS = "safetensors"  # Hugging Face safetensors
    SHARDED = "sharded"          # Distributed sharded checkpoint


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint manager."""

    # Checkpoint directory
    checkpoint_dir: str = "checkpoints"

    # Checkpoint frequency
    save_every_n_steps: int = 1000
    save_every_n_minutes: int = 30

    # Checkpoint retention
    max_checkpoints: int = 5
    keep_best_n: int = 3

    # Checkpoint format
    format: CheckpointFormat = CheckpointFormat.PYTORCH

    # Distributed settings
    save_on_rank_0_only: bool = True

    # Metadata
    include_metrics: bool = True
    include_training_state: bool = True

    # Validation
    verify_checkpoints: bool = True

    # Async saving (experimental)
    async_save: bool = False

    # Compression
    compress: bool = False


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""

    step: int
    epoch: int
    timestamp: str
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    # Training state
    global_step: int = 0
    learning_rate: float = 0.0

    # Performance
    training_time_seconds: float = 0.0
    tokens_processed: int = 0

    # Validation info
    checksum: Optional[str] = None
    is_best: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        return cls(**data)


class DistributedCheckpointManager:
    """
    Manages distributed checkpoints for HPC training.

    Features:
    - Automatic checkpoint saving on schedule
    - Best checkpoint tracking
    - Checkpoint rotation and cleanup
    - Distributed synchronization
    - Resume from latest/best checkpoint
    - SLURM preemption handling
    """

    def __init__(self, config: CheckpointConfig, rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.checkpoints: List[CheckpointMetadata] = []
        self.best_metric: Optional[float] = None
        self.best_checkpoint: Optional[str] = None
        self.last_save_time = time.time()
        self.last_save_step = 0

        # Load existing checkpoint metadata
        self._load_checkpoint_registry()

        _ensure_torch()

        if not TORCH_AVAILABLE:
            logger.warning("Checkpoint manager running without PyTorch - save/load disabled")

        logger.info(f"Checkpoint manager initialized: dir={self.checkpoint_dir}, "
                   f"rank={rank}/{world_size}")

    def _load_checkpoint_registry(self):
        """Load registry of existing checkpoints."""
        registry_path = self.checkpoint_dir / "checkpoint_registry.json"

        if registry_path.exists():
            try:
                with open(registry_path) as f:
                    data = json.load(f)

                self.checkpoints = [
                    CheckpointMetadata.from_dict(cp) for cp in data.get("checkpoints", [])
                ]
                self.best_checkpoint = data.get("best_checkpoint")
                self.best_metric = data.get("best_metric")

                logger.info(f"Loaded {len(self.checkpoints)} checkpoints from registry")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint registry: {e}")

    def _save_checkpoint_registry(self):
        """Save registry of checkpoints."""
        if self.config.save_on_rank_0_only and self.rank != 0:
            return

        registry_path = self.checkpoint_dir / "checkpoint_registry.json"

        data = {
            "checkpoints": [cp.to_dict() for cp in self.checkpoints],
            "best_checkpoint": self.best_checkpoint,
            "best_metric": self.best_metric,
            "last_updated": datetime.now().isoformat(),
        }

        with open(registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def should_save(self, step: int) -> bool:
        """Check if a checkpoint should be saved."""
        # Check step-based saving
        if step - self.last_save_step >= self.config.save_every_n_steps:
            return True

        # Check time-based saving
        elapsed_minutes = (time.time() - self.last_save_time) / 60
        if elapsed_minutes >= self.config.save_every_n_minutes:
            return True

        return False

    def save(
        self,
        step: int,
        epoch: int,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        scheduler_state: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> Optional[str]:
        """
        Save a checkpoint.

        Args:
            step: Current training step
            epoch: Current epoch
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            scheduler_state: LR scheduler state dict
            metrics: Training metrics
            extra_state: Additional state to save
            is_best: Whether this is the best checkpoint

        Returns:
            Checkpoint path if saved, None otherwise
        """
        # Synchronize across ranks
        if self.world_size > 1:
            dist.barrier()

        # Only rank 0 saves if configured
        if self.config.save_on_rank_0_only and self.rank != 0:
            return None

        # Create checkpoint directory
        checkpoint_name = f"checkpoint_step_{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Prepare checkpoint data
        checkpoint = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
        }

        if scheduler_state:
            checkpoint["scheduler_state_dict"] = scheduler_state

        if extra_state:
            checkpoint["extra_state"] = extra_state

        # Save checkpoint
        checkpoint_file = checkpoint_path / "checkpoint.pt"

        if self.config.format == CheckpointFormat.PYTORCH:
            torch.save(checkpoint, checkpoint_file)
        elif self.config.format == CheckpointFormat.SAFETENSORS:
            self._save_safetensors(checkpoint, checkpoint_path)

        # Compute checksum for verification
        checksum = None
        if self.config.verify_checkpoints:
            checksum = self._compute_checksum(checkpoint_file)

        # Create metadata
        metadata = CheckpointMetadata(
            step=step,
            epoch=epoch,
            timestamp=datetime.now().isoformat(),
            metrics=metrics or {},
            global_step=step,
            checksum=checksum,
            is_best=is_best,
        )

        # Save metadata
        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Update tracking
        self.checkpoints.append(metadata)
        self.last_save_time = time.time()
        self.last_save_step = step

        # Track best checkpoint
        if is_best:
            self.best_checkpoint = str(checkpoint_path)
            if metrics and "loss" in metrics:
                self.best_metric = metrics["loss"]

            # Create symlink to best
            best_link = self.checkpoint_dir / "best"
            if best_link.exists() or best_link.is_symlink():
                best_link.unlink()
            best_link.symlink_to(checkpoint_path.name)

        # Create symlink to latest
        latest_link = self.checkpoint_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(checkpoint_path.name)

        # Save registry
        self._save_checkpoint_registry()

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        logger.info(f"Checkpoint saved: {checkpoint_path} (step={step}, is_best={is_best})")

        return str(checkpoint_path)

    def load(
        self,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False,
        load_latest: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Specific checkpoint to load
            load_best: Load best checkpoint
            load_latest: Load latest checkpoint (default)

        Returns:
            Checkpoint dict or None if not found
        """
        # Determine which checkpoint to load
        if checkpoint_path:
            path = Path(checkpoint_path)
        elif load_best:
            path = self.checkpoint_dir / "best"
        elif load_latest:
            path = self.checkpoint_dir / "latest"
        else:
            return None

        # Resolve symlink
        if path.is_symlink():
            path = path.resolve()

        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return None

        checkpoint_file = path / "checkpoint.pt"

        if not checkpoint_file.exists():
            logger.warning(f"Checkpoint file not found: {checkpoint_file}")
            return None

        # Verify checksum if enabled
        if self.config.verify_checkpoints:
            metadata_file = path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)

                expected_checksum = metadata.get("checksum")
                if expected_checksum:
                    actual_checksum = self._compute_checksum(checkpoint_file)
                    if actual_checksum != expected_checksum:
                        raise ValueError(f"Checkpoint checksum mismatch: {path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_file, map_location="cpu")

        logger.info(f"Checkpoint loaded: {path} (step={checkpoint.get('step', 'unknown')})")

        return checkpoint

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        latest_link = self.checkpoint_dir / "latest"
        if latest_link.exists():
            return str(latest_link.resolve())
        return None

    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        best_link = self.checkpoint_dir / "best"
        if best_link.exists():
            return str(best_link.resolve())
        return self.best_checkpoint

    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256 = hashlib.sha256()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        return sha256.hexdigest()

    def _save_safetensors(self, checkpoint: Dict[str, Any], path: Path):
        """Save checkpoint in safetensors format."""
        try:
            from safetensors.torch import save_file

            # Separate tensors from other state
            tensors = {}
            other_state = {}

            for key, value in checkpoint.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        if hasattr(v, "cpu"):
                            tensors[f"{key}.{k}"] = v.cpu()
                        else:
                            if key not in other_state:
                                other_state[key] = {}
                            other_state[key][k] = v
                elif hasattr(value, "cpu"):
                    tensors[key] = value.cpu()
                else:
                    other_state[key] = value

            # Save tensors
            save_file(tensors, path / "model.safetensors")

            # Save other state
            with open(path / "state.json", "w") as f:
                json.dump(other_state, f)

        except ImportError:
            logger.warning("safetensors not available, falling back to PyTorch format")
            torch.save(checkpoint, path / "checkpoint.pt")

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints exceeding max_checkpoints."""
        if len(self.checkpoints) <= self.config.max_checkpoints:
            return

        # Sort by step (ascending)
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x.step)

        # Identify checkpoints to keep
        best_checkpoints = sorted(
            [cp for cp in sorted_checkpoints if cp.is_best],
            key=lambda x: x.metrics.get("loss", float("inf"))
        )[:self.config.keep_best_n]

        keep_set = set(cp.step for cp in best_checkpoints)

        # Keep most recent
        recent = sorted_checkpoints[-self.config.max_checkpoints:]
        keep_set.update(cp.step for cp in recent)

        # Delete old checkpoints
        for cp in sorted_checkpoints:
            if cp.step not in keep_set:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{cp.step}"
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
                    logger.info(f"Deleted old checkpoint: {checkpoint_path}")

                self.checkpoints.remove(cp)

        self._save_checkpoint_registry()

    def create_preemption_handler(self, save_callback: Callable[[], Dict[str, Any]]):
        """
        Create signal handler for SLURM preemption.

        Args:
            save_callback: Function that returns state dict to save
        """
        import signal

        def handler(signum, frame):
            logger.warning(f"Received preemption signal {signum}, saving checkpoint...")

            try:
                state = save_callback()
                self.save(
                    step=state.get("step", self.last_save_step + 1),
                    epoch=state.get("epoch", 0),
                    model_state=state.get("model_state_dict", {}),
                    optimizer_state=state.get("optimizer_state_dict", {}),
                    scheduler_state=state.get("scheduler_state_dict"),
                    metrics=state.get("metrics"),
                    extra_state={"preempted": True},
                )
                logger.info("Preemption checkpoint saved")
            except Exception as e:
                logger.error(f"Failed to save preemption checkpoint: {e}")

            # Allow normal termination
            signal.default_int_handler(signum, frame)

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGUSR1, handler)

        logger.info("Preemption signal handlers installed")

"""Capibara Slim — checkpointing (T5.4).

Saves and loads model + optimizer state so training can be resumed.

Checkpoint layout:
    <output_dir>/
        step_<N>/
            model.pt       — model state dict
            optimizer.pt   — optimizer state dict
            meta.json      — step, config, timestamp
        latest -> step_<N> (symlink to last checkpoint)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


def save_checkpoint(
    model,
    optimizer,
    step: int,
    output_dir: str,
    config_dict: Optional[dict] = None,
) -> Path:
    if not _TORCH:
        raise ImportError("Checkpointing requires PyTorch.")

    ckpt_dir = Path(output_dir) / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

    meta = {
        "step": step,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config_dict or {},
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Update 'latest' symlink
    latest = Path(output_dir) / "latest"
    if latest.is_symlink():
        latest.unlink()
    latest.symlink_to(ckpt_dir.name)

    logger.info("Checkpoint saved: %s", ckpt_dir)
    return ckpt_dir


def load_checkpoint(
    model,
    optimizer=None,
    checkpoint_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: str = "cpu",
) -> int:
    """Load model (and optionally optimizer) from checkpoint. Returns step."""
    if not _TORCH:
        raise ImportError("Checkpointing requires PyTorch.")

    if checkpoint_path:
        ckpt_dir = Path(checkpoint_path)
    elif output_dir:
        latest = Path(output_dir) / "latest"
        if not latest.exists():
            raise FileNotFoundError(f"No 'latest' checkpoint in {output_dir}")
        ckpt_dir = latest.resolve()
    else:
        raise ValueError("Provide checkpoint_path or output_dir.")

    model.load_state_dict(
        torch.load(ckpt_dir / "model.pt", map_location=device)
    )
    if optimizer is not None and (ckpt_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(
            torch.load(ckpt_dir / "optimizer.pt", map_location=device)
        )

    meta = json.loads((ckpt_dir / "meta.json").read_text())
    step = meta.get("step", 0)
    logger.info("Checkpoint loaded from %s (step %d)", ckpt_dir, step)
    return step


def list_checkpoints(output_dir: str) -> list[Path]:
    """Return all step checkpoint directories, sorted by step."""
    root = Path(output_dir)
    return sorted(root.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))

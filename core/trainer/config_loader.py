"""
Recipe loader for the canonical pretraining trainer (BACKLOG-009).

Single entry point: `load_recipe(path)` parses a TOML file and returns a
`Recipe` dataclass holding four sub-configs:

    Recipe(
        model    : core.model_factory.ModelConfig,    # [model]
        training : TrainingConfig,                    # [training]
        checkpoint: CheckpointConfig,                 # [checkpoint]
        logging  : LoggingConfig,                     # [logging]
    )

The model section is delegated to `ModelConfig.from_toml` (which is the
single source of truth for shape + arch) so we never duplicate parsing
logic across the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

# tomllib is stdlib in 3.11+; tomli is the backport for 3.10.
try:
    import tomllib                          # type: ignore[import-not-found]
except ModuleNotFoundError:                 # pragma: no cover
    import tomli as tomllib                 # type: ignore[no-redef]

from core.model_factory.config import ModelConfig


# ---------------------------------------------------------------------------
# sub-configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """All knobs the trainer needs to drive optimization. Mirrors [training]."""
    seed: int = 42
    seq_len: int = 4096
    batch_size_global: int = 1024
    grad_accum_steps: int = 1
    total_steps: int = 100_000
    warmup_steps: int = 2_000
    lr_peak: float = 3.0e-4
    lr_min_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1.0e-8
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    grad_ckpt_ratio: float = 1.0
    # Schedule kind ("cosine" by default; "constant" or "linear" also accepted).
    schedule_type: str = "cosine"
    schedule_init_value: float = 0.0
    schedule_end_value: float | None = None  # if None, derived as lr_peak * lr_min_ratio

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if self.warmup_steps > self.total_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) > total_steps ({self.total_steps})"
            )
        if not 0.0 < self.lr_peak:
            raise ValueError("lr_peak must be positive")
        if not 0.0 < self.lr_min_ratio <= 1.0:
            raise ValueError("lr_min_ratio must be in (0, 1]")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be > 0")
        if self.batch_size_global <= 0:
            raise ValueError("batch_size_global must be > 0")
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be > 0")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if not 0.0 <= self.grad_ckpt_ratio <= 1.0:
            raise ValueError("grad_ckpt_ratio must be in [0, 1]")
        if self.schedule_type not in ("cosine", "constant", "linear"):
            raise ValueError(
                f"unknown schedule_type={self.schedule_type!r}; "
                "expected one of: cosine, constant, linear"
            )

    @property
    def effective_end_value(self) -> float:
        if self.schedule_end_value is not None:
            return self.schedule_end_value
        return self.lr_peak * self.lr_min_ratio

    @property
    def per_device_batch_size(self) -> int:
        """Per-device batch size, accounting for grad accumulation but not data parallel."""
        if self.batch_size_global % self.grad_accum_steps != 0:
            raise ValueError(
                f"batch_size_global ({self.batch_size_global}) not divisible by "
                f"grad_accum_steps ({self.grad_accum_steps})"
            )
        return self.batch_size_global // self.grad_accum_steps


@dataclass(frozen=True)
class CheckpointConfig:
    """Mirrors [checkpoint]."""
    out_dir: str = "checkpoints/run"
    keep_last: int = 3
    save_every: int = 1_000

    def __post_init__(self) -> None:
        if self.keep_last <= 0:
            raise ValueError("keep_last must be > 0")
        if self.save_every <= 0:
            raise ValueError("save_every must be > 0")


@dataclass(frozen=True)
class LoggingConfig:
    """Mirrors [logging]. wandb_* are optional; a value of None disables W&B."""
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    log_every: int = 50
    eval_every: int = 1_000
    eval_seq_count: int = 256

    def __post_init__(self) -> None:
        if self.log_every <= 0:
            raise ValueError("log_every must be > 0")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be > 0")
        if self.eval_seq_count <= 0:
            raise ValueError("eval_seq_count must be > 0")


# ---------------------------------------------------------------------------
# Recipe (root)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    """A fully resolved training recipe. The single object passed around."""
    model: ModelConfig
    training: TrainingConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.model.name


# ---------------------------------------------------------------------------
# parse helpers
# ---------------------------------------------------------------------------


def _parse_training(raw: Dict[str, Any]) -> TrainingConfig:
    section = dict(raw.get("training", {}))
    sched = section.pop("schedule", None)
    if sched is None:
        return TrainingConfig(
            **{k: v for k, v in section.items() if k in TrainingConfig.__dataclass_fields__}
        )
    # nested [training.schedule] table - flatten relevant fields.
    extras = {
        "schedule_type": sched.get("type", "cosine"),
        "schedule_init_value": float(sched.get("init_value", 0.0)),
    }
    if "end_value" in sched:
        extras["schedule_end_value"] = float(sched["end_value"])
    merged = {
        **{k: v for k, v in section.items() if k in TrainingConfig.__dataclass_fields__},
        **extras,
    }
    return TrainingConfig(**merged)


def _parse_checkpoint(raw: Dict[str, Any]) -> CheckpointConfig:
    section = raw.get("checkpoint", {})
    return CheckpointConfig(
        **{k: v for k, v in section.items() if k in CheckpointConfig.__dataclass_fields__}
    )


def _parse_logging(raw: Dict[str, Any]) -> LoggingConfig:
    section = raw.get("logging", {})
    return LoggingConfig(
        **{k: v for k, v in section.items() if k in LoggingConfig.__dataclass_fields__}
    )


# ---------------------------------------------------------------------------
# public entrypoint
# ---------------------------------------------------------------------------


def load_recipe(path: str | Path) -> Recipe:
    """Load a full training recipe from a single TOML file.

    The TOML must contain at least a [model] section parseable by
    `ModelConfig.from_toml`. [training], [checkpoint], [logging] are all
    optional - their dataclass defaults kick in when absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    model = ModelConfig.from_toml(path)
    train = _parse_training(raw)
    ckpt = _parse_checkpoint(raw)
    log = _parse_logging(raw)

    return Recipe(model=model, training=train, checkpoint=ckpt, logging=log, raw=raw)


__all__ = [
    "TrainingConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "Recipe",
    "load_recipe",
]

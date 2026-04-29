"""Capibara Slim — training loop (T5.3).

Single-node training loop with:
  - AdamW + cosine LR schedule with linear warmup
  - bf16 / fp16 mixed precision (via torch.amp)
  - Gradient clipping
  - Gradient accumulation
  - Periodic evaluation + checkpointing

Usage:
    cfg  = SlimConfig.preset("1.5b")
    model = SlimModel(cfg)
    train_cfg = TrainConfig(
        output_dir="checkpoints/1.5b",
        max_steps=100_000,
        batch_size=4,
        grad_accum=8,          # effective batch = 32
        lr=3e-4,
        warmup_steps=2_000,
        dtype="bf16",
    )
    trainer = SlimTrainer(model, train_cfg, train_loader, eval_loader)
    trainer.train()
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    _TORCH = True
except ImportError:
    _TORCH = False


def _require_torch() -> None:
    if not _TORCH:
        raise ImportError("SlimTrainer requires PyTorch. Install with: pip install torch")


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    output_dir: str = "checkpoints"
    max_steps: int = 100_000
    batch_size: int = 4
    grad_accum: int = 8           # gradient accumulation steps
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2_000
    eval_every: int = 1_000
    save_every: int = 5_000
    log_every: int = 100
    dtype: str = "bf16"           # "fp32", "fp16", "bf16"
    device: str = "auto"          # "auto", "cpu", "cuda", "mps"
    seed: int = 42

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum


# ---------------------------------------------------------------------------
# SlimTrainer
# ---------------------------------------------------------------------------

if _TORCH:
    class SlimTrainer:
        def __init__(
            self,
            model: nn.Module,
            cfg: TrainConfig,
            train_loader,
            eval_loader=None,
        ) -> None:
            _require_torch()
            self.model = model
            self.cfg = cfg
            self.train_loader = train_loader
            self.eval_loader = eval_loader

            self.device = self._resolve_device(cfg.device)
            self.dtype = self._resolve_dtype(cfg.dtype)
            self.scaler = (
                torch.cuda.amp.GradScaler()
                if self.dtype == torch.float16 and self.device.type == "cuda"
                else None
            )

            self.model.to(self.device)
            torch.manual_seed(cfg.seed)

            self.optimizer = AdamW(
                self._param_groups(),
                lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2),
                eps=1e-8,
            )
            self.scheduler = self._build_scheduler()

            self.step = 0
            self._train_iter = iter(self.train_loader)

            logger.info(
                "SlimTrainer ready | device=%s dtype=%s params=%s",
                self.device, cfg.dtype,
                f"{model.num_params() / 1e9:.2f}B" if hasattr(model, "num_params") else "?",
            )

        # ------------------------------------------------------------------
        # Public API
        # ------------------------------------------------------------------

        def train(self) -> None:
            self.model.train()
            t0 = time.monotonic()
            accum_loss = 0.0

            while self.step < self.cfg.max_steps:
                batch = self._next_batch()
                loss = self._forward(batch)
                loss_val = loss.item() / self.cfg.grad_accum
                accum_loss += loss_val
                self._backward(loss)

                if (self.step + 1) % self.cfg.grad_accum == 0:
                    self._optimizer_step()
                    if self.step % self.cfg.log_every == 0:
                        elapsed = time.monotonic() - t0
                        lr_now = self.scheduler.get_last_lr()[0]
                        logger.info(
                            "step=%d loss=%.4f lr=%.2e elapsed=%.1fs",
                            self.step, accum_loss, lr_now, elapsed,
                        )
                        accum_loss = 0.0

                    if self.eval_loader and self.step % self.cfg.eval_every == 0:
                        val_loss = self.evaluate()
                        logger.info("step=%d val_loss=%.4f", self.step, val_loss)
                        self.model.train()

                    if self.step % self.cfg.save_every == 0:
                        from training.checkpoint import save_checkpoint
                        save_checkpoint(self.model, self.optimizer, self.step, self.cfg.output_dir)

                self.step += 1

            logger.info("Training complete at step %d", self.step)

        def evaluate(self) -> float:
            if self.eval_loader is None:
                return float("nan")
            self.model.eval()
            total, n = 0.0, 0
            with torch.no_grad():
                for batch in self.eval_loader:
                    loss = self._forward(batch)
                    total += loss.item()
                    n += 1
            return total / max(n, 1)

        def train_step(self, batch) -> float:
            """Single step — useful for custom loops."""
            loss = self._forward(batch)
            self._backward(loss)
            self._optimizer_step()
            self.step += 1
            return loss.item()

        # ------------------------------------------------------------------
        # Internals
        # ------------------------------------------------------------------

        def _forward(self, batch) -> "torch.Tensor":
            input_ids = batch["input_ids"].to(self.device)
            labels    = batch["labels"].to(self.device)

            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
                logits = self.model(input_ids)
                # (B, L, V) → cross-entropy over vocab
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            return loss / self.cfg.grad_accum

        def _backward(self, loss: "torch.Tensor") -> None:
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        def _optimizer_step(self) -> None:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        def _next_batch(self) -> dict:
            try:
                return next(self._train_iter)
            except StopIteration:
                self._train_iter = iter(self.train_loader)
                return next(self._train_iter)

        def _param_groups(self) -> list[dict]:
            decay = {n for n, p in self.model.named_parameters()
                     if p.dim() >= 2 and p.requires_grad}
            no_decay = {n for n, p in self.model.named_parameters()
                        if p.dim() < 2 and p.requires_grad}
            params = dict(self.model.named_parameters())
            return [
                {"params": [params[n] for n in sorted(decay)],    "weight_decay": self.cfg.weight_decay},
                {"params": [params[n] for n in sorted(no_decay)], "weight_decay": 0.0},
            ]

        def _build_scheduler(self) -> LambdaLR:
            warmup = self.cfg.warmup_steps
            total = self.cfg.max_steps
            min_ratio = self.cfg.min_lr / self.cfg.lr

            def _lr(step: int) -> float:
                if step < warmup:
                    return step / max(warmup, 1)
                progress = (step - warmup) / max(total - warmup, 1)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return max(min_ratio, cosine)

            return LambdaLR(self.optimizer, _lr)

        @staticmethod
        def _resolve_device(device: str) -> "torch.device":
            if device == "auto":
                if torch.cuda.is_available():
                    return torch.device("cuda")
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return torch.device("mps")
                return torch.device("cpu")
            return torch.device(device)

        @staticmethod
        def _resolve_dtype(dtype: str):
            return {"fp32": torch.float32, "fp16": torch.float16,
                    "bf16": torch.bfloat16}.get(dtype, torch.float32)

else:
    class SlimTrainer:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require_torch()


__all__ = ["TrainConfig", "SlimTrainer"]

"""L-MTP two-stage trainer (arXiv:2505.17505, NeurIPS 2025).

Implements the training recipe from Section 3.1 of the paper:

Stage 1 — Head warm-up
    Backbone + original lm_head are frozen.
    Only the new L-MTP heads are trained on self-distillation data
    (the model's own outputs serve as soft supervision targets).
    Loss: cross-entropy at each leap position (Eq. 4).

Stage 2 — Full model tuning
    All parameters are updated (or LoRA adapters if lora_rank > 0).
    Loss: L_NTP + β · L_LMTP  (Eq. 5).

Usage
-----
    from models.lmtp import LMTPConfig, wrap_with_lmtp
    from models.architecture import SlimConfig, SlimModel
    from training.lmtp_trainer import LMTPTrainConfig, LMTPTrainer

    backbone = SlimModel(SlimConfig.preset("1.5b"))
    lmtp_cfg = LMTPConfig(n_head=4, leap_k=2, beta=0.1)
    model = wrap_with_lmtp(backbone, lmtp_cfg)

    train_cfg = LMTPTrainConfig(output_dir="checkpoints/lmtp")
    trainer = LMTPTrainer(model, train_cfg, train_loader)
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
        raise ImportError("LMTPTrainer requires PyTorch. Install with: pip install torch")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LMTPTrainConfig:
    """Training configuration for both L-MTP stages.

    Attributes:
        output_dir:       Checkpoint directory.
        # Stage 1 (head warm-up)
        warmup_steps:     Gradient steps for Stage 1 (heads only, frozen backbone).
        warmup_lr:        Peak learning rate for Stage 1.
        warmup_batch:     Batch size for Stage 1.
        # Stage 2 (full tuning)
        full_steps:       Gradient steps for Stage 2 (all parameters or LoRA).
        full_lr:          Peak learning rate for Stage 2.
        full_batch:       Batch size for Stage 2.
        # Shared
        beta:             Auxiliary L-MTP loss weight (Eq. 5).
        grad_accum:       Gradient accumulation steps.
        grad_clip:        Max gradient norm.
        weight_decay:     AdamW weight decay.
        warmup_ratio:     Cosine warmup fraction (0.1 = 10 % of steps).
        dtype:            Mixed-precision dtype: "fp32", "fp16", "bf16".
        device:           "auto", "cpu", "cuda", "mps".
        eval_every:       Evaluate every N optimiser steps.
        save_every:       Checkpoint every N optimiser steps.
        log_every:        Log every N optimiser steps.
        # LoRA (Stage 2)
        lora_rank:        LoRA rank; 0 = full fine-tuning (no LoRA).
        lora_alpha:       LoRA alpha scaling.
    """
    output_dir: str = "checkpoints/lmtp"
    # Stage 1
    warmup_steps: int = 5_000
    warmup_lr: float = 1e-4
    warmup_batch: int = 4
    # Stage 2
    full_steps: int = 3_000
    full_lr: float = 1e-5
    full_batch: int = 4
    # Shared
    beta: float = 0.1
    grad_accum: int = 4
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    dtype: str = "bf16"
    device: str = "auto"
    eval_every: int = 500
    save_every: int = 1_000
    log_every: int = 50
    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 16
    seed: int = 42


# ---------------------------------------------------------------------------
# Minimal LoRA injection (no external dependency)
# ---------------------------------------------------------------------------

if _TORCH:
    class _LoRALinear(nn.Module):
        """Low-rank adapter injected in-place around an existing nn.Linear.

        y = W x + (B A x) * scale,   where A ∈ R^{rank×in}, B ∈ R^{out×rank}.
        Only A and B are trainable; W is frozen.
        """

        def __init__(self, linear: "nn.Linear", rank: int, alpha: int) -> None:
            super().__init__()
            out, inp = linear.weight.shape
            self.linear = linear
            self.lora_A = nn.Parameter(torch.randn(rank, inp) * 0.02)
            self.lora_B = nn.Parameter(torch.zeros(out, rank))
            self.scale = alpha / rank
            # Freeze original weights
            for p in linear.parameters():
                p.requires_grad_(False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            base = self.linear(x)
            delta = (x @ self.lora_A.T) @ self.lora_B.T
            return base + delta * self.scale

    def _apply_lora(module: "nn.Module", rank: int, alpha: int) -> None:
        """Replace all 2-D Linear layers in module with LoRA-wrapped versions."""
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child.weight.dim() == 2:
                setattr(module, name, _LoRALinear(child, rank, alpha))
            else:
                _apply_lora(child, rank, alpha)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

if _TORCH:
    class LMTPTrainer:
        """Two-stage trainer for L-MTP fine-tuning.

        Wraps an LMTPWrapper and runs:
            Stage 1: head warm-up (backbone frozen, cross-entropy on leap heads)
            Stage 2: full tuning (LoRA or full, combined NTP + L-MTP loss)
        """

        def __init__(
            self,
            model: "nn.Module",    # LMTPWrapper
            cfg: LMTPTrainConfig,
            train_loader,
            eval_loader=None,
        ) -> None:
            _require_torch()
            self.model = model
            self.cfg = cfg
            self.train_loader = train_loader
            self.eval_loader = eval_loader
            self._train_iter = iter(train_loader)

            self.device = self._resolve_device(cfg.device)
            self.dtype = self._resolve_dtype(cfg.dtype)
            self.scaler = (
                torch.cuda.amp.GradScaler()
                if cfg.dtype == "fp16" and self.device.type == "cuda"
                else None
            )

            self.model.to(self.device)
            Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------ #
        # Public entry point
        # ------------------------------------------------------------------ #

        def train(self) -> None:
            """Run Stage 1 then Stage 2."""
            logger.info("L-MTP Stage 1: head warm-up (%d steps)", self.cfg.warmup_steps)
            self._run_stage1()

            logger.info("L-MTP Stage 2: full model tuning (%d steps)", self.cfg.full_steps)
            self._run_stage2()

            logger.info("L-MTP training complete.")

        # ------------------------------------------------------------------ #
        # Stage 1 — head warm-up (backbone frozen)
        # ------------------------------------------------------------------ #

        def _run_stage1(self) -> None:
            # Freeze backbone; only heads are trainable
            self.model.backbone.requires_grad_(False)
            head_params = list(self.model.lmtp_heads.parameters())

            opt = AdamW(head_params, lr=self.cfg.warmup_lr, weight_decay=0.0)
            sched = self._cosine_schedule(opt, self.cfg.warmup_steps, self.cfg.warmup_ratio)

            self.model.train()
            step, accum = 0, 0
            loss_sum = 0.0
            t0 = time.monotonic()

            while step < self.cfg.warmup_steps:
                batch = self._next_batch()
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.autocast(self.device.type, dtype=self.dtype,
                                    enabled=self.dtype != torch.float32):
                    logits, lmtp_logits = self.model(input_ids, return_lmtp_logits=True)
                    loss = self._lmtp_only_loss(lmtp_logits, labels)
                    loss = loss / self.cfg.grad_accum

                self._backward(loss)
                loss_sum += loss.item()
                accum += 1

                if accum == self.cfg.grad_accum:
                    self._step(opt, sched, head_params)
                    accum = 0
                    if step % self.cfg.log_every == 0:
                        elapsed = time.monotonic() - t0
                        logger.info(
                            "[warmup] step=%d  loss=%.4f  elapsed=%.1fs",
                            step, loss_sum, elapsed,
                        )
                        loss_sum = 0.0
                    if step % self.cfg.save_every == 0 and step > 0:
                        self._save(f"stage1_step{step}")
                    step += 1

        # ------------------------------------------------------------------ #
        # Stage 2 — full model tuning
        # ------------------------------------------------------------------ #

        def _run_stage2(self) -> None:
            # Unfreeze backbone
            self.model.backbone.requires_grad_(True)

            if self.cfg.lora_rank > 0:
                # Apply LoRA to backbone only (heads are already small MLPs)
                _apply_lora(self.model.backbone, self.cfg.lora_rank, self.cfg.lora_alpha)
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            else:
                trainable_params = list(self.model.parameters())

            opt = AdamW(trainable_params, lr=self.cfg.full_lr,
                        weight_decay=self.cfg.weight_decay)
            sched = self._cosine_schedule(opt, self.cfg.full_steps, self.cfg.warmup_ratio)

            self.model.train()
            step, accum = 0, 0
            loss_sum = 0.0
            t0 = time.monotonic()

            while step < self.cfg.full_steps:
                batch = self._next_batch()
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.autocast(self.device.type, dtype=self.dtype,
                                    enabled=self.dtype != torch.float32):
                    logits, lmtp_logits = self.model(input_ids, return_lmtp_logits=True)
                    from models.lmtp import compute_lmtp_loss
                    loss = compute_lmtp_loss(
                        logits, lmtp_logits, labels,
                        leap_k=self.model.lmtp_cfg.leap_k,
                        beta=self.cfg.beta,
                    )
                    loss = loss / self.cfg.grad_accum

                self._backward(loss)
                loss_sum += loss.item()
                accum += 1

                if accum == self.cfg.grad_accum:
                    self._step(opt, sched, trainable_params)
                    accum = 0
                    if step % self.cfg.log_every == 0:
                        elapsed = time.monotonic() - t0
                        logger.info(
                            "[full] step=%d  loss=%.4f  elapsed=%.1fs",
                            step, loss_sum, elapsed,
                        )
                        loss_sum = 0.0
                    if self.eval_loader and step % self.cfg.eval_every == 0 and step > 0:
                        val = self._evaluate()
                        logger.info("[full] step=%d  val_loss=%.4f", step, val)
                        self.model.train()
                    if step % self.cfg.save_every == 0 and step > 0:
                        self._save(f"stage2_step{step}")
                    step += 1

            self._save("final")

        # ------------------------------------------------------------------ #
        # Loss helpers
        # ------------------------------------------------------------------ #

        def _lmtp_only_loss(
            self,
            lmtp_logits: list,
            labels: "torch.Tensor",
        ) -> "torch.Tensor":
            """Stage-1 loss: cross-entropy only on the auxiliary heads."""
            vocab_size = lmtp_logits[0].size(-1)
            leap_k = self.model.lmtp_cfg.leap_k
            total = lmtp_logits[0].new_zeros(())
            for i, head_logits in enumerate(lmtp_logits):
                shift = leap_k * (i + 1)
                seq_len = head_logits.shape[1]
                if shift >= seq_len:
                    continue
                h_logits = head_logits[:, :-shift].contiguous()
                h_labels = labels[:, shift:].contiguous()
                total = total + nn.functional.cross_entropy(
                    h_logits.view(-1, vocab_size),
                    h_labels.view(-1),
                    ignore_index=-100,
                )
            return total

        # ------------------------------------------------------------------ #
        # Optimiser helpers
        # ------------------------------------------------------------------ #

        def _backward(self, loss: "torch.Tensor") -> None:
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        def _step(self, opt, sched, params) -> None:
            if self.scaler:
                self.scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
                self.scaler.step(opt)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
                opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

        def _cosine_schedule(self, opt, total_steps: int, warmup_ratio: float) -> "LambdaLR":
            warmup = int(total_steps * warmup_ratio)

            def _lr(step: int) -> float:
                if step < warmup:
                    return step / max(warmup, 1)
                progress = (step - warmup) / max(total_steps - warmup, 1)
                return max(1e-2, 0.5 * (1.0 + math.cos(math.pi * progress)))

            return LambdaLR(opt, _lr)

        # ------------------------------------------------------------------ #
        # Evaluation
        # ------------------------------------------------------------------ #

        def _evaluate(self) -> float:
            self.model.eval()
            total, n = 0.0, 0
            with torch.no_grad():
                for batch in self.eval_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    logits, lmtp_logits = self.model(input_ids, return_lmtp_logits=True)
                    from models.lmtp import compute_lmtp_loss
                    loss = compute_lmtp_loss(
                        logits, lmtp_logits, labels,
                        leap_k=self.model.lmtp_cfg.leap_k,
                        beta=self.cfg.beta,
                    )
                    total += loss.item()
                    n += 1
            return total / max(n, 1)

        # ------------------------------------------------------------------ #
        # Checkpointing
        # ------------------------------------------------------------------ #

        def _save(self, tag: str) -> None:
            path = Path(self.cfg.output_dir) / f"lmtp_{tag}.pt"
            torch.save({
                "lmtp_heads": self.model.lmtp_heads.state_dict(),
                "backbone": self.model.backbone.state_dict(),
            }, path)
            logger.info("Saved checkpoint: %s", path)

        # ------------------------------------------------------------------ #
        # Data iteration
        # ------------------------------------------------------------------ #

        def _next_batch(self) -> dict:
            try:
                return next(self._train_iter)
            except StopIteration:
                self._train_iter = iter(self.train_loader)
                return next(self._train_iter)

        # ------------------------------------------------------------------ #
        # Device / dtype helpers
        # ------------------------------------------------------------------ #

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
    class LMTPTrainer:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            _require_torch()


__all__ = ["LMTPTrainConfig", "LMTPTrainer"]

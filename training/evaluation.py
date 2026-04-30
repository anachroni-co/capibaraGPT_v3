"""Capibara Slim — evaluation metrics (T6.5).

Perplexity:
    Standard language-model quality metric.
    Lower = better. Random baseline for 32k vocab ≈ 32000.
    Well-trained LLMs: 10–30 on held-out text.

Usage (with torch):
    evaluator = Evaluator(model, tokenizer)
    result = evaluator.evaluate(eval_loader)
    print(result)
    # {"perplexity": 24.3, "loss": 3.19, "num_tokens": 51200, "num_batches": 100}

Usage (without torch — no-op for testing):
    Evaluator raises ImportError at construction time.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


@dataclass
class EvalResult:
    perplexity: float
    loss: float
    num_tokens: int
    num_batches: int

    def __str__(self) -> str:
        return (
            f"perplexity={self.perplexity:.2f}  "
            f"loss={self.loss:.4f}  "
            f"tokens={self.num_tokens:,}  "
            f"batches={self.num_batches}"
        )


if _TORCH:
    class Evaluator:
        def __init__(self, model: "nn.Module", device: str = "cpu") -> None:
            self.model = model
            self.device = torch.device(device)
            self.model.to(self.device)

        def evaluate(self, dataloader, max_batches: int = 0) -> EvalResult:
            self.model.eval()
            total_loss = 0.0
            total_tokens = 0
            n_batches = 0

            with torch.no_grad():
                for batch in dataloader:
                    if max_batches and n_batches >= max_batches:
                        break

                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    logits = self.model(input_ids)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        reduction="sum",
                        ignore_index=-100,
                    )
                    valid_tokens = (labels != -100).sum().item()
                    total_loss += loss.item()
                    total_tokens += valid_tokens
                    n_batches += 1

            if total_tokens == 0:
                return EvalResult(perplexity=float("inf"), loss=float("inf"),
                                  num_tokens=0, num_batches=0)

            avg_loss = total_loss / total_tokens
            ppl = math.exp(min(avg_loss, 100))         # cap to avoid overflow
            result = EvalResult(
                perplexity=ppl,
                loss=avg_loss,
                num_tokens=total_tokens,
                num_batches=n_batches,
            )
            logger.info("Eval: %s", result)
            return result

else:
    class Evaluator:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError("Evaluator requires PyTorch.")

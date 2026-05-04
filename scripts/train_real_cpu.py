#!/usr/bin/env python3
"""
scripts/train_real_cpu.py

Minimal byte-level language model trained on the repo's own source code.

Uses ByteLevelTokenizer + ByteLevelDataLoader from training.byte_level_training
for data loading (real corpus: .py and .md files in this repo), and a
pure-NumPy 2-layer model (embedding → ReLU → linear → softmax) with manual
SGD+momentum for training — no JAX, no PyTorch required.

Architecture
    vocab  = 256 bytes + special tokens (262 total)
    model  = Embedding[vocab→H] + Linear[H→vocab]   (bigram-style LM)
    loss   = cross-entropy next-byte prediction
    optim  = SGD with momentum

Run
    python scripts/train_real_cpu.py [--steps N] [--hidden H] [--batch B]
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List

import numpy as np

# Real data loading from the existing infrastructure
from training.byte_level_training import (
    ByteLevelConfig,
    ByteLevelDataLoader,
    ByteLevelTokenizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Minimal NumPy model — 2-layer byte LM with manual backprop
# ---------------------------------------------------------------------------

class ByteLM:
    """Embedding → ReLU → Linear byte-level language model."""

    def __init__(self, vocab: int, hidden: int, lr: float = 0.05, momentum: float = 0.9):
        self.vocab = vocab
        self.H = hidden
        self.lr = lr

        scale = 0.01
        self.W_emb = np.random.randn(vocab, hidden).astype(np.float32) * scale
        self.W_out = np.random.randn(hidden, vocab).astype(np.float32) * scale

        # SGD momentum buffers
        self.v_emb = np.zeros_like(self.W_emb)
        self.v_out = np.zeros_like(self.W_out)
        self.momentum = momentum

    @property
    def num_params(self) -> int:
        return self.W_emb.size + self.W_out.size

    def forward(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        ids: (B, T) int32
        Returns (logits (B, T, vocab), H (B, T, hidden))
        """
        H = self.W_emb[ids]                        # (B, T, hidden)
        H_act = np.maximum(H, 0)                   # ReLU
        logits = H_act @ self.W_out                # (B, T, vocab)
        return logits, H_act

    def loss_and_grad(
        self, ids: np.ndarray, targets: np.ndarray, mask: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        ids, targets, mask: (B, T) int32 / float32
        Returns (scalar_loss, dW_emb, dW_out)
        """
        B, T = ids.shape
        logits, H_act = self.forward(ids)           # (B, T, vocab)

        # Numerically stable softmax
        shift = logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(logits - shift)
        probs = exp_l / exp_l.sum(axis=-1, keepdims=True)  # (B, T, vocab)

        # Cross-entropy loss (masked, mean over non-pad tokens)
        tgt_probs = probs[np.arange(B)[:, None], np.arange(T)[None, :], targets]
        tgt_probs = np.clip(tgt_probs, 1e-9, None)
        loss_per_tok = -np.log(tgt_probs) * mask
        n_valid = mask.sum() + 1e-8
        loss = float(loss_per_tok.sum() / n_valid)

        # Backward
        d_logits = probs.copy()                                    # (B, T, vocab)
        d_logits[np.arange(B)[:, None], np.arange(T)[None, :], targets] -= 1
        d_logits *= mask[..., None] / n_valid                      # scale + mask

        # dW_out: H_act^T @ d_logits  averaged over B,T
        dW_out = (H_act.reshape(-1, self.H).T
                  @ d_logits.reshape(-1, self.vocab))              # (H, vocab)

        # dH_act → dH (through ReLU)
        d_H_act = d_logits @ self.W_out.T                         # (B, T, H)
        d_H = d_H_act * (H_act > 0)                               # ReLU gate

        # dW_emb: scatter-add
        dW_emb = np.zeros_like(self.W_emb)
        np.add.at(dW_emb, ids.flatten(), d_H.reshape(-1, self.H))

        return loss, dW_emb, dW_out

    def step(self, dW_emb: np.ndarray, dW_out: np.ndarray) -> None:
        """SGD with momentum update."""
        self.v_emb = self.momentum * self.v_emb + dW_emb
        self.v_out = self.momentum * self.v_out + dW_out
        self.W_emb -= self.lr * self.v_emb
        self.W_out -= self.lr * self.v_out


# ---------------------------------------------------------------------------
# Data sampling
# ---------------------------------------------------------------------------

def build_corpus(data_dir: Path, extensions: list[str], min_bytes: int = 200) -> np.ndarray:
    """Concatenate all eligible files into a single byte array."""
    cfg = ByteLevelConfig(
        file_extensions=extensions,
        min_file_size_bytes=min_bytes,
        max_file_size_mb=5,
    )
    tokenizer = ByteLevelTokenizer(cfg)
    loader = ByteLevelDataLoader(cfg, tokenizer)

    files = loader.load_files_from_directory(data_dir)
    if not files:
        raise RuntimeError(f"No files found in {data_dir}")

    chunks: List[np.ndarray] = []
    for fp in files:
        raw = loader.load_file_as_bytes(fp)
        if raw is None:
            continue
        chunks.append(np.frombuffer(raw, dtype=np.uint8).astype(np.int32))

    corpus = np.concatenate(chunks)
    logger.info(
        "Corpus: %d files, %d bytes (%.1f MB)",
        len(files), len(corpus), len(corpus) / 1e6,
    )
    return corpus


def sample_batch(
    corpus: np.ndarray, batch_size: int, seq_len: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random windows from the corpus."""
    max_start = len(corpus) - seq_len - 1
    starts = rng.integers(0, max_start, size=batch_size)
    ids = np.stack([corpus[s: s + seq_len] for s in starts])
    targets = np.stack([corpus[s + 1: s + seq_len + 1] for s in starts])
    mask = np.ones((batch_size, seq_len), dtype=np.float32)
    return ids, targets, mask


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    steps: int = 100,
    hidden: int = 128,
    batch_size: int = 8,
    seq_len: int = 256,
    lr: float = 0.05,
    log_every: int = 10,
    data_dirs: list[str] | None = None,
) -> None:
    rng = np.random.default_rng(42)

    # Build corpus from real repo source files
    dirs = data_dirs or [
        str(REPO_ROOT / "core"),
        str(REPO_ROOT / "training"),
        str(REPO_ROOT / "inference"),
        str(REPO_ROOT / "data"),
    ]
    chunks = []
    for d in dirs:
        p = Path(d)
        if p.exists():
            chunks.append(build_corpus(p, [".py", ".md"]))

    corpus = np.concatenate(chunks)
    logger.info("Total corpus: %d bytes (%.2f MB)", len(corpus), len(corpus) / 1e6)

    # vocab = 256 (raw bytes, 0–255); special tokens above that are not in corpus
    vocab = 256
    model = ByteLM(vocab=vocab, hidden=hidden, lr=lr)
    logger.info(
        "Model: vocab=%d hidden=%d params=%d (%.1f KB)",
        vocab, hidden, model.num_params, model.num_params * 4 / 1024,
    )
    logger.info(
        "Training: steps=%d batch=%d seq_len=%d",
        steps, batch_size, seq_len,
    )

    # Baseline: random model loss should be ~log(256) ≈ 5.55 bits/byte
    logger.info("Baseline loss (random model) ≈ %.3f", np.log(vocab))

    losses: list[float] = []
    t0_total = time.perf_counter()

    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        ids, targets, mask = sample_batch(corpus, batch_size, seq_len, rng)

        loss, dW_emb, dW_out = model.loss_and_grad(ids, targets, mask)
        model.step(dW_emb, dW_out)

        losses.append(loss)
        elapsed = time.perf_counter() - t0

        if step % log_every == 0 or step == 1:
            recent = losses[-log_every:]
            avg = sum(recent) / len(recent)
            tokens_per_s = (batch_size * seq_len) / elapsed
            logger.info(
                "step %4d/%d | loss=%.4f | avg(last %d)=%.4f | %.0f tok/s",
                step, steps, loss, len(recent), avg, tokens_per_s,
            )

    total_time = time.perf_counter() - t0_total
    first_loss = losses[0]
    last_avg = sum(losses[-10:]) / min(10, len(losses))
    delta = first_loss - last_avg

    logger.info("=" * 60)
    logger.info("Training complete in %.1fs", total_time)
    logger.info("Initial loss : %.4f", first_loss)
    logger.info("Final avg    : %.4f  (Δ = %.4f)", last_avg, delta)
    logger.info("Throughput   : %.0f tok/s avg", (steps * batch_size * seq_len) / total_time)

    if delta > 0.05:
        logger.info("Loss decreased — model is learning byte patterns from real source code.")
    else:
        logger.info("Loss decrease small — increase --steps or --hidden for more capacity.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Byte-level CPU training on repo source")
    p.add_argument("--steps",   type=int,   default=100,  help="Training steps")
    p.add_argument("--hidden",  type=int,   default=128,  help="Hidden size")
    p.add_argument("--batch",   type=int,   default=8,    help="Batch size (sequences)")
    p.add_argument("--seq-len", type=int,   default=256,  help="Sequence length (bytes)")
    p.add_argument("--lr",      type=float, default=0.05, help="Learning rate")
    p.add_argument("--log-every", type=int, default=10,   help="Log every N steps")
    args = p.parse_args()

    train(
        steps=args.steps,
        hidden=args.hidden,
        batch_size=args.batch,
        seq_len=args.seq_len,
        lr=args.lr,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()

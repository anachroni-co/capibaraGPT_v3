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
    """Embedding → [ReLU → Hidden]* → Linear byte-level language model.

    Supports multiple hidden layers via the `num_layers` parameter.
    Each hidden layer is a square W[H, H] matrix with ReLU activation.
    """

    def __init__(
        self,
        vocab: int,
        hidden: int,
        num_layers: int = 1,
        lr: float = 0.05,
        momentum: float = 0.9,
    ):
        self.vocab = vocab
        self.H = hidden
        self.num_layers = num_layers
        self.lr = lr
        self.momentum = momentum

        scale = 0.02
        rng = np.random.default_rng(42)

        self.W_emb = (rng.standard_normal((vocab, hidden)) * scale).astype(np.float32)
        # Hidden layers (may be empty if num_layers == 1)
        self.W_hidden = [
            (rng.standard_normal((hidden, hidden)) * scale).astype(np.float32)
            for _ in range(num_layers - 1)
        ]
        self.W_out = (rng.standard_normal((hidden, vocab)) * scale).astype(np.float32)

        # SGD momentum buffers
        self.v_emb = np.zeros_like(self.W_emb)
        self.v_hidden = [np.zeros_like(w) for w in self.W_hidden]
        self.v_out = np.zeros_like(self.W_out)

    @property
    def num_params(self) -> int:
        return (self.W_emb.size
                + sum(w.size for w in self.W_hidden)
                + self.W_out.size)

    def forward(self, ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        ids: (B, T) int32
        Returns (logits (B, T, vocab), activations list)
        """
        acts = []
        h = self.W_emb[ids]           # (B, T, H)
        h = np.maximum(h, 0)          # ReLU
        acts.append(h)

        for W in self.W_hidden:
            h = h @ W                 # (B, T, H)
            h = np.maximum(h, 0)      # ReLU
            acts.append(h)

        logits = h @ self.W_out       # (B, T, vocab)
        return logits, acts

    def loss_and_grad(
        self, ids: np.ndarray, targets: np.ndarray, mask: np.ndarray
    ) -> tuple[float, np.ndarray, list[np.ndarray], np.ndarray]:
        """
        Returns (scalar_loss, dW_emb, dW_hidden list, dW_out)
        """
        B, T = ids.shape
        logits, acts = self.forward(ids)

        # Numerically stable softmax
        shift = logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(logits - shift)
        probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

        # Cross-entropy loss
        tgt_probs = probs[np.arange(B)[:, None], np.arange(T)[None, :], targets]
        tgt_probs = np.clip(tgt_probs, 1e-9, None)
        n_valid = mask.sum() + 1e-8
        loss = float((-np.log(tgt_probs) * mask).sum() / n_valid)

        # Backprop through output layer
        d = probs.copy()
        d[np.arange(B)[:, None], np.arange(T)[None, :], targets] -= 1
        d *= mask[..., None] / n_valid                             # (B, T, vocab)

        flat_acts = acts[-1].reshape(-1, self.H)                   # (B*T, H)
        dW_out = flat_acts.T @ d.reshape(-1, self.vocab)           # (H, vocab)

        d_h = d @ self.W_out.T                                     # (B, T, H)
        d_h *= (acts[-1] > 0)                                      # ReLU gate

        # Backprop through hidden layers (reverse order)
        dW_hidden = []
        for i in range(len(self.W_hidden) - 1, -1, -1):
            prev_act = acts[i]                                     # input to layer i+1
            dW = prev_act.reshape(-1, self.H).T @ d_h.reshape(-1, self.H)
            dW_hidden.insert(0, dW)
            d_h = d_h @ self.W_hidden[i].T
            d_h *= (prev_act > 0)                                  # ReLU gate on prev

        # dW_emb via scatter-add
        dW_emb = np.zeros_like(self.W_emb)
        np.add.at(dW_emb, ids.flatten(), d_h.reshape(-1, self.H))

        return loss, dW_emb, dW_hidden, dW_out

    def step(
        self,
        dW_emb: np.ndarray,
        dW_hidden: list[np.ndarray],
        dW_out: np.ndarray,
    ) -> None:
        """SGD with momentum update."""
        self.v_emb = self.momentum * self.v_emb + dW_emb
        self.W_emb -= self.lr * self.v_emb

        for i, (dW, v) in enumerate(zip(dW_hidden, self.v_hidden)):
            self.v_hidden[i] = self.momentum * v + dW
            self.W_hidden[i] -= self.lr * self.v_hidden[i]

        self.v_out = self.momentum * self.v_out + dW_out
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
    num_layers: int = 1,
    batch_size: int = 8,
    seq_len: int = 256,
    lr: float = 0.05,
    log_every: int = 10,
    data_dirs: list[str] | None = None,
) -> None:
    rng = np.random.default_rng(42)

    # Build corpus — full repo by default (skip node_modules / .git / venv)
    _skip = {"node_modules", ".git", ".venv", "venv", "__pycache__", "ui"}
    repo_dirs = data_dirs or [
        d for d in sorted(REPO_ROOT.iterdir())
        if d.is_dir() and d.name not in _skip
    ]
    chunks = []
    for d in repo_dirs:
        p = Path(d)
        if p.exists():
            try:
                chunk = build_corpus(p, [".py", ".md"])
                chunks.append(chunk)
            except RuntimeError as exc:
                logger.debug("Skipping %s: %s", d, exc)

    corpus = np.concatenate(chunks)
    logger.info("Total corpus: %d bytes (%.2f MB)", len(corpus), len(corpus) / 1e6)

    vocab = 256
    model = ByteLM(vocab=vocab, hidden=hidden, num_layers=num_layers, lr=lr)
    layers_str = f"{num_layers}×[{hidden}→{hidden}]" if num_layers > 1 else f"[{hidden}]"
    logger.info(
        "Model: vocab=%d  emb→%s→vocab  params=%d (%.1f KB)",
        vocab, layers_str, model.num_params, model.num_params * 4 / 1024,
    )
    logger.info(
        "Training: steps=%d  batch=%d  seq_len=%d  lr=%.4f",
        steps, batch_size, seq_len, lr,
    )
    logger.info("Baseline loss (random) ≈ %.4f nats/byte", np.log(vocab))

    losses: list[float] = []
    t0_total = time.perf_counter()

    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        ids, targets, mask = sample_batch(corpus, batch_size, seq_len, rng)

        loss, dW_emb, dW_hidden, dW_out = model.loss_and_grad(ids, targets, mask)
        model.step(dW_emb, dW_hidden, dW_out)

        losses.append(loss)
        elapsed = time.perf_counter() - t0

        if step % log_every == 0 or step == 1:
            recent = losses[-log_every:]
            avg = sum(recent) / len(recent)
            tokens_per_s = (batch_size * seq_len) / elapsed
            logger.info(
                "step %5d/%d | loss=%.4f | avg_%d=%.4f | %.0f tok/s",
                step, steps, loss, len(recent), avg, tokens_per_s,
            )

    total_time = time.perf_counter() - t0_total
    first_loss = losses[0]
    last_avg = sum(losses[-20:]) / min(20, len(losses))
    delta = first_loss - last_avg

    logger.info("=" * 62)
    logger.info("Training complete in %.1fs  (%.0f tok/s avg)",
                total_time, (steps * batch_size * seq_len) / total_time)
    logger.info("Initial loss : %.4f nats/byte", first_loss)
    logger.info("Final avg    : %.4f nats/byte  (Δ = %.4f)", last_avg, delta)
    pct = delta / first_loss * 100
    logger.info("Improvement  : %.1f%%", pct)
    logger.info(
        "In bits/byte : %.3f → %.3f",
        first_loss / np.log(2), last_avg / np.log(2),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Byte-level CPU training on repo source")
    p.add_argument("--steps",   type=int,   default=100,  help="Training steps")
    p.add_argument("--hidden",  type=int,   default=128,  help="Hidden size per layer")
    p.add_argument("--layers",  type=int,   default=1,    help="Number of hidden layers")
    p.add_argument("--batch",   type=int,   default=8,    help="Batch size (sequences)")
    p.add_argument("--seq-len", type=int,   default=256,  help="Sequence length (bytes)")
    p.add_argument("--lr",      type=float, default=0.05, help="Learning rate")
    p.add_argument("--log-every", type=int, default=10,   help="Log every N steps")
    args = p.parse_args()

    train(
        steps=args.steps,
        hidden=args.hidden,
        num_layers=args.layers,
        batch_size=args.batch,
        seq_len=args.seq_len,
        lr=args.lr,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()

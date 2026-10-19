#!/usr/bin/env python3
"""
scripts/train_lmtp_cpu.py

L-MTP (Look-backward Multiple Token Prediction) training on CPU.
Pure NumPy — no JAX, no PyTorch required.

L-MTP paper: arXiv:2505.17505 (NeurIPS 2025)
  - n_head prediction heads; head i predicts token at offset i*leap_k+1
  - Look-backward inference: at each step, both h_{t-1} and h_t are fed to
    every head, producing leap_k*(n_head-1)+1 candidate tokens per step.

Two-stage training
  Stage 1 — head warm-up : backbone frozen, only L-MTP heads trained
  Stage 2 — full tuning  : all parameters (backbone + heads) trained jointly

Run
    python scripts/train_lmtp_cpu.py [--steps N] [--hidden H] [--n-head K]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-use corpus helpers from train_real_cpu
# ---------------------------------------------------------------------------

from training.byte_level_training import (
    ByteLevelConfig,
    ByteLevelDataLoader,
    ByteLevelTokenizer,
)


def build_corpus(data_dir: Path, extensions: list[str], min_bytes: int = 200) -> np.ndarray:
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
    return np.concatenate(chunks)


def sample_batch(
    corpus: np.ndarray, batch_size: int, seq_len: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_start = len(corpus) - seq_len - 1
    starts = rng.integers(0, max_start, size=batch_size)
    ids = np.stack([corpus[s: s + seq_len] for s in starts])
    targets = np.stack([corpus[s + 1: s + seq_len + 1] for s in starts])
    mask = np.ones((batch_size, seq_len), dtype=np.float32)
    return ids, targets, mask


# ---------------------------------------------------------------------------
# ByteLM backbone  (identical to train_real_cpu — embedding→ReLU→linear)
# ---------------------------------------------------------------------------

class ByteLM:
    def __init__(self, vocab: int, hidden: int, lr: float = 0.05, momentum: float = 0.9):
        self.vocab = vocab
        self.H = hidden
        self.lr = lr
        self.momentum = momentum
        scale = 0.02
        rng = np.random.default_rng(42)
        self.W_emb = (rng.standard_normal((vocab, hidden)) * scale).astype(np.float32)
        self.W_out = (rng.standard_normal((hidden, vocab)) * scale).astype(np.float32)
        self.v_emb = np.zeros_like(self.W_emb)
        self.v_out = np.zeros_like(self.W_out)

    @property
    def num_params(self) -> int:
        return self.W_emb.size + self.W_out.size

    def forward(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (logits (B,T,V), hidden (B,T,H))."""
        h = np.maximum(self.W_emb[ids], 0)   # (B, T, H)  embedding + ReLU
        logits = h @ self.W_out               # (B, T, V)
        return logits, h

    def ntp_loss_and_grad(
        self, ids: np.ndarray, targets: np.ndarray, mask: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Cross-entropy NTP loss; returns (loss, d_logits, dW_emb, dW_out)."""
        B, T = ids.shape
        logits, h = self.forward(ids)

        shift = logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(logits - shift)
        probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

        tgt_p = probs[np.arange(B)[:, None], np.arange(T)[None, :], targets]
        tgt_p = np.clip(tgt_p, 1e-9, None)
        n_valid = mask.sum() + 1e-8
        loss = float((-np.log(tgt_p) * mask).sum() / n_valid)

        d = probs.copy()
        d[np.arange(B)[:, None], np.arange(T)[None, :], targets] -= 1
        d *= mask[..., None] / n_valid

        dW_out = h.reshape(-1, self.H).T @ d.reshape(-1, self.vocab)
        d_h = (d @ self.W_out.T) * (h > 0)

        dW_emb = np.zeros_like(self.W_emb)
        np.add.at(dW_emb, ids.flatten(), d_h.reshape(-1, self.H))

        return loss, d, dW_emb, dW_out

    def step_backbone(self, dW_emb: np.ndarray, dW_out: np.ndarray) -> None:
        self.v_emb = self.momentum * self.v_emb + dW_emb
        self.W_emb -= self.lr * self.v_emb
        self.v_out = self.momentum * self.v_out + dW_out
        self.W_out -= self.lr * self.v_out


# ---------------------------------------------------------------------------
# L-MTP heads  — one small Linear per future-offset prediction head
# ---------------------------------------------------------------------------

class LMTPHeads:
    """n_head prediction heads; head i predicts token at offset (i+1)*leap_k."""

    def __init__(
        self,
        hidden: int,
        vocab: int,
        n_head: int = 4,
        leap_k: int = 2,
        lr: float = 0.05,
        momentum: float = 0.9,
    ):
        self.H = hidden
        self.vocab = vocab
        self.n_head = n_head
        self.leap_k = leap_k
        self.lr = lr
        self.momentum = momentum

        scale = 0.02
        rng = np.random.default_rng(7)
        # Each head: (2H → V) — takes [h_prev; h_curr] concatenated (look-backward)
        self.W = [
            (rng.standard_normal((2 * hidden, vocab)) * scale).astype(np.float32)
            for _ in range(n_head)
        ]
        self.v = [np.zeros_like(w) for w in self.W]

    @property
    def num_params(self) -> int:
        return sum(w.size for w in self.W)

    def tokens_per_step(self) -> int:
        """Tokens produced per autoregressive step in look-backward inference."""
        return self.leap_k * (self.n_head - 1) + 1

    def forward_head(
        self, h_prev: np.ndarray, h_curr: np.ndarray, head_idx: int
    ) -> np.ndarray:
        """(B, T, V) logits from head head_idx given concatenated [h_prev; h_curr]."""
        x = np.concatenate([h_prev, h_curr], axis=-1)   # (B, T, 2H)
        return x.reshape(-1, 2 * self.H) @ self.W[head_idx]  # reshaped for matmul

    def loss_and_grads(
        self,
        h_prev: np.ndarray,   # (B, T, H) backbone hidden at t-1
        h_curr: np.ndarray,   # (B, T, H) backbone hidden at t
        ids: np.ndarray,      # (B, T+max_offset) token ids (extended)
        mask: np.ndarray,     # (B, T) valid positions
    ) -> tuple[float, list[np.ndarray]]:
        """
        Compute total L-MTP loss and per-head weight gradients.

        For head i, the target at position t is ids[t + (i+1)*leap_k].
        """
        B, T = mask.shape
        total_loss = 0.0
        dW_list = [np.zeros_like(w) for w in self.W]

        x = np.concatenate([h_prev, h_curr], axis=-1)  # (B, T, 2H)
        x_flat = x.reshape(-1, 2 * self.H)              # (BT, 2H)

        for i, W in enumerate(self.W):
            offset = (i + 1) * self.leap_k
            if offset >= ids.shape[1] - T + 1:
                continue  # not enough tokens for this offset

            # Targets: shift ids by offset, take first T positions
            tgt = ids[:, offset: offset + T]           # (B, T)
            tgt = np.clip(tgt, 0, self.vocab - 1)

            logits_flat = x_flat @ W                   # (BT, V)
            logits = logits_flat.reshape(B, T, self.vocab)

            shift = logits.max(axis=-1, keepdims=True)
            exp_l = np.exp(logits - shift)
            probs = exp_l / exp_l.sum(axis=-1, keepdims=True)

            tp = probs[np.arange(B)[:, None], np.arange(T)[None, :], tgt]
            tp = np.clip(tp, 1e-9, None)
            n_valid = mask.sum() + 1e-8
            head_loss = float((-np.log(tp) * mask).sum() / n_valid)
            total_loss += head_loss

            d = probs.copy()
            d[np.arange(B)[:, None], np.arange(T)[None, :], tgt] -= 1
            d *= mask[..., None] / n_valid

            dW_list[i] = x_flat.T @ d.reshape(-1, self.vocab)  # (2H, V)

        return total_loss, dW_list

    def step(self, dW_list: list[np.ndarray]) -> None:
        for i, dW in enumerate(dW_list):
            self.v[i] = self.momentum * self.v[i] + dW
            self.W[i] -= self.lr * self.v[i]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    steps_warmup: int = 200,
    steps_full: int = 200,
    hidden: int = 128,
    n_head: int = 4,
    leap_k: int = 2,
    batch_size: int = 8,
    seq_len: int = 256,
    lr_backbone: float = 0.05,
    lr_heads: float = 0.05,
    log_every: int = 20,
    data_dirs: list[str] | None = None,
) -> None:
    rng = np.random.default_rng(42)

    # ── Build corpus ──────────────────────────────────────────────────────
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
                chunks.append(build_corpus(p, [".py", ".md"]))
            except RuntimeError as exc:
                logger.debug("Skipping %s: %s", d, exc)
    corpus = np.concatenate(chunks)
    logger.info("Corpus: %d bytes (%.2f MB)", len(corpus), len(corpus) / 1e6)

    vocab = 256
    # Extra tokens needed for the farthest head offset
    max_offset = n_head * leap_k
    extended_seq = seq_len + max_offset

    # ── Models ────────────────────────────────────────────────────────────
    backbone = ByteLM(vocab=vocab, hidden=hidden, lr=lr_backbone)
    heads = LMTPHeads(hidden=hidden, vocab=vocab, n_head=n_head, leap_k=leap_k, lr=lr_heads)

    total_params = backbone.num_params + heads.num_params
    logger.info(
        "Backbone params: %d (%.1f KB) | Head params: %d (%.1f KB) | Total: %d",
        backbone.num_params, backbone.num_params * 4 / 1024,
        heads.num_params, heads.num_params * 4 / 1024,
        total_params,
    )
    logger.info(
        "L-MTP: n_head=%d  leap_k=%d  tokens/step=%d",
        n_head, leap_k, heads.tokens_per_step(),
    )
    logger.info("Baseline NTP loss ≈ %.4f nats/byte", np.log(vocab))

    def _sample(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_start = len(corpus) - extended_seq - 1
        starts = rng.integers(0, max_start, size=batch_size)
        ids_ext = np.stack([corpus[s: s + extended_seq] for s in starts])  # (B, extended)
        ids = ids_ext[:, :seq_len]
        tgt_ntp = ids_ext[:, 1: seq_len + 1]
        mask = np.ones((batch_size, seq_len), dtype=np.float32)
        return ids_ext, ids, tgt_ntp, mask

    # ── Stage 1 — head warm-up ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 1 — L-MTP head warm-up  (backbone FROZEN)")
    logger.info("=" * 60)

    lmtp_losses: list[float] = []
    t0 = time.perf_counter()

    for step in range(1, steps_warmup + 1):
        ids_ext, ids, tgt_ntp, mask = _sample(rng)

        # Backbone forward only (no grad update for backbone in stage 1)
        _, h_curr = backbone.forward(ids)

        # h_prev: shift hidden right by one (use zeros for first token)
        h_prev = np.zeros_like(h_curr)
        h_prev[:, 1:] = h_curr[:, :-1]

        lmtp_loss, dW_list = heads.loss_and_grads(h_prev, h_curr, ids_ext, mask)
        heads.step(dW_list)
        lmtp_losses.append(lmtp_loss)

        if step % log_every == 0 or step == 1:
            avg = sum(lmtp_losses[-log_every:]) / len(lmtp_losses[-log_every:])
            elapsed = time.perf_counter() - t0
            toks_per_s = (step * batch_size * seq_len) / elapsed
            logger.info(
                "warm-up %4d/%d | lmtp_loss=%.4f | avg_%d=%.4f | %.0f tok/s",
                step, steps_warmup, lmtp_loss, len(lmtp_losses[-log_every:]), avg, toks_per_s,
            )

    warmup_first = lmtp_losses[0]
    warmup_last = sum(lmtp_losses[-20:]) / min(20, len(lmtp_losses))
    logger.info(
        "Stage 1 done — lmtp_loss: %.2f → %.2f  (Δ=%.2f)",
        warmup_first, warmup_last, warmup_first - warmup_last,
    )

    # ── Stage 2 — full tuning ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 2 — Full tuning  (backbone + heads)")
    logger.info("=" * 60)

    ntp_losses: list[float] = []
    lmtp_losses2: list[float] = []
    t0 = time.perf_counter()

    for step in range(1, steps_full + 1):
        ids_ext, ids, tgt_ntp, mask = _sample(rng)

        # NTP loss + backbone gradients
        ntp_loss, _, dW_emb, dW_out = backbone.ntp_loss_and_grad(ids, tgt_ntp, mask)

        # Backbone hidden states (re-use from forward that was inside ntp_loss_and_grad)
        _, h_curr = backbone.forward(ids)
        h_prev = np.zeros_like(h_curr)
        h_prev[:, 1:] = h_curr[:, :-1]

        lmtp_loss, dW_list = heads.loss_and_grads(h_prev, h_curr, ids_ext, mask)

        # Combined loss
        total_loss = ntp_loss + lmtp_loss

        # Update both backbone and heads
        backbone.step_backbone(dW_emb, dW_out)
        heads.step(dW_list)

        ntp_losses.append(ntp_loss)
        lmtp_losses2.append(lmtp_loss)

        if step % log_every == 0 or step == 1:
            win = lmtp_losses2[-log_every:]
            nwin = ntp_losses[-log_every:]
            avg_ntp = sum(nwin) / len(nwin)
            avg_lmtp = sum(win) / len(win)
            elapsed = time.perf_counter() - t0
            toks_per_s = (step * batch_size * seq_len) / elapsed
            logger.info(
                "full  %4d/%d | ntp=%.4f | lmtp=%.4f | total=%.4f | %.0f tok/s",
                step, steps_full, avg_ntp, avg_lmtp, avg_ntp + avg_lmtp, toks_per_s,
            )

    ntp_first = ntp_losses[0]
    ntp_last = sum(ntp_losses[-20:]) / min(20, len(ntp_losses))
    lmtp_first = lmtp_losses2[0]
    lmtp_last = sum(lmtp_losses2[-20:]) / min(20, len(lmtp_losses2))

    logger.info("=" * 60)
    logger.info("Training complete")
    logger.info("=" * 60)
    logger.info(
        "NTP   loss : %.4f → %.4f  (Δ=%.4f, %.1f%% improvement)",
        ntp_first, ntp_last, ntp_first - ntp_last,
        (ntp_first - ntp_last) / ntp_first * 100,
    )
    logger.info(
        "L-MTP loss : %.4f → %.4f  (Δ=%.4f, %.1f%% improvement)",
        lmtp_first, lmtp_last, lmtp_first - lmtp_last,
        (lmtp_first - lmtp_last) / lmtp_first * 100,
    )
    logger.info(
        "Total loss : %.4f → %.4f  (Δ=%.4f)",
        ntp_first + lmtp_first, ntp_last + lmtp_last,
        (ntp_first + lmtp_first) - (ntp_last + lmtp_last),
    )

    # ── Look-backward inference demo ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Look-backward inference demo")
    logger.info("=" * 60)

    tps = heads.tokens_per_step()
    logger.info("tokens_per_step = leap_k * (n_head - 1) + 1 = %d * (%d - 1) + 1 = %d",
                leap_k, n_head, tps)

    # Encode a small prompt from the corpus
    prompt = corpus[:32]
    prompt_ids = prompt.reshape(1, -1)       # (1, 32)

    generated = list(prompt_ids[0])
    h_prev_demo = np.zeros((1, 1, hidden), dtype=np.float32)

    max_gen = 3   # decode steps (each produces tps tokens)
    for step in range(max_gen):
        ctx = np.array(generated[-32:], dtype=np.int32).reshape(1, -1)
        _, h_ctx = backbone.forward(ctx)
        h_curr_demo = h_ctx[:, -1:, :]     # last hidden  (1, 1, H)

        step_tokens = []
        for i in range(n_head):
            x = np.concatenate([h_prev_demo, h_curr_demo], axis=-1)  # (1,1,2H)
            logits = (x.reshape(1, -1) @ heads.W[i]).reshape(1, heads.vocab)
            tok = int(np.argmax(logits, axis=-1)[0])
            step_tokens.append(tok)

        # Fill leap_k-1 gaps with greedy NTP
        full_step: list[int] = []
        for i, tok in enumerate(step_tokens):
            full_step.append(tok)
            if i < n_head - 1:
                for _ in range(leap_k - 1):
                    full_step.append(tok)  # simple repeat-fill for demo

        generated.extend(full_step[:tps])
        h_prev_demo = h_curr_demo

        logger.info(
            "  decode step %d: produced %d tokens → [%s]",
            step + 1, tps,
            ", ".join(str(t) for t in full_step[:tps]),
        )

    logger.info(
        "Total tokens generated: %d (prompt=%d + new=%d)",
        len(generated), len(prompt_ids[0]), len(generated) - len(prompt_ids[0]),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="L-MTP CPU training (pure NumPy)")
    p.add_argument("--warmup-steps", type=int, default=200, help="Head warm-up steps")
    p.add_argument("--full-steps",   type=int, default=200, help="Full tuning steps")
    p.add_argument("--hidden",       type=int, default=128, help="Backbone hidden size")
    p.add_argument("--n-head",       type=int, default=4,   help="Number of L-MTP heads")
    p.add_argument("--leap-k",       type=int, default=2,   help="Leap stride per head")
    p.add_argument("--batch",        type=int, default=8,   help="Batch size")
    p.add_argument("--seq-len",      type=int, default=256, help="Sequence length")
    p.add_argument("--lr",           type=float, default=0.05, help="Learning rate")
    p.add_argument("--log-every",    type=int, default=20,  help="Log every N steps")
    args = p.parse_args()

    train(
        steps_warmup=args.warmup_steps,
        steps_full=args.full_steps,
        hidden=args.hidden,
        n_head=args.n_head,
        leap_k=args.leap_k,
        batch_size=args.batch,
        seq_len=args.seq_len,
        lr_backbone=args.lr,
        lr_heads=args.lr,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/benchmark_cpu_training.py

CPU training benchmarks: NTP baseline vs L-MTP (arXiv:2505.17505).

Runs a matrix of configurations, each consuming ~TARGET_TOKENS tokens, and
reports convergence (bits-per-byte), throughput (tok/s), and inference
look-backward speedup.

Results are printed as an ASCII table and saved to
  benchmarks/cpu_training_bench.json

Usage
    python scripts/benchmark_cpu_training.py [--tokens 10_000_000]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------

def _build_corpus() -> np.ndarray:
    import warnings; warnings.filterwarnings("ignore")
    import logging as _l; _l.disable(_l.CRITICAL)
    from training.byte_level_training import (
        ByteLevelConfig, ByteLevelDataLoader, ByteLevelTokenizer,
    )
    _l.disable(_l.NOTSET)

    cfg = ByteLevelConfig(file_extensions=[".py", ".md"],
                          min_file_size_bytes=200, max_file_size_mb=5)
    tok = ByteLevelTokenizer(cfg)
    loader = ByteLevelDataLoader(cfg, tok)
    _skip = {"node_modules", ".git", ".venv", "venv", "__pycache__", "ui"}
    chunks = []
    for d in sorted(REPO_ROOT.iterdir()):
        if d.is_dir() and d.name not in _skip:
            files = loader.load_files_from_directory(d)
            for fp in files:
                raw = loader.load_file_as_bytes(fp)
                if raw is not None:
                    chunks.append(np.frombuffer(raw, dtype=np.uint8).astype(np.int32))
    return np.concatenate(chunks)


def _sample(corpus: np.ndarray, B: int, T: int, rng: np.random.Generator,
            extra: int = 0) -> np.ndarray:
    """Return (B, T+extra) windows from corpus."""
    max_start = len(corpus) - T - extra - 1
    starts = rng.integers(0, max_start, size=B)
    return np.stack([corpus[s: s + T + extra] for s in starts])


# ---------------------------------------------------------------------------
# Minimal NTP model
# ---------------------------------------------------------------------------

class ByteLM:
    def __init__(self, vocab: int, hidden: int, lr: float = 0.05, mom: float = 0.9):
        self.vocab = vocab; self.H = hidden; self.lr = lr; self.mom = mom
        r = np.random.default_rng(42); s = 0.02
        self.W_emb = (r.standard_normal((vocab, hidden)) * s).astype(np.float32)
        self.W_out = (r.standard_normal((hidden, vocab)) * s).astype(np.float32)
        self.ve = np.zeros_like(self.W_emb)
        self.vo = np.zeros_like(self.W_out)

    @property
    def num_params(self) -> int:
        return self.W_emb.size + self.W_out.size

    def forward(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.maximum(self.W_emb[ids], 0)
        return h @ self.W_out, h

    def train_step(self, ids: np.ndarray, tgt: np.ndarray,
                   mask: np.ndarray) -> float:
        B, T = ids.shape
        logits, h = self.forward(ids)
        shift = logits.max(-1, keepdims=True)
        exp_l = np.exp(logits - shift)
        probs = exp_l / exp_l.sum(-1, keepdims=True)
        n = mask.sum() + 1e-8
        tp = np.clip(probs[np.arange(B)[:, None], np.arange(T)[None, :], tgt], 1e-9, None)
        loss = float((-np.log(tp) * mask).sum() / n)
        d = probs.copy()
        d[np.arange(B)[:, None], np.arange(T)[None, :], tgt] -= 1
        d *= mask[..., None] / n
        dWo = h.reshape(-1, self.H).T @ d.reshape(-1, self.vocab)
        dh = (d @ self.W_out.T) * (h > 0)
        dWe = np.zeros_like(self.W_emb)
        np.add.at(dWe, ids.flatten(), dh.reshape(-1, self.H))
        self.ve = self.mom * self.ve + dWe; self.W_emb -= self.lr * self.ve
        self.vo = self.mom * self.vo + dWo; self.W_out -= self.lr * self.vo
        return loss


# ---------------------------------------------------------------------------
# L-MTP heads
# ---------------------------------------------------------------------------

class LMTPHeads:
    def __init__(self, hidden: int, vocab: int, n_head: int, leap_k: int,
                 lr: float = 0.05, mom: float = 0.9):
        self.H = hidden; self.vocab = vocab
        self.n_head = n_head; self.leap_k = leap_k
        self.lr = lr; self.mom = mom
        r = np.random.default_rng(7)
        self.W = [(r.standard_normal((2 * hidden, vocab)) * 0.02).astype(np.float32)
                  for _ in range(n_head)]
        self.v = [np.zeros_like(w) for w in self.W]

    @property
    def num_params(self) -> int:
        return sum(w.size for w in self.W)

    def tokens_per_step(self) -> int:
        return self.leap_k * (self.n_head - 1) + 1

    def train_step(self, h_prev: np.ndarray, h_curr: np.ndarray,
                   ids_ext: np.ndarray, mask: np.ndarray) -> float:
        B, T = mask.shape
        x = np.concatenate([h_prev, h_curr], axis=-1)
        x_flat = x.reshape(-1, 2 * self.H)
        total = 0.0
        for i, W in enumerate(self.W):
            offset = (i + 1) * self.leap_k
            if offset >= ids_ext.shape[1] - T + 1:
                continue
            tgt = np.clip(ids_ext[:, offset: offset + T], 0, self.vocab - 1)
            logits_flat = x_flat @ W
            logits = logits_flat.reshape(B, T, self.vocab)
            shift = logits.max(-1, keepdims=True)
            exp_l = np.exp(logits - shift)
            probs = exp_l / exp_l.sum(-1, keepdims=True)
            n = mask.sum() + 1e-8
            tp = np.clip(probs[np.arange(B)[:, None], np.arange(T)[None, :], tgt], 1e-9, None)
            head_loss = float((-np.log(tp) * mask).sum() / n)
            total += head_loss
            d = probs.copy()
            d[np.arange(B)[:, None], np.arange(T)[None, :], tgt] -= 1
            d *= mask[..., None] / n
            dW = x_flat.T @ d.reshape(-1, self.vocab)
            self.v[i] = self.mom * self.v[i] + dW
            self.W[i] -= self.lr * self.v[i]
        return total


# ---------------------------------------------------------------------------
# Benchmark run
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    name: str
    hidden: int
    batch: int
    seq_len: int
    n_head: int = 0      # 0 = NTP baseline
    leap_k: int = 2
    lr: float = 0.05
    warmup_frac: float = 0.25   # fraction of steps used for head warm-up


@dataclass
class BenchResult:
    name: str
    hidden: int
    batch: int
    seq_len: int
    n_head: int
    leap_k: int
    total_tokens: int
    num_params: int
    # convergence
    initial_loss: float       # first 1% of steps, avg
    final_loss: float         # last 5% of steps, avg
    initial_bpb: float
    final_bpb: float
    pct_improvement: float
    # throughput
    train_tok_per_s: float
    # L-MTP extras
    tokens_per_step: int = 0
    infer_tok_per_s: float = 0.0
    speedup_vs_ntp: float = 0.0
    # curve (loss every ~10% of steps)
    loss_curve: list[float] = field(default_factory=list)
    step_curve: list[int] = field(default_factory=list)


def run_benchmark(cfg: RunConfig, corpus: np.ndarray,
                  target_tokens: int) -> BenchResult:
    rng = np.random.default_rng(42)
    vocab = 256
    max_offset = max(cfg.n_head * cfg.leap_k, 1)
    extended = cfg.seq_len + max_offset

    steps = max(10, target_tokens // (cfg.batch * cfg.seq_len))
    warmup_steps = int(steps * cfg.warmup_frac) if cfg.n_head > 0 else 0
    total_steps = warmup_steps + steps

    backbone = ByteLM(vocab=vocab, hidden=cfg.hidden, lr=cfg.lr)
    heads = (LMTPHeads(cfg.hidden, vocab, cfg.n_head, cfg.leap_k, lr=cfg.lr)
             if cfg.n_head > 0 else None)

    num_params = backbone.num_params + (heads.num_params if heads else 0)
    record_every = max(1, total_steps // 10)

    losses: list[float] = []
    curve_loss: list[float] = []
    curve_step: list[int] = []

    print(f"  [{cfg.name}] {total_steps} steps "
          f"({warmup_steps} warm-up + {steps} full) "
          f"· {num_params:,} params", flush=True)

    t0 = time.perf_counter()

    # --- Stage 1: head warm-up (backbone frozen) ---
    for step in range(1, warmup_steps + 1):
        win = _sample(corpus, cfg.batch, cfg.seq_len, rng, max_offset)
        ids = win[:, :cfg.seq_len]
        mask = np.ones((cfg.batch, cfg.seq_len), dtype=np.float32)
        _, h_curr = backbone.forward(ids)
        h_prev = np.zeros_like(h_curr); h_prev[:, 1:] = h_curr[:, :-1]
        heads.train_step(h_prev, h_curr, win, mask)  # type: ignore[union-attr]

    # --- Stage 2: full training ---
    for step in range(1, steps + 1):
        global_step = warmup_steps + step
        win = _sample(corpus, cfg.batch, cfg.seq_len, rng, max_offset)
        ids = win[:, :cfg.seq_len]
        tgt = win[:, 1: cfg.seq_len + 1]
        mask = np.ones((cfg.batch, cfg.seq_len), dtype=np.float32)

        ntp_loss = backbone.train_step(ids, tgt, mask)

        if heads:
            _, h_curr = backbone.forward(ids)
            h_prev = np.zeros_like(h_curr); h_prev[:, 1:] = h_curr[:, :-1]
            lmtp_loss = heads.train_step(h_prev, h_curr, win, mask)
            loss = ntp_loss + lmtp_loss
        else:
            loss = ntp_loss

        losses.append(loss)
        if step % record_every == 0 or step == 1:
            avg = float(np.mean(losses[-record_every:]))
            curve_loss.append(avg)
            curve_step.append(global_step)

    elapsed = time.perf_counter() - t0
    total_tok = steps * cfg.batch * cfg.seq_len
    train_tps = total_tok / elapsed

    first_n = max(1, len(losses) // 20)
    last_n = max(1, len(losses) // 20)
    init_loss = float(np.mean(losses[:first_n]))
    fin_loss = float(np.mean(losses[-last_n:]))
    init_bpb = init_loss / np.log(2)
    fin_bpb = fin_loss / np.log(2)
    pct = (init_loss - fin_loss) / init_loss * 100

    # --- Inference throughput ---
    tps_i = 0
    speedup = 0.0
    tps_val = heads.tokens_per_step() if heads else 0

    if heads:
        # measure look-backward inference speed
        ctx = corpus[:64].reshape(1, -1).astype(np.int32)
        n_infer = 200
        h_prev_i = np.zeros((1, 1, cfg.hidden), dtype=np.float32)
        t_i = time.perf_counter()
        for _ in range(n_infer):
            _, h_ctx = backbone.forward(ctx)
            h_curr_i = h_ctx[:, -1:, :]
            for W in heads.W:
                x = np.concatenate([h_prev_i, h_curr_i], axis=-1)
                _ = (x.reshape(1, -1) @ W).argmax(-1)
            h_prev_i = h_curr_i
        dt_i = time.perf_counter() - t_i
        tps_i = n_infer * tps_val / dt_i

        # NTP baseline speed (single token per step)
        t_n = time.perf_counter()
        for _ in range(n_infer):
            _, _ = backbone.forward(ctx)
        dt_n = time.perf_counter() - t_n
        ntp_tps = n_infer / dt_n
        speedup = tps_i / ntp_tps if ntp_tps > 0 else 0.0

    return BenchResult(
        name=cfg.name, hidden=cfg.hidden, batch=cfg.batch, seq_len=cfg.seq_len,
        n_head=cfg.n_head, leap_k=cfg.leap_k,
        total_tokens=total_tok, num_params=num_params,
        initial_loss=init_loss, final_loss=fin_loss,
        initial_bpb=init_bpb, final_bpb=fin_bpb,
        pct_improvement=pct, train_tok_per_s=train_tps,
        tokens_per_step=tps_val, infer_tok_per_s=tps_i, speedup_vs_ntp=speedup,
        loss_curve=curve_loss, step_curve=curve_step,
    )


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_table(results: list[BenchResult]) -> None:
    cols = [
        ("Config",         "name",           "<", 22),
        ("Params",         "num_params",     ">", 9),
        ("Tokens",         "total_tokens",   ">", 10),
        ("Init bpb",       "initial_bpb",    ">", 9),
        ("Final bpb",      "final_bpb",      ">", 9),
        ("Δ%",             "pct_improvement",">", 7),
        ("Train tok/s",    "train_tok_per_s",">", 12),
        ("Inf tok/s",      "infer_tok_per_s",">", 11),
        ("Speedup",        "speedup_vs_ntp", ">", 8),
    ]
    header = "  ".join(f"{h:{align}{w}}" for h, _, align, w in cols)
    sep    = "  ".join("─" * w for _, _, _, w in cols)
    print()
    print(header)
    print(sep)
    for r in results:
        vals = {
            "name": r.name,
            "num_params": f"{r.num_params:,}",
            "total_tokens": f"{r.total_tokens/1e6:.1f}M",
            "initial_bpb": f"{r.initial_bpb:.4f}",
            "final_bpb": f"{r.final_bpb:.4f}",
            "pct_improvement": f"{r.pct_improvement:.1f}%",
            "train_tok_per_s": f"{r.train_tok_per_s:,.0f}",
            "infer_tok_per_s": f"{r.infer_tok_per_s:,.0f}" if r.infer_tok_per_s else "—",
            "speedup_vs_ntp": f"{r.speedup_vs_ntp:.2f}×" if r.speedup_vs_ntp else "1.00×",
        }
        row = "  ".join(f"{vals[k]:{align}{w}}" for _, k, align, w in cols)
        print(row)
    print()


def print_curves(results: list[BenchResult]) -> None:
    print("Loss curves (avg loss at ~10% checkpoints):")
    print()
    for r in results:
        pts = "  ".join(f"{v:.3f}" for v in r.loss_curve)
        print(f"  {r.name:<22} [{pts}]")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CONFIGS = [
    # NTP baselines
    RunConfig("NTP h=128 b=8  s=256",  hidden=128, batch=8,  seq_len=256),
    RunConfig("NTP h=256 b=8  s=256",  hidden=256, batch=8,  seq_len=256),
    RunConfig("NTP h=128 b=16 s=512",  hidden=128, batch=16, seq_len=512),
    RunConfig("NTP h=256 b=16 s=512",  hidden=256, batch=16, seq_len=512),
    # L-MTP (n_head=4, leap_k=2) → 7 tok/step
    RunConfig("LMTP h=128 n=4 k=2",   hidden=128, batch=8,  seq_len=256, n_head=4, leap_k=2),
    RunConfig("LMTP h=256 n=4 k=2",   hidden=256, batch=8,  seq_len=256, n_head=4, leap_k=2),
    # L-MTP ablations
    RunConfig("LMTP h=128 n=2 k=2",   hidden=128, batch=8,  seq_len=256, n_head=2, leap_k=2),
    RunConfig("LMTP h=128 n=4 k=4",   hidden=128, batch=8,  seq_len=256, n_head=4, leap_k=4),
]


def main() -> None:
    p = argparse.ArgumentParser(description="CPU training benchmark: NTP vs L-MTP")
    p.add_argument("--tokens", type=int, default=10_000_000,
                   help="Target training tokens per config (default: 10M)")
    p.add_argument("--out", type=str,
                   default=str(REPO_ROOT / "benchmarks" / "cpu_training_bench.json"),
                   help="Output JSON path")
    args = p.parse_args()

    print("=" * 72)
    print(f" CPU Training Benchmark — NTP vs L-MTP")
    print(f" Target: {args.tokens/1e6:.0f}M tokens per config")
    print(f" Corpus: repo .py + .md files")
    print("=" * 72)

    print("\nBuilding corpus...", end=" ", flush=True)
    corpus = _build_corpus()
    print(f"{len(corpus):,} bytes ({len(corpus)/1e6:.2f} MB)")

    results: list[BenchResult] = []
    t_all = time.perf_counter()

    for cfg in CONFIGS:
        result = run_benchmark(cfg, corpus, args.tokens)
        results.append(result)

    elapsed_all = time.perf_counter() - t_all
    print(f"\nAll configs done in {elapsed_all:.1f}s")

    print_table(results)
    print_curves(results)

    # Summarise NTP vs L-MTP
    ntp_base = next((r for r in results if r.name.startswith("NTP h=128 b=8")), None)
    lmtp_base = next((r for r in results if r.name.startswith("LMTP h=128 n=4 k=2")), None)
    if ntp_base and lmtp_base:
        print("NTP vs L-MTP comparison (hidden=128, batch=8, seq=256):")
        print(f"  NTP  final bpb : {ntp_base.final_bpb:.4f}")
        print(f"  LMTP final bpb : {lmtp_base.final_bpb:.4f}  "
              f"(NTP component only)")
        print(f"  L-MTP tokens/step : {lmtp_base.tokens_per_step}")
        print(f"  Inference speedup : {lmtp_base.speedup_vs_ntp:.2f}×")
        print()

    # Save JSON
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_tokens": args.tokens,
        "corpus_bytes": int(len(corpus)),
        "results": [asdict(r) for r in results],
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"Results saved → {out}")


if __name__ == "__main__":
    main()

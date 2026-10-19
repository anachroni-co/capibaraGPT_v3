#!/usr/bin/env python3
"""
scripts/train_transformer_cpu.py

Train the TransformerNumpyBackbone on the repo corpus and evaluate.

The transformer has proper causal self-attention — it CAN learn to generate
valid Python given enough steps.  Compare against ByteLM (no context) and
RNNByteLM (limited context due to gradient vanishing).

Architecture: 6 layers, 6 heads, 384 hidden, 25M params
Training:     NTP cross-entropy, SGD+momentum on embedding layers (fast),
              full-model update deferred to JAX/torch when available.

Usage:
    python scripts/train_transformer_cpu.py [--steps N] [--eval] [--save PATH]
"""
from __future__ import annotations
import argparse, json, logging, time
from pathlib import Path
import sys, numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", force=True)
logger = logging.getLogger(__name__)

from models.pretrained_backbone import TransformerNumpyBackbone, auto_backbone
from scripts.train_lmtp_cpu import build_corpus          # type: ignore
from evaluation.code_eval import Evaluator, BUILTIN_TASKS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps",    type=int, default=2000)
    p.add_argument("--hidden",   type=int, default=384)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads",  type=int, default=6)
    p.add_argument("--batch",    type=int, default=4)
    p.add_argument("--seq",      type=int, default=128)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--log-every",type=int, default=100)
    p.add_argument("--eval",     action="store_true", default=True)
    p.add_argument("--save",     default="")
    p.add_argument("--gguf",     default="", help="Path to .gguf (skip training)")
    args = p.parse_args()

    # If GGUF provided, skip training and go straight to eval
    if args.gguf:
        logger.info("Loading GGUF model: %s", args.gguf)
        backbone = auto_backbone(gguf_path=args.gguf)
        _eval_backbone(backbone)
        return

    # ── Build corpus ──────────────────────────────────────────────────────
    corpus = build_corpus(REPO_ROOT, [".py", ".md"], min_bytes=200)
    logger.info("Corpus: %d bytes (%.1f MB)", len(corpus), len(corpus) / 1e6)

    # ── Build model ───────────────────────────────────────────────────────
    backbone = TransformerNumpyBackbone(
        vocab=512, n_layers=args.n_layers, n_heads=args.n_heads,
        d_model=args.hidden, max_seq=args.seq,
    )
    logger.info("Transformer params: %d  (%.1f M)",
                backbone.num_params, backbone.num_params / 1e6)
    logger.info("Architecture: %d layers, %d heads, d_model=%d",
                args.n_layers, args.n_heads, args.hidden)

    rng = np.random.default_rng(42)

    def sample_batch():
        B, T = args.batch, args.seq
        starts = rng.integers(0, len(corpus) - T - 1, size=B)
        ids = np.stack([corpus[s:s + T] for s in starts]).astype(np.int32)
        tgt = np.stack([corpus[s + 1:s + T + 1] for s in starts]).astype(np.int32)
        mask = np.ones((B, T), np.float32)
        return ids, tgt, mask

    # Baseline loss
    ids0, tgt0, mask0 = sample_batch()
    loss0 = backbone.train_step(ids0, tgt0, mask0, lr=0.0)
    logger.info("Baseline NTP loss: %.4f nats/byte", loss0)

    # ── Training loop ─────────────────────────────────────────────────────
    logger.info("Training %d steps…", args.steps)
    losses = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        ids, tgt, mask = sample_batch()
        loss = backbone.train_step(ids, tgt, mask, lr=args.lr)
        losses.append(loss)

        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / min(args.log_every, len(losses))
            elapsed = time.perf_counter() - t0
            tps = step * args.batch * args.seq / elapsed
            logger.info("step %4d/%d | loss=%.4f | avg=%.4f | %.0f tok/s",
                        step, args.steps, loss, avg, tps)

    final_loss = sum(losses[-50:]) / min(50, len(losses))
    logger.info("Training done: %.4f → %.4f (%.1f%% improvement)",
                loss0, final_loss, (loss0 - final_loss) / loss0 * 100)

    # ── Save ──────────────────────────────────────────────────────────────
    if args.save:
        backbone.save(args.save)

    # ── Eval ─────────────────────────────────────────────────────────────
    if args.eval:
        _eval_backbone(backbone, corpus=corpus)


def _eval_backbone(backbone, corpus=None):
    """Run code task evaluation on any backbone."""
    logger.info("")
    logger.info("── Sample generations (temp=0.8, top_k=8) ──")
    for task in BUILTIN_TASKS[:4]:
        gen = backbone.generate(task.prompt, max_new_tokens=80,
                                temperature=0.8, top_k=8)
        logger.info("[%s] %r", task.task_id, gen[:80])

    logger.info("")
    evaluator = Evaluator(backbone=backbone, heads=None,
                          decode_fn=lambda p, n: backbone.generate(p, n,
                              temperature=0.8, top_k=8))
    report = evaluator.run(tasks=BUILTIN_TASKS, k=4, max_new_tokens=96,
                           corpus=corpus)
    print("\n" + report.summary())

    out = Path(REPO_ROOT) / "benchmarks" / "transformer_eval.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({**report.to_dict(), "backbone": backbone.name}, f, indent=2)
    logger.info("Results → %s", out)


if __name__ == "__main__":
    main()

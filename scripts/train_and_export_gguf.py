#!/usr/bin/env python3
"""
scripts/train_and_export_gguf.py

Train TransformerNumpyBackbone on the repo corpus, then export the
trained weights to a valid GPT-2 GGUF file loadable by llama.cpp.

Weight mapping (NumPy layout x→y via right-multiply vs GGUF [d_out,d_in]):
  wte[0:N_BYTE]        → token_embd.weight  [N_BYTE, D]   (first 256 rows)
  wpe                  → position_embd.weight [max_seq, D]
  blk.ln1_g/b          → blk.i.attn_norm.weight/bias
  attn_qkv.T           → blk.i.attn_qkv.weight   [3D, D]  (transpose!)
  attn_proj.T          → blk.i.attn_output.weight [D, D]   (transpose!)
  blk.ln2_g/b          → blk.i.ffn_norm.weight/bias
  ff_w1.T              → blk.i.ffn_up.weight   [d_ff, D]   (transpose!)
  ff_b1                → blk.i.ffn_up.bias     [d_ff]
  ff_w2.T              → blk.i.ffn_down.weight [D, d_ff]   (transpose!)
  ff_b2                → blk.i.ffn_down.bias   [D]
  ln_f_g/b             → output_norm.weight/bias
  wte[0:N_BYTE]        → output.weight (weight-tied)

Usage:
    python scripts/train_and_export_gguf.py [--steps 2000] [--out models/capibara.gguf]
"""
from __future__ import annotations
import argparse, logging, time
from pathlib import Path
import sys, numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", force=True)
log = logging.getLogger(__name__)

from models.pretrained_backbone import TransformerNumpyBackbone
from scripts.train_lmtp_cpu import build_corpus  # type: ignore

# Number of byte tokens exported to GGUF (we use the 256 raw-byte tokens)
N_BYTE = 256


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(steps: int, hidden: int, n_layers: int, n_heads: int,
          seq: int, batch: int, lr: float) -> TransformerNumpyBackbone:

    corpus = build_corpus(REPO_ROOT, [".py", ".md"], min_bytes=200)
    log.info("Corpus: %d bytes (%.1f MB)", len(corpus), len(corpus) / 1e6)

    backbone = TransformerNumpyBackbone(
        vocab=512, n_layers=n_layers, n_heads=n_heads,
        d_model=hidden, max_seq=seq,
    )
    log.info("Params: %d  (%.1f M) | arch: %dL/%dH/d%d",
             backbone.num_params, backbone.num_params / 1e6,
             n_layers, n_heads, hidden)

    rng = np.random.default_rng(42)

    def sample_batch():
        starts = rng.integers(0, len(corpus) - seq - 1, size=batch)
        ids = np.stack([corpus[s:s + seq] for s in starts]).astype(np.int32)
        tgt = np.stack([corpus[s + 1:s + seq + 1] for s in starts]).astype(np.int32)
        return ids, tgt, np.ones((batch, seq), np.float32)

    # Baseline
    ids0, tgt0, msk0 = sample_batch()
    loss0 = backbone.train_step(ids0, tgt0, msk0, lr=0.0)
    log.info("Baseline loss: %.4f", loss0)

    losses, t0 = [], time.perf_counter()
    log_every = max(1, steps // 15)

    for step in range(1, steps + 1):
        ids, tgt, msk = sample_batch()
        loss = backbone.train_step(ids, tgt, msk, lr=lr)
        losses.append(loss)
        if step % log_every == 0 or step == 1:
            avg = sum(losses[-log_every:]) / min(log_every, len(losses))
            tps = step * batch * seq / (time.perf_counter() - t0)
            log.info("step %4d/%d | loss=%.4f | avg=%.4f | %.0f tok/s",
                     step, steps, loss, avg, tps)

    final = sum(losses[-50:]) / min(50, len(losses))
    log.info("Training done: %.4f → %.4f  (%.1f%% improvement)",
             loss0, final, (loss0 - final) / loss0 * 100)
    return backbone


# ---------------------------------------------------------------------------
# GGUF export
# ---------------------------------------------------------------------------

def _gpt2_b2u() -> dict:
    """GPT-2 bytes-to-unicode mapping."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, cs))


def export_gguf(backbone: TransformerNumpyBackbone, out_path: str,
                ctx_export: int = 512) -> None:
    """Export backbone weights to GGUF.

    ctx_export: declared context length in the GGUF (must be >= backbone.max_seq).
    We pad position_embd.weight by tiling so llama.cpp can use any n_ctx <= ctx_export.
    """
    try:
        import gguf
    except ImportError:
        log.error("pip install gguf")
        raise

    D   = backbone.d_model
    DFF = backbone.d_ff
    V   = N_BYTE + 1   # 256 byte-tokens + 1 BPE merge (required by llama.cpp)
    CTX = max(ctx_export, backbone.max_seq)

    log.info("Exporting GGUF: vocab=%d  d=%d  layers=%d  ctx=%d → %s",
             V, D, backbone.n_layers, CTX, out_path)

    writer = gguf.GGUFWriter(out_path, arch="gpt2")

    # ── Architecture metadata ────────────────────────────────────────────
    writer.add_uint32("gpt2.context_length",               CTX)
    writer.add_uint32("gpt2.embedding_length",             D)
    writer.add_uint32("gpt2.block_count",                  backbone.n_layers)
    writer.add_uint32("gpt2.feed_forward_length",          DFF)
    writer.add_uint32("gpt2.attention.head_count",         backbone.n_heads)
    writer.add_uint32("gpt2.attention.head_count_kv",      backbone.n_heads)
    writer.add_float32("gpt2.attention.layer_norm_epsilon", 1e-5)

    # ── Tokenizer (GPT-2 byte-level, no real BPE) ────────────────────────
    b2u = _gpt2_b2u()
    tokens     = [chr(b2u[i]) for i in range(N_BYTE)] + ["ab"]
    scores     = [0.0] * V
    tok_types  = [1]   * V   # NORMAL

    writer.add_string("tokenizer.ggml.model",        "gpt2")
    writer.add_array("tokenizer.ggml.tokens",         tokens)
    writer.add_array("tokenizer.ggml.scores",         scores)
    writer.add_array("tokenizer.ggml.token_type",     tok_types)
    writer.add_array("tokenizer.ggml.merges",         ["a b"])   # required
    writer.add_uint32("tokenizer.ggml.bos_token_id",  1)
    writer.add_uint32("tokenizer.ggml.eos_token_id",  2)
    writer.add_bool("tokenizer.ggml.add_space_prefix", False)

    # ── Tensors ──────────────────────────────────────────────────────────
    # wte: our shape is (vocab=512, D); export first N_BYTE+1 rows
    # GGUF convention: weight stored as [d_out, d_in], forward = x @ W.T
    # Our forward:     x @ W  (right-multiply) → GGUF W = our_W.T

    wte_export = backbone.wte[:V].astype(np.float32)        # (V, D)
    writer.add_tensor("token_embd.weight", wte_export)      # [V, D]

    # Pad position embeddings to CTX by tiling backbone.wpe (trained on max_seq rows).
    # This lets llama.cpp run at any n_ctx <= CTX without an out-of-bounds crash.
    wpe_src = backbone.wpe.astype(np.float32)               # [max_seq, D]
    if CTX > backbone.max_seq:
        reps = -(-CTX // backbone.max_seq)                  # ceil division
        wpe_ext = np.tile(wpe_src, (reps, 1))[:CTX]        # [CTX, D]
    else:
        wpe_ext = wpe_src[:CTX]
    writer.add_tensor("position_embd.weight", wpe_ext)      # [CTX, D]

    for i, blk in enumerate(backbone.blocks):
        p = f"blk.{i}"

        # Layer norms (no transpose needed — element-wise)
        writer.add_tensor(f"{p}.attn_norm.weight", blk["ln1_g"].astype(np.float32))
        writer.add_tensor(f"{p}.attn_norm.bias",   blk["ln1_b"].astype(np.float32))

        # Attention QKV: our shape (D, 3D), GGUF expects (3D, D)
        writer.add_tensor(f"{p}.attn_qkv.weight",
                          blk["attn_qkv"].T.astype(np.float32))       # (3D, D)
        writer.add_tensor(f"{p}.attn_qkv.bias",
                          np.zeros(3 * D, np.float32))

        # Attention output projection: our shape (D, D), GGUF expects (D, D)
        writer.add_tensor(f"{p}.attn_output.weight",
                          blk["attn_proj"].T.astype(np.float32))      # (D, D)
        writer.add_tensor(f"{p}.attn_output.bias",
                          np.zeros(D, np.float32))

        # FFN layer norm
        writer.add_tensor(f"{p}.ffn_norm.weight", blk["ln2_g"].astype(np.float32))
        writer.add_tensor(f"{p}.ffn_norm.bias",   blk["ln2_b"].astype(np.float32))

        # FFN up: our shape (D, d_ff), GGUF expects (d_ff, D)
        writer.add_tensor(f"{p}.ffn_up.weight",
                          blk["ff_w1"].T.astype(np.float32))          # (d_ff, D)
        writer.add_tensor(f"{p}.ffn_up.bias",
                          blk["ff_b1"].astype(np.float32))

        # FFN down: our shape (d_ff, D), GGUF expects (D, d_ff)
        writer.add_tensor(f"{p}.ffn_down.weight",
                          blk["ff_w2"].T.astype(np.float32))          # (D, d_ff)
        writer.add_tensor(f"{p}.ffn_down.bias",
                          blk["ff_b2"].astype(np.float32))

    # Final layer norm + output (weight-tied to wte)
    writer.add_tensor("output_norm.weight", backbone.ln_f_g.astype(np.float32))
    writer.add_tensor("output_norm.bias",   backbone.ln_f_b.astype(np.float32))
    writer.add_tensor("output.weight",      wte_export)   # weight tying

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = Path(out_path).stat().st_size / 1e6
    log.info("GGUF saved: %s  (%.1f MB)", out_path, size_mb)


# ---------------------------------------------------------------------------
# Evaluation with LlamaCppBackbone
# ---------------------------------------------------------------------------

def evaluate(gguf_path: str, n_ctx: int = 512) -> None:
    from models.pretrained_backbone import LlamaCppBackbone
    from evaluation.code_eval import Evaluator, BUILTIN_TASKS

    log.info("\n── Loading GGUF with llama.cpp (n_ctx=%d) ──", n_ctx)
    backbone = LlamaCppBackbone(gguf_path, n_ctx=n_ctx)
    log.info("Backend: %s", backbone.name)

    log.info("\n── Sample generations ──")
    for task in BUILTIN_TASKS[:4]:
        out = backbone.generate(task.prompt, max_new_tokens=60,
                                temperature=0.8, top_k=20)
        log.info("[%s] %r", task.task_id, out[:80])

    evaluator = Evaluator(
        backbone=None, heads=None,
        decode_fn=lambda p, n: backbone.generate(p, n, temperature=0.8, top_k=20),
    )
    report = evaluator.run(tasks=BUILTIN_TASKS, k=4, max_new_tokens=96)
    print("\n" + report.summary())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps",    type=int,   default=2000)
    p.add_argument("--hidden",   type=int,   default=384)
    p.add_argument("--n-layers", type=int,   default=6)
    p.add_argument("--n-heads",  type=int,   default=6)
    p.add_argument("--seq",      type=int,   default=128)
    p.add_argument("--batch",    type=int,   default=4)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--out",      default=str(REPO_ROOT / "models" / "capibara_trained.gguf"))
    p.add_argument("--eval",     action="store_true", default=True)
    args = p.parse_args()

    # Train
    backbone = train(
        steps=args.steps, hidden=args.hidden,
        n_layers=args.n_layers, n_heads=args.n_heads,
        seq=args.seq, batch=args.batch, lr=args.lr,
    )

    # Export
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ctx_export = max(512, args.seq)   # always export >= 512 positions
    export_gguf(backbone, args.out, ctx_export=ctx_export)

    # Eval — use ctx_export so n_ctx matches the exported position table
    if args.eval:
        evaluate(args.out, n_ctx=ctx_export)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/create_tiny_gguf.py

Creates a minimal valid GGUF file (GPT-2 architecture, 2 layers, d=64)
with random weights so the full llama.cpp pipeline can be tested without
a network connection.

The model generates gibberish (random weights) but exercises:
  - LlamaCppBackbone.from_gguf()
  - auto_backbone(gguf_path=...)
  - backbone.generate()
  - scripts/train_transformer_cpu.py --gguf PATH

Usage:
    python scripts/create_tiny_gguf.py [--out models/tiny_test.gguf]
"""
from __future__ import annotations
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Tiny model hyper-parameters
# ---------------------------------------------------------------------------
N_VOCAB    = 257        # 256 byte tokens + 1 merge result
N_CTX      = 512        # context length (must match position_embd.weight rows)
N_EMBD     = 64         # embedding dim
N_LAYER    = 2          # transformer layers
N_HEAD     = 4          # attention heads
N_FF       = N_EMBD * 4 # FFN intermediate
HEAD_DIM   = N_EMBD // N_HEAD


def _rand(shape, scale=0.02, seed=None, rng=None):
    """Return float32 random tensor."""
    if rng is None:
        rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _ones(shape):
    return np.ones(shape, dtype=np.float32)


def _zeros(shape):
    return np.zeros(shape, dtype=np.float32)


# ---------------------------------------------------------------------------
# Build byte-level GPT-2 tokenizer lists
# ---------------------------------------------------------------------------

def _gpt2_b2u() -> dict:
    """GPT-2 bytes-to-unicode mapping (maps each byte to a printable code point)."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, cs))


def build_byte_tokenizer():
    """257-token GPT-2-style byte vocabulary: 256 byte tokens + 1 merge result.
    GPT-2 tokenizer requires at least one BPE merge to satisfy llama.cpp's
    strict-mode check; we add the trivial merge 'a b' → 'ab' (token 256).
    """
    b2u = _gpt2_b2u()
    tokens = [chr(b2u[i]) for i in range(256)] + ["ab"]
    scores = [0.0] * N_VOCAB
    token_types = [1] * N_VOCAB  # GGUF_TOKEN_TYPE_NORMAL = 1
    return tokens, scores, token_types


# ---------------------------------------------------------------------------
# Write GGUF
# ---------------------------------------------------------------------------

def create_gguf(out_path: str) -> None:
    try:
        import gguf
    except ImportError:
        print("ERROR: pip install gguf")
        sys.exit(1)

    rng = np.random.default_rng(42)
    out_path = str(out_path)

    writer = gguf.GGUFWriter(out_path, arch="gpt2")

    # ── Metadata ──────────────────────────────────────────────────────────
    writer.add_uint32("gpt2.context_length",             N_CTX)
    writer.add_uint32("gpt2.embedding_length",            N_EMBD)
    writer.add_uint32("gpt2.block_count",                 N_LAYER)
    writer.add_uint32("gpt2.feed_forward_length",         N_FF)
    writer.add_uint32("gpt2.attention.head_count",        N_HEAD)
    writer.add_uint32("gpt2.attention.head_count_kv",     N_HEAD)
    writer.add_float32("gpt2.attention.layer_norm_epsilon", 1e-5)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokens, scores, token_types = build_byte_tokenizer()
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.scores", scores)
    writer.add_array("tokenizer.ggml.token_type", token_types)
    writer.add_uint32("tokenizer.ggml.bos_token_id", 1)
    writer.add_uint32("tokenizer.ggml.eos_token_id", 2)
    writer.add_bool("tokenizer.ggml.add_space_prefix", False)
    writer.add_array("tokenizer.ggml.merges", ["a b"])  # one merge: a+b→ab (token 256)

    # ── Tensors ───────────────────────────────────────────────────────────
    # Embeddings
    writer.add_tensor("token_embd.weight",    _rand([N_VOCAB, N_EMBD],  rng=rng))
    writer.add_tensor("position_embd.weight", _rand([N_CTX,   N_EMBD],  rng=rng))

    # Per-layer
    for i in range(N_LAYER):
        p = f"blk.{i}"
        # Attention norm (layer norm weights + bias)
        writer.add_tensor(f"{p}.attn_norm.weight", _ones([N_EMBD]))
        writer.add_tensor(f"{p}.attn_norm.bias",   _zeros([N_EMBD]))
        # QKV projection (combined)
        writer.add_tensor(f"{p}.attn_qkv.weight",  _rand([3 * N_EMBD, N_EMBD], rng=rng))
        writer.add_tensor(f"{p}.attn_qkv.bias",    _zeros([3 * N_EMBD]))
        # Output projection
        writer.add_tensor(f"{p}.attn_output.weight", _rand([N_EMBD, N_EMBD], rng=rng))
        writer.add_tensor(f"{p}.attn_output.bias",   _zeros([N_EMBD]))
        # FFN norm
        writer.add_tensor(f"{p}.ffn_norm.weight",  _ones([N_EMBD]))
        writer.add_tensor(f"{p}.ffn_norm.bias",    _zeros([N_EMBD]))
        # FFN up/down
        writer.add_tensor(f"{p}.ffn_up.weight",    _rand([N_FF,   N_EMBD], rng=rng))
        writer.add_tensor(f"{p}.ffn_up.bias",      _zeros([N_FF]))
        writer.add_tensor(f"{p}.ffn_down.weight",  _rand([N_EMBD, N_FF],   rng=rng))
        writer.add_tensor(f"{p}.ffn_down.bias",    _zeros([N_EMBD]))

    # Output head
    writer.add_tensor("output_norm.weight", _ones([N_EMBD]))
    writer.add_tensor("output_norm.bias",   _zeros([N_EMBD]))
    writer.add_tensor("output.weight",      _rand([N_VOCAB, N_EMBD], rng=rng))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"Created: {out_path}  ({size_mb:.2f} MB)")
    print(f"  arch={N_LAYER}L/{N_HEAD}H/d{N_EMBD}  vocab={N_VOCAB}  ctx={N_CTX}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(REPO_ROOT / "models" / "tiny_test.gguf"))
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    create_gguf(args.out)


if __name__ == "__main__":
    main()

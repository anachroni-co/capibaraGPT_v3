#!/usr/bin/env python3
"""Cascade speculative decoding inference for cunca-v2 models.

2-model cascade: Small (draft) → Large+LoRA (verify)

Algorithm: Leviathan et al. 2023 "Fast Inference from Transformers via
Speculative Decoding" (arXiv:2211.17192).

At each step:
  1. Draft model generates K tokens autoregressively (cheap).
  2. Target model runs ONE forward pass over the K draft tokens (parallel).
  3. Each draft token i is accepted with prob min(1, p_target / q_draft).
  4. First rejection → resample from corrected distribution; discard rest.
  5. All accepted → sample one bonus token from target.

Expected tokens per target call: (1 - α^K) / (1 - α) ≈ K·α + 1
Expected speedup: (K·α + 1) / (K·c + 1)  where c = cost(draft)/cost(target)

Usage:
    python scripts/cascade_inference.py \\
        --draft  checkpoints/cunca_v2_small_gl/soup_uniform.pkl \\
        --target checkpoints/cunca_v2_gl/soup_uniform.pkl \\
        --lora   checkpoints/lora/gestoria_gl/lora_final.pkl \\
        --prompt "Como me dou de alta como autónomo en Galicia?" \\
        --k 5 --max-tokens 512 --temperature 0.7

    # Benchmark: acceptance rate + speedup estimate
    python scripts/cascade_inference.py \\
        --draft  checkpoints/cunca_v2_small_gl/soup_uniform.pkl \\
        --target checkpoints/cunca_v2_gl/soup_uniform.pkl \\
        --benchmark --n-samples 20

    # Without LoRA (base cunca-v2 target)
    python scripts/cascade_inference.py \\
        --draft  checkpoints/cunca_v2_small_gl/soup_uniform.pkl \\
        --target checkpoints/cunca_v2_gl/soup_uniform.pkl \\
        --prompt "Que é o ITP en Galicia?"
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cascade")

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Presets (must match training) ────────────────────────────────────────────

PRESETS = {
    "smoke":  dict(hidden_size=256,  num_layers=4,  num_heads=4,  seq_len=256),
    "small":  dict(hidden_size=512,  num_layers=8,  num_heads=8,  seq_len=512),
    "medium": dict(hidden_size=768,  num_layers=12, num_heads=12, seq_len=1024),
    "full":   dict(hidden_size=1024, num_layers=12, num_heads=16, seq_len=2048),
    "large":  dict(hidden_size=1280, num_layers=24, num_heads=20, seq_len=1024),
}

VOCAB_SIZE = 512
PAD_ID     = 256
BOS_ID     = 257
EOS_ID     = 258


# ── Tokenizer (byte-level) ────────────────────────────────────────────

def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(ids: list[int]) -> str:
    return bytes(t for t in ids if t < 256).decode("utf-8", errors="replace")


# ── Model loading ─────────────────────────────────────────────────

class LoadedModel(NamedTuple):
    params:      dict
    model:       object
    preset:      str
    seq_len:     int
    apply_jit:   object = None
    bucket_fns:  dict   = None   # {bucket_size: jit_fn}


def _detect_preset(params: dict) -> str:
    """Infer preset from embedding weight shape."""
    try:
        import jax
        leaves = jax.tree_util.tree_leaves(params)
        # Embedding table: (vocab_size, hidden_size)
        for leaf in leaves:
            if hasattr(leaf, "shape") and len(leaf.shape) == 2:
                if leaf.shape[0] == VOCAB_SIZE:
                    hidden = leaf.shape[1]
                    for name, cfg in PRESETS.items():
                        if cfg["hidden_size"] == hidden:
                            return name
    except Exception:
        pass
    return "large"


def load_model(ckpt_path: str, lora_path: str | None = None) -> LoadedModel:
    """Load model from checkpoint, optionally merging a LoRA adapter."""
    import jax
    from models.slim_200m import Slim200M, ModelConfig

    logger.info("Loading %s", ckpt_path)
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    params = ckpt.get("params", ckpt)

    preset_name = _detect_preset(params)
    preset = PRESETS[preset_name]
    logger.info("Detected preset: %s (hidden=%d)", preset_name, preset["hidden_size"])

    if lora_path:
        logger.info("Merging LoRA from %s", lora_path)
        from scripts.lora_finetune import merge_lora
        with open(lora_path, "rb") as f:
            lora_ckpt = pickle.load(f)
        params = merge_lora(params, lora_ckpt["lora"])

    cfg = ModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=preset["hidden_size"],
        num_layers=preset["num_layers"],
        num_heads=preset["num_heads"],
        max_seq_len=preset["seq_len"],
        dropout_rate=0.0,
    )
    model = Slim200M(cfg)

    # Compile one JIT function per bucket — avoids recompilation AND
    # avoids padding to full seq_len for short contexts.
    import jax
    seq_len = preset["seq_len"]
    raw_buckets = [64, 128, 256, 512, 1024]
    buckets = [b for b in raw_buckets if b <= seq_len]
    if buckets[-1] < seq_len:
        buckets.append(seq_len)

    bucket_fns: dict = {}
    _apply_jit = jax.jit(model.apply)
    for b in buckets:
        logger.info("  Compiling bucket=%d ...", b)
        dummy = np.zeros((1, b), dtype=np.int32)
        _ = _apply_jit(params, dummy).block_until_ready()
        bucket_fns[b] = _apply_jit
    logger.info("Model ready (%s) — %d buckets %s", preset_name, len(buckets), buckets)

    return LoadedModel(params=params, model=model,
                       preset=preset_name, seq_len=seq_len,
                       apply_jit=_apply_jit, bucket_fns=bucket_fns)


# ── Sampling helpers ────────────────────────────────────────────────

def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max()
    e = np.exp(logits)
    return e / e.sum()


def _sample(probs: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if temperature <= 0.0:
        return int(np.argmax(probs))
    logits = np.log(probs + 1e-10) / temperature
    p = _softmax(logits)
    return int(rng.choice(len(p), p=p))


def _pad_and_apply(m: LoadedModel, input_ids: np.ndarray) -> np.ndarray:
    """Pad to smallest bucket >= T and run cached JIT. Shape: (1, bucket, vocab)"""
    import jax.numpy as jnp
    T = len(input_ids)
    if m.bucket_fns:
        bucket = next((b for b in sorted(m.bucket_fns) if b >= T), m.seq_len)
    else:
        bucket = m.seq_len
    padded = np.zeros((bucket,), dtype=np.int32)
    padded[:T] = input_ids
    ids = jnp.array(padded[np.newaxis, :])
    fn = m.bucket_fns.get(bucket, m.apply_jit) if m.bucket_fns else m.model.apply
    return np.array(fn(m.params, ids))           # (1, bucket, vocab)


def _get_logits(m: LoadedModel, input_ids: np.ndarray) -> np.ndarray:
    """Return logits at last real token position. Shape: (vocab,)"""
    T = len(input_ids)
    logits = _pad_and_apply(m, input_ids)
    return logits[0, T - 1, :]


def _get_logits_all(m: LoadedModel, input_ids: np.ndarray) -> np.ndarray:
    """Return logits at ALL real token positions. Shape: (T, vocab)"""
    T = len(input_ids)
    logits = _pad_and_apply(m, input_ids)
    return logits[0, :T, :]


# ── Core speculative decoding step ─────────────────────────────────────────

def speculative_step(
    context: list[int],
    draft_m: LoadedModel,
    target_m: LoadedModel,
    k: int,
    temperature: float,
    rng: np.random.Generator,
) -> tuple[list[int], int, int]:
    """One speculative decoding step.

    Returns:
        new_tokens  — accepted (+ possibly bonus) tokens
        n_accepted  — number of draft tokens accepted (for stats)
        n_draft     — number of draft tokens proposed (k)
    """
    seq_len = min(draft_m.seq_len, target_m.seq_len)

    # ── Phase 1: draft model generates k tokens ─────────────────────────────────
    draft_tokens: list[int] = []
    draft_probs:  list[float] = []

    draft_seq_len = draft_m.seq_len
    ctx = list(context[-draft_seq_len:])
    for _ in range(k):
        ids = np.array(ctx[-draft_seq_len:], dtype=np.int32)
        logits = _get_logits(draft_m, ids)
        p = _softmax(logits)
        tok = _sample(p, temperature, rng)
        draft_tokens.append(tok)
        draft_probs.append(float(p[tok]))
        ctx.append(tok)
        if tok == EOS_ID:
            break

    # ── Phase 2: target model — single forward pass over context + k drafts ──
    target_ctx = np.array((context + draft_tokens)[-seq_len:], dtype=np.int32)
    target_logits_all = _get_logits_all(target_m, target_ctx)

    # Positions in target_logits_all corresponding to each draft token
    # draft_token[i] was generated at position len(context)-1+i in the sequence
    # target logits at position j predict token j+1
    context_trimmed_len = len((context + draft_tokens)[-seq_len:]) - len(draft_tokens)
    # logit[context_trimmed_len - 1 + i] predicts draft_tokens[i]
    base = context_trimmed_len - 1

    # ── Phase 3: accept / reject ────────────────────────────────────────────
    accepted: list[int] = []
    n_accepted = 0

    for i, (tok, q_tok) in enumerate(zip(draft_tokens, draft_probs)):
        target_p_all = _softmax(target_logits_all[base + i])
        p_tok = float(target_p_all[tok])

        accept_prob = min(1.0, p_tok / (q_tok + 1e-10))
        if rng.random() < accept_prob:
            accepted.append(tok)
            n_accepted += 1
            if tok == EOS_ID:
                return accepted, n_accepted, len(draft_tokens)
        else:
            # Resample from corrected distribution: max(0, p - q) normalised
            target_p_full = _softmax(target_logits_all[base + i])
            import jax.numpy as jnp
            draft_ids = np.array(ctx[:context_trimmed_len + i], dtype=np.int32)
            draft_logits = _get_logits(draft_m, draft_ids)
            q_full = _softmax(draft_logits)
            corrected = np.maximum(0.0, target_p_full - q_full)
            s = corrected.sum()
            if s > 1e-10:
                corrected /= s
                tok_new = _sample(corrected, 1.0, rng)
            else:
                tok_new = _sample(target_p_full, temperature, rng)
            accepted.append(tok_new)
            return accepted, n_accepted, len(draft_tokens)

    # All k accepted → bonus token from target
    bonus_logits = _softmax(target_logits_all[base + len(draft_tokens)])
    bonus = _sample(bonus_logits, temperature, rng)
    accepted.append(bonus)

    return accepted, n_accepted, len(draft_tokens)


# ── Main generation loop ──────────────────────────────────────────────────

def cascade_generate(
    prompt: str,
    draft_m: LoadedModel,
    target_m: LoadedModel,
    k: int = 5,
    max_tokens: int = 512,
    temperature: float = 0.7,
    seed: int = 42,
    stream: bool = True,
) -> tuple[str, dict]:
    """Generate text using 2-model speculative decoding.

    Returns:
        text      — generated string
        stats     — acceptance_rate, speedup_estimate, tokens_per_step, elapsed_s
    """
    rng = np.random.default_rng(seed)

    seq_len = min(draft_m.seq_len, target_m.seq_len)
    context = ([BOS_ID] + encode(prompt))[-seq_len:]
    generated: list[int] = []

    total_drafted  = 0
    total_accepted = 0
    total_steps    = 0
    t0 = time.perf_counter()

    if stream:
        print(f"\n[Prompt] {prompt}\n[Response] ", end="", flush=True)

    while len(generated) < max_tokens:
        new_toks, n_acc, n_draft = speculative_step(
            context, draft_m, target_m, k, temperature, rng
        )

        total_drafted  += n_draft
        total_accepted += n_acc
        total_steps    += 1

        for tok in new_toks:
            generated.append(tok)
            if stream:
                ch = decode([tok])
                print(ch, end="", flush=True)
            if tok == EOS_ID or len(generated) >= max_tokens:
                break

        context = context + new_toks
        if generated and generated[-1] == EOS_ID:
            break

    elapsed = time.perf_counter() - t0
    if stream:
        print()

    alpha = total_accepted / total_drafted if total_drafted else 0.0
    toks_per_step = len(generated) / total_steps if total_steps else 0.0

    # Theoretical speedup: (toks_per_step) / 1  compared to target-only (1 tok/step)
    speedup = toks_per_step  # relative to 1 token per target forward pass

    stats = {
        "acceptance_rate":   round(alpha, 3),
        "tokens_per_step":   round(toks_per_step, 2),
        "speedup_estimate":  round(speedup, 2),
        "total_tokens":      len(generated),
        "total_steps":       total_steps,
        "elapsed_s":         round(elapsed, 2),
        "tok_per_s":         round(len(generated) / elapsed, 1) if elapsed > 0 else 0,
    }

    return decode(generated), stats


# ── Benchmark ─────────────────────────────────────────────────────────────────

BENCHMARK_PROMPTS = [
    "Como me dou de alta como autónomo en Galicia?",
    "Que deducións autonómicas de Galicia podo aplicar no IRPF?",
    "Canto custa constituír unha SRL en Galicia?",
    "Como presento o modelo 303 do IVE?",
    "Que bonificacións ten o imposto de sucesións en Galicia?",
    "¿Cómo funciona la baja por enfermedad común para autónomos?",
    "¿Qué tipos de contrato existen tras la reforma laboral de 2022?",
    "Redacta un contrato de arrendamento de vivenda.",
    "Como solicito unha axuda á rehabilitación de vivenda na Xunta?",
    "¿Cuál es la diferencia entre autónomo y SL en términos fiscales?",
]


def run_benchmark(
    draft_m: LoadedModel,
    target_m: LoadedModel,
    n_samples: int = 10,
    k: int = 5,
    max_tokens: int = 100,
) -> dict:
    logger.info("Benchmark: %d samples, k=%d, max_tokens=%d", n_samples, k, max_tokens)

    all_alpha:   list[float] = []
    all_speedup: list[float] = []
    all_tps:     list[float] = []

    prompts = (BENCHMARK_PROMPTS * 10)[:n_samples]

    for i, prompt in enumerate(prompts):
        _, stats = cascade_generate(
            prompt, draft_m, target_m,
            k=k, max_tokens=max_tokens, temperature=0.7,
            seed=i, stream=False,
        )
        all_alpha.append(stats["acceptance_rate"])
        all_speedup.append(stats["speedup_estimate"])
        all_tps.append(stats["tok_per_s"])
        logger.info(
            "  [%2d/%d] α=%.2f  toks/step=%.1f  speedup=%.1fx  tok/s=%.0f",
            i + 1, n_samples,
            stats["acceptance_rate"],
            stats["tokens_per_step"],
            stats["speedup_estimate"],
            stats["tok_per_s"],
        )

    result = {
        "mean_acceptance_rate":  round(float(np.mean(all_alpha)), 3),
        "mean_speedup":          round(float(np.mean(all_speedup)), 2),
        "mean_tok_per_s":        round(float(np.mean(all_tps)), 1),
        "cascade_viable":        float(np.mean(all_alpha)) >= 0.60,
    }

    print("\n── Benchmark Results ────────────────────────────────────")
    print(f"  Mean acceptance rate α : {result['mean_acceptance_rate']:.1%}")
    print(f"  Mean speedup           : {result['mean_speedup']:.1f}x tokens/target-step")
    print(f"  Mean throughput        : {result['mean_tok_per_s']:.0f} tok/s")
    print(f"  Cascade viable (α≥60%) : {'✓ SÍ' if result['cascade_viable'] else '✗ NO — considerar DAPT adicional'}")
    print("────────────────────────────────────────────────────────\n")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--draft",  required=True,
                        help="Draft model checkpoint (Small)")
    parser.add_argument("--target", required=True,
                        help="Target model checkpoint (Large)")
    parser.add_argument("--lora",       default=None,
                        help="LoRA adapter to merge into target (optional)")
    parser.add_argument("--draft-lora", default=None,
                        help="LoRA adapter to merge into draft (optional)")
    parser.add_argument("--prompt", default=None,
                        help="Prompt to generate from")
    parser.add_argument("--k",            type=int,   default=5,
                        help="Draft tokens per step (default: 5)")
    parser.add_argument("--max-tokens",   type=int,   default=512)
    parser.add_argument("--temperature",  type=float, default=0.7)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--no-stream",    action="store_true",
                        help="Disable streaming output")
    parser.add_argument("--benchmark",    action="store_true",
                        help="Run acceptance rate benchmark instead of generating")
    parser.add_argument("--n-samples",    type=int,   default=10,
                        help="Benchmark samples (default: 10)")
    args = parser.parse_args()

    # JAX CPU setup
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("OMP_NUM_THREADS", "32")
    os.environ.setdefault("XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=true "
        "--xla_cpu_enable_fast_math=true "
        "--xla_force_host_platform_device_count=1")

    draft_m  = load_model(args.draft,  lora_path=args.draft_lora)
    target_m = load_model(args.target, lora_path=args.lora)

    if args.benchmark:
        run_benchmark(draft_m, target_m,
                      n_samples=args.n_samples, k=args.k,
                      max_tokens=min(args.max_tokens, 150))
        return

    if not args.prompt:
        parser.error("--prompt required (or --benchmark)")

    text, stats = cascade_generate(
        args.prompt, draft_m, target_m,
        k=args.k,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        stream=not args.no_stream,
    )

    print("\n── Stats ──────────────────────────────────────────────────")
    print(f"  Acceptance rate α : {stats['acceptance_rate']:.1%}")
    print(f"  Tokens / step     : {stats['tokens_per_step']:.1f}  (k={args.k})")
    print(f"  Speedup estimate  : {stats['speedup_estimate']:.1f}x")
    print(f"  Throughput        : {stats['tok_per_s']:.0f} tok/s")
    print(f"  Total tokens      : {stats['total_tokens']} in {stats['elapsed_s']}s")
    print("────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()

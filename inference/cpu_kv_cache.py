"""inference/cpu_kv_cache.py

Incremental hidden-state cache for CPU autoregressive inference.

Problem solved
--------------
Without a cache, each decode step calls backbone.forward(full_context),
recomputing embeddings for all previous tokens even though they haven't
changed.  For a context of length n the work is O(n) per step → O(n²)
total for a sequence.  With the cache it is O(1) per step → O(n) total.

Design
------
ByteLM has no cross-token attention: h[t] = ReLU(W_emb[id_t]) depends only
on id_t.  The cache stores every h[t] as it is produced.  At step t+1 we
compute only the new token's embedding, then read h[t] (h_prev) and h[t+1]
(h_curr) from the cache to drive the L-MTP heads.

Generalisation note
-------------------
A Transformer / Mamba backbone would override _step() to do a proper KV
lookup or SSM state update.  The interface (push / h_prev / h_curr) is
intentionally the same so callers don't change when the backbone changes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Cache container
# ---------------------------------------------------------------------------

class HiddenStateCache:
    """Ring-buffer of hidden states for one sequence.

    Stores up to `max_len` states.  Older states are evicted when full
    (sliding-window attention / SSM models only look back max_len tokens).

    For ByteLM with no attention, max_len can be set to 2 (only h_prev and
    h_curr needed) for minimal memory.  Set to None or 0 to keep all states.
    """

    def __init__(self, hidden: int, max_len: int = 0) -> None:
        self.H = hidden
        self.max_len = max_len  # 0 = unlimited
        self._states: list[np.ndarray] = []   # each (H,)
        self._total_pushed = 0

    # ------------------------------------------------------------------

    def push(self, h: np.ndarray) -> None:
        """Append hidden state h (shape (H,)) to the cache."""
        self._states.append(h.astype(np.float32))
        self._total_pushed += 1
        if self.max_len and len(self._states) > self.max_len:
            self._states.pop(0)

    def get(self, offset: int) -> Optional[np.ndarray]:
        """Return state at offset from the *end* (0 = most recent)."""
        idx = -(offset + 1)
        if abs(idx) > len(self._states):
            return None
        return self._states[idx]

    @property
    def length(self) -> int:
        return self._total_pushed

    @property
    def h_curr(self) -> Optional[np.ndarray]:
        return self.get(0)

    @property
    def h_prev(self) -> Optional[np.ndarray]:
        return self.get(1)

    def reset(self) -> None:
        self._states.clear()
        self._total_pushed = 0


# ---------------------------------------------------------------------------
# Cached decoder
# ---------------------------------------------------------------------------

@dataclass
class CacheDecodeConfig:
    max_new_tokens: int = 128
    greedy: bool = True          # False = multinomial sampling
    temperature: float = 1.0
    top_k: int = 0               # 0 = disabled
    eos_token_id: Optional[int] = None
    cache_max_len: int = 0       # 0 = keep all hidden states


class LMTPCachedDecoder:
    """O(1)-per-step autoregressive decoder for ByteLM + LMTPHeads.

    Usage::

        decoder = LMTPCachedDecoder(backbone, heads)
        tokens  = decoder.generate(prompt_ids, max_new_tokens=256)
        stats   = decoder.last_stats
    """

    def __init__(self, backbone, heads, cfg: Optional[CacheDecodeConfig] = None):
        self.backbone = backbone
        self.heads = heads
        self.cfg = cfg or CacheDecodeConfig()
        self.last_stats: dict = {}

    # ------------------------------------------------------------------
    # Core step — O(1) regardless of sequence length
    # ------------------------------------------------------------------

    def _step_single(self, token_id: int, cache: HiddenStateCache) -> tuple[int, np.ndarray]:
        """Process one new token; return (next_token_id, h_curr).

        Only computes the embedding for token_id — does NOT reprocess
        previous tokens.
        """
        # O(1): single embedding lookup + ReLU
        h_curr = np.maximum(self.backbone.W_emb[token_id], 0)  # (H,)
        cache.push(h_curr)

        # NTP logits for this position only
        logits = h_curr @ self.backbone.W_out   # (V,)
        next_tok = self._sample(logits)
        return next_tok, h_curr

    def _sample(self, logits: np.ndarray) -> int:
        cfg = self.cfg
        if cfg.temperature != 1.0:
            logits = logits / max(cfg.temperature, 1e-8)
        if cfg.top_k > 0:
            thresh = np.sort(logits)[-cfg.top_k]
            logits = np.where(logits < thresh, -1e9, logits)
        if not cfg.greedy:
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            return int(np.random.choice(len(probs), p=probs))
        return int(np.argmax(logits))

    # ------------------------------------------------------------------
    # L-MTP multi-token step
    # ------------------------------------------------------------------

    def _lmtp_step(self, cache: HiddenStateCache) -> list[int]:
        """Produce tokens_per_step tokens using L-MTP heads.

        Reads h_prev and h_curr directly from the cache — zero recomputation.
        """
        h_curr = cache.h_curr
        h_prev = cache.h_prev
        if h_prev is None:
            h_prev = np.zeros_like(h_curr)

        toks: list[int] = []
        # Head 0 prediction (offset leap_k)
        for i, W in enumerate(self.heads.W):
            x = np.concatenate([h_prev, h_curr])  # (2H,)
            logits = x @ W                          # (V,)
            toks.append(self._sample(logits))

        # Fill leap_k-1 gap tokens between head predictions via NTP
        full: list[int] = []
        for i, anchor_tok in enumerate(toks):
            full.append(anchor_tok)
            if i < self.heads.n_head - 1:
                # Intermediate positions: greedy NTP from anchor hidden
                h_anc = np.maximum(self.backbone.W_emb[anchor_tok], 0)
                for _ in range(self.heads.leap_k - 1):
                    ntp_logits = h_anc @ self.backbone.W_out
                    nxt = int(np.argmax(ntp_logits))
                    full.append(nxt)
                    h_anc = np.maximum(self.backbone.W_emb[nxt], 0)
        return full[: self.heads.tokens_per_step()]

    # ------------------------------------------------------------------
    # Public generate
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_ids: list[int] | np.ndarray,
        max_new_tokens: Optional[int] = None,
    ) -> list[int]:
        """Generate tokens after the prompt.

        Returns the full sequence (prompt + generated).
        """
        prompt_ids = list(prompt_ids)
        budget = max_new_tokens or self.cfg.max_new_tokens
        cache = HiddenStateCache(self.backbone.H, self.cfg.cache_max_len)

        # --- Prefill: push all prompt tokens into cache (O(n), one time) ---
        t_prefill = time.perf_counter()
        for tid in prompt_ids:
            h = np.maximum(self.backbone.W_emb[tid], 0)
            cache.push(h)
        prefill_ms = (time.perf_counter() - t_prefill) * 1000

        generated = list(prompt_ids)
        tokens_produced = 0
        decode_steps = 0
        t_decode = time.perf_counter()

        while tokens_produced < budget:
            # Use L-MTP if heads available, else plain NTP
            if self.heads is not None:
                step_toks = self._lmtp_step(cache)
            else:
                last_tok = generated[-1]
                nxt, _ = self._step_single(last_tok, cache)
                step_toks = [nxt]

            remaining = budget - tokens_produced
            step_toks = step_toks[:remaining]

            # Push each generated token into the cache
            for tok in step_toks:
                h = np.maximum(self.backbone.W_emb[tok], 0)
                cache.push(h)

            generated.extend(step_toks)
            tokens_produced += len(step_toks)
            decode_steps += 1

            if self.cfg.eos_token_id is not None:
                if self.cfg.eos_token_id in step_toks:
                    break

        decode_ms = (time.perf_counter() - t_decode) * 1000

        self.last_stats = {
            "prompt_len": len(prompt_ids),
            "tokens_generated": tokens_produced,
            "decode_steps": decode_steps,
            "tokens_per_step": self.heads.tokens_per_step() if self.heads else 1,
            "prefill_ms": round(prefill_ms, 2),
            "decode_ms": round(decode_ms, 2),
            "tok_per_s": round(tokens_produced / (decode_ms / 1000 + 1e-9), 1),
        }
        return generated


# ---------------------------------------------------------------------------
# Benchmark: cached vs. naive (full recompute each step)
# ---------------------------------------------------------------------------

def benchmark_cache_vs_naive(backbone, heads, prompt_len: int = 64,
                              new_tokens: int = 256, n_runs: int = 5) -> dict:
    """Compare throughput of cached decoder vs. naive full-recompute.

    Returns dict with tok/s and speedup ratio.
    """
    import time

    prompt = list(np.random.randint(0, backbone.vocab, size=prompt_len))

    # --- Cached ---
    decoder = LMTPCachedDecoder(backbone, heads)
    # Warm-up
    decoder.generate(prompt, max_new_tokens=new_tokens)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        decoder.generate(prompt, max_new_tokens=new_tokens)
    cached_s = (time.perf_counter() - t0) / n_runs
    cached_tps = new_tokens / cached_s

    # --- Naive (full context recompute each step) ---
    tps_naive = heads.tokens_per_step() if heads else 1

    def naive_generate(prompt, new_tokens):
        generated = list(prompt)
        h_prev = np.zeros((1, 1, backbone.H), dtype=np.float32)
        for _ in range(max(1, new_tokens // tps_naive)):
            ctx = np.array(generated[-64:], dtype=np.int32).reshape(1, -1)
            _, h_ctx = backbone.forward(ctx)   # O(len(ctx)) every step
            h_curr = h_ctx[:, -1:, :]
            # L-MTP heads
            step_toks = []
            for i, W in enumerate(heads.W):
                x = np.concatenate([h_prev, h_curr], axis=-1).reshape(1, -1)
                logits = (x @ W).flatten()
                step_toks.append(int(np.argmax(logits)))
            full = step_toks[:tps_naive]
            generated.extend(full)
            h_prev = h_curr
        return generated

    # Warm-up
    naive_generate(prompt, new_tokens)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        naive_generate(prompt, new_tokens)
    naive_s = (time.perf_counter() - t0) / n_runs
    naive_tps = new_tokens / naive_s

    return {
        "cached_tok_per_s": round(cached_tps, 1),
        "naive_tok_per_s": round(naive_tps, 1),
        "speedup": round(cached_tps / naive_tps, 2),
        "new_tokens": new_tokens,
        "prompt_len": prompt_len,
    }

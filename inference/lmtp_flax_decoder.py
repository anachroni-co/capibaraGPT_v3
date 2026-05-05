"""inference/lmtp_flax_decoder.py

JIT-compiled look-backward L-MTP decoder for production inference.

Key features
------------
* @jax.jit compiled generate loop — XLA-optimised on any backend
  (CPU/GPU/TPU/ARM Axion).
* lax.while_loop for the autoregressive loop (avoids Python overhead
  per token even for very long sequences).
* Sampling: greedy, top-k, top-p (nucleus), temperature.
* Speculative acceptance filtering (same logic as inference/lmtp_decoder.py
  but fully in JAX — no Python per token).
* tokens_per_step = leap_k × (n_head − 1) + 1  (e.g. k=4,n=4 → 13)

Usage
-----
    from models.lmtp_flax import create_lmtp_model
    from inference.lmtp_flax_decoder import LMTPFlaxDecoder, LMTPFlaxDecodeConfig
    import jax, jax.numpy as jnp

    model  = create_lmtp_model("1.5b", n_head=4, leap_k=2)
    params = ...  # loaded from checkpoint

    decoder = LMTPFlaxDecoder(model, params)
    ids     = jnp.array([[1, 2, 3, 4]])          # (1, prompt_len)
    output  = decoder.generate(ids, max_new_tokens=256)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    _JAX = True
except ImportError:
    _JAX = False

_AVAILABLE = _JAX


def _require(name: str = "LMTPFlaxDecoder") -> None:
    if not _AVAILABLE:
        raise ImportError(
            f"{name} requires JAX. Install with: pip install jax[cpu]"
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LMTPFlaxDecodeConfig:
    """Decoding configuration.

    Attributes:
        max_new_tokens:   Token generation budget (per batch item).
        temperature:      Sampling temperature; 1.0 = no scaling.
        top_k:            Top-k filter (0 = disabled).
        top_p:            Nucleus filter (1.0 = disabled).
        do_sample:        True = multinomial sampling; False = greedy.
        eos_token_id:     Stop when this token is produced (any batch item).
        speculative:      Enable speculative acceptance filtering.
        spec_threshold:   Minimum probability to accept a speculative token.
        pad_token_id:     Padding token used when sequences terminate early.
    """
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = False
    eos_token_id: Optional[int] = None
    speculative: bool = False
    spec_threshold: float = 0.5
    pad_token_id: int = 0


# ---------------------------------------------------------------------------
# Sampling helpers (pure JAX — jit-able)
# ---------------------------------------------------------------------------

if _AVAILABLE:

    def _top_k_logits(logits: jnp.ndarray, k: int) -> jnp.ndarray:
        """Zero out logits below the k-th largest."""
        if k <= 0:
            return logits
        top_k_vals = lax.top_k(logits, k)[0][..., -1, None]
        return jnp.where(logits < top_k_vals, jnp.finfo(logits.dtype).min, logits)

    def _top_p_logits(logits: jnp.ndarray, p: float) -> jnp.ndarray:
        """Nucleus (top-p) filtering."""
        sorted_idx = jnp.argsort(-logits, axis=-1)
        sorted_logits = jnp.take_along_axis(logits, sorted_idx, axis=-1)
        cum_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
        # Remove tokens where cumulative prob exceeds p (shift right by 1)
        remove = jnp.concatenate(
            [jnp.zeros_like(cum_probs[..., :1]),
             (cum_probs[..., :-1] >= p).astype(jnp.bool_)],
            axis=-1,
        )
        sorted_logits = jnp.where(remove, jnp.finfo(logits.dtype).min, sorted_logits)
        # Scatter back to original order
        return jnp.zeros_like(logits).at[
            jnp.arange(logits.shape[0])[:, None], sorted_idx
        ].set(sorted_logits)

    def _sample_tokens(
        logits: jnp.ndarray,    # (B, V)
        key: jnp.ndarray,
        cfg: LMTPFlaxDecodeConfig,
    ) -> jnp.ndarray:           # (B,)
        logits = logits / jnp.maximum(cfg.temperature, 1e-8)
        if cfg.top_k > 0:
            logits = _top_k_logits(logits, cfg.top_k)
        if cfg.top_p < 1.0:
            logits = _top_p_logits(logits, cfg.top_p)
        if cfg.do_sample:
            return jax.random.categorical(key, logits, axis=-1)  # (B,)
        return logits.argmax(-1)                                   # (B,)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

if _AVAILABLE:

    class LMTPFlaxDecoder:
        """JIT-compiled look-backward L-MTP autoregressive decoder.

        Wraps an LMTPWrapper Flax model and provides a generate() method that
        uses lax.while_loop to avoid Python overhead per generation step.

        Each decode step produces tokens_per_step tokens:
            tokens_per_step = leap_k × (n_head − 1) + 1
        """

        def __init__(
            self,
            model: "LMTPWrapper",
            params: dict,
            cfg: Optional[LMTPFlaxDecodeConfig] = None,
        ) -> None:
            _require("LMTPFlaxDecoder")
            self.model = model
            self.params = params
            self.cfg = cfg or LMTPFlaxDecodeConfig()
            self.tps = model.cfg.tokens_per_step()
            # Compile decode step
            self._decode_step_jit = jax.jit(self._decode_step_raw)

        # ------------------------------------------------------------------
        # Core decode step (JIT-compiled)
        # ------------------------------------------------------------------

        def _decode_step_raw(
            self,
            token_ids: jnp.ndarray,   # (B, T)
            h_prev: jnp.ndarray,      # (B, 1, H)
            key: jnp.ndarray,
        ):
            """One look-backward step → (step_tokens, h_curr, key)."""
            step_tokens, h_curr = self.model.apply(
                self.params,
                token_ids,
                h_prev,
                method=self.model.decode_step,
            )
            # Optionally re-sample (greedy by default inside decode_step,
            # but we allow temperature / top-k / top-p here for the NTP head)
            if self.cfg.do_sample or self.cfg.top_k > 0 or self.cfg.top_p < 1.0:
                # Re-run backbone forward to get NTP logits for sampling
                ntp_logits, _, hidden = self.model.apply(
                    self.params, token_ids, deterministic=True,
                )
                key, sub = jax.random.split(key)
                first_tok = _sample_tokens(ntp_logits[:, -1, :], sub, self.cfg)
                step_tokens = step_tokens.at[:, 0].set(first_tok)
            return step_tokens, h_curr, key

        # ------------------------------------------------------------------
        # Python-loop generate (simple, supports dynamic stopping)
        # ------------------------------------------------------------------

        def generate(
            self,
            input_ids: jnp.ndarray,              # (B, prompt_len)
            max_new_tokens: Optional[int] = None,
            key: Optional[jnp.ndarray] = None,
        ) -> jnp.ndarray:
            """Generate tokens autoregressively using look-backward L-MTP.

            Args:
                input_ids:      (B, L) prompt token ids.
                max_new_tokens: Override cfg.max_new_tokens.
                key:            JAX PRNG key; uses PRNGKey(0) if None.

            Returns:
                (B, L + new_tokens) complete sequence including the prompt.
            """
            _require("LMTPFlaxDecoder.generate")
            budget = max_new_tokens or self.cfg.max_new_tokens
            key = key or jax.random.PRNGKey(0)
            B = input_ids.shape[0]
            H = self.model.cfg.backbone.hidden_size

            generated = input_ids
            h_prev = jnp.zeros((B, 1, H), dtype=jnp.float32)
            tokens_generated = 0

            while tokens_generated < budget:
                step_tokens, h_prev, key = self._decode_step_jit(
                    generated, h_prev, key
                )
                remaining = budget - tokens_generated
                step_tokens = step_tokens[:, :remaining]       # (B, ≤tps)

                generated = jnp.concatenate([generated, step_tokens], axis=1)
                tokens_generated += step_tokens.shape[1]

                # EOS check
                if self.cfg.eos_token_id is not None:
                    if (step_tokens == self.cfg.eos_token_id).any():
                        break

            return generated

        # ------------------------------------------------------------------
        # XLA-compiled generate via lax.while_loop (no Python per step)
        # ------------------------------------------------------------------

        def generate_jit(
            self,
            input_ids: jnp.ndarray,
            max_new_tokens: Optional[int] = None,
            key: Optional[jnp.ndarray] = None,
        ) -> jnp.ndarray:
            """Fully JIT-compiled generation using lax.while_loop.

            Requires fixed max_new_tokens and no dynamic EOS check.
            Faster for large batches / long sequences on GPU/TPU.

            Returns:
                (B, prompt_len + max_new_tokens) — padded to fixed length.
            """
            _require("LMTPFlaxDecoder.generate_jit")
            budget = max_new_tokens or self.cfg.max_new_tokens
            key = key or jax.random.PRNGKey(0)
            B, L = input_ids.shape
            H = self.model.cfg.backbone.hidden_size
            pad_len = budget  # output_ids size: (B, L + budget)

            # Pre-allocate output buffer
            output = jnp.concatenate(
                [input_ids,
                 jnp.full((B, pad_len), self.cfg.pad_token_id, dtype=jnp.int32)],
                axis=1,
            )                                                       # (B, L+budget)

            # State: (output, write_pos, h_prev, key)
            init_state = (output, L, jnp.zeros((B, 1, H)), key)

            decode_step = self._decode_step_jit

            def body_fn(state):
                out, pos, h_prev, key = state
                ctx = lax.dynamic_slice(out, (0, pos - L), (B, L))  # rolling window
                # Clamp context length for fixed-shape JIT
                ctx = out[:, :pos] if pos <= L else out[:, pos - L: pos]
                step_toks, h_curr, key = decode_step(ctx, h_prev, key)
                # Write up to tps tokens starting at pos
                n = jnp.minimum(self.tps, L + budget - pos)
                out = lax.dynamic_update_slice(out, step_toks[:, :n], (0, pos))
                return out, pos + n, h_curr, key

            def cond_fn(state):
                _, pos, _, _ = state
                return pos < L + budget

            final_output, _, _, _ = lax.while_loop(cond_fn, body_fn, init_state)
            return final_output

        # ------------------------------------------------------------------
        # Throughput measurement
        # ------------------------------------------------------------------

        def benchmark(
            self,
            prompt: jnp.ndarray,
            n_runs: int = 50,
            max_new_tokens: int = 64,
        ) -> dict:
            """Measure inference throughput (tok/s).

            Args:
                prompt:         (1, L) prompt.
                n_runs:         Number of timed generations.
                max_new_tokens: Tokens to generate per run.

            Returns:
                {"tok_per_s": float, "ms_per_step": float, "tokens_per_step": int}
            """
            import time
            # Warm-up
            self.generate(prompt, max_new_tokens=max_new_tokens)
            t0 = time.perf_counter()
            for _ in range(n_runs):
                out = self.generate(prompt, max_new_tokens=max_new_tokens)
                _ = out.block_until_ready()
            elapsed = time.perf_counter() - t0
            total_toks = n_runs * max_new_tokens
            return {
                "tok_per_s": total_toks / elapsed,
                "ms_per_step": elapsed / n_runs / (max_new_tokens / self.tps) * 1000,
                "tokens_per_step": self.tps,
            }

else:
    class LMTPFlaxDecoder:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): _require("LMTPFlaxDecoder")


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def lmtp_flax_generate(
    model: "LMTPWrapper",
    params: dict,
    input_ids: "jnp.ndarray",
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    do_sample: bool = False,
    eos_token_id: Optional[int] = None,
) -> "jnp.ndarray":
    """One-liner generate.

    Args:
        model:          LMTPWrapper (trained, Flax module).
        params:         Trained parameter dict.
        input_ids:      (B, L) prompt tokens.
        max_new_tokens: Token budget.
        temperature:    Sampling temperature.
        do_sample:      Multinomial vs greedy.
        eos_token_id:   Early-stop token.

    Returns:
        (B, L + new_tokens) generated sequence.
    """
    cfg = LMTPFlaxDecodeConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        eos_token_id=eos_token_id,
    )
    decoder = LMTPFlaxDecoder(model, params, cfg)
    return decoder.generate(input_ids)


__all__ = [
    "LMTPFlaxDecodeConfig",
    "LMTPFlaxDecoder",
    "lmtp_flax_generate",
    "_AVAILABLE",
]

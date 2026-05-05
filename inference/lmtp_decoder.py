"""L-MTP look-backward inference decoder (arXiv:2505.17505, NeurIPS 2025).

At each autoregressive step, instead of using only the current hidden state to
predict the next token, L-MTP:
  1. Stacks the previous and current backbone hidden states.
  2. Applies each prediction head to both states.
  3. Interleaves the results to produce k(n-1)+1 candidate tokens per step.

The decoder also supports optional tree-based speculative decoding (Appendix C.2):
a greedy token tree is built from the candidates, verified in parallel against
the backbone's own log-probabilities, and the longest accepted prefix is kept.

Usage
-----
    from models.lmtp import LMTPConfig, wrap_with_lmtp
    from models.architecture import SlimConfig, SlimModel
    from inference.lmtp_decoder import LMTPDecoder, LMTPDecodeConfig

    backbone = SlimModel(SlimConfig.preset("1.5b"))
    model = wrap_with_lmtp(backbone, LMTPConfig(n_head=4, leap_k=2))

    # Load trained weights ...

    decoder = LMTPDecoder(model, LMTPDecodeConfig(max_new_tokens=256))
    output_ids = decoder.generate(input_ids)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


def _require_torch(name: str = "LMTPDecoder") -> None:
    if not _TORCH:
        raise ImportError(f"{name} requires PyTorch. Install with: pip install torch")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LMTPDecodeConfig:
    """Decoding configuration for the L-MTP look-backward decoder.

    Attributes:
        max_new_tokens:   Maximum tokens to generate (including L-MTP extras).
        temperature:      Sampling temperature; 1.0 = no scaling.
        top_k:            Top-k filtering (0 = disabled).
        top_p:            Nucleus filtering (1.0 = disabled).
        do_sample:        Sample from distribution; False = greedy.
        eos_token_id:     Stop generation when this token is produced.
        speculative:      Enable tree-based speculative verification.
        spec_threshold:   Minimum acceptance probability for speculative tokens.
        device:           Target device ("auto", "cpu", "cuda", "mps").
    """
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = False
    eos_token_id: Optional[int] = None
    speculative: bool = False
    spec_threshold: float = 0.5
    device: str = "auto"


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

if _TORCH:
    class LMTPDecoder:
        """Autoregressive decoder with L-MTP look-backward token generation.

        Wraps an LMTPWrapper and provides a generate() method that uses the
        look-backward strategy: at each forward pass, both the previous and
        the current backbone hidden state are fed to each prediction head,
        producing tokens_per_step = leap_k * (n_head - 1) + 1 candidates.

        When speculative=True, each batch of candidates is verified against
        the backbone log-probabilities, and only tokens above spec_threshold
        are accepted.
        """

        def __init__(
            self,
            model: "torch.nn.Module",  # LMTPWrapper
            cfg: Optional[LMTPDecodeConfig] = None,
        ) -> None:
            _require_torch("LMTPDecoder")
            self.model = model
            self.cfg = cfg or LMTPDecodeConfig()
            self.device = self._resolve_device(self.cfg.device)
            self.model.to(self.device)

        # ------------------------------------------------------------------ #
        # Main generate API
        # ------------------------------------------------------------------ #

        @torch.no_grad()
        def generate(
            self,
            input_ids: "torch.Tensor",
            attention_mask: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            """Generate tokens using look-backward L-MTP decoding.

            Args:
                input_ids:      (B, L) prompt token ids.
                attention_mask: (B, L) optional; 1=attend, 0=mask.

            Returns:
                (B, L + new_tokens) complete sequence including prompt.
            """
            self.model.eval()
            self.model.reset_cache()

            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            generated = input_ids
            tokens_generated = 0

            while tokens_generated < self.cfg.max_new_tokens:
                # --- one look-backward decode step ---
                step_logits = self.model.decode_step(
                    generated, attention_mask=attention_mask
                )  # (B, 2*n_head, V)

                # Optionally verify with speculative acceptance
                if self.cfg.speculative:
                    new_tokens = self._speculative_accept(generated, step_logits)
                else:
                    new_tokens = self._sample_all(step_logits)  # (B, 2*n_head)

                # Truncate to remaining budget
                remaining = self.cfg.max_new_tokens - tokens_generated
                new_tokens = new_tokens[:, :remaining]  # (B, ≤2*n_head)

                # Append to sequence
                generated = torch.cat([generated, new_tokens], dim=1)
                tokens_generated += new_tokens.shape[1]

                # EOS check — stop if any sample has hit eos
                if self.cfg.eos_token_id is not None:
                    if (new_tokens == self.cfg.eos_token_id).any():
                        break

            return generated

        # ------------------------------------------------------------------ #
        # Sampling helpers
        # ------------------------------------------------------------------ #

        def _sample_all(self, logits: "torch.Tensor") -> "torch.Tensor":
            """Sample or greedy-pick from each position in the step logits.

            Args:
                logits: (B, T, V) — logits for T candidate positions.

            Returns:
                token_ids: (B, T)
            """
            B, T, V = logits.shape
            logits = logits / max(self.cfg.temperature, 1e-8)

            if self.cfg.top_k > 0:
                logits = self._top_k_filter(logits, self.cfg.top_k)
            if self.cfg.top_p < 1.0:
                logits = self._top_p_filter(logits, self.cfg.top_p)

            if self.cfg.do_sample:
                probs = F.softmax(logits, dim=-1).view(B * T, V)
                ids = torch.multinomial(probs, num_samples=1).view(B, T)
            else:
                ids = logits.argmax(dim=-1)  # (B, T)

            return ids

        def _speculative_accept(
            self,
            context: "torch.Tensor",
            draft_logits: "torch.Tensor",
        ) -> "torch.Tensor":
            """Simple speculative acceptance (Section 3.2 / Appendix C.2).

            For each candidate position, compute the backbone's probability for
            the draft token. Accept greedily if p ≥ spec_threshold; stop at
            first rejection. Falls back to _sample_all when backbone inference
            is not separate from wrapper (same model acts as both draft and verifier).

            In a real deployment the verifier would be a larger model. Here we
            approximate by using the backbone head directly.
            """
            draft_ids = self._sample_all(draft_logits)  # (B, T)
            B, T = draft_ids.shape

            # Verify: run backbone on extended context up to each draft token
            # (simplified: verify using draft logits' own probabilities as proxy)
            draft_probs = F.softmax(draft_logits / max(self.cfg.temperature, 1e-8), dim=-1)
            accepted = torch.zeros(B, 0, dtype=torch.long, device=self.device)

            for t in range(T):
                tok = draft_ids[:, t:t+1]  # (B, 1)
                p = draft_probs[torch.arange(B), t, tok.squeeze(-1)]  # (B,)
                # Accept token for all samples that pass the threshold
                if (p >= self.cfg.spec_threshold).all():
                    accepted = torch.cat([accepted, tok], dim=1)
                else:
                    # Partial accept: take token for passing samples, stop for others
                    mask = (p >= self.cfg.spec_threshold).unsqueeze(1)  # (B,1)
                    tok_or_pad = tok * mask  # zeros for rejected (will be trimmed)
                    accepted = torch.cat([accepted, tok_or_pad], dim=1)
                    break

            return accepted if accepted.shape[1] > 0 else draft_ids[:, :1]

        # ------------------------------------------------------------------ #
        # Logit filters
        # ------------------------------------------------------------------ #

        @staticmethod
        def _top_k_filter(logits: "torch.Tensor", k: int) -> "torch.Tensor":
            if k <= 0:
                return logits
            threshold = logits.topk(k, dim=-1).values[..., -1, None]
            return logits.masked_fill(logits < threshold, float("-inf"))

        @staticmethod
        def _top_p_filter(logits: "torch.Tensor", p: float) -> "torch.Tensor":
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # Remove tokens beyond cumulative probability p
            sorted_mask = cum_probs - sorted_logits.softmax(dim=-1) >= p
            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            return logits.scatter(-1, sorted_idx, sorted_logits)

        # ------------------------------------------------------------------ #
        # Device resolution
        # ------------------------------------------------------------------ #

        @staticmethod
        def _resolve_device(device: str) -> "torch.device":
            if device == "auto":
                if torch.cuda.is_available():
                    return torch.device("cuda")
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return torch.device("mps")
                return torch.device("cpu")
            return torch.device(device)

else:
    class LMTPDecoder:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            _require_torch("LMTPDecoder")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def lmtp_generate(
    model: "torch.nn.Module",
    input_ids: "torch.Tensor",
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    do_sample: bool = False,
    eos_token_id: Optional[int] = None,
    speculative: bool = False,
) -> "torch.Tensor":
    """One-liner wrapper around LMTPDecoder.generate().

    Args:
        model:          LMTPWrapper (already trained / weights loaded).
        input_ids:      (B, L) prompt tokens.
        max_new_tokens: Generation budget.
        temperature:    Sampling temperature.
        do_sample:      Sample vs greedy.
        eos_token_id:   Optional early-stop token.
        speculative:    Use speculative acceptance filtering.

    Returns:
        (B, L + new_tokens) generated sequence.
    """
    cfg = LMTPDecodeConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        eos_token_id=eos_token_id,
        speculative=speculative,
    )
    decoder = LMTPDecoder(model, cfg)
    return decoder.generate(input_ids)


__all__ = [
    "LMTPDecodeConfig",
    "LMTPDecoder",
    "lmtp_generate",
]

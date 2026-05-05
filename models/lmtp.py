"""L-MTP: Leap Multi-Token Prediction (arXiv:2505.17505, NeurIPS 2025).

Extends next-token prediction with additional MLP heads that supervise tokens
at leaping (non-adjacent) positions during training, and use a "look-backward"
decode strategy to emit k(n-1)+1 tokens per forward pass at inference time.

Architecture
------------
Training objective (Eq. 5):
    L = L_NTP + β * Σ_i L_CE(head_i(hidden[:, :-k*(i+1)]),  labels[:, k*(i+1):])

Inference "look-backward":
    At step t, stack [past_hidden, current_hidden] and apply all heads →
    predictions from the past hidden fill the k-1 gap tokens while predictions
    from the current hidden fill the leap positions, giving k(n-1)+1 tokens/step.

Reference:
    Liu et al., "L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context
    for Large Language Models", NeurIPS 2025.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False


def _require_torch(name: str = "L-MTP") -> None:
    if not _TORCH:
        raise ImportError(f"{name} requires PyTorch. Install with: pip install torch")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LMTPConfig:
    """Hyperparameters for Leap Multi-Token Prediction.

    Attributes:
        n_head:  Number of additional prediction heads (beyond the backbone's
                 lm_head). Paper default: 4 with leap_k=2.
        leap_k:  Stride between supervised token positions. leap_k=1 degrades
                 to standard MTP; leap_k=2 is the paper default.
        beta:    Loss weight for auxiliary L-MTP heads (Eq. 5). Paper: 0.1.
        warmup_lr: Learning rate used in the head warm-up stage (Stage 1).
        full_lr:   Learning rate for full model tuning (Stage 2).
        warmup_steps: Number of gradient steps for Stage 1.
        full_steps:   Number of gradient steps for Stage 2.
        lora_rank: LoRA rank for Stage 2. 0 = full fine-tuning (no LoRA).
        lora_alpha: LoRA alpha (scaling = lora_alpha / lora_rank).
    """
    n_head: int = 4
    leap_k: int = 2
    beta: float = 0.1
    warmup_lr: float = 1e-4
    full_lr: float = 1e-5
    warmup_steps: int = 5_000
    full_steps: int = 3_000
    lora_rank: int = 32
    lora_alpha: int = 16

    def tokens_per_step(self) -> int:
        """Extra tokens per decode step with look-backward decoding."""
        return self.leap_k * (self.n_head - 1) + 1


# ---------------------------------------------------------------------------
# Prediction Head (Appendix B.5)
# ---------------------------------------------------------------------------

if _TORCH:
    class LMTPHead(nn.Module):
        """Single L-MTP prediction head.

        Applies a residual SiLU transform then maps to vocabulary logits:
            z' = z + SiLU(W z + b)
            logits = W_head z'

        W_head is cloned from the backbone lm_head weight so that, with the
        transform initialized to zero, each head initially reproduces the
        original next-token distribution (maximising acceptance at warm-up).
        """

        def __init__(
            self,
            hidden_size: int,
            vocab_size: int,
            lm_head_weight: Optional["torch.Tensor"] = None,
        ) -> None:
            super().__init__()
            self.transform = nn.Linear(hidden_size, hidden_size, bias=True)
            self.vocab_proj = nn.Linear(hidden_size, vocab_size, bias=False)

            # Zero-init: head is initially an identity copy of lm_head
            nn.init.zeros_(self.transform.weight)
            nn.init.zeros_(self.transform.bias)

            if lm_head_weight is not None:
                self.vocab_proj.weight = nn.Parameter(lm_head_weight.detach().clone())
            else:
                nn.init.normal_(self.vocab_proj.weight, std=0.02)

        def forward(self, hidden_states: "torch.Tensor") -> "torch.Tensor":
            z = hidden_states + F.silu(self.transform(hidden_states))
            return self.vocab_proj(z)

else:
    class LMTPHead:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            _require_torch("LMTPHead")


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

if _TORCH:
    class LMTPWrapper(nn.Module):
        """Wraps a backbone (SlimModel or CUNCAModel) with L-MTP heads.

        The backbone must implement:
            forward(input_ids, attention_mask=None, return_hidden_states=False)
            → logits          when return_hidden_states=False
            → (logits, hidden) when return_hidden_states=True

        Look-backward inference state (_past_hidden) is stored as a regular
        tensor attribute (not a parameter). Call reset_cache() between sequences.
        """

        def __init__(self, backbone: "nn.Module", lmtp_cfg: LMTPConfig) -> None:
            super().__init__()
            self.backbone = backbone
            self.lmtp_cfg = lmtp_cfg

            lm_head: "nn.Linear" = backbone.lm_head
            hidden_size = lm_head.in_features
            vocab_size = lm_head.out_features

            self.lmtp_heads = nn.ModuleList([
                LMTPHead(hidden_size, vocab_size, lm_head.weight)
                for _ in range(lmtp_cfg.n_head)
            ])

            self._past_hidden: Optional["torch.Tensor"] = None

        # ------------------------------------------------------------------ #
        # Cache management
        # ------------------------------------------------------------------ #

        def reset_cache(self) -> None:
            """Clear look-backward state between independent sequences."""
            self._past_hidden = None

        # ------------------------------------------------------------------ #
        # Training forward
        # ------------------------------------------------------------------ #

        def forward(
            self,
            input_ids: "torch.Tensor",
            attention_mask=None,
            return_lmtp_logits: bool = False,
        ) -> Union["torch.Tensor", Tuple["torch.Tensor", List["torch.Tensor"]]]:
            """Training / evaluation forward pass.

            Returns:
                logits                              when return_lmtp_logits=False
                (logits, [head_0_logits, ...])      when return_lmtp_logits=True
                  Each head_i_logits has shape (B, L, V).
            """
            logits, hidden = self.backbone(
                input_ids,
                attention_mask=attention_mask,
                return_hidden_states=True,
            )
            if not return_lmtp_logits:
                return logits
            lmtp_logits = [head(hidden) for head in self.lmtp_heads]
            return logits, lmtp_logits

        # ------------------------------------------------------------------ #
        # Look-backward inference decoding step
        # ------------------------------------------------------------------ #

        def decode_step(
            self,
            input_ids: "torch.Tensor",
            attention_mask=None,
        ) -> "torch.Tensor":
            """One look-backward decoding step (Section 3.2 / Appendix C).

            Given a batch of input_ids (B, L), produces logits for
            tokens_per_step = k*(n-1)+1 token positions by combining
            predictions from the current and previous backbone hidden state.

            The interleaving order (past_head_i, curr_head_i for each i)
            fills the k-1 gap tokens first, then the leap positions, giving
            a contiguous span of tokens_per_step.

            Returns:
                lmtp_logits: (B, 2*n_head, V)
            """
            _, hidden = self.backbone(
                input_ids,
                attention_mask=attention_mask,
                return_hidden_states=True,
            )
            # Take only the last token position for efficient decoding
            current = hidden[:, -1:, :]  # (B, 1, D)

            if self._past_hidden is None:
                past = hidden[:, -2:-1, :] if hidden.shape[1] > 1 else current
            else:
                past = self._past_hidden

            # Apply every head to both past and current hidden states.
            # Interleave: [past_h0, curr_h0, past_h1, curr_h1, ...]
            # Past  predictions fill gap positions;
            # Current predictions fill leap positions.
            all_logits: List["torch.Tensor"] = []
            for head in self.lmtp_heads:
                all_logits.append(head(past))     # (B, 1, V)
                all_logits.append(head(current))  # (B, 1, V)

            self._past_hidden = current
            return torch.cat(all_logits, dim=1)  # (B, 2*n_head, V)

else:
    class LMTPWrapper:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            _require_torch("LMTPWrapper")


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def compute_lmtp_loss(
    ntp_logits: "torch.Tensor",
    lmtp_logits_list: List["torch.Tensor"],
    labels: "torch.Tensor",
    leap_k: int,
    beta: float,
    ignore_index: int = -100,
) -> "torch.Tensor":
    """Combined NTP + L-MTP training loss (Eq. 5 in the paper).

    Args:
        ntp_logits:       (B, L, V) — backbone output logits.
        lmtp_logits_list: list of n tensors (B, L, V), one per leap head.
        labels:           (B, L) — target token ids (with -100 for padding).
        leap_k:           stride between supervised positions.
        beta:             weight applied to the auxiliary L-MTP loss.
        ignore_index:     label value excluded from cross-entropy.

    Returns:
        Scalar loss: L_NTP + β * Σ_i L_CE(head_i logits, shifted labels).
    """
    _require_torch("compute_lmtp_loss")
    import torch.nn.functional as F  # noqa: PLC0415 (re-import inside fn ok)

    vocab_size = ntp_logits.size(-1)

    ntp_loss = F.cross_entropy(
        ntp_logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=ignore_index,
    )

    lmtp_loss = ntp_logits.new_zeros(())
    for i, head_logits in enumerate(lmtp_logits_list):
        shift = leap_k * (i + 1)
        seq_len = head_logits.shape[1]
        if shift >= seq_len:
            continue
        h_logits = head_logits[:, :-shift].contiguous()
        h_labels = labels[:, shift:].contiguous()
        lmtp_loss = lmtp_loss + F.cross_entropy(
            h_logits.view(-1, vocab_size),
            h_labels.view(-1),
            ignore_index=ignore_index,
        )

    return ntp_loss + beta * lmtp_loss


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def wrap_with_lmtp(backbone: "nn.Module", cfg: Optional[LMTPConfig] = None) -> "LMTPWrapper":
    """Attach L-MTP heads to an existing backbone model.

    Args:
        backbone: SlimModel or CUNCAModel (must have .lm_head and support
                  return_hidden_states=True in forward()).
        cfg:      LMTPConfig; uses defaults when omitted.

    Returns:
        LMTPWrapper ready for two-stage fine-tuning.
    """
    _require_torch("wrap_with_lmtp")
    cfg = cfg or LMTPConfig()
    return LMTPWrapper(backbone, cfg)


__all__ = [
    "LMTPConfig",
    "LMTPHead",
    "LMTPWrapper",
    "compute_lmtp_loss",
    "wrap_with_lmtp",
]

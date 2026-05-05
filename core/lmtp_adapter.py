"""HuggingFace → LMTPWrapper adapter for CapibaraGPT main branch.

Bridges the HuggingFace AutoModelForCausalLM API to the interface expected
by LMTPWrapper (models/lmtp.py), which requires:
    backbone.lm_head           — nn.Linear  (hidden → vocab)
    backbone.forward(input_ids, ..., return_hidden_states=True)
        → (logits, last_hidden_state)

HuggingFace models expose hidden states via output_hidden_states=True in
their forward call; this adapter maps that to the LMTPWrapper convention.

Usage
-----
    from core.lmtp_adapter import HFLMTPAdapter
    from models.lmtp import LMTPConfig, wrap_with_lmtp
    from core.hf_model import HuggingFaceCausalLM, HuggingFaceConfig

    hf = HuggingFaceCausalLM(HuggingFaceConfig(model_path="path/to/llama"))
    adapter = HFLMTPAdapter(hf.model)
    lmtp_model = wrap_with_lmtp(adapter, LMTPConfig(n_head=4, leap_k=2))
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


def _require_torch(name: str = "HFLMTPAdapter") -> None:
    if not _TORCH:
        raise ImportError(f"{name} requires PyTorch. Install with: pip install torch")


if _TORCH:
    class HFLMTPAdapter(nn.Module):
        """Thin wrapper that makes a HuggingFace CausalLM compatible with LMTPWrapper.

        Exposes:
            .lm_head       — the underlying model's lm_head (or equivalent)
            .forward(...)  — returns (logits, last_hidden) when return_hidden_states=True
        """

        def __init__(self, hf_model: "nn.Module") -> None:
            super().__init__()
            _require_torch("HFLMTPAdapter")
            self.hf_model = hf_model
            # Resolve lm_head: standard attribute in GPT-2, Llama, Mistral, etc.
            if hasattr(hf_model, "lm_head"):
                self.lm_head: "nn.Linear" = hf_model.lm_head
            elif hasattr(hf_model, "embed_out"):
                # Falcon / MPT naming convention
                self.lm_head = hf_model.embed_out  # type: ignore[assignment]
            elif hasattr(hf_model, "output_projection"):
                self.lm_head = hf_model.output_projection  # type: ignore[assignment]
            else:
                raise AttributeError(
                    "Cannot locate lm_head on the provided HuggingFace model. "
                    "Expected attribute 'lm_head', 'embed_out', or 'output_projection'."
                )

        def forward(
            self,
            input_ids: "torch.Tensor",
            attention_mask: Optional["torch.Tensor"] = None,
            return_hidden_states: bool = False,
        ) -> Union["torch.Tensor", Tuple["torch.Tensor", "torch.Tensor"]]:
            """Forward pass through the HuggingFace model.

            Args:
                input_ids:           (B, L) token ids.
                attention_mask:      (B, L) optional.
                return_hidden_states: when True returns (logits, last_hidden).

            Returns:
                logits               when return_hidden_states=False
                (logits, last_hidden) when True;
                  last_hidden has shape (B, L, hidden_size).
            """
            outputs = self.hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=return_hidden_states,
            )
            logits = outputs.logits  # (B, L, V)
            if not return_hidden_states:
                return logits
            # hidden_states is a tuple (embedding, layer_0, ..., layer_N);
            # take the last layer's output
            last_hidden = outputs.hidden_states[-1]  # (B, L, D)
            return logits, last_hidden

        def parameters(self, recurse: bool = True):
            return self.hf_model.parameters(recurse=recurse)

        def named_parameters(self, prefix: str = "", recurse: bool = True):
            return self.hf_model.named_parameters(prefix=prefix, recurse=recurse)

        def named_children(self):
            return self.hf_model.named_children()

        def to(self, *args, **kwargs):
            self.hf_model = self.hf_model.to(*args, **kwargs)
            return self

        def train(self, mode: bool = True):
            self.hf_model.train(mode)
            return self

        def eval(self):
            self.hf_model.eval()
            return self

        def requires_grad_(self, requires_grad: bool = True):
            self.hf_model.requires_grad_(requires_grad)
            return self

else:
    class HFLMTPAdapter:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            _require_torch("HFLMTPAdapter")


__all__ = ["HFLMTPAdapter"]

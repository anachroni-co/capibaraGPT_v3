"""Capibara Slim — Transformer backend via HuggingFace.

Uses core/hf_model.py which already handles the torch/transformers
optional-dependency dance. Falls back gracefully when torch is absent.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TransformerBackend:
    """HuggingFace causal LM backend."""

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model = None
        self._available = False
        self._load()

    def _load(self) -> None:
        try:
            from core.hf_model import HuggingFaceCausalLM, HuggingFaceConfig
            cfg = HuggingFaceConfig(model_path=self._model_path)
            self._model = HuggingFaceCausalLM(cfg)
            self._available = True
            logger.info("TransformerBackend loaded: %s", self._model_path)
        except Exception as exc:
            logger.warning("TransformerBackend unavailable (%s), falling back to stub", exc)
            self._available = False

    @property
    def name(self) -> str:
        return "transformer"

    @property
    def is_available(self) -> bool:
        return self._available

    def generate(
        self,
        input_text: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        if not self._available or self._model is None:
            raise RuntimeError("TransformerBackend not available — torch/transformers missing or model not found")

        encoded = self._model.encode_with_mask(input_text)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        output_ids = self._model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self._model.tokenizer.eos_token_id,
        )
        # Decode only the newly generated tokens
        new_ids = output_ids[0][input_ids.shape[-1]:]
        output_text = self._model.decode(new_ids)

        return {
            "output": output_text,
            "model": self.name,
            "tokens_used": int(input_ids.shape[-1]),
        }

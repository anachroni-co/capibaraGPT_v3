"""
HuggingFace causal LM wrapper for CapibaraGPT core inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

torch = None  # type: ignore
AutoModelForCausalLM = None  # type: ignore
AutoTokenizer = None  # type: ignore
_TRANSFORMERS_AVAILABLE = False
_IMPORT_ERROR: Optional[Exception] = None


def _torch_import_works() -> bool:
    import subprocess
    import sys
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _try_import_transformers() -> bool:
    global torch, AutoModelForCausalLM, AutoTokenizer, _TRANSFORMERS_AVAILABLE, _IMPORT_ERROR
    if _TRANSFORMERS_AVAILABLE:
        return True
    if not _torch_import_works():
        _IMPORT_ERROR = ImportError("torch import failed")
        return False
    try:
        import torch as _torch
        from transformers import AutoModelForCausalLM as _AutoModel, AutoTokenizer as _AutoTokenizer
        torch = _torch
        AutoModelForCausalLM = _AutoModel  # type: ignore
        AutoTokenizer = _AutoTokenizer  # type: ignore
        _TRANSFORMERS_AVAILABLE = True
        return True
    except Exception as exc:  # pragma: no cover - optional dependency
        _TRANSFORMERS_AVAILABLE = False
        _IMPORT_ERROR = exc
        torch = None  # type: ignore
        AutoModelForCausalLM = None  # type: ignore
        AutoTokenizer = None  # type: ignore
        return False


def _resolve_dtype(dtype: str | None):
    if torch is None:
        return None
    if dtype is None or dtype == "auto":
        return None
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    return None


@dataclass
class HuggingFaceConfig:
    model_path: str
    tokenizer_path: Optional[str] = None
    device: Optional[str] = None
    dtype: str | None = "auto"
    trust_remote_code: bool = True


class HuggingFaceCausalLM:
    """Thin wrapper around AutoModelForCausalLM with explicit dependencies."""

    def __init__(self, config: HuggingFaceConfig):
        if not _TRANSFORMERS_AVAILABLE and not _try_import_transformers():
            raise ImportError(
                "transformers/torch are required for HuggingFaceCausalLM. "
                f"Import error: {_IMPORT_ERROR}"
            )

        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        tokenizer_path = Path(config.tokenizer_path or config.model_path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")

        device = config.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.dtype = _resolve_dtype(config.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
            trust_remote_code=config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            trust_remote_code=config.trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info("HuggingFace model loaded on %s", self.device)

    def encode(self, text: str):
        tokens = self.encode_with_mask(text)
        return tokens["input_ids"]

    def encode_with_mask(self, text: str):
        tokens = self.tokenizer(text, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device) if "attention_mask" in tokens else None
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def forward(self, input_ids):
        with torch.no_grad():
            return self.model(input_ids=input_ids)

    def generate(self, input_ids, **kwargs):
        with torch.no_grad():
            return self.model.generate(input_ids, **kwargs)


__all__ = [
    "HuggingFaceConfig",
    "HuggingFaceCausalLM",
]

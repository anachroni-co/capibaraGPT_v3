"""Capibara Slim — tokenizer wrapper (T5.2).

Priority:
  1. HuggingFace tokenizer from a local model path (BPE/SentencePiece)
  2. Whitespace fallback (no external deps — development/test only)

Usage:
    tok = SlimTokenizer("models/tiny-gpt2")
    ids = tok.encode("hello world")   # [15339, 995]
    text = tok.decode(ids)            # "hello world"
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

_HF_AVAILABLE = False
try:
    from transformers import AutoTokenizer as _AutoTokenizer
    _HF_AVAILABLE = True
except ImportError:
    pass


class _WhitespaceTokenizer:
    """Minimal word-level tokenizer — for testing only."""

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self._inv: dict[int, str] = {v: k for k, v in self._vocab.items()}
        self.vocab_size = 4

    def _tok(self, word: str) -> int:
        if word not in self._vocab:
            idx = len(self._vocab)
            self._vocab[word] = idx
            self._inv[idx] = word
            self.vocab_size = len(self._vocab)
        return self._vocab[word]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self._tok(w) for w in text.split()]
        if add_special_tokens:
            ids = [self.BOS_ID] + ids + [self.EOS_ID]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        special = {self.PAD_ID, self.BOS_ID, self.EOS_ID} if skip_special_tokens else set()
        return " ".join(self._inv.get(i, "<unk>") for i in ids if i not in special)

    @property
    def pad_token_id(self) -> int:
        return self.PAD_ID

    @property
    def eos_token_id(self) -> int:
        return self.EOS_ID


class SlimTokenizer:
    """Unified tokenizer for Capibara Slim (inference + training)."""

    def __init__(self, model_path: Union[str, None] = None) -> None:
        self._inner = None
        self._is_hf = False

        if model_path and _HF_AVAILABLE:
            p = Path(model_path)
            if p.exists():
                try:
                    self._inner = _AutoTokenizer.from_pretrained(
                        str(p), use_fast=True, trust_remote_code=True
                    )
                    if self._inner.pad_token is None:
                        self._inner.pad_token = self._inner.eos_token
                    self._is_hf = True
                    logger.info("SlimTokenizer: loaded HF tokenizer from %s", p)
                except Exception as exc:
                    logger.warning("SlimTokenizer: HF load failed (%s) — using whitespace fallback", exc)

        if not self._is_hf:
            logger.info("SlimTokenizer: using whitespace fallback tokenizer")
            self._inner = _WhitespaceTokenizer()

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if self._is_hf:
            return self._inner.encode(text, add_special_tokens=add_special_tokens)
        return self._inner.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._inner.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        if self._is_hf:
            return len(self._inner)
        return self._inner.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self._inner.pad_token_id

    @property
    def eos_token_id(self) -> int:
        if self._is_hf:
            return self._inner.eos_token_id
        return self._inner.eos_token_id

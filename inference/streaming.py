"""Capibara Slim — token-by-token streaming generator (T6.3).

Yields one decoded token at a time so the API can push SSE events
without waiting for the full sequence to be generated.

Works with any backend:
  - stub:        yields fake tokens character by character (demo)
  - transformer: uses model.generate with streamer (requires torch)
  - mamba:       same as transformer path
"""
from __future__ import annotations

import logging
import time
from typing import Generator

from config.slim_loader import get as cfg_get
from safety.input_filter import InputFilter
from tools.detector import detect_tool

logger = logging.getLogger(__name__)


def _stub_stream(text: str, delay: float = 0.0) -> Generator[str, None, None]:
    """Stub streamer: emits one word at a time from a canned reply."""
    words = f"[stub-stream] echoing: {text}".split()
    for word in words:
        yield word + " "
        if delay:
            time.sleep(delay)


def _hf_stream(
    input_text: str,
    model_path: str,
    max_tokens: int,
    temperature: float,
) -> Generator[str, None, None]:
    """Stream tokens from a HuggingFace model using TextIteratorStreamer."""
    try:
        import torch
        from transformers import TextIteratorStreamer
        from core.hf_model import HuggingFaceCausalLM, HuggingFaceConfig
        import threading
    except ImportError as exc:
        logger.warning("HF streaming unavailable (%s); falling back to stub stream", exc)
        yield from _stub_stream(input_text)
        return

    cfg = HuggingFaceConfig(model_path=model_path)
    model = HuggingFaceCausalLM(cfg)

    streamer = TextIteratorStreamer(
        model.tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    encoded = model.encode_with_mask(input_text)

    gen_kwargs = dict(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=model.tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.model.generate, kwargs=gen_kwargs, daemon=True)
    thread.start()

    for token_text in streamer:
        yield token_text

    thread.join()


class SlimStreamer:
    """Public streaming interface used by the API route."""

    def __init__(self) -> None:
        self._backend: str = cfg_get("model", "backend", "auto")
        self._model_path: str = cfg_get("model", "path", "models/tiny-gpt2")
        self._input_filter = InputFilter(enabled=cfg_get("safety", "input_filter", True))

    def stream(
        self,
        input_text: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """Yield decoded text fragments until generation is complete."""
        text = input_text.strip()

        # Safety check
        check = self._input_filter.check(text)
        if not check.allowed:
            yield f"[blocked] {check.reason}"
            return

        # Tool calls are not streamable — run sync and yield result
        tool_match = detect_tool(text)
        if tool_match is not None:
            from tools.executor import ToolExecutor
            tool_name, tool_input = tool_match
            result = ToolExecutor().execute(tool_name, tool_input)
            yield result["result"] if result["error"] is None else f"[tool error] {result['error']}"
            return

        effective_backend = self._backend
        if effective_backend == "auto":
            word_count = len(text.split())
            effective_backend = "mamba" if word_count < 128 else "transformer"

        if effective_backend in ("transformer", "mamba"):
            yield from _hf_stream(text, self._model_path, max_tokens, temperature)
        else:
            yield from _stub_stream(text)

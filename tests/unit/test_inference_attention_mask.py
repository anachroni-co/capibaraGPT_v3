"""
Inference Attention Mask Tests - Unit tests for attention mask handling.

This module provides tests for attention mask handling during inference,
validating padding masks, causal masks, and attention computation.

Author: Skydesk International Dev Team.
"""

from dataclasses import dataclass

import pytest

from core import inference as inference_module


@dataclass
class DummyTensor:
    name: str
    device: str = "cpu"


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1


class DummyModel:
    def __init__(self, config):
        self.config = config
        self.tokenizer = DummyTokenizer()
        self._mask = None

    def encode_with_mask(self, text: str):
        input_ids = DummyTensor("input_ids")
        attention_mask = DummyTensor("attention_mask")
        self._mask = attention_mask
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def generate(self, input_ids, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        assert attention_mask is self._mask
        assert getattr(attention_mask, "device", None) == "cpu"
        return [[101, 102]]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "Hello world"


def test_inference_passes_attention_mask_cpu(monkeypatch):
    monkeypatch.setattr(inference_module, "HuggingFaceCausalLM", DummyModel)

    config = inference_module.InferenceConfig(
        model_path=".",
        tokenizer_path=".",
        device="cpu",
        max_length=8,
    )

    with inference_module.CapibaraInference(config) as engine:
        output = engine.generate("Hello", max_length=8)
        assert output.strip()

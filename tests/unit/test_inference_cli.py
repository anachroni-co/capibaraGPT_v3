"""
Inference CLI Tests - Unit tests for command-line inference interface.

This module provides tests for the CLI inference interface, validating
command-line argument parsing, inference execution, and output formatting.

Author: Skydesk International Dev Team.
"""

import sys

from core import inference as inference_module


class DummyInference:
    last_instance = None

    def __init__(self, config):
        self.config = config
        self.prompts = []
        DummyInference.last_instance = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def generate(self, prompt):
        self.prompts.append(prompt)
        return "ok"

    def health_check(self):
        return {"status": "ok", "checks": {}}

    def get_system_stats(self):
        return {"system": {}}


def test_inference_cli_prompt(monkeypatch):
    monkeypatch.setattr(inference_module, "CapibaraInference", DummyInference)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference.py",
            "--model-path",
            "models/tiny-gpt2",
            "--tokenizer-path",
            "models/tiny-gpt2",
            "--prompt",
            "Hello",
            "--device",
            "cpu",
            "--log-level",
            "ERROR",
        ],
    )

    inference_module.main()

    instance = DummyInference.last_instance
    assert instance is not None
    assert instance.config.device == "cpu"
    assert instance.prompts == ["Hello"]

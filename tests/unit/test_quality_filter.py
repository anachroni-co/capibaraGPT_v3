"""
Quality Filter Tests - Unit tests for data quality filtering.

This module provides tests for quality filtering functionality,
validating toxicity detection, perplexity filtering, and content
quality assessment.

Author: Skydesk International Dev Team.
"""

import importlib.util
from pathlib import Path


def load_quality_filter_module():
    module_path = Path(__file__).resolve().parents[2] / "training" / "data_preprocessing" / "quality_filter.py"
    spec = importlib.util.spec_from_file_location("quality_filter", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyClassifier:
    def __init__(self, responses):
        self.responses = responses

    def __call__(self, text):
        return self.responses(text)


def test_toxicity_filter_handles_case_insensitive_labels():
    quality_filter = load_quality_filter_module()
    config = quality_filter.QualityConfig(use_toxicity_filter=False, toxicity_threshold=0.5)
    advanced_filter = quality_filter.AdvancedFilter(config)
    advanced_filter.config.use_toxicity_filter = True
    advanced_filter.toxicity_classifier = DummyClassifier(
        lambda text: [{"label": "toxic", "score": 0.9}] if "bad" in text else [{"label": "safe", "score": 0.1}]
    )

    docs = [{"text": "bad text"}, {"text": "ok text"}]
    filtered = advanced_filter.filter_by_toxicity(docs)

    assert len(filtered) == 1
    assert filtered[0]["text"] == "ok text"


def test_toxicity_filter_ignores_non_toxic_labels():
    quality_filter = load_quality_filter_module()
    config = quality_filter.QualityConfig(use_toxicity_filter=False, toxicity_threshold=0.5)
    advanced_filter = quality_filter.AdvancedFilter(config)
    advanced_filter.config.use_toxicity_filter = True
    advanced_filter.toxicity_classifier = DummyClassifier(
        lambda text: [{"label": "non-toxic", "score": 0.99}]
    )

    docs = [{"text": "safe text"}]
    filtered = advanced_filter.filter_by_toxicity(docs)

    assert len(filtered) == 1
    assert filtered[0]["text"] == "safe text"


def test_toxicity_filter_accepts_dict_payload():
    quality_filter = load_quality_filter_module()
    config = quality_filter.QualityConfig(use_toxicity_filter=False, toxicity_threshold=0.5)
    advanced_filter = quality_filter.AdvancedFilter(config)
    advanced_filter.config.use_toxicity_filter = True
    advanced_filter.toxicity_classifier = DummyClassifier(
        lambda text: {"label": "TOXIC", "score": 0.9}
    )

    docs = [{"text": "bad text"}]
    filtered = advanced_filter.filter_by_toxicity(docs)

    assert filtered == []

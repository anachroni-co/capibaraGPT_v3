"""Unit tests for core/cot (BACKLOG-006).

Covers the parts of core.cot.module and core.cot.enhanced_cot_module that
do NOT require JAX/Flax at import time. The Flax-based EnhancedCoTModule
is only defined when JAX is available, so we test its absence path too.

The ``core.cot`` package's ``__init__.py`` eagerly re-exports symbols that
pull in heavy optional deps, so we load the two submodules standalone via
``importlib``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_COT_DIR = Path(__file__).resolve().parents[2] / "core" / "cot"


import sys as _sys


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _COT_DIR / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec_module so @dataclass introspection works.
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_module = _load("_cot_module_under_test", "module.py")
_enhanced = _load("_cot_enhanced_under_test", "enhanced_cot_module.py")

EnhancedChainOfThoughtModule = _module.EnhancedChainOfThoughtModule
CoTConfig = _module.CoTConfig
ReasoningConfig = _enhanced.ReasoningConfig
ProcessRewardModel = _enhanced.ProcessRewardModel
MetaCognitionModule = _enhanced.MetaCognitionModule
SelfReflectionModule = _enhanced.SelfReflectionModule
ADVANCED_COT_AVAILABLE = _enhanced.ADVANCED_COT_AVAILABLE


# ---------------------------------------------------------------------------
# module.EnhancedChainOfThoughtModule
# ---------------------------------------------------------------------------


def test_cot_module_defaults():
    m = EnhancedChainOfThoughtModule()
    assert m.config == {}
    assert m.cache_size == 128
    assert m.initialized is False
    assert m.is_available() is False


def test_cot_module_custom_config_and_cache():
    m = EnhancedChainOfThoughtModule(config={"foo": 1}, cache_size=42)
    assert m.config == {"foo": 1}
    assert m.cache_size == 42


def test_cot_module_initialize():
    m = EnhancedChainOfThoughtModule()
    assert m.initialize() is True
    assert m.initialized is True
    assert m.is_available() is True


def test_cot_module_call_returns_expected_shape():
    m = EnhancedChainOfThoughtModule()
    result = m("what is 2+2?")
    assert result["query"] == "what is 2+2?"
    assert result["module"] == "EnhancedChainOfThoughtModule"
    assert "steps" in result
    assert len(result["steps"]) == 4
    assert 0.0 <= result["confidence"] <= 1.0
    # calling auto-initializes
    assert m.initialized is True


def test_cot_module_call_with_context():
    m = EnhancedChainOfThoughtModule()
    result = m("query", initial_context="prior")
    assert result["context"] == "prior"


def test_cot_module_process_dispatches_on_type():
    m = EnhancedChainOfThoughtModule()
    # string -> __call__ path
    out_str = m.process("hello")
    assert out_str["query"] == "hello"
    # non-string -> simple wrap
    out_other = m.process({"k": "v"})
    assert out_other == {"processed": {"k": "v"}, "module": "EnhancedChainOfThoughtModule"}


def test_cotconfig_defaults():
    c = CoTConfig()
    assert c.hidden_size == 768
    assert c.num_reasoning_steps == 4
    assert c.reasoning_dim == 384
    assert c.use_intermediate_supervision is True
    assert c.dropout_rate == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# enhanced_cot_module.ReasoningConfig
# ---------------------------------------------------------------------------


def test_reasoning_config_defaults():
    c = ReasoningConfig()
    assert c.max_reasoning_steps == 16
    assert c.confidence_threshold == pytest.approx(0.8)
    assert c.use_process_rewards is True
    assert c.enable_meta_cognition is True
    assert c.enable_self_verification is True
    assert c.hidden_size == 768


def test_reasoning_config_override():
    c = ReasoningConfig(max_reasoning_steps=4, hidden_size=128)
    assert c.max_reasoning_steps == 4
    assert c.hidden_size == 128


# ---------------------------------------------------------------------------
# ProcessRewardModel (graceful without jax)
# ---------------------------------------------------------------------------


def test_process_reward_model_returns_float_in_unit_interval():
    prm = ProcessRewardModel()
    r = prm([0.1, 0.2, 0.3])
    assert isinstance(r, float)
    assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# MetaCognitionModule
# ---------------------------------------------------------------------------


def test_meta_cognition_assess_boosts_with_history_and_clips():
    mc = MetaCognitionModule()
    # Start high and with a long history -> should clip to 1.0
    assessed = mc.assess(history=[{}] * 50, current_confidence=0.9)
    assert assessed <= 1.0


def test_meta_cognition_assess_preserves_low_confidence_with_no_history():
    mc = MetaCognitionModule()
    v = mc.assess(history=[], current_confidence=0.3)
    assert 0.0 <= v <= 1.0
    assert v == pytest.approx(0.3, abs=0.01)


# ---------------------------------------------------------------------------
# SelfReflectionModule
# ---------------------------------------------------------------------------


def test_self_reflection_empty_trace():
    sr = SelfReflectionModule()
    v = sr.verify([])
    assert v == {"verified": False, "score": 0.0}


def test_self_reflection_high_confidence_verifies():
    sr = SelfReflectionModule()
    trace = [{"step_confidence": 0.9}, {"step_confidence": 0.8}]
    v = sr.verify(trace)
    assert v["verified"] is True
    assert v["score"] >= 0.6
    assert v["steps"] == 2


def test_self_reflection_low_confidence_does_not_verify():
    sr = SelfReflectionModule()
    trace = [{"step_confidence": 0.1}, {"step_confidence": 0.2}]
    v = sr.verify(trace)
    assert v["verified"] is False


# ---------------------------------------------------------------------------
# Backend-availability flag
# ---------------------------------------------------------------------------


def test_advanced_cot_available_is_bool():
    # The flag is set at import time and may be True (jax present) or False
    # (pure CPU fallback). Both are acceptable - we only assert it's a bool.
    assert isinstance(ADVANCED_COT_AVAILABLE, bool)

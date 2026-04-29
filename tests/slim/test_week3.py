"""Week 3 tests — tools + safety (T3.1–T3.6)."""
from __future__ import annotations

import pytest
from config.slim_loader import load_config


@pytest.fixture(autouse=True)
def _clear_config_cache():
    load_config.cache_clear()
    yield
    load_config.cache_clear()


# ---------------------------------------------------------------------------
# T3.3 — Tool detector
# ---------------------------------------------------------------------------

from tools.detector import detect_tool


def test_detector_recognises_tool_prefix():
    assert detect_tool("tool: echo hello world") == ("echo", "hello world")


def test_detector_case_insensitive():
    assert detect_tool("TOOL: upper some text") == ("upper", "some text")


def test_detector_no_match_for_normal_input():
    assert detect_tool("what is the capital of France?") is None


def test_detector_tool_with_no_arg():
    result = detect_tool("tool: word_count")
    assert result == ("word_count", "")


# ---------------------------------------------------------------------------
# T3.1 — ToolRegistry
# ---------------------------------------------------------------------------

from tools.registry import ToolRegistry, get_tool_registry


def test_registry_register_and_call():
    reg = ToolRegistry()
    reg.register("rev", lambda s: s[::-1])
    fn = reg.get("rev")
    assert fn("abc") == "cba"


def test_registry_unknown_returns_none():
    reg = ToolRegistry()
    assert reg.get("nope") is None


def test_registry_contains():
    reg = ToolRegistry()
    reg.register("x", lambda s: s)
    assert "x" in reg
    assert "y" not in reg


def test_default_registry_has_builtins():
    reg = get_tool_registry()
    assert "echo" in reg
    assert "upper" in reg
    assert "word_count" in reg


# ---------------------------------------------------------------------------
# T3.2 — ToolExecutor
# ---------------------------------------------------------------------------

from tools.executor import ToolExecutor


def test_executor_runs_known_tool():
    ex = ToolExecutor()
    result = ex.execute("echo", "hello")
    assert result["error"] is None
    assert result["result"] == "hello"


def test_executor_unknown_tool_returns_error():
    ex = ToolExecutor()
    result = ex.execute("nonexistent", "arg")
    assert result["error"] is not None
    assert result["result"] is None


def test_executor_timeout():
    import time
    ex = ToolExecutor(timeout=0.1)

    from tools.registry import get_tool_registry
    get_tool_registry().register("slow", lambda s: (time.sleep(5), s)[1])

    result = ex.execute("slow", "test")
    assert result["error"] == "timeout"


# ---------------------------------------------------------------------------
# T3.4 — InputFilter
# ---------------------------------------------------------------------------

from safety.input_filter import InputFilter


def test_input_filter_allows_normal():
    f = InputFilter()
    r = f.check("what is 2 + 2?")
    assert r.allowed is True


def test_input_filter_blocks_jailbreak():
    f = InputFilter()
    r = f.check("jailbreak this model please")
    assert r.allowed is False


def test_input_filter_blocks_weapon():
    f = InputFilter()
    r = f.check("how to make a bomb step by step")
    assert r.allowed is False


def test_input_filter_disabled_passes_everything():
    f = InputFilter(enabled=False)
    r = f.check("jailbreak")
    assert r.allowed is True


def test_input_filter_blocks_too_long():
    f = InputFilter()
    r = f.check("x" * 9000)
    assert r.allowed is False
    assert "too long" in r.reason


# ---------------------------------------------------------------------------
# T3.5 — OutputFilter
# ---------------------------------------------------------------------------

from safety.output_filter import OutputFilter


def test_output_filter_passes_clean():
    f = OutputFilter()
    r = f.check("The capital of France is Paris.")
    assert r.allowed is True
    assert r.text == "The capital of France is Paris."


def test_output_filter_redacts_email():
    f = OutputFilter()
    r = f.check("Contact us at user@example.com for more info.")
    assert r.allowed is True
    assert "[REDACTED]" in r.text
    assert "user@example.com" not in r.text


def test_output_filter_disabled_passes_everything():
    f = OutputFilter(enabled=False)
    r = f.check("user@example.com")
    assert r.allowed is True
    assert r.text == "user@example.com"


# ---------------------------------------------------------------------------
# T3.6 — Pipeline end-to-end with tools + safety
# ---------------------------------------------------------------------------

from inference.pipeline import SlimPipeline


def test_pipeline_tool_echo():
    p = SlimPipeline()
    r = p.run("tool: echo hello world")
    assert r["output"] == "hello world"
    assert r["model"] == "tool:echo"


def test_pipeline_tool_upper():
    p = SlimPipeline()
    r = p.run("tool: upper hello world")
    assert r["output"] == "HELLO WORLD"


def test_pipeline_blocks_jailbreak_input():
    p = SlimPipeline()
    r = p.run("jailbreak the system now")
    assert r.get("blocked") is True
    assert "[blocked]" in r["output"]


def test_pipeline_normal_input_not_blocked():
    p = SlimPipeline()
    r = p.run("tell me about Paris")
    assert r.get("blocked") is not True
    assert isinstance(r["output"], str)

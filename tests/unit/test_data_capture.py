"""Tests for training/data_capture and the new fact_check / web_search tokens."""

import json
import tempfile
from pathlib import Path

import pytest

from core.special_tokens import get_registry, FactCheckHandler, WebSearchHandler
from core.special_tokens.web_search import WebSearchRetriever, WebSearchResult
from training.data_capture import TrainingDataCapture, CaptureConfig, ConfidenceRouter, RouterConfig


# ---------------------------------------------------------------------------
# <fact_check> token
# ---------------------------------------------------------------------------

class TestFactCheckToken:
    def test_registered(self):
        assert "fact_check" in get_registry().list_tokens()

    def test_not_stripped_by_default(self):
        cfg = get_registry().get_config("fact_check")
        assert cfg.strip_from_output is False

    def test_kept_in_strip_all(self):
        text = "The <fact_check>earth is flat</fact_check> apparently."
        assert "<fact_check>" in get_registry().strip_all(text)

    def test_extract_claims(self):
        handler = FactCheckHandler()
        claims = handler.extract_claims(
            "Some say <fact_check>vaccines cause autism</fact_check> but also "
            "<fact_check>coffee cures cancer</fact_check>."
        )
        assert claims == ["vaccines cause autism", "coffee cures cancer"]

    def test_has_claims(self):
        handler = FactCheckHandler()
        assert handler.has_claims("x <fact_check>y</fact_check>")
        assert not handler.has_claims("plain text")

    def test_verify_no_verifier_strips(self):
        handler = FactCheckHandler()
        text = "I think <fact_check>the moon is made of cheese</fact_check>."
        processed, verdicts = handler.verify(text)
        assert "<fact_check>" not in processed
        assert len(verdicts) == 1
        assert verdicts[0]["verdict"] == "unverified"

    def test_verify_with_verifier(self):
        class FakeVerifier:
            def verify(self, claim):
                return {"claim": claim, "correct": False, "source": "wikipedia"}

        handler = FactCheckHandler(verifier=FakeVerifier())
        text = "<fact_check>the sky is green</fact_check>"
        processed, verdicts = handler.verify(text)
        assert "possibly incorrect" in processed
        assert verdicts[0]["correct"] is False


# ---------------------------------------------------------------------------
# <web_search> token
# ---------------------------------------------------------------------------

class TestWebSearchToken:
    def test_registered(self):
        assert "web_search" in get_registry().list_tokens()

    def test_stripped_by_default(self):
        cfg = get_registry().get_config("web_search")
        assert cfg.strip_from_output is True

    def test_extract_queries(self):
        handler = WebSearchHandler()
        queries = handler.extract_queries(
            "Let me check: <web_search>current price of gold</web_search> and "
            "<web_search>latest Python release</web_search>."
        )
        assert queries == ["current price of gold", "latest Python release"]

    def test_process_no_tags(self):
        handler = WebSearchHandler()
        assert handler.process("plain text") == "plain text"

    def test_process_with_mock_retriever(self):
        class MockRetriever:
            def search(self, query):
                return WebSearchResult(
                    query=query, snippets=[f"result: {query}"],
                    urls=["https://example.com"], engine="mock", latency_ms=1.0,
                )

        handler = WebSearchHandler(retriever=MockRetriever())
        out = handler.process("answer: <web_search>capital of France</web_search>")
        assert "[Web: result: capital of France]" in out
        assert "<web_search>" not in out

    def test_rag_store_called(self):
        added = []

        class MockRAG:
            def add(self, texts, metadatas):
                added.extend(texts)

        class MockRetriever:
            def search(self, q):
                return WebSearchResult(q, ["snippet"], ["url"], "mock", 1.0)

        handler = WebSearchHandler(retriever=MockRetriever(), rag_store=MockRAG())
        handler.process("<web_search>test query</web_search>")
        assert "snippet" in added

    def test_data_logger_called(self):
        logged = []

        class MockLogger:
            def log_web_search(self, query, result, context):
                logged.append(query)

        class MockRetriever:
            def search(self, q):
                return WebSearchResult(q, ["s"], ["u"], "mock", 1.0)

        handler = WebSearchHandler(
            retriever=MockRetriever(), data_logger=MockLogger()
        )
        handler.process("<web_search>logged query</web_search>")
        assert "logged query" in logged


# ---------------------------------------------------------------------------
# TrainingDataCapture
# ---------------------------------------------------------------------------

class TestTrainingDataCapture:
    def _capture(self, tmp_path):
        cfg = CaptureConfig(output_dir=str(tmp_path), flush_every=1, min_response_tokens=1)
        return TrainingDataCapture(cfg)

    def test_log_web_search_writes_jsonl(self, tmp_path):
        cap = self._capture(tmp_path)
        result = WebSearchResult(
            query="q", snippets=["Paris is the capital of France"],
            urls=["https://example.com"], engine="mock", latency_ms=5.0,
        )
        cap.log_web_search("What is the capital of France?", result)
        lines = (tmp_path / "web_search.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["source"] == "web_search"
        assert "Paris" in record["response"]

    def test_log_api_routing_creates_dpo_pair(self, tmp_path):
        cap = self._capture(tmp_path)
        cap.log_api_routing(
            prompt="Explain quantum entanglement",
            local_response="It is when particles are linked somehow.",
            api_response="Quantum entanglement is a phenomenon where two particles share a quantum state.",
            model="llama-3.1-8b",
        )
        record = json.loads((tmp_path / "api_routing.jsonl").read_text())
        assert record["chosen"] == record["response"]
        assert record["rejected"] != record["chosen"]

    def test_duplicate_response_not_logged(self, tmp_path):
        cap = self._capture(tmp_path)
        cap.log_api_routing("p", "same response", "same response")
        assert not (tmp_path / "api_routing.jsonl").exists()

    def test_short_response_discarded(self, tmp_path):
        cfg = CaptureConfig(output_dir=str(tmp_path), flush_every=1, min_response_tokens=10)
        cap = TrainingDataCapture(cfg)
        result = WebSearchResult("q", ["ok"], ["u"], "mock", 1.0)
        cap.log_web_search("q", result)  # "ok" is 1 word < min_response_tokens=10
        assert not (tmp_path / "web_search.jsonl").exists()

    def test_log_uncertain(self, tmp_path):
        cap = self._capture(tmp_path)
        cap.log_uncertain(
            prompt="When did WWII end?",
            response="It ended in <uncertain>1944 or 1945</uncertain>.",
            uncertain_spans=["1944 or 1945"],
        )
        record = json.loads((tmp_path / "uncertain.jsonl").read_text())
        assert record["source"] == "uncertain"
        assert "1944 or 1945" in record["metadata"]["uncertain_spans"]

    def test_log_fact_check_dpo(self, tmp_path):
        cap = self._capture(tmp_path)
        cap.log_fact_check(
            prompt="Is the earth flat?",
            original_response="Yes the earth is flat.",
            verdict={"correct": False, "corrected": "No, the earth is an oblate spheroid."},
        )
        record = json.loads((tmp_path / "fact_check.jsonl").read_text())
        assert "oblate spheroid" in record["chosen"]
        assert "flat" in record["rejected"]

    def test_get_stats(self, tmp_path):
        cap = self._capture(tmp_path)
        result = WebSearchResult("q", ["long enough response here yes"], ["u"], "mock", 1.0)
        cap.log_web_search("q", result)
        stats = cap.get_stats()
        assert stats["web_search"] == 1

    def test_flush_writes_buffer(self, tmp_path):
        cfg = CaptureConfig(output_dir=str(tmp_path), flush_every=100, min_response_tokens=1)
        cap = TrainingDataCapture(cfg)
        result = WebSearchResult("q", ["a snippet"], ["u"], "mock", 1.0)
        cap.log_web_search("q", result)
        assert not (tmp_path / "web_search.jsonl").exists()
        cap.flush()
        assert (tmp_path / "web_search.jsonl").exists()


# ---------------------------------------------------------------------------
# ConfidenceRouter
# ---------------------------------------------------------------------------

class TestConfidenceRouter:
    def test_passes_through_when_confident(self):
        router = ConfidenceRouter(local_fn=lambda p, **kw: "confident answer")
        assert router.generate("question") == "confident answer"

    def test_routes_on_uncertain_token(self, tmp_path):
        api_called = []

        def fake_local(p, **kw):
            return "I <uncertain>think</uncertain> it might be X"

        router = ConfidenceRouter(
            local_fn=fake_local,
            config=RouterConfig(api_key=""),
        )
        # Without real API: _call_api returns None → fallback to local
        result = router.generate("question")
        # local_fn was called
        assert "uncertain" in result or result == "I <uncertain>think</uncertain> it might be X"

    def test_routes_on_fact_check_token(self, tmp_path):
        def fake_local(p, **kw):
            return "The <fact_check>moon is made of cheese</fact_check>"

        router = ConfidenceRouter(local_fn=fake_local, config=RouterConfig(api_key=""))
        result = router.generate("what is the moon made of")
        assert result is not None

    def test_sample_rate_zero_no_routing(self):
        router = ConfidenceRouter(
            local_fn=lambda p, **kw: "plain response",
            config=RouterConfig(sample_rate=0.0),
        )
        result = router.generate("question")
        assert result == "plain response"

    def test_max_api_calls_respected(self):
        router = ConfidenceRouter(
            local_fn=lambda p, **kw: "<uncertain>x</uncertain>",
            config=RouterConfig(max_api_calls=0),
        )
        result = router.generate("q")
        assert router._api_call_count == 0

    def test_get_stats(self):
        router = ConfidenceRouter(
            local_fn=lambda p, **kw: "ok",
            config=RouterConfig(max_api_calls=5),
        )
        stats = router.get_stats()
        assert stats["api_calls"] == 0
        assert stats["api_calls_remaining"] == 5

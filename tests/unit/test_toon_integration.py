"""Tests for TOON integration in SearchTokenHandler and WebSearchHandler."""

import pytest
from core.special_tokens.search import SearchTokenHandler, _TOON_AVAILABLE
from core.special_tokens.web_search import (
    WebSearchHandler, WebSearchResult, _format_results_toon,
)


class TestToonAvailability:
    def test_toon_module_importable(self):
        assert _TOON_AVAILABLE is True

    def test_toon_format_array_produces_header(self):
        from utils.jsonld_toon import _format_array
        rows = [
            {"snippet": "Paris is the capital", "url": "https://a.com"},
            {"snippet": "Population 2M", "url": "https://b.com"},
        ]
        out = _format_array("results", rows, indent=0)
        assert "results[2]{snippet,url}:" in out
        assert "Paris is the capital" in out
        assert "https://a.com" in out


class TestSearchTokenHandlerToon:
    def test_single_result_no_toon(self):
        class FakeRetriever:
            def retrieve(self, q):
                return ["single result text"]

        handler = SearchTokenHandler(retriever=FakeRetriever(), use_toon=True)
        out = handler.process("<search>query</search>")
        assert "[Retrieved: single result text]" == out

    def test_multiple_results_toon_format(self):
        class FakeRetriever:
            def retrieve(self, q):
                return [
                    {"text": "Result one", "score": "0.9"},
                    {"text": "Result two", "score": "0.8"},
                    {"text": "Result three", "score": "0.7"},
                ]

        handler = SearchTokenHandler(retriever=FakeRetriever(), use_toon=True)
        out = handler.process("<search>capital of France</search>")
        assert "[Retrieved:" in out
        assert "results[3]{" in out      # TOON tabular header
        assert "Result one" in out

    def test_multiple_results_toon_disabled(self):
        class FakeRetriever:
            def retrieve(self, q):
                return [
                    {"text": "Result one", "score": "0.9"},
                    {"text": "Result two", "score": "0.8"},
                ]

        handler = SearchTokenHandler(retriever=FakeRetriever(), use_toon=False)
        out = handler.process("<search>query</search>")
        assert "results[2]{" not in out   # no TOON tabular header
        assert "Result one" in out

    def test_non_uniform_dicts_fallback(self):
        class FakeRetriever:
            def retrieve(self, q):
                return [
                    {"text": "one", "extra": "x"},
                    {"text": "two"},           # different keys
                ]

        handler = SearchTokenHandler(retriever=FakeRetriever(), use_toon=True)
        out = handler.process("<search>query</search>")
        # Falls back to plain first result
        assert "[Retrieved: " in out
        assert "results[2]{" not in out


class TestWebSearchHandlerToon:
    def _mock_result(self, n=3):
        return WebSearchResult(
            query="q",
            snippets=[f"Snippet {i}" for i in range(n)],
            urls=[f"https://example{i}.com" for i in range(n)],
            engine="mock",
            latency_ms=1.0,
        )

    def test_format_results_toon_multiple(self):
        result = self._mock_result(3)
        out = _format_results_toon(result.snippets, result.urls, "q")
        assert "results[3]{snippet,url}:" in out
        assert "Snippet 0" in out

    def test_format_results_toon_single_fallback(self):
        result = self._mock_result(1)
        out = _format_results_toon(result.snippets, result.urls, "q")
        # Single result → plain text, no tabular header
        assert "results[1]{" not in out
        assert "Snippet 0" in out

    def test_handler_uses_toon_for_multiple(self):
        class MockRetriever:
            def search(self, q):
                return WebSearchResult(
                    q,
                    snippets=["First result", "Second result", "Third result"],
                    urls=["https://a.com", "https://b.com", "https://c.com"],
                    engine="mock",
                    latency_ms=1.0,
                )

        handler = WebSearchHandler(retriever=MockRetriever(), use_toon=True)
        out = handler.process("<web_search>latest news</web_search>")
        assert "[Web:" in out
        assert "results[3]{snippet,url}:" in out
        assert "First result" in out

    def test_handler_single_result_no_toon(self):
        class MockRetriever:
            def search(self, q):
                return WebSearchResult(q, ["only one result"], ["https://x.com"], "mock", 1.0)

        handler = WebSearchHandler(retriever=MockRetriever(), use_toon=True)
        out = handler.process("<web_search>query</web_search>")
        assert out == "[Web: only one result]"

    def test_handler_toon_disabled(self):
        class MockRetriever:
            def search(self, q):
                return WebSearchResult(
                    q, ["r1", "r2", "r3"], ["u1", "u2", "u3"], "mock", 1.0
                )

        handler = WebSearchHandler(retriever=MockRetriever(), use_toon=False)
        out = handler.process("<web_search>q</web_search>")
        assert "results[3]{" not in out
        assert out == "[Web: r1]"

    def test_empty_results(self):
        class MockRetriever:
            def search(self, q):
                return WebSearchResult(q, [], [], "mock", 1.0)

        handler = WebSearchHandler(retriever=MockRetriever(), use_toon=True)
        out = handler.process("before <web_search>q</web_search> after")
        assert "<web_search>" not in out
        assert out.strip() == "before  after"

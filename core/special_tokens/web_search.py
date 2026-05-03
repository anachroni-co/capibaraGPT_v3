"""
<web_search> / </web_search> — real-time internet search trigger.

Semantically distinct from <search> (which queries the local RAG index):
  <search>       → local RAG retrieval (fast, private, offline)
  <web_search>   → live internet search (real-time, external, slower)

The model inserts <web_search>query</web_search> when it needs
up-to-date information that the local knowledge base may not have
(current events, prices, recent papers, live data…).

WebSearchRetriever is the pluggable backend. Supported engines:
  - Brave Search API  (recommended: privacy-respecting, generous free tier)
  - Serper (Google)
  - DuckDuckGo (no key required, rate-limited)

Results returned by WebSearchHandler are:
  1. Injected inline as [Web: …] context in the response
  2. Optionally indexed into the local RAG store for future queries
  3. Logged as (query, result) pairs for training-data capture

strip_from_output=True: the raw tag is stripped; retrieved context
replaces it inline.

Seed tokens: "internet", "web", "online", "current", "today", "latest"
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base import SpecialTokenConfig, SpecialTokenProcessor
from .registry import register_token

WEB_SEARCH_TOKEN = SpecialTokenConfig(
    name="web_search",
    open_tag="<web_search>",
    close_tag="</web_search>",
    seed_tokens=["internet", "web", "online", "current", "today", "latest"],
    alpha=0.5,
    boundary_token="<im_start>",
    strip_from_output=True,
    description="Real-time internet search trigger — distinct from local RAG <search>",
)

register_token(WEB_SEARCH_TOKEN)

_FULL_PATTERN = re.compile(
    re.escape(WEB_SEARCH_TOKEN.open_tag) + r"(.*?)" + re.escape(WEB_SEARCH_TOKEN.close_tag),
    re.DOTALL,
)


@dataclass
class WebSearchResult:
    query: str
    snippets: List[str]
    urls: List[str]
    engine: str
    latency_ms: float
    timestamp: float = field(default_factory=time.time)

    @property
    def top_snippet(self) -> str:
        return self.snippets[0] if self.snippets else ""


class WebSearchRetriever:
    """
    Pluggable web-search backend.

    Supports Brave, Serper (Google), and DuckDuckGo.
    Pass engine="brave"|"serper"|"duckduckgo" and the appropriate api_key.
    """

    def __init__(
        self,
        engine: str = "duckduckgo",
        api_key: Optional[str] = None,
        max_results: int = 3,
        timeout: float = 5.0,
    ) -> None:
        self.engine = engine
        self.api_key = api_key
        self.max_results = max_results
        self.timeout = timeout

    def search(self, query: str) -> WebSearchResult:
        t0 = time.time()
        try:
            if self.engine == "brave":
                snippets, urls = self._brave(query)
            elif self.engine == "serper":
                snippets, urls = self._serper(query)
            else:
                snippets, urls = self._duckduckgo(query)
        except Exception as e:
            snippets, urls = [f"[search error: {e}]"], []

        return WebSearchResult(
            query=query,
            snippets=snippets[: self.max_results],
            urls=urls[: self.max_results],
            engine=self.engine,
            latency_ms=(time.time() - t0) * 1000,
        )

    def _brave(self, query: str) -> Tuple[List[str], List[str]]:
        import urllib.request, json
        url = f"https://api.search.brave.com/res/v1/web/search?q={urllib.parse.quote(query)}&count={self.max_results}"
        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key or "",
        })
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        results = data.get("web", {}).get("results", [])
        return [r.get("description", "") for r in results], [r.get("url", "") for r in results]

    def _serper(self, query: str) -> Tuple[List[str], List[str]]:
        import urllib.request, urllib.parse, json
        payload = json.dumps({"q": query, "num": self.max_results}).encode()
        req = urllib.request.Request(
            "https://google.serper.dev/search",
            data=payload,
            headers={"X-API-KEY": self.api_key or "", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        results = data.get("organic", [])
        return [r.get("snippet", "") for r in results], [r.get("link", "") for r in results]

    def _duckduckgo(self, query: str) -> Tuple[List[str], List[str]]:
        import urllib.request, urllib.parse, json
        encoded = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
        with urllib.request.urlopen(url, timeout=self.timeout) as r:
            data = json.loads(r.read())
        results = data.get("RelatedTopics", [])[:self.max_results]
        snippets = [r.get("Text", "") for r in results if "Text" in r]
        urls = [r.get("FirstURL", "") for r in results if "FirstURL" in r]
        return snippets, urls


class WebSearchHandler:
    """
    Replace <web_search>query</web_search> blocks with live web results.

    Optionally indexes results into a RAG store and logs (query, result)
    pairs for training-data capture.
    """

    def __init__(
        self,
        retriever: Optional[WebSearchRetriever] = None,
        rag_store=None,
        data_logger=None,
    ) -> None:
        self._retriever = retriever or WebSearchRetriever()
        self._rag_store = rag_store       # rag.retriever.Retriever or compatible
        self._data_logger = data_logger   # TrainingDataCapture (built next)
        self._proc = SpecialTokenProcessor(WEB_SEARCH_TOKEN)

    def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Replace each <web_search> block with retrieved web context inline.
        Side-effects: RAG indexing + training-data logging if configured.
        """
        if not _FULL_PATTERN.search(text):
            return text

        def _replace(match: re.Match) -> str:
            query = match.group(1).strip()
            result = self._retriever.search(query)

            # Index into RAG store for future local queries
            if self._rag_store is not None and result.snippets:
                try:
                    self._rag_store.add(
                        texts=result.snippets,
                        metadatas=[{"url": u, "query": query} for u in result.urls],
                    )
                except Exception:
                    pass

            # Log for training-data capture
            if self._data_logger is not None:
                try:
                    self._data_logger.log_web_search(
                        query=query, result=result, context=context
                    )
                except Exception:
                    pass

            if result.top_snippet:
                return f"[Web: {result.top_snippet}]"
            return ""

        return _FULL_PATTERN.sub(_replace, text).strip()

    def extract_queries(self, text: str) -> List[str]:
        return [m.strip() for m in _FULL_PATTERN.findall(text)]

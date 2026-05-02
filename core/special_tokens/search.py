"""
<search> / </search> — on-demand RAG trigger.

The model inserts <search>query</search> at the exact token position
where it needs external knowledge — instead of retrieving context
blindly at the start of every request.

SearchTokenHandler integrates with rag.retriever.Retriever:
  - parses all <search> blocks from generated text
  - queries the retriever for each block's content
  - replaces the block with [Retrieved: …] context inline
  - strips the tags from the final output

Seed tokens: "find", "lookup", "retrieve", "search", "query"
Stripped from final output (strip_from_output=True).
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from .base import SpecialTokenConfig, SpecialTokenProcessor
from .registry import register_token

SEARCH_TOKEN = SpecialTokenConfig(
    name="search",
    open_tag="<search>",
    close_tag="</search>",
    seed_tokens=["find", "lookup", "retrieve", "search", "query"],
    alpha=0.5,
    boundary_token="<im_start>",
    strip_from_output=True,
    description="Search/retrieval trigger: signals where the model needs external knowledge",
)

register_token(SEARCH_TOKEN)

_FULL_PATTERN = re.compile(
    re.escape(SEARCH_TOKEN.open_tag) + r"(.*?)" + re.escape(SEARCH_TOKEN.close_tag),
    re.DOTALL,
)


class SearchTokenHandler:
    """
    Replace <search>query</search> blocks with retrieved context.

    If no retriever is provided the blocks are simply stripped.
    """

    def __init__(self, retriever: Optional[Any] = None) -> None:
        self._retriever = retriever
        self._proc = SpecialTokenProcessor(SEARCH_TOKEN)

    def process(self, text: str) -> str:
        """
        Replace each <search> block inline with retrieved context,
        or strip the block if no retriever is configured.
        """
        if self._retriever is None:
            return self._proc.strip(text)

        def _replace(match: re.Match) -> str:
            query = match.group(1).strip()
            try:
                results = self._retriever.retrieve(query)
                if results:
                    ctx = results[0] if isinstance(results, list) else results
                    return f"[Retrieved: {ctx}]"
            except Exception:
                pass
            return ""

        return _FULL_PATTERN.sub(_replace, text).strip()

    def extract_queries(self, text: str) -> List[str]:
        """Return all query strings inside <search> blocks."""
        return [m.strip() for m in _FULL_PATTERN.findall(text)]

    def count_searches(self, text: str) -> int:
        return len(_FULL_PATTERN.findall(text))

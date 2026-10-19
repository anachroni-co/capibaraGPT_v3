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
from typing import Any, List, Optional

from .base import SpecialTokenConfig, SpecialTokenProcessor
from .registry import register_token

# TOON serialization for compact RAG context injection
try:
    from utils.jsonld_toon import _format_array as _toon_format_array
    _TOON_AVAILABLE = True
except Exception:
    _TOON_AVAILABLE = False

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

    When multiple results are returned and use_toon=True (default),
    results are serialized as TOON tabular format to reduce prompt
    token overhead by ~30-40% vs JSON or repeated key-value pairs.

    If no retriever is provided the blocks are simply stripped.
    """

    def __init__(
        self,
        retriever: Optional[Any] = None,
        use_toon: bool = True,
    ) -> None:
        self._retriever = retriever
        self._use_toon = use_toon and _TOON_AVAILABLE
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
                if not results:
                    return ""
                if isinstance(results, list):
                    if self._use_toon and len(results) > 1:
                        # Normalize to list of dicts if needed
                        rows = [
                            r if isinstance(r, dict) else {"text": str(r)}
                            for r in results
                        ]
                        # Only use tabular TOON if all rows share same keys
                        keys = [set(r.keys()) for r in rows]
                        if all(k == keys[0] for k in keys):
                            try:
                                toon = _toon_format_array("results", rows, indent=0)
                                return f"[Retrieved:\n{toon}]"
                            except Exception:
                                pass
                    ctx = results[0]
                    return f"[Retrieved: {ctx}]"
                return f"[Retrieved: {results}]"
            except Exception:
                pass
            return ""

        return _FULL_PATTERN.sub(_replace, text).strip()

    def extract_queries(self, text: str) -> List[str]:
        """Return all query strings inside <search> blocks."""
        return [m.strip() for m in _FULL_PATTERN.findall(text)]

    def count_searches(self, text: str) -> int:
        return len(_FULL_PATTERN.findall(text))

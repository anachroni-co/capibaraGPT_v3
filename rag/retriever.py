"""RAG retriever — query embedding + similarity search + context augmentation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .ingestion import embed_query
from .store import Document, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    document: Document
    score: float

    @property
    def text(self) -> str:
        return self.document.text

    @property
    def metadata(self) -> dict:
        return self.document.metadata


class Retriever:
    """Query the VectorStore and augment prompts with retrieved context."""

    def __init__(
        self,
        store: VectorStore,
        top_k: int = 5,
        min_score: float = 0.3,
        context_template: str = (
            "Relevant context:\n{context}\n\n"
            "---\n"
            "Question: {query}"
        ),
    ):
        self.store = store
        self.top_k = top_k
        self.min_score = min_score
        self.context_template = context_template

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Embed the query and return matching documents above min_score."""
        if len(self.store) == 0:
            return []
        q_vec = embed_query(query)
        hits = self.store.search(q_vec, top_k=top_k or self.top_k, min_score=self.min_score)
        results = [RetrievalResult(document=doc, score=score) for doc, score in hits]
        logger.debug("retrieve: %d results for query=%.60s", len(results), query)
        return results

    def augment(self, query: str, top_k: Optional[int] = None) -> str:
        """Return the query augmented with retrieved context.

        If nothing is retrieved above min_score, returns the raw query unchanged.
        """
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return query
        context = "\n\n".join(r.text for r in results)
        return self.context_template.format(context=context, query=query)

    def retrieve_texts(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Convenience method — returns just the text strings."""
        return [r.text for r in self.retrieve(query, top_k=top_k)]

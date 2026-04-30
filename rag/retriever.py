"""Capibara Slim — RAG retriever: augments prompts with retrieved context (T6.4).

Usage:
    retriever = Retriever(store)

    # Augment a user query before sending to the model
    augmented = retriever.augment("What is the capital of France?")
    # → "Context:\\n[...retrieved chunks...]\\n\\nQuestion: What is the capital..."

    # Or retrieve raw chunks
    chunks = retriever.retrieve("capital of France", top_k=3)
"""
from __future__ import annotations

import logging
from typing import Optional

from rag.store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)

_CONTEXT_TEMPLATE = (
    "Context information:\n"
    "{context}\n\n"
    "Question: {query}"
)


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        top_k: int = 3,
        min_score: float = 0.0,
        template: Optional[str] = None,
    ) -> None:
        self._store = store
        self._top_k = top_k
        self._min_score = min_score
        self._template = template or _CONTEXT_TEMPLATE

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievedChunk]:
        k = top_k if top_k is not None else self._top_k
        chunks = self._store.search(query, top_k=k)
        return [c for c in chunks if c.score >= self._min_score]

    def augment(self, query: str, top_k: Optional[int] = None) -> str:
        chunks = self.retrieve(query, top_k=top_k)
        if not chunks:
            return query
        context = "\n---\n".join(
            f"[{c.source or 'doc'}] {c.text}" for c in chunks
        )
        return self._template.format(context=context, query=query)

    def __len__(self) -> int:
        return len(self._store)

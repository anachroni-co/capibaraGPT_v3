"""Capibara Slim — in-memory vector store (T6.4).

Stores document chunks as dense vectors and retrieves the top-k most
similar chunks via cosine similarity.

Dependencies: numpy (always available). For better quality use
sentence-transformers — falls back to a simple TF-IDF-style bag-of-words
vector when sentence-transformers is absent.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer as _ST
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Embedding backend
# ---------------------------------------------------------------------------

class _BagOfWordsEmbedder:
    """Minimal BoW embedder — no external deps, lower quality."""

    def __init__(self, dim: int = 512) -> None:
        self._dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for text in texts:
            vec = np.zeros(self._dim, dtype=np.float32)
            for word in text.lower().split():
                idx = hash(word) % self._dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            vecs.append(vec / norm if norm > 0 else vec)
        return np.stack(vecs)


class _STEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = _ST(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True)


def _build_embedder(model_name: Optional[str] = None):
    if _ST_AVAILABLE:
        name = model_name or "all-MiniLM-L6-v2"
        logger.info("RAG: using SentenceTransformer '%s'", name)
        return _STEmbedder(name)
    logger.info("RAG: sentence-transformers absent — using BoW embedder")
    return _BagOfWordsEmbedder()


# ---------------------------------------------------------------------------
# Document + store
# ---------------------------------------------------------------------------

@dataclass
class Document:
    text: str
    source: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


class VectorStore:
    """In-memory vector store with cosine similarity retrieval."""

    def __init__(self, embedder=None) -> None:
        self._embedder = embedder or _build_embedder()
        self._texts: list[str] = []
        self._sources: list[str] = []
        self._vectors: Optional[np.ndarray] = None

    def add(self, documents: list[Document]) -> None:
        if not documents:
            return
        texts = [d.text for d in documents]
        sources = [d.source for d in documents]
        new_vecs = self._embedder.encode(texts)

        self._texts.extend(texts)
        self._sources.extend(sources)

        if self._vectors is None:
            self._vectors = new_vecs
        else:
            self._vectors = np.vstack([self._vectors, new_vecs])

        logger.info("VectorStore: added %d docs, total=%d", len(documents), len(self._texts))

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if self._vectors is None or len(self._texts) == 0:
            return []

        q_vec = self._embedder.encode([query])[0]
        scores = self._vectors @ q_vec          # cosine sim (vectors normalised)

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(
                text=self._texts[i],
                source=self._sources[i],
                score=float(scores[i]),
            )
            for i in top_idx
        ]

    def __len__(self) -> int:
        return len(self._texts)

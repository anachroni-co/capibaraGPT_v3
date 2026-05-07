"""Vector store — numpy in-memory cosine similarity search.

Accepts pre-computed float32 vectors from any embedding backend.
No ML dependency required for the store itself.
"""
from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Document:
    text: str
    embedding: np.ndarray          # float32, shape (dim,)
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """In-memory vector store with cosine similarity retrieval."""

    def __init__(self, dim: Optional[int] = None):
        self._docs: List[Document] = []
        self._matrix: Optional[np.ndarray] = None  # (N, dim) float32, L2-normalised
        self._dim = dim
        self._dirty = True

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, doc: Document) -> None:
        if self._dim is None:
            self._dim = doc.embedding.shape[-1]
        if doc.embedding.shape[-1] != self._dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self._dim}, got {doc.embedding.shape[-1]}"
            )
        self._docs.append(doc)
        self._dirty = True

    def add_many(self, docs: List[Document]) -> None:
        for doc in docs:
            self.add(doc)

    def clear(self) -> None:
        self._docs.clear()
        self._matrix = None
        self._dirty = True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        if not self._docs:
            self._matrix = None
            return
        mat = np.stack([d.embedding for d in self._docs]).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._matrix = mat / norms
        self._dirty = False

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Tuple[Document, float]]:
        """Return top-k documents by cosine similarity, score in [0, 1]."""
        if not self._docs:
            return []
        if self._dirty:
            self._rebuild()
        if self._matrix is None:
            return []

        q = query_vector.astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        scores = self._matrix @ q                       # (N,)
        scores = (scores + 1.0) / 2.0                  # map [-1,1] → [0,1]

        top_k = min(top_k, len(self._docs))
        idx = np.argpartition(scores, -top_k)[-top_k:]
        idx = idx[np.argsort(scores[idx])[::-1]]

        results = []
        for i in idx:
            s = float(scores[i])
            if s >= min_score:
                results.append((self._docs[i], s))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        texts = [d.text for d in self._docs]
        metas = [d.metadata for d in self._docs]
        vecs  = np.stack([d.embedding for d in self._docs]).astype(np.float32) if self._docs else np.empty((0,))
        np.save(path / "vectors.npy", vecs)
        (path / "documents.json").write_text(
            json.dumps({"texts": texts, "metas": metas}), encoding="utf-8"
        )
        logger.info("VectorStore saved: %d documents → %s", len(self._docs), path)

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        path = Path(path)
        vecs = np.load(path / "vectors.npy")
        data = json.loads((path / "documents.json").read_text(encoding="utf-8"))
        store = cls()
        for text, meta, vec in zip(data["texts"], data["metas"], vecs):
            store.add(Document(text=text, embedding=vec, metadata=meta))
        logger.info("VectorStore loaded: %d documents ← %s", len(store._docs), path)
        return store

    def __len__(self) -> int:
        return len(self._docs)

    def __repr__(self) -> str:
        return f"VectorStore(docs={len(self._docs)}, dim={self._dim})"

"""Memory extraction — auto-ingest high-confidence responses into the RAG store.

When the ``memory_extraction`` feature flag is enabled, every verified
response whose confidence score exceeds the configured threshold is
chunked and embedded into the VectorStore, making it available for
future retrieval.

Usage:
    from rag.memory import MemoryExtractor

    extractor = MemoryExtractor(store)
    extractor.maybe_ingest(query="What is X?", response="X is Y.", score=0.9)
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

from .store import VectorStore
from .ingestion import ingest_text

logger = logging.getLogger(__name__)

# Default confidence threshold — override via config / feature_flags
_DEFAULT_THRESHOLD = 0.8


@dataclass
class MemoryEntry:
    query: str
    response: str
    score: float
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        raw = (self.query + self.response).encode()
        self.digest = hashlib.sha256(raw).hexdigest()[:16]


class MemoryExtractor:
    """Ingest model responses into a VectorStore when confidence is high enough.

    Parameters
    ----------
    store:
        The VectorStore to write into.
    threshold:
        Minimum confidence score required for ingestion.
    chunk_size / overlap:
        Passed through to ``ingest_text``.
    deduplicate:
        When True (default), skip entries whose digest was already ingested.
    """

    def __init__(
        self,
        store: VectorStore,
        threshold: float = _DEFAULT_THRESHOLD,
        chunk_size: int = 256,
        overlap: int = 32,
        deduplicate: bool = True,
    ) -> None:
        self.store = store
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.deduplicate = deduplicate
        self._seen: set[str] = set()

    def maybe_ingest(
        self,
        query: str,
        response: str,
        score: float,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Ingest the (query, response) pair if score >= threshold.

        Returns True when the text was ingested, False otherwise.
        """
        if score < self.threshold:
            logger.debug(
                "memory_extractor: skipping (score=%.3f < threshold=%.3f)",
                score,
                self.threshold,
            )
            return False

        entry = MemoryEntry(query=query, response=response, score=score)

        if self.deduplicate and entry.digest in self._seen:
            logger.debug("memory_extractor: duplicate, skipping digest=%s", entry.digest)
            return False

        text = f"Q: {query}\nA: {response}"
        meta = {"source": "memory", "digest": entry.digest, "score": score}
        if metadata:
            meta.update(metadata)

        n = ingest_text(
            text,
            self.store,
            metadata=meta,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

        if n > 0:
            self._seen.add(entry.digest)
            logger.info(
                "memory_extractor: ingested %d chunk(s) (score=%.3f, digest=%s)",
                n,
                score,
                entry.digest,
            )
            return True

        return False

    @property
    def seen_count(self) -> int:
        """Number of unique entries ingested so far."""
        return len(self._seen)

    def clear_dedup_cache(self) -> None:
        """Reset the deduplication set (allows re-ingesting same content)."""
        self._seen.clear()


def build_memory_extractor(store: VectorStore) -> MemoryExtractor | None:
    """Build a MemoryExtractor from feature flags config, or return None if disabled."""
    try:
        from config.feature_flags import is_enabled, flag_config
        if not is_enabled("memory_extraction"):
            return None
        cfg = flag_config("memory_extraction")
        threshold = float(cfg.get("confidence_threshold", _DEFAULT_THRESHOLD))
        return MemoryExtractor(store, threshold=threshold)
    except Exception as exc:
        logger.warning("memory_extractor: could not read config (%s), using defaults", exc)
        return MemoryExtractor(store)

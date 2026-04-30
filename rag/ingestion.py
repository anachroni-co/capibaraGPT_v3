"""Capibara Slim — document ingestion pipeline (T6.4).

Reads plain text files, splits them into overlapping chunks, and
adds them to a VectorStore.

Usage:
    store = VectorStore()
    ingest_file("data/docs/manual.txt", store, chunk_size=256, overlap=32)
    ingest_directory("data/docs/", store)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from rag.store import Document, VectorStore

logger = logging.getLogger(__name__)


def _split_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into word-level chunks with overlap."""
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def ingest_text(
    text: str,
    store: VectorStore,
    source: str = "",
    chunk_size: int = 256,
    overlap: int = 32,
) -> int:
    """Chunk `text` and add to store. Returns number of chunks added."""
    chunks = _split_chunks(text, chunk_size, overlap)
    docs = [Document(text=c, source=source) for c in chunks]
    store.add(docs)
    return len(docs)


def ingest_file(
    path: Union[str, Path],
    store: VectorStore,
    chunk_size: int = 256,
    overlap: int = 32,
) -> int:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    n = ingest_text(text, store, source=str(path), chunk_size=chunk_size, overlap=overlap)
    logger.info("Ingested %d chunks from %s", n, path)
    return n


def ingest_directory(
    directory: Union[str, Path],
    store: VectorStore,
    glob: str = "**/*.txt",
    chunk_size: int = 256,
    overlap: int = 32,
) -> int:
    total = 0
    for p in sorted(Path(directory).glob(glob)):
        total += ingest_file(p, store, chunk_size=chunk_size, overlap=overlap)
    logger.info("Ingested %d total chunks from %s", total, directory)
    return total

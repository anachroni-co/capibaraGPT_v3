"""Document ingestion with EmbedAnything → sentence-transformers → BoW fallback.

Usage:
    from rag.ingestion import ingest_file, ingest_directory, ingest_text
    from rag.store import VectorStore

    store = VectorStore()
    ingest_file("docs/manual.pdf", store)
    ingest_directory("docs/", store)
    ingest_text("Some raw text chunk.", store, metadata={"source": "manual"})
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .store import Document, VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_BACKEND: str = "bow"          # "embed_anything" | "sentence_transformers" | "bow"
_EA_MODEL = None               # embed_anything EmbeddingModel
_ST_MODEL = None               # SentenceTransformer

try:
    import embed_anything
    from embed_anything import EmbeddingModel, WhichModel, TextEmbedConfig
    _EA_MODEL = EmbeddingModel.from_pretrained_hf(
        model_id="sentence-transformers/all-MiniLM-L6-v2"
    )
    _BACKEND = "embed_anything"
    logger.debug("RAG backend: embed_anything (Rust-native, no PyTorch)")
except Exception:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        _BACKEND = "sentence_transformers"
        logger.debug("RAG backend: sentence_transformers")
    except Exception:
        logger.debug("RAG backend: BoW/TF-IDF (numpy only)")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_BOW_VOCAB: Dict[str, int] = {}
_BOW_DIM = 4096


def _bow_embed(text: str) -> np.ndarray:
    """Deterministic BoW embedding with fixed 4096-dim vocabulary via hashing."""
    vec = np.zeros(_BOW_DIM, dtype=np.float32)
    for token in re.findall(r"\w+", text.lower()):
        idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % _BOW_DIM
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts → (N, dim) float32 array."""
    if not texts:
        return np.empty((0, _BOW_DIM), dtype=np.float32)

    if _BACKEND == "embed_anything" and _EA_MODEL is not None:
        try:
            results = embed_anything.embed_query(texts, _EA_MODEL)
            return np.array([r.embedding for r in results], dtype=np.float32)
        except Exception as exc:
            logger.warning("embed_anything failed, falling back to BoW: %s", exc)

    if _BACKEND == "sentence_transformers" and _ST_MODEL is not None:
        try:
            vecs = _ST_MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return vecs.astype(np.float32)
        except Exception as exc:
            logger.warning("sentence_transformers failed, falling back to BoW: %s", exc)

    return np.stack([_bow_embed(t) for t in texts])


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string → (dim,) float32 array."""
    return embed_texts([text])[0]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    strategy: str = "sentence",
) -> List[str]:
    """Split text into chunks.

    strategy:
        "sentence"  — split on sentence boundaries, then merge up to chunk_size chars
        "word"      — split on whitespace, then merge up to chunk_size chars
    """
    text = text.strip()
    if not text:
        return []

    if strategy == "sentence":
        units = re.split(r"(?<=[.!?])\s+", text)
    else:
        units = text.split()

    chunks: List[str] = []
    buf = ""
    for unit in units:
        candidate = (buf + " " + unit).strip() if buf else unit
        if len(candidate) > chunk_size and buf:
            chunks.append(buf)
            buf = text[max(0, len(buf) - overlap):].split(" ", 1)[-1] + " " + unit
            buf = buf.strip()
        else:
            buf = candidate
    if buf:
        chunks.append(buf)
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_text(
    text: str,
    store: VectorStore,
    metadata: Optional[Dict] = None,
    chunk_size: int = 512,
    overlap: int = 64,
    strategy: str = "sentence",
) -> int:
    """Chunk, embed and add raw text to the store. Returns number of chunks added."""
    chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap, strategy=strategy)
    if not chunks:
        return 0
    vecs = embed_texts(chunks)
    meta = metadata or {}
    for chunk, vec in zip(chunks, vecs):
        store.add(Document(text=chunk, embedding=vec, metadata=meta))
    logger.debug("ingest_text: added %d chunks", len(chunks))
    return len(chunks)


def ingest_file(
    path: str | Path,
    store: VectorStore,
    chunk_size: int = 512,
    overlap: int = 64,
    strategy: str = "sentence",
) -> int:
    """Ingest a file (TXT, MD, PDF via embed_anything, source code).

    Returns number of chunks added.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("ingest_file: path not found: %s", path)
        return 0

    suffix = path.suffix.lower()
    meta = {"source": str(path), "filename": path.name}

    # PDF via embed_anything (only if available)
    if suffix == ".pdf" and _BACKEND == "embed_anything" and _EA_MODEL is not None:
        try:
            results = embed_anything.embed_file(
                str(path),
                embedder=_EA_MODEL,
                config=embed_anything.TextEmbedConfig(
                    chunk_size=chunk_size,
                    splitting_strategy="sentence",
                ),
            )
            for r in results:
                m = dict(meta)
                m.update(r.metadata or {})
                store.add(Document(text=r.text, embedding=np.array(r.embedding, dtype=np.float32), metadata=m))
            logger.debug("ingest_file (PDF/EA): added %d chunks from %s", len(results), path)
            return len(results)
        except Exception as exc:
            logger.warning("embed_anything PDF ingestion failed: %s — falling back to text read", exc)

    # Text-based ingestion (TXT, MD, PY, JS, …)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("ingest_file: cannot read %s: %s", path, exc)
        return 0

    return ingest_text(text, store, metadata=meta, chunk_size=chunk_size,
                       overlap=overlap, strategy=strategy)


def ingest_directory(
    directory: str | Path,
    store: VectorStore,
    extensions: Optional[List[str]] = None,
    chunk_size: int = 512,
    overlap: int = 64,
    recursive: bool = True,
) -> int:
    """Ingest all files in a directory. Returns total chunks added."""
    directory = Path(directory)
    if not directory.is_dir():
        logger.warning("ingest_directory: not a directory: %s", directory)
        return 0

    extensions = extensions or [".txt", ".md", ".py", ".js", ".ts", ".pdf", ".rst"]
    pattern = "**/*" if recursive else "*"
    total = 0
    for p in directory.glob(pattern):
        if p.is_file() and p.suffix.lower() in extensions:
            total += ingest_file(p, store, chunk_size=chunk_size, overlap=overlap)
    logger.info("ingest_directory: %d chunks from %s", total, directory)
    return total


def ingest_webpage(
    url: str,
    store: VectorStore,
    chunk_size: int = 512,
    overlap: int = 64,
) -> int:
    """Ingest a webpage via embed_anything (if available) or httpx fallback."""
    meta = {"source": url}

    if _BACKEND == "embed_anything" and _EA_MODEL is not None:
        try:
            results = embed_anything.embed_webpage(url)
            for r in results:
                store.add(Document(
                    text=r.text,
                    embedding=np.array(r.embedding, dtype=np.float32),
                    metadata={**meta, **(r.metadata or {})},
                ))
            logger.debug("ingest_webpage (EA): %d chunks from %s", len(results), url)
            return len(results)
        except Exception as exc:
            logger.warning("embed_anything webpage ingestion failed: %s", exc)

    # Fallback: fetch HTML and strip tags
    try:
        import httpx
        resp = httpx.get(url, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        text = re.sub(r"<[^>]+>", " ", resp.text)
        text = re.sub(r"\s+", " ", text).strip()
        return ingest_text(text, store, metadata=meta, chunk_size=chunk_size, overlap=overlap)
    except Exception as exc:
        logger.warning("ingest_webpage: failed to fetch %s: %s", url, exc)
        return 0


def get_backend() -> str:
    """Return the active embedding backend name."""
    return _BACKEND

#!/usr/bin/env python3
"""Build a vector index over the Capibara Legal raw-text corpus.

The index powers RAG retrieval at inference time: relevant chunks are
prepended to model input as bytes (seq_len=1024, so keep context short).

Embedder auto-detection (highest priority first):
  1. sentence-transformers  paraphrase-multilingual-MiniLM-L12-v2  (117M, fast)
  2. sentence-transformers  paraphrase-multilingual-mpnet-base-v2   (278M, best quality)
  3. sklearn TfidfVectorizer  (no heavy deps)
  4. rank_bm25 BM25Okapi      (pure Python)
  5. character n-gram hashing  (pure Python, zero deps)

Index format (--output dir):
  index.faiss  OR  index.npy  — embedding matrix
  chunks.jsonl                 — one JSON per line: {id, text, source, offset}
  meta.json                    — build metadata

Usage:
    # Build from default location
    python scripts/rag_indexer.py \\
        --input-dir data/raw/legal/ \\
        --output    data/rag_index/ \\
        --chunk-size 300 \\
        --chunk-overlap 50

    # Rebuild from scratch (ignore existing index)
    python scripts/rag_indexer.py \\
        --input-dir data/raw/legal/ \\
        --output    data/rag_index/ \\
        --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag_indexer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".txt", ".md", ".adoc"}
MIN_CHUNK_CHARS = 100
EMBED_BATCH_SIZE = 256

# Sentinel strings stored in meta.json so the retriever can reload the same model
EMBEDDER_SBERT_MINILM   = "sbert:paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDER_SBERT_MPNET    = "sbert:paraphrase-multilingual-mpnet-base-v2"
EMBEDDER_TFIDF          = "tfidf"
EMBEDDER_BM25           = "bm25"
EMBEDDER_NGRAM          = "ngram"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _iter_chunks(
    text: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Iterator[dict]:
    """Yield overlapping character-window chunks from *text*."""
    step = max(1, chunk_size - chunk_overlap)
    offset = 0
    while offset < len(text):
        chunk = text[offset: offset + chunk_size]
        # Strip leading/trailing whitespace but keep the raw offset for provenance
        chunk_stripped = chunk.strip()
        if len(chunk_stripped) >= MIN_CHUNK_CHARS:
            yield {
                "text": chunk_stripped,
                "source": source,
                "offset": offset,
            }
        offset += step


def collect_chunks(
    input_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """Walk *input_dir* recursively and return all text chunks."""
    chunks: list[dict] = []
    files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        logger.warning("No supported files found in %s (extensions: %s)", input_dir, SUPPORTED_EXTENSIONS)
        return chunks

    logger.info("Found %d files in %s", len(files), input_dir)
    for fpath in files:
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", fpath, exc)
            continue
        source = str(fpath)
        for chunk in _iter_chunks(text, source, chunk_size, chunk_overlap):
            chunks.append(chunk)
        if len(chunks) % 1000 < 10:  # rough log every ~1000 chunks
            pass  # logged below in batches

    logger.info("Collected %d chunks from %d files", len(chunks), len(files))
    return chunks


# ---------------------------------------------------------------------------
# Embedder detection
# ---------------------------------------------------------------------------

class SBertEmbedder:
    """Wraps a sentence-transformers model."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
        logger.info("Loading sentence-transformers model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedder_id = f"sbert:{model_name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class TfidfEmbedder:
    """Wraps sklearn TfidfVectorizer with L2-normalised dense output."""

    embedder_id = EMBEDDER_TFIDF

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]
        self._cls = TfidfVectorizer
        self.vectorizer = None  # fitted lazily during build

    def fit(self, texts: list[str]) -> None:
        logger.info("Fitting TF-IDF vectorizer on %d texts…", len(texts))
        self.vectorizer = self._cls(
            max_features=16_384,
            sublinear_tf=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
        )
        self.vectorizer.fit(texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("TfidfEmbedder.fit() must be called before embed()")
        mat = self.vectorizer.transform(texts)
        # Convert sparse → dense and L2-normalise
        dense = mat.toarray().astype(np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return dense / norms

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class BM25Embedder:
    """Wraps rank_bm25 with a dense pseudo-score vector over the corpus vocabulary."""

    embedder_id = EMBEDDER_BM25

    def __init__(self):
        from rank_bm25 import BM25Okapi  # type: ignore[import]
        self._cls = BM25Okapi
        self.bm25 = None
        self._vocab: list[str] = []
        self._corpus_tokens: list[list[str]] = []

    def fit(self, texts: list[str]) -> None:
        logger.info("Building BM25 index over %d texts…", len(texts))
        tokenized = [t.lower().split() for t in texts]
        self._corpus_tokens = tokenized
        # Build vocabulary (top 8k tokens by frequency)
        from collections import Counter
        counter: Counter = Counter(tok for doc in tokenized for tok in doc)
        self._vocab = [w for w, _ in counter.most_common(8192)]
        self._vocab_idx = {w: i for i, w in enumerate(self._vocab)}
        self.bm25 = self._cls(tokenized)

    def _scores_for(self, query_tokens: list[str]) -> np.ndarray:
        scores = self.bm25.get_scores(query_tokens).astype(np.float32)
        norm = np.linalg.norm(scores)
        return scores / norm if norm > 0 else scores

    def embed(self, texts: list[str]) -> np.ndarray:
        # For the corpus itself we return identity-position vectors; query uses scores
        # Practical shortcut: embed as TF-style bag-of-vocab-words
        dim = len(self._vocab)
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = text.lower().split()
            for tok in tokens:
                idx = self._vocab_idx.get(tok)
                if idx is not None:
                    out[i, idx] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class NgramEmbedder:
    """Character n-gram hashing embedder — zero runtime dependencies."""

    embedder_id = EMBEDDER_NGRAM

    DIM = 4096  # hash space dimension
    NGRAM_SIZES = (3, 4, 5)

    def __init__(self):
        logger.info("Using character n-gram hash embedder (dim=%d)", self.DIM)

    def _embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.DIM, dtype=np.float32)
        text_lower = text.lower()
        for n in self.NGRAM_SIZES:
            for i in range(len(text_lower) - n + 1):
                gram = text_lower[i: i + n]
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16) % self.DIM
                vec[h] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed_text(t) for t in texts])

    def embed_one(self, text: str) -> np.ndarray:
        return self._embed_text(text)

    def fit(self, texts: list[str]) -> None:
        pass  # stateless


def _load_best_embedder():
    """Try embedders in priority order; return the first that loads successfully."""

    # 1. sentence-transformers MiniLM
    try:
        embedder = SBertEmbedder("paraphrase-multilingual-MiniLM-L12-v2")
        logger.info("Embedder selected: %s", embedder.embedder_id)
        return embedder
    except Exception as exc:
        logger.debug("sentence-transformers MiniLM unavailable: %s", exc)

    # 2. sentence-transformers mpnet
    try:
        embedder = SBertEmbedder("paraphrase-multilingual-mpnet-base-v2")
        logger.info("Embedder selected: %s", embedder.embedder_id)
        return embedder
    except Exception as exc:
        logger.debug("sentence-transformers mpnet unavailable: %s", exc)

    # 3. TF-IDF
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
        embedder = TfidfEmbedder()
        logger.info("Embedder selected: %s", embedder.embedder_id)
        return embedder
    except ImportError as exc:
        logger.debug("sklearn unavailable: %s", exc)

    # 4. BM25
    try:
        from rank_bm25 import BM25Okapi  # noqa: F401
        embedder = BM25Embedder()
        logger.info("Embedder selected: %s", embedder.embedder_id)
        return embedder
    except ImportError as exc:
        logger.debug("rank_bm25 unavailable: %s", exc)

    # 5. Character n-gram (always available)
    embedder = NgramEmbedder()
    logger.info("Embedder selected: %s", embedder.embedder_id)
    return embedder


# ---------------------------------------------------------------------------
# Embedding pass
# ---------------------------------------------------------------------------

def embed_chunks(embedder, chunks: list[dict]) -> np.ndarray:
    """Embed all chunks in batches; return float32 matrix (n_chunks, dim)."""
    texts = [c["text"] for c in chunks]
    n = len(texts)
    all_vecs: list[np.ndarray] = []

    for batch_start in range(0, n, EMBED_BATCH_SIZE):
        batch = texts[batch_start: batch_start + EMBED_BATCH_SIZE]
        vecs = embedder.embed(batch)
        all_vecs.append(vecs)

        processed = min(batch_start + EMBED_BATCH_SIZE, n)
        if processed % 1000 == 0 or processed == n:
            logger.info("  Embedded %d / %d chunks", processed, n)

    return np.vstack(all_vecs).astype(np.float32)


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def _try_build_faiss(vectors: np.ndarray, output_dir: Path) -> bool:
    """Build FAISS IndexFlatIP and save to output_dir/index.faiss.
    Returns True on success."""
    try:
        import faiss  # type: ignore[import]
    except ImportError:
        logger.debug("faiss-cpu not installed — using numpy fallback")
        return False

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    # Vectors must already be L2-normalised for inner product == cosine similarity
    index.add(vectors)
    faiss_path = output_dir / "index.faiss"
    faiss.write_index(index, str(faiss_path))
    size_mb = faiss_path.stat().st_size / 1e6
    logger.info("FAISS index saved → %s (%.1f MB)", faiss_path, size_mb)
    return True


def _build_numpy(vectors: np.ndarray, output_dir: Path) -> None:
    """Save raw numpy matrix to output_dir/index.npy."""
    npy_path = output_dir / "index.npy"
    np.save(npy_path, vectors)
    size_mb = npy_path.stat().st_size / 1e6
    logger.info("Numpy index saved → %s (%.1f MB)", npy_path, size_mb)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_chunks(chunks: list[dict], output_dir: Path) -> None:
    chunks_path = output_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fh:
        for i, chunk in enumerate(chunks):
            record = {
                "id": i,
                "text": chunk["text"],
                "source": chunk["source"],
                "offset": chunk["offset"],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Chunks saved → %s (%d records)", chunks_path, len(chunks))


def _save_meta(
    output_dir: Path,
    embedder_id: str,
    chunk_size: int,
    chunk_overlap: int,
    n_chunks: int,
    index_type: str,
) -> None:
    meta = {
        "embedder": embedder_id,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "n_chunks": n_chunks,
        "index_type": index_type,   # "faiss" or "numpy"
        "created": datetime.now(tz=timezone.utc).isoformat(),
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Meta saved → %s", meta_path)


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_index(
    input_dir: Path,
    output_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    force: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guard: skip if already built (unless --force)
    meta_path = output_dir / "meta.json"
    if meta_path.exists() and not force:
        logger.info("Index already exists at %s (use --force to rebuild)", output_dir)
        return

    t0 = time.time()

    # 1. Collect chunks
    chunks = collect_chunks(input_dir, chunk_size, chunk_overlap)
    if not chunks:
        logger.error("No chunks collected — aborting.")
        sys.exit(1)

    # 2. Load embedder
    embedder = _load_best_embedder()

    # 3. Fit corpus-level models (TF-IDF, BM25, n-gram)
    if hasattr(embedder, "fit"):
        texts = [c["text"] for c in chunks]
        embedder.fit(texts)

    # 4. Embed all chunks
    logger.info("Embedding %d chunks (batch=%d)…", len(chunks), EMBED_BATCH_SIZE)
    vectors = embed_chunks(embedder, chunks)
    logger.info("Embedding complete: matrix shape %s", vectors.shape)

    # 5. Save chunks.jsonl
    _save_chunks(chunks, output_dir)

    # 6. Build index (FAISS preferred, numpy fallback)
    used_faiss = _try_build_faiss(vectors, output_dir)
    if not used_faiss:
        _build_numpy(vectors, output_dir)
    index_type = "faiss" if used_faiss else "numpy"

    # 7. Save metadata
    _save_meta(
        output_dir,
        embedder_id=embedder.embedder_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        n_chunks=len(chunks),
        index_type=index_type,
    )

    # 8. Summary
    elapsed = time.time() - t0
    total_bytes = sum(
        p.stat().st_size
        for p in output_dir.iterdir()
        if p.is_file()
    )
    logger.info("=" * 60)
    logger.info("Index build complete in %.1f s", elapsed)
    logger.info("  Chunks    : %d", len(chunks))
    logger.info("  Embedder  : %s", embedder.embedder_id)
    logger.info("  Index type: %s", index_type)
    logger.info("  Dimensions: %d", vectors.shape[1])
    logger.info("  Disk size : %.1f MB", total_bytes / 1e6)
    logger.info("  Output    : %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw/legal/",
        help="Directory with raw .txt/.md/.adoc files (default: data/raw/legal/)",
    )
    parser.add_argument(
        "--output",
        default="data/rag_index/",
        help="Output directory for the index (default: data/rag_index/)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="Characters per chunk (default: 300)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap characters between consecutive chunks (default: 50)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if index already exists",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    build_index(
        input_dir=input_dir,
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force=args.force,
    )


if __name__ == "__main__":
    main()

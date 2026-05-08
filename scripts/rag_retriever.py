#!/usr/bin/env python3
"""Load a RAG index built by rag_indexer.py and retrieve relevant chunks at inference time.

The retrieved context is formatted as a short byte string to be prepended
to the model input.  Because the Large preset uses seq_len=1024 bytes,
max_context_bytes defaults to 500 so roughly half the context window is
reserved for the actual query + model output.

RAGRetriever API
----------------
    retriever = RAGRetriever("data/rag_index/", top_k=3, max_context_bytes=500)

    # Returns a ready-to-prepend string (UTF-8 encodeable)
    context = retriever.retrieve("recurso de amparo requisitos")

    # Returns raw scored chunks for programmatic use
    results = retriever.retrieve_raw("plazo interposición recurso")
    # [{"text": "...", "source": "...", "score": 0.87}, ...]

Output format of retrieve():
    [CONTEXTO LEGAL]
    §1 (data/raw/legal/tc-sentencias/BOE-T-2023-1234.txt):
    El recurso de amparo procede cuando se vulneran derechos...

    §2 (data/raw/legal/legalize-es/ley_enjuiciamiento.md):
    El plazo para interponer el recurso es de veinte días...

    [FIN CONTEXTO]

CLI usage (for testing):
    python scripts/rag_retriever.py \\
        --index  data/rag_index/ \\
        --query  "recurso de amparo tribunal constitucional requisitos" \\
        --top-k  3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag_retriever")

# Embedder ID constants — must match rag_indexer.py
EMBEDDER_SBERT_MINILM   = "sbert:paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDER_SBERT_MPNET    = "sbert:paraphrase-multilingual-mpnet-base-v2"
EMBEDDER_TFIDF          = "tfidf"
EMBEDDER_BM25           = "bm25"
EMBEDDER_NGRAM          = "ngram"


# ---------------------------------------------------------------------------
# Embedder loaders (mirrors rag_indexer.py, but query-only path)
# ---------------------------------------------------------------------------

class _SBertQueryEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
        logger.info("Loading sentence-transformers: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def embed_one(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return np.array(vec[0], dtype=np.float32)


class _TfidfQueryEmbedder:
    def __init__(self, vectorizer: Any):
        self._vectorizer = vectorizer

    def embed_one(self, text: str) -> np.ndarray:
        mat = self._vectorizer.transform([text])
        dense = mat.toarray().astype(np.float32)[0]
        norm = np.linalg.norm(dense)
        return dense / norm if norm > 0 else dense


class _BM25QueryEmbedder:
    def __init__(self, vocab_idx: dict[str, int], dim: int):
        self._vocab_idx = vocab_idx
        self._dim = dim

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for tok in text.lower().split():
            idx = self._vocab_idx.get(tok)
            if idx is not None:
                vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class _NgramQueryEmbedder:
    DIM = 4096
    NGRAM_SIZES = (3, 4, 5)

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.DIM, dtype=np.float32)
        text_lower = text.lower()
        for n in self.NGRAM_SIZES:
            for i in range(len(text_lower) - n + 1):
                gram = text_lower[i: i + n]
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16) % self.DIM
                vec[h] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


# ---------------------------------------------------------------------------
# RAGRetriever
# ---------------------------------------------------------------------------

class RAGRetriever:
    """Retrieves relevant legal text chunks from a pre-built vector index.

    Parameters
    ----------
    index_dir:
        Path to the directory produced by rag_indexer.py.
    top_k:
        Number of chunks to retrieve per query.
    max_context_bytes:
        Hard cap on the byte length of the formatted context string.
        The last chunk is truncated (or dropped) if needed.
        At least one chunk is always included.
    """

    def __init__(
        self,
        index_dir: str,
        top_k: int = 3,
        max_context_bytes: int = 500,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.top_k = top_k
        self.max_context_bytes = max_context_bytes

        self._meta: dict = {}
        self._chunks: list[dict] = []
        self._query_embedder: Any = None

        # FAISS index or numpy matrix
        self._faiss_index: Any = None
        self._np_vectors: np.ndarray | None = None

        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load meta, chunks, vectors, and initialise the query embedder."""
        meta_path = self.index_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"meta.json not found in {self.index_dir}. "
                "Run rag_indexer.py first."
            )

        self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
        embedder_id: str = self._meta.get("embedder", EMBEDDER_NGRAM)
        index_type: str = self._meta.get("index_type", "numpy")
        logger.info(
            "Loading RAG index: embedder=%s  index_type=%s  n_chunks=%s",
            embedder_id,
            index_type,
            self._meta.get("n_chunks", "?"),
        )

        # Load chunks
        chunks_path = self.index_dir / "chunks.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(f"chunks.jsonl not found in {self.index_dir}")
        with chunks_path.open(encoding="utf-8") as fh:
            self._chunks = [json.loads(line) for line in fh if line.strip()]
        logger.info("Loaded %d chunks", len(self._chunks))

        # Load vector index
        if index_type == "faiss":
            self._load_faiss()
        else:
            self._load_numpy()

        # Load query embedder — must match what was used during indexing
        self._query_embedder = self._build_query_embedder(embedder_id)

    def _load_faiss(self) -> None:
        faiss_path = self.index_dir / "index.faiss"
        try:
            import faiss  # type: ignore[import]
            self._faiss_index = faiss.read_index(str(faiss_path))
            logger.info("FAISS index loaded (%d vectors)", self._faiss_index.ntotal)
        except ImportError:
            logger.warning(
                "faiss-cpu not installed — falling back to numpy brute-force. "
                "Install with: pip install faiss-cpu"
            )
            # Try to load the numpy matrix if available, otherwise fail
            npy_path = self.index_dir / "index.npy"
            if npy_path.exists():
                self._np_vectors = np.load(npy_path)
            else:
                # Re-read FAISS index raw bytes with numpy is not feasible;
                # we must reconstruct from the faiss file via faiss itself.
                raise RuntimeError(
                    "faiss-cpu is required to load index.faiss but is not installed. "
                    "Alternatively, rebuild the index without faiss."
                )

    def _load_numpy(self) -> None:
        npy_path = self.index_dir / "index.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"index.npy not found in {self.index_dir}")
        self._np_vectors = np.load(npy_path)
        logger.info("Numpy index loaded: shape=%s", self._np_vectors.shape)

    def _build_query_embedder(self, embedder_id: str) -> Any:
        """Reconstruct the query embedder that matches *embedder_id*."""
        if embedder_id.startswith("sbert:"):
            model_name = embedder_id[len("sbert:"):]
            try:
                return _SBertQueryEmbedder(model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load sentence-transformers model '{model_name}'. "
                    f"Original error: {exc}"
                ) from exc

        if embedder_id == EMBEDDER_TFIDF:
            return self._load_tfidf_embedder()

        if embedder_id == EMBEDDER_BM25:
            return self._load_bm25_embedder()

        if embedder_id == EMBEDDER_NGRAM:
            logger.info("Using character n-gram query embedder")
            return _NgramQueryEmbedder()

        raise ValueError(f"Unknown embedder id in meta.json: {embedder_id!r}")

    def _load_tfidf_embedder(self) -> _TfidfQueryEmbedder:
        """Re-fit a TF-IDF vectorizer on the stored chunks (stateless index)."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("sklearn is required for tfidf embedder") from exc

        logger.info("Re-fitting TF-IDF vectorizer on %d chunks…", len(self._chunks))
        texts = [c["text"] for c in self._chunks]
        vec = TfidfVectorizer(
            max_features=16_384,
            sublinear_tf=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
        )
        vec.fit(texts)
        return _TfidfQueryEmbedder(vec)

    def _load_bm25_embedder(self) -> _BM25QueryEmbedder:
        """Re-derive the BM25 vocabulary from stored chunks."""
        from collections import Counter
        texts = [c["text"] for c in self._chunks]
        tokenized = [t.lower().split() for t in texts]
        counter: Counter = Counter(tok for doc in tokenized for tok in doc)
        vocab = [w for w, _ in counter.most_common(8192)]
        vocab_idx = {w: i for i, w in enumerate(vocab)}
        logger.info("BM25 vocabulary: %d tokens", len(vocab))
        return _BM25QueryEmbedder(vocab_idx, dim=len(vocab))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _search(self, query_vec: np.ndarray) -> list[tuple[int, float]]:
        """Return list of (chunk_idx, score) sorted by descending score."""
        k = min(self.top_k, len(self._chunks))

        if self._faiss_index is not None:
            q = query_vec.reshape(1, -1).astype(np.float32)
            scores, indices = self._faiss_index.search(q, k)
            return [(int(indices[0][i]), float(scores[0][i])) for i in range(k)]

        if self._np_vectors is not None:
            # Brute-force cosine similarity (vectors already L2-normalised)
            sims = (self._np_vectors @ query_vec).astype(np.float32)
            top_idx = np.argpartition(sims, -k)[-k:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
            return [(int(idx), float(sims[idx])) for idx in top_idx]

        raise RuntimeError("Neither FAISS nor numpy index loaded")

    def retrieve_raw(self, query: str) -> list[dict]:
        """Return top-k chunks as a list of {text, source, score} dicts."""
        query_vec = self._query_embedder.embed_one(query)
        hits = self._search(query_vec)
        results = []
        for chunk_idx, score in hits:
            if chunk_idx < 0 or chunk_idx >= len(self._chunks):
                continue
            chunk = self._chunks[chunk_idx]
            results.append({
                "text": chunk["text"],
                "source": chunk.get("source", ""),
                "score": score,
            })
        return results

    def retrieve(self, query: str) -> str:
        """Return formatted context string ready to prepend to model input.

        Format:
            [CONTEXTO LEGAL]
            §1 (source_path):
            chunk text...

            §2 (source_path):
            chunk text...

            [FIN CONTEXTO]

        If the total byte length of the formatted string exceeds
        max_context_bytes, trailing chunks are truncated or dropped.
        At least one chunk is always included.
        """
        raw = self.retrieve_raw(query)
        if not raw:
            logger.warning("No chunks retrieved for query: %r", query[:80])
            return ""

        header = "[CONTEXTO LEGAL]\n"
        footer = "\n[FIN CONTEXTO]"

        sections: list[str] = []
        for i, hit in enumerate(raw, start=1):
            section = f"§{i} ({hit['source']}):\n{hit['text']}"
            sections.append(section)

        # Enforce max_context_bytes: try full context first, then drop/truncate
        overhead = len(header.encode("utf-8")) + len(footer.encode("utf-8"))
        budget = self.max_context_bytes - overhead

        kept: list[str] = []
        used_bytes = 0
        truncated = False

        for idx, section in enumerate(sections):
            section_bytes = section.encode("utf-8")
            # separator "\n\n" between sections
            sep_bytes = 2 if kept else 0
            if used_bytes + sep_bytes + len(section_bytes) <= budget:
                kept.append(section)
                used_bytes += sep_bytes + len(section_bytes)
            elif idx == 0:
                # Always include at least the first chunk, even if it exceeds the limit
                truncated = True
                available = budget
                # Truncate to fit: decode back char-by-char from UTF-8 boundary
                encoded = section_bytes[:available]
                # Walk back to a valid UTF-8 boundary
                while encoded:
                    try:
                        encoded.decode("utf-8")
                        break
                    except UnicodeDecodeError:
                        encoded = encoded[:-1]
                kept.append(encoded.decode("utf-8", errors="replace"))
                used_bytes += len(encoded)
                break
            else:
                truncated = True
                break

        if truncated:
            logger.warning(
                "RAG context truncated to fit max_context_bytes=%d "
                "(kept %d / %d chunks)",
                self.max_context_bytes,
                len(kept),
                len(sections),
            )

        body = "\n\n".join(kept)
        return header + body + footer


# ---------------------------------------------------------------------------
# CLI (for testing / debugging)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--index",
        default="data/rag_index/",
        help="Index directory produced by rag_indexer.py (default: data/rag_index/)",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Query string to retrieve against",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve (default: 3)",
    )
    parser.add_argument(
        "--max-context-bytes",
        type=int,
        default=500,
        help="Hard cap on formatted context byte length (default: 500)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw JSON results instead of formatted context",
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

    index_dir = Path(args.index)
    if not index_dir.exists():
        logger.error("Index directory not found: %s", index_dir)
        sys.exit(1)

    retriever = RAGRetriever(
        index_dir=str(index_dir),
        top_k=args.top_k,
        max_context_bytes=args.max_context_bytes,
    )

    if args.raw:
        results = retriever.retrieve_raw(args.query)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        context = retriever.retrieve(args.query)
        print(context)
        print()
        print(f"[context byte length: {len(context.encode('utf-8'))}]")


if __name__ == "__main__":
    main()

"""Week-6 slim tests: RAG pipeline (store, ingestion, retriever)."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class TestVectorStore:
    def _make_store(self):
        from rag.store import VectorStore, Document
        store = VectorStore()
        return store, Document

    def test_add_and_len(self):
        store, Document = self._make_store()
        doc = Document(text="hello", embedding=np.array([1.0, 0.0], dtype=np.float32))
        store.add(doc)
        assert len(store) == 1

    def test_dim_mismatch_raises(self):
        from rag.store import VectorStore, Document
        store = VectorStore()
        store.add(Document(text="a", embedding=np.array([1.0, 0.0], dtype=np.float32)))
        with pytest.raises(ValueError, match="dim"):
            store.add(Document(text="b", embedding=np.array([1.0], dtype=np.float32)))

    def test_search_returns_sorted(self):
        from rag.store import VectorStore, Document
        store = VectorStore()
        store.add(Document(text="cat", embedding=np.array([1.0, 0.0], dtype=np.float32)))
        store.add(Document(text="dog", embedding=np.array([0.0, 1.0], dtype=np.float32)))
        query = np.array([1.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=2)
        assert results[0][0].text == "cat"
        assert results[0][1] > results[1][1]

    def test_search_empty_store(self):
        from rag.store import VectorStore
        store = VectorStore()
        results = store.search(np.array([1.0, 0.0], dtype=np.float32))
        assert results == []

    def test_min_score_filter(self):
        from rag.store import VectorStore, Document
        store = VectorStore()
        store.add(Document(text="opposite", embedding=np.array([-1.0, 0.0], dtype=np.float32)))
        query = np.array([1.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=1, min_score=0.9)
        assert results == []

    def test_save_and_load(self, tmp_path):
        from rag.store import VectorStore, Document
        store = VectorStore()
        store.add(Document(text="persist", embedding=np.array([0.5, 0.5], dtype=np.float32), metadata={"k": "v"}))
        store.save(tmp_path / "vs")
        loaded = VectorStore.load(tmp_path / "vs")
        assert len(loaded) == 1
        assert loaded._docs[0].text == "persist"
        assert loaded._docs[0].metadata == {"k": "v"}

    def test_clear(self):
        from rag.store import VectorStore, Document
        store = VectorStore()
        store.add(Document(text="x", embedding=np.array([1.0], dtype=np.float32)))
        store.clear()
        assert len(store) == 0

    def test_add_many(self):
        from rag.store import VectorStore, Document
        store = VectorStore()
        docs = [Document(text=str(i), embedding=np.array([float(i)], dtype=np.float32)) for i in range(1, 4)]
        store.add_many(docs)
        assert len(store) == 3

    def test_repr(self):
        from rag.store import VectorStore
        store = VectorStore()
        assert "VectorStore" in repr(store)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class TestIngestion:
    def test_embed_texts_returns_array(self):
        from rag.ingestion import embed_texts
        vecs = embed_texts(["hello world", "foo bar"])
        assert isinstance(vecs, np.ndarray)
        assert vecs.shape[0] == 2
        assert vecs.dtype == np.float32

    def test_embed_query_shape(self):
        from rag.ingestion import embed_query
        vec = embed_query("test query")
        assert vec.ndim == 1
        assert vec.dtype == np.float32

    def test_embed_texts_empty(self):
        from rag.ingestion import embed_texts
        vecs = embed_texts([])
        assert vecs.shape[0] == 0

    def test_get_backend_returns_string(self):
        from rag.ingestion import get_backend
        backend = get_backend()
        assert isinstance(backend, str)
        assert backend in {"embed_anything", "sentence_transformers", "bow"}

    def test_ingest_text_adds_chunks(self):
        from rag.ingestion import ingest_text
        from rag.store import VectorStore
        store = VectorStore()
        text = "The quick brown fox. " * 50
        n = ingest_text(text, store)
        assert n > 0
        assert len(store) == n

    def test_ingest_text_empty(self):
        from rag.ingestion import ingest_text
        from rag.store import VectorStore
        store = VectorStore()
        n = ingest_text("", store)
        assert n == 0
        assert len(store) == 0

    def test_ingest_text_metadata(self):
        from rag.ingestion import ingest_text
        from rag.store import VectorStore
        store = VectorStore()
        ingest_text("Some text.", store, metadata={"source": "test"})
        assert store._docs[0].metadata["source"] == "test"

    def test_ingest_file_txt(self, tmp_path):
        from rag.ingestion import ingest_file
        from rag.store import VectorStore
        f = tmp_path / "doc.txt"
        f.write_text("Hello from file. " * 20, encoding="utf-8")
        store = VectorStore()
        n = ingest_file(f, store)
        assert n > 0
        assert store._docs[0].metadata["filename"] == "doc.txt"

    def test_ingest_file_missing(self, tmp_path):
        from rag.ingestion import ingest_file
        from rag.store import VectorStore
        store = VectorStore()
        n = ingest_file(tmp_path / "nonexistent.txt", store)
        assert n == 0

    def test_ingest_directory(self, tmp_path):
        from rag.ingestion import ingest_directory
        from rag.store import VectorStore
        (tmp_path / "a.txt").write_text("Alpha text chunk here. " * 10, encoding="utf-8")
        (tmp_path / "b.md").write_text("Beta markdown content. " * 10, encoding="utf-8")
        store = VectorStore()
        total = ingest_directory(tmp_path, store, extensions=[".txt", ".md"])
        assert total > 0
        assert len(store) == total

    def test_ingest_directory_nonexistent(self, tmp_path):
        from rag.ingestion import ingest_directory
        from rag.store import VectorStore
        store = VectorStore()
        n = ingest_directory(tmp_path / "does_not_exist", store)
        assert n == 0

    def test_chunk_text_sentence_strategy(self):
        from rag.ingestion import _chunk_text
        text = "First sentence. Second sentence. Third sentence."
        chunks = _chunk_text(text, chunk_size=30, overlap=5, strategy="sentence")
        assert isinstance(chunks, list)
        assert all(isinstance(c, str) for c in chunks)

    def test_chunk_text_word_strategy(self):
        from rag.ingestion import _chunk_text
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = _chunk_text(text, chunk_size=50, overlap=10, strategy="word")
        assert len(chunks) > 1

    def test_chunk_text_empty(self):
        from rag.ingestion import _chunk_text
        assert _chunk_text("") == []
        assert _chunk_text("   ") == []

    def test_bow_embed_deterministic(self):
        from rag.ingestion import _bow_embed
        v1 = _bow_embed("hello world")
        v2 = _bow_embed("hello world")
        np.testing.assert_array_equal(v1, v2)

    def test_bow_embed_normalized(self):
        from rag.ingestion import _bow_embed
        vec = _bow_embed("some test text")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_bow_embed_empty(self):
        from rag.ingestion import _bow_embed
        vec = _bow_embed("")
        assert np.all(vec == 0)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class TestRetriever:
    def _setup(self):
        from rag.store import VectorStore
        from rag.ingestion import ingest_text
        from rag.retriever import Retriever
        store = VectorStore()
        ingest_text("The capital of France is Paris.", store)
        ingest_text("Python is a programming language.", store)
        ingest_text("Dogs are friendly animals.", store)
        retriever = Retriever(store, top_k=2, min_score=0.0)
        return retriever, store

    def test_retrieve_returns_results(self):
        retriever, _ = self._setup()
        results = retriever.retrieve("What is the capital of France?")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_retrieve_result_has_score(self):
        retriever, _ = self._setup()
        results = retriever.retrieve("programming language")
        assert all(hasattr(r, "score") for r in results)
        assert all(0.0 <= r.score <= 1.0 for r in results)

    def test_retrieve_empty_store(self):
        from rag.store import VectorStore
        from rag.retriever import Retriever
        store = VectorStore()
        retriever = Retriever(store)
        results = retriever.retrieve("anything")
        assert results == []

    def test_augment_adds_context(self):
        retriever, _ = self._setup()
        augmented = retriever.augment("capital of France")
        assert "Relevant context:" in augmented
        assert "capital of France" in augmented

    def test_augment_no_results_returns_query(self):
        from rag.store import VectorStore
        from rag.retriever import Retriever
        store = VectorStore()
        retriever = Retriever(store)
        query = "something unique"
        assert retriever.augment(query) == query

    def test_retrieve_texts_list_of_strings(self):
        retriever, _ = self._setup()
        texts = retriever.retrieve_texts("France")
        assert isinstance(texts, list)
        assert all(isinstance(t, str) for t in texts)

    def test_top_k_override(self):
        retriever, _ = self._setup()
        results = retriever.retrieve("language", top_k=1)
        assert len(results) <= 1

    def test_min_score_filters(self):
        from rag.store import VectorStore
        from rag.ingestion import ingest_text
        from rag.retriever import Retriever
        store = VectorStore()
        ingest_text("Completely unrelated content xyz.", store)
        retriever = Retriever(store, min_score=0.99)
        results = retriever.retrieve("France capital city")
        assert results == []

    def test_result_text_property(self):
        retriever, _ = self._setup()
        results = retriever.retrieve("dogs animals")
        for r in results:
            assert isinstance(r.text, str)

    def test_result_metadata_property(self):
        retriever, _ = self._setup()
        results = retriever.retrieve("France")
        for r in results:
            assert isinstance(r.metadata, dict)

    def test_custom_context_template(self):
        from rag.store import VectorStore
        from rag.ingestion import ingest_text
        from rag.retriever import Retriever
        store = VectorStore()
        ingest_text("Context document.", store)
        retriever = Retriever(store, context_template="CTX:{context}\nQ:{query}", min_score=0.0)
        aug = retriever.augment("query text")
        assert aug.startswith("CTX:")
        assert "Q:query text" in aug


# ---------------------------------------------------------------------------
# RAG package exports
# ---------------------------------------------------------------------------

class TestRAGPackage:
    def test_imports(self):
        import rag
        assert hasattr(rag, "VectorStore")
        assert hasattr(rag, "Document")
        assert hasattr(rag, "Retriever")
        assert hasattr(rag, "ingest_text")
        assert hasattr(rag, "ingest_file")
        assert hasattr(rag, "ingest_directory")
        assert hasattr(rag, "ingest_webpage")

    def test_end_to_end(self, tmp_path):
        from rag import VectorStore, Retriever, ingest_text
        store = VectorStore()
        ingest_text("Capibara Slim is a hybrid transformer-mamba model.", store)
        ingest_text("It supports JAX/Flax for TPU training.", store)
        retriever = Retriever(store, top_k=1, min_score=0.0)
        aug = retriever.augment("What is Capibara Slim?")
        assert "Relevant context:" in aug
        assert "What is Capibara Slim?" in aug

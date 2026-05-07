"""RAG package for Capibara Slim.

Backends (in order of preference):
  1. embed_anything  — Rust-native, no PyTorch required (pip install embed-anything)
  2. sentence_transformers — PyTorch-based (pip install sentence-transformers)
  3. BoW / TF-IDF fallback — always available (numpy only)
"""
from .store import VectorStore, Document
from .ingestion import ingest_text, ingest_file, ingest_directory, ingest_webpage
from .retriever import Retriever

__all__ = [
    "VectorStore",
    "Document",
    "ingest_text",
    "ingest_file",
    "ingest_directory",
    "ingest_webpage",
    "Retriever",
]

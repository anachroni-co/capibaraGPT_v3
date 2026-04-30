"""Capibara Slim — retrieval-augmented generation subsystem."""
from rag.store import Document, VectorStore, RetrievedChunk
from rag.ingestion import ingest_text, ingest_file, ingest_directory
from rag.retriever import Retriever

__all__ = [
    "Document", "VectorStore", "RetrievedChunk",
    "ingest_text", "ingest_file", "ingest_directory",
    "Retriever",
]

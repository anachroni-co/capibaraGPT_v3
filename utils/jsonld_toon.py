"""
JSON-LD TOON Module - JSON-LD normalization for RAG contexts.

This module provides JSON-LD (Linked Data) normalization utilities for
structuring documents and chunks in RAG (Retrieval-Augmented Generation)
contexts. It implements the TOON (Text Object Ontology Notation) format.

Key Components:
    - JsonLdConfig: Configuration dataclass for JSON-LD settings
    - normalize_document_jsonld: Normalize documents to JSON-LD format
    - normalize_chunk_jsonld: Normalize text chunks to JSON-LD format
    - build_rag_jsonld_context: Build complete RAG context in JSON-LD

Author: Skydesk International Dev Team.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class JsonLdConfig:
    base_context: Dict[str, Any]
    document_type: str = "Document"
    chunk_type: str = "DocumentChunk"
    rag_context_type: str = "RAGContext"
    id_namespace: str = "rag"


def normalize_document_jsonld(
    doc: Mapping[str, Any],
    doc_id: str,
    config: JsonLdConfig,
) -> Dict[str, Any]:
    metadata = dict(doc.get("metadata", {}) or {})
    content = {
        "@id": doc_id,
        "@type": config.document_type,
        "text": doc.get("text", ""),
        "metadata": metadata,
    }
    if "timestamp" in doc:
        content["timestamp"] = doc["timestamp"]
    return content


def normalize_chunk_jsonld(
    chunk: Mapping[str, Any],
    chunk_id: str,
    parent_id: str,
    config: JsonLdConfig,
) -> Dict[str, Any]:
    metadata = dict(chunk.get("metadata", {}) or {})
    content = {
        "@id": chunk_id,
        "@type": config.chunk_type,
        "text": chunk.get("text", ""),
        "metadata": metadata,
        "parent": {"@id": parent_id},
    }
    if "timestamp" in chunk:
        content["timestamp"] = chunk["timestamp"]
    return content


def build_rag_jsonld_context(
    query: str,
    context_text: str,
    documents: Iterable[Mapping[str, Any]],
    config: Optional[JsonLdConfig] = None,
) -> Dict[str, Any]:
    config = config or JsonLdConfig(base_context={})
    doc_list: List[Dict[str, Any]] = []
    chunk_list: List[Dict[str, Any]] = []

    for index, doc in enumerate(documents):
        doc_id = f"{config.id_namespace}:doc:{index}"
        doc_jsonld = normalize_document_jsonld(doc, doc_id, config)
        doc_list.append(doc_jsonld)
        for chunk_index, chunk in enumerate(doc.get("chunks", []) or []):
            chunk_id = f"{config.id_namespace}:chunk:{index}:{chunk_index}"
            chunk_list.append(normalize_chunk_jsonld(chunk, chunk_id, doc_id, config))

    fingerprint = sha1(f"{query}|{len(doc_list)}|{len(chunk_list)}".encode("utf-8")).hexdigest()[:12]
    return {
        "@context": config.base_context,
        "@id": f"{config.id_namespace}:context:{fingerprint}",
        "@type": config.rag_context_type,
        "query": query,
        "context": context_text,
        "documents": doc_list,
        "chunks": chunk_list,
    }


def _is_tabular_array(items: List[Any]) -> bool:
    if not items:
        return False
    if not all(isinstance(item, Mapping) for item in items):
        return False
    keys = list(items[0].keys())
    if any(list(item.keys()) != keys for item in items):
        return False
    for item in items:
        if any(isinstance(value, (Mapping, list)) for value in item.values()):
            return False
    return True


def _sorted_keys(obj: Mapping[str, Any]) -> List[str]:
    keys = list(obj.keys())
    pinned = [key for key in ("@id", "@type") if key in keys]
    remainder = sorted(key for key in keys if key not in pinned)
    return pinned + remainder


def jsonld_to_toon(data: Any, indent: int = 0, field_name: Optional[str] = None) -> str:
    pad = "  " * indent
    if isinstance(data, Mapping):
        lines: List[str] = []
        if field_name is not None:
            lines.append(f"{pad}{field_name}:")
            pad = "  " * (indent + 1)
        for key in _sorted_keys(data):
            value = data[key]
            if isinstance(value, Mapping):
                lines.append(f"{pad}{key}:")
                lines.append(jsonld_to_toon(value, indent + 1))
            elif isinstance(value, list):
                lines.append(_format_array(key, value, indent))
            else:
                lines.append(f"{pad}{key}: {value}")
        return "\n".join(lines)
    if isinstance(data, list):
        return _format_array(field_name or "items", data, indent)
    return f"{pad}{data}"


def _format_array(field_name: str, value: List[Any], indent: int) -> str:
    pad = "  " * indent
    if _is_tabular_array(value):
        fields = list(value[0].keys())
        header = f"{pad}{field_name}[{len(value)}]{{{','.join(fields)}}}:"
        rows = [
            f"{pad}  {','.join(str(item.get(field, '')) for field in fields)}" for item in value
        ]
        return "\n".join([header] + rows)
    inline_items = ", ".join(str(item) for item in value)
    return f"{pad}{field_name}: [{inline_items}]"

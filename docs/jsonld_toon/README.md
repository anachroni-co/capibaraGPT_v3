# JSON-LD + TOON Integration

Define JSON-LD as the canonical format for persistent/structured data and TOON as the compact LLM-facing view. This preserves semantic traceability (JSON-LD) while improving token efficiency (TOON).

## Key features
- Canonical JSON-LD entities with `@context`, `@id`, `@type`.
- Deterministic and lossless TOON serialization.
- Clear layer separation between storage, retrieval, and prompt rendering.

## Flow
1. Ingestion and normalization to JSON-LD.
2. Internal RAG retrieval/filter/merge on JSON-LD.
3. TOON serialization for LLM prompts.

## Initial implementation
- Utility module for JSON-LD normalization and TOON serialization.
- Integration entry points:
- `core/pipelines/advanced_rag_pipeline.py`
- `core/pipelines/rag_pipeline.py` (`use_toon` option)
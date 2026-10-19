# JSON-LD + TOON Assessment

## Summary
The current RAG pipelines primarily use flat text and metadata structures. Adopting JSON-LD as the canonical representation and TOON as the LLM-facing compact view is architecturally viable, but requires dedicated normalization and serialization layers.

## Current state
- Documents/chunks are stored in flat structures (`text`, `metadata`, `chunks`, `timestamp`).
- Retrieval and prompt composition are text-centric.
- No canonical semantic contract is enforced yet (`@id`, `@type`, `@context`).

## Recommendation
Proceed in phases:
1. Define the JSON-LD schema contract and validators.
2. Add JSON-LD normalization at ingestion.
3. Add deterministic TOON serialization.
4. Measure token savings and latency before/after.

## Risks
- Added pipeline complexity.
- Serialization overhead if not optimized.
- Migration effort for legacy records.

## Success criteria
- Consistent semantic schema across persisted context.
- Lossless TOON round-trip for selected entities.
- Measurable prompt token reduction with acceptable latency overhead.
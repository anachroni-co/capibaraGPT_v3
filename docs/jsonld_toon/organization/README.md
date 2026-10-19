# Data Organization and Expected Savings (JSON-LD + TOON)

## Data organization
- Canonical JSON-LD document with minimum fields: `@context`, `@id`, `@type`.
- Chunks represented as linked entities (`parent` relationship).
- Deterministic TOON output with stable ordering.

## Suggested metrics
- Token count before vs after TOON.
- Serialization time (`JSON-LD -> TOON`) by context size.
- End-to-end latency (retrieval + serialization + inference).
- Round-trip integrity (TOON -> JSON-LD reconstruction).
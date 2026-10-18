"""
Inference optimizations for CapibaraGPT.

Includes:
- Speculative Decoding: 2-3x speedup using draft model verification
- (Future) KV-Cache Compression
- (Future) Continuous Batching
"""

from .speculative_decoding import (
    SpeculativeConfig,
    SpeculativeStats,
    SpeculativeDecoder,
    DraftModel,
    DraftModelType,
    MedusaHead,
    create_speculative_config,
    create_draft_model,
)

__all__ = [
    # Speculative Decoding
    'SpeculativeConfig',
    'SpeculativeStats',
    'SpeculativeDecoder',
    'DraftModel',
    'DraftModelType',
    'MedusaHead',
    'create_speculative_config',
    'create_draft_model',
]

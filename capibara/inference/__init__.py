"""
Inference optimizations for CapibaraGPT.

Includes:
- Speculative Decoding: 2-3x speedup using draft model verification
- Watermarking: Cryptographic proof of content origin (AI Act compliance)
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

from .watermarking import (
    WatermarkConfig,
    WatermarkType,
    WatermarkPayload,
    DetectionResult,
    DetectionOutput,
    GreenRedListWatermark,
    MultiBitWatermark,
    WatermarkProcessor,
    create_watermark_config,
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
    # Watermarking
    'WatermarkConfig',
    'WatermarkType',
    'WatermarkPayload',
    'DetectionResult',
    'DetectionOutput',
    'GreenRedListWatermark',
    'MultiBitWatermark',
    'WatermarkProcessor',
    'create_watermark_config',
]

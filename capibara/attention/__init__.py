"""
Attention mechanisms for CapibaraGPT.

Includes:
- RingX: Ring Attention for HPC long-context training
- Standard attention variants
"""

from .ring_attention import (
    RingAttentionConfig,
    RingAttention,
    RingCommunicator,
    RingAttentionTrainer,
    CommunicationBackend,
    create_ring_attention_config,
)

__all__ = [
    'RingAttentionConfig',
    'RingAttention',
    'RingCommunicator',
    'RingAttentionTrainer',
    'CommunicationBackend',
    'create_ring_attention_config',
]

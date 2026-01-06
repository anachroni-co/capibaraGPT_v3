"""
Memory subsystem for CapibaraGPT

This package implements advanced memory management systems including:
- Continuum Memory System (multi-scale temporal memory from Nested Learning)
- Memory consolidation and cross-temporal attention
- Different update frequencies for different memory components
"""

from capibara.core.memory.continuum_memory import (
    ContinuumMemorySystem,
    ContinuumMemoryConfig,
    MemoryBank,
    MemoryBankConfig,
    MemoryEntry,
    CrossTemporalAttention,
    create_continuum_memory,
    get_global_continuum_memory,
)

__all__ = [
    'ContinuumMemorySystem',
    'ContinuumMemoryConfig',
    'MemoryBank',
    'MemoryBankConfig',
    'MemoryEntry',
    'CrossTemporalAttention',
    'create_continuum_memory',
    'get_global_continuum_memory',
]

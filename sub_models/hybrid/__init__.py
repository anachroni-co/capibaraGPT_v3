"""
Hybrid Attention Module for CapibaraGPT-v2.

Módulo híbrido inteligente que selecciona automáticamente entre:
- Transformer (O(n²)) para secuencias cortas (mejor precisión)
- Mamba (O(n)) para secuencias largas (mejor eficiencia)
"""

from .hybrid_attention_module import HybridAttentionModule, HybridConfig

__all__ = ['HybridAttentionModule', 'HybridConfig']

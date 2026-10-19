"""
Mamba (Selective State Space Model) Module for CapibaraGPT v3.

Proporciona procesamiento con complejidad O(n) para secuencias largas,
en contraste con la complejidad O(n²) de Transformers tradicionales.
"""

from .mamba_module import MambaModule, MambaConfig

__all__ = ['MambaModule', 'MambaConfig']

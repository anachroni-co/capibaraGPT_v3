"""
CapibaraGPT-v2 Mathematics Datasets
Datasets matemáticos and algorítmicos
"""

try:
    from . import math_datasets
    MATH_DATASETS_AVAILABLE = True
except ImportError:
    MATH_DATASETS_AVAILABLE = False
    math_datasets = None

__all__ = []

if MATH_DATASETS_AVAILABLE:
    __all__.append('math_datasets')
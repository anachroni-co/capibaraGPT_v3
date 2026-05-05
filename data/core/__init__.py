"""
Core components for data handling in CapibaraGPT v3.
"""

from .dataset import Dataset
from .data_loader import DataLoader
from .data_processing import DataProcessor
from .multi_dataset_loader import MultiDatasetLoader
from .unified_data_pipeline import UnifiedDataPipeline

try:
    from .jax_data_processing import JaxDataProcessor
    _JAX_DATA_AVAILABLE = True
except ImportError:
    _JAX_DATA_AVAILABLE = False
    JaxDataProcessor = None  # type: ignore[assignment,misc]

__all__ = [
    'Dataset',
    'DataLoader',
    'DataProcessor',
    'JaxDataProcessor',
    'UnifiedDataPipeline',
    'MultiDatasetLoader',
]

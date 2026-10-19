"""
Spanish Government Datasets for CapibaraGPT v3

Access to Spanish government open data including BOE, BORME, and regional data.
"""

from .spanish_government_datasets import SpanishGovernmentDatasets
from .boe_datasets import BOEDatasets
from .regional_spain_datasets import RegionalSpainDatasets

__all__ = [
    'SpanishGovernmentDatasets',
    'BOEDatasets',
    'RegionalSpainDatasets',
]

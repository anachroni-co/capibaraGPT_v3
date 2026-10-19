"""
Specialized Research Datasets Module for CapibaraGPT v3

This module provides access to specialized research datasets from multiple domains,
including archaeology, computer science bibliography, and other academic fields.

Key Features:
- Archaeological datasets and digital heritage
- Computer science bibliographic data (DBLP)
- Cross-disciplinary research opportunities
- Bibliometric and scientometric analysis
- Historical and temporal data analysis
"""

from .archaeology_datasets import get_archaeology_datasets, ArchaeologyDatasets
from .dblp_computer_science_datasets import get_dblp_computer_science_datasets, DBLPComputerScienceDatasets

__all__ = [
    "get_archaeology_datasets",
    "ArchaeologyDatasets",
    "get_dblp_computer_science_datasets",
    "DBLPComputerScienceDatasets"
]

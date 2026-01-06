"""Data Management Module for CapibaraGPT.

This module provides comprehensive data management functionality including dataset
loading, preprocessing, registry management, and data pipeline utilities for training
and inference workflows.

See Also:
    - capibara.data.loaders: Data loading utilities
    - capibara.data.tools: Dataset validation and setup tools
    - capibara.training.data_preprocessing: Advanced preprocessing
"""

import os
import sys
from pathlib import Path

def get_project_root():
    """Get the root path of the project."""
    return Path(__file__).parent.parent

# Version information
__version__ = "1.0.0"
__author__ = "CapibaraGPT Team"

# Module exports
__all__ = []

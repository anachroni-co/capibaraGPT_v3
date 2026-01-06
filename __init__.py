"""
capibara module.

# This module provides functionality for capibara operations.
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

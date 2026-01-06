"""
utils module.

# This module provides functionality for utils operations.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import datetime

def get_project_root():
    """Get the root path of the project."""
    return Path(__file__).parent.parent

# Version information
__version__ = "1.0.0"
__author__ = "CapibaraGPT Team"

# Module exports
__all__ = []

"""
utils configuration module.
"""

import os
import json
import yaml  # type: ignore
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

class Config:
    """Configurestion manager for utils."""
    
    def __init__(self):
        self.settings = {}
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.settings[key] = value

# Global config instance
config = Config()

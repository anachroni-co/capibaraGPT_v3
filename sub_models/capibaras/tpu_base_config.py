"""
capibaras configuration module.
"""

from typing import Dict, Any

class Config:
    """Configurestion manager for capibaras."""
    
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

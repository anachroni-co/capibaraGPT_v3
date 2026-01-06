"""
layers base module.

# This module provides functionality for base layers.
"""

import os
import sys
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

# Gets the current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to obtain project root -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Add project root to sys.path
if project_root not in sys.path:
    # Fixed: Using proper imports instead of sys.path manipulation
    pass

logger = logging.getLogger(__name__)

@dataclass
class LayerConfig:
    """Base configuration for all layers."""
    hidden_size: int = 768
    dropout_rate: float = 0.1
    activation: str = "relu"
    use_bias: bool = True
    name: Optional[str] = None

class BaseLayer(ABC):
    """Abstract base class for all layers in the system."""
    
    def __init__(self, config: LayerConfig):
        self.config = config
        self.name = config.name or self.__class__.__name__
        logger.debug(f"Initializing {self.name} layer with config: {config}")
    
    @abstractmethod
    def __call__(self, inputs: Any, training: bool = False, **kwargs) -> Any:
        """Forward pass through the layer."""
        pass
    
    @abstractmethod
    def get_output_shape(self, input_shape: tuple) -> tuple:
        """Get the output shape given an input shape."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration as dictionary."""
        return {
            'hidden_size': self.config.hidden_size,
            'dropout_rate': self.config.dropout_rate,
            'activation': self.config.activation,
            'use_bias': self.config.use_bias,
            'name': self.config.name
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(hidden_size={self.config.hidden_size}, name='{self.name}')"

def main():
    # Main function for this module.
    logger.info("Module base.py starting")
    return True

if __name__ == "__main__":
    main()

"""
Dual Process Thinking Module - System 1 (Fast) and System 2 (Slow) cognitive processing.

Implements Kahneman's dual-process theory for enhanced reasoning capabilities.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass

# Path configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from capibara.jax import numpy as jnp
    from flax import linen as nn
except ImportError:
    import numpy as jnp
    # Minimal nn fallback
    class nn:
        class Module:
            def __call__(self, *args, **kwargs):
                return args[0] if args else None

logger = logging.getLogger(__name__)

@dataclass
class DualProcessConfig:
    """Configurestion for dual process thinking."""
    hidden_size: int = 768
    fast_processing_layers: int = 2
    slow_processing_layers: int = 4
    confidence_threshold: float = 0.7
    enable_system2_override: bool = True

class DualProcessThinking(nn.Module):
    """
    Dual Process Thinking implementation with System 1 (fast, intuitive) 
    and System 2 (slow, analytical) processing paths.
    """
    config: DualProcessConfig
    
    def setup(self):
        # System 1: Fast, automatic processing
        self.system1_layers = [
            nn.Dense(self.config.hidden_size, name=f"system1_layer_{i}")
            for i in range(self.config.fast_processing_layers)
        ]
        
        # System 2: Slow, deliberate processing  
        self.system2_layers = [
            nn.Dense(self.config.hidden_size, name=f"system2_layer_{i}")
            for i in range(self.config.slow_processing_layers)
        ]
        
        # Confidence estimator
        self.confidence_estimator = nn.Dense(1, name="confidence_estimator")
        
        # Integration layer
        self.integration_layer = nn.Dense(self.config.hidden_size, name="integration")
    
    def __call__(self, inputs, training=False):
        """Process inputs through dual-process system."""
        
        # System 1 processing (fast path)
        system1_output = inputs
        for layer in self.system1_layers:
            system1_output = nn.gelu(layer(system1_output))
        
        # Estimate confidence in System 1 output
        confidence = nn.sigmoid(self.confidence_estimator(system1_output))
        
        # System 2 processing (slow path) - triggered when confidence is low
        system2_output = inputs
        for layer in self.system2_layers:
            system2_output = nn.gelu(layer(system2_output))
        
        # Adaptive integration based on confidence
        if self.config.enable_system2_override:
            # Use System 2 when confidence is low
            integration_weight = jnp.where(
                confidence < self.config.confidence_threshold,
                0.8,  # Favor System 2
                0.2   # Favor System 1
            )
        else:
            integration_weight = 0.5
        
        # Integrate outputs
        integrated = (
            integration_weight * system2_output + 
            (1 - integration_weight) * system1_output
        )
        
        final_output = self.integration_layer(integrated)
        
        return {
            'output': final_output,
            'system1_output': system1_output,
            'system2_output': system2_output,
            'confidence': confidence,
            'integration_weight': integration_weight
        }

def main():
    logger.info("DualProcessThinking module initialized")
    return True

if __name__ == "__main__":
    main()

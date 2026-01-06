"""
Interfaces for sub-models module.

Defines the base interfaces and contracts for sub-models in the CapibaraGPT system.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, NamedTuple
from enum import Enum

logger = logging.getLogger(__name__)

class PrecisionMode(Enum):
    """Precision modes for model operations."""
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"

class ExpertActivationMode(Enum):
    """Modes for expert activation."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    CONDITIONAL = "conditional"
    THRESHOLD_BASED = "threshold_based"

@dataclass
class ConfigTPU:
    """Configuration for TPU optimization."""
    use_tpu: bool = True
    precision_mode: PrecisionMode = PrecisionMode.BFLOAT16
    mesh_shape: tuple = (1, 1)
    batch_size: int = 32
    sequence_length: int = 512

@dataclass
class ExpertContext:
    """Context information for expert sub-models."""
    text: str
    task_hint: str = "general"
    constraints: Dict[str, Any] = field(default_factory=dict)
    kb_hooks: Dict[str, Any] = field(default_factory=dict)
    flags: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExpertResult:
    """Result from expert sub-model processing."""
    success: bool
    output: Any
    confidence: float
    processing_time: float
    expert_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

class ISubModelExpert(ABC):
    """Base interface for sub-model experts."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the expert name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Return the expert version."""
        pass
    
    @abstractmethod
    def supports(self, context: ExpertContext) -> bool:
        """Check if this expert supports the given context."""
        pass
    
    @abstractmethod
    async def process(self, context: ExpertContext) -> ExpertResult:
        """Process the input context and return results."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this expert provides."""
        pass
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the expert with configuration."""
        return True
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

class ISubModel(ABC):
    """Base interface for sub-models."""
    
    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """Make predictions with the sub-model."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get sub-model configuration."""
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """Load model from path."""
        pass

class IExperimentalModel(ABC):
    """Interface for experimental models."""
    
    @abstractmethod
    def experiment(self, data: Any) -> Any:
        """Run experimental model inference."""
        pass
    
    @abstractmethod
    def validate_experiment(self, results: Any) -> bool:
        """Validate experimental results."""
        pass

class ICounterfactualExpert(ISubModelExpert):
    """Interface for counterfactual analysis experts."""
    
    @abstractmethod
    def generate_hypotheses(self, context: ExpertContext, max_hypotheses: int = 6) -> List[Any]:
        """Generate counterfactual hypotheses."""
        pass
    
    @abstractmethod
    def evaluate_scenarios(self, context: ExpertContext, hypotheses: List[Any]) -> List[Any]:
        """Evaluate counterfactual scenarios."""
        pass

def main():
    """Main function for this module."""
    logger.info("Module isub_models.py starting")
    return True

if __name__ == "__main__":
    main()
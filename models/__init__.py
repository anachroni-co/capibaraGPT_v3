"""Model backends for capibaraGPT CPU inference."""
from .pretrained_backbone import (
    BaseBackbone,
    LlamaCppBackbone,
    HuggingFaceBackbone,
    TransformerNumpyBackbone,
    auto_backbone,
)
__all__ = [
    "BaseBackbone", "LlamaCppBackbone", "HuggingFaceBackbone",
    "TransformerNumpyBackbone", "auto_backbone",
]

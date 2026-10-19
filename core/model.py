"""
Model module alias for CapibaraGPT core.

Exposes the minimal model implementation from core/_model.py so that
imports like `capibara.core.model` resolve correctly.
"""

from ._model import ModelCore, create_model

__all__ = ["ModelCore", "create_model"]

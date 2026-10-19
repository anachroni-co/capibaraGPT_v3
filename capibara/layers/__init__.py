"""
CapibaraGPT v3 layers package shim.

This module aliases the top-level `layers` package so that both
`layers.*` and `capibara.layers.*` imports work.
"""

from __future__ import annotations

import importlib
import sys


_layers = importlib.import_module("layers")

# Expose as capibara.layers
sys.modules[__name__] = _layers

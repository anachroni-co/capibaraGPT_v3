"""
CapibaraGPT v3 pipeline package shim.

This module aliases the top-level `pipeline` package so that both
`pipeline.*` and `capibara.pipeline.*` imports work.
"""

from __future__ import annotations

import importlib
import sys


_pipeline = importlib.import_module("pipeline")

# Expose as capibara.pipeline
sys.modules[__name__] = _pipeline

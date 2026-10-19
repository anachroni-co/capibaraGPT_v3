"""
CapibaraGPT v3 modules package shim.

This module aliases the top-level `modules` package so that both
`modules.*` and `capibara.modules.*` imports work.
"""

from __future__ import annotations

import importlib
import sys


_modules = importlib.import_module("modules")

# Expose as capibara.modules
sys.modules[__name__] = _modules

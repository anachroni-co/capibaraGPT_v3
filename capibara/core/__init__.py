"""
CapibaraGPT v3 core package shim.

This module aliases the top-level `core` package so that both
`core.*` and `capibara.core.*` imports work.
"""

from __future__ import annotations

import importlib
import sys


_core = importlib.import_module("core")

# Expose as capibara.core
sys.modules[__name__] = _core

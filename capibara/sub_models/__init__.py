"""
CapibaraGPT v3 sub_models package shim.

This module aliases the top-level `sub_models` package so that both
`sub_models.*` and `capibara.sub_models.*` imports work.
"""

from __future__ import annotations

import importlib
import sys


_sub_models = importlib.import_module("sub_models")

# Expose as capibara.sub_models
sys.modules[__name__] = _sub_models

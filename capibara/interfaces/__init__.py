"""
CapibaraGPT v3 interfaces package shim.

This module aliases the top-level `interfaces` package so that both
`interfaces.*` and `capibara.interfaces.*` imports work.
"""

from __future__ import annotations

import importlib
import sys


_interfaces = importlib.import_module("interfaces")

# Expose as capibara.interfaces
sys.modules[__name__] = _interfaces

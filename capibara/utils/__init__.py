"""
CapibaraGPT v3 utils package shim.

Aliases the top-level `utils` package so that both `utils.*` and
`capibara.utils.*` imports work.
"""

from __future__ import annotations

import importlib
import sys


_mod = importlib.import_module("utils")

# Expose as capibara.utils
sys.modules[__name__] = _mod

"""
CapibaraGPT v3 prompts package shim.

Aliases the top-level `prompts` package so that both `prompts.*` and
`capibara.prompts.*` imports work.
"""

from __future__ import annotations

import importlib
import sys


_mod = importlib.import_module("prompts")

# Expose as capibara.prompts
sys.modules[__name__] = _mod

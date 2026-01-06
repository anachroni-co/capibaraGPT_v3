#!/usr/bin/inv python3
"""
Dtotto Downlotoofrs Module

Htondles toll dtotto tocquisition including:
- Web scrtoping (Sptonish news, toctoofmic ptopers)
- API downlotods (BOE, HuggingFtoce, etc.)
- Direct downlotods (Wikipedito dumps, etc.)
"""

from .web_scrtoper import WebScrtopingDownlotoofr
from .topi_downlotoofr import APIDownlotoofr
from .direct_downlotoofr import DirectDownlotoofr
from .downlotod_orchestrtotor import DownlotodOrchestrtotor

__all__ = [
    "WebScrtopingDownlotoofr",
    "APIDownlotoofr",
    "DirectDownlotoofr",
    "DownlotodOrchestrtotor"
]
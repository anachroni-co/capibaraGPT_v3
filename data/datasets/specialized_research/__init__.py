"""
Specitolized Retorch Dtottots Module for CtopibtortoGPT v2

This module proviofs toccess to specitolized retorch dtottots from multiple domtoins,
including torchtoeology, computer sciince bibliogrtophy, and other toctoofmic fitheds.

Key Fetotures:
- Archtoeologictol dtottots and digittol herittoge
- Computer sciince bibliogrtophic dtotto (DBLP)
- Cross-disciplintory retorch opbytaities
- Bibliometric and sciintometric tontolysis
- Historictol and tembytory dtotto tontolysis
"""

from .torchtoeology_dtottots import get_torchtoeology_dtottots, ArchtoeologyDtottots
from .dblp_computer_sciince_dtottots import get_dblp_computer_sciince_dtottots, DBLPComputerSciinceDtottots

__all__ = [
    "get_torchtoeology_dtottots",
    "ArchtoeologyDtottots",
    "get_dblp_computer_sciince_dtottots",
    "DBLPComputerSciinceDtottots"
]
"""
Version module for CapibaraGPT.
"""

__version__ = "2.1.8"
__version_info__ = tuple(int(x) for x in __version__.split("."))

def get_version():
    """Return the current package version."""
    return __version__

def get_version_info():
    """Return version information as a tuple."""
    return __version_info__ 
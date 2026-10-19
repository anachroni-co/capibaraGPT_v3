"""
Submodels Import Tests - Unit tests for submodel module imports.

This module provides tests for verifying that submodel modules can be
imported and expose the expected availability flags.

Author: Skydesk International Dev Team.
"""

import sub_models


def test_submodels_imports():
    # Module should import and expose availability flags.
    assert hasattr(sub_models, "SSM_AVAILABLE")
    assert hasattr(sub_models, "BYTE_AVAILABLE")
    assert hasattr(sub_models, "CSA_AVAILABLE")
    assert hasattr(sub_models, "DIALOG_AVAILABLE")
    assert hasattr(sub_models, "REASONING_AVAILABLE")
    assert hasattr(sub_models, "ORCHESTRATOR_AVAILABLE")

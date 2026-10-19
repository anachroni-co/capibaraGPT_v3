"""
Modules Import Tests - Unit tests for module import functionality.

This module provides tests for verifying that modules can be imported
correctly and that basic functionality works as expected.

Author: Skydesk International Dev Team.
"""

import pytest


@pytest.mark.unit
def test_modules_importable():
    import modules
    import capibara.modules as cap_modules

    assert modules is cap_modules
    status = modules.get_module_status()
    assert isinstance(status, dict)


@pytest.mark.unit
def test_hierarchical_reasoning_runs():
    from modules.hierarchical_reasoning import HierarchicalReasoning

    reasoning = HierarchicalReasoning()
    result = reasoning.process("Plan a solution with steps.")
    assert isinstance(result, dict)
    assert result.get("plan")


@pytest.mark.unit
def test_specialized_processors_cpu():
    from modules.specialized_processors import (
        create_processor_manager,
        create_default_processors,
    )

    manager = create_processor_manager()
    create_default_processors(manager)
    result = manager.process("text_analyzer", "Hello world.")
    assert "error" not in result

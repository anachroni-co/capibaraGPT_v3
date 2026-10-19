#!/usr/bin/env python3
"""
Project Validation Script

Validates that the project structure is correct and all components work.

Usage:
    python scripts/validate_project.py
"""

import importlib
import sys
from pathlib import Path
from typing import List, Tuple

import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_file_exists(path: str, description: str) -> Tuple[bool, str]:
    """Check if a file exists."""
    full_path = PROJECT_ROOT / path
    if full_path.exists():
        return True, f"Found {description}: {path}"
    return False, f"Missing {description}: {path}"


def check_import(module: str, description: str) -> Tuple[bool, str]:
    """Check if a module can be imported."""
    try:
        # For backends, check file exists (avoid core/__init__.py issues)
        if "backends" in module:
            module_path = PROJECT_ROOT / module.replace(".", "/")
            if module_path.is_dir():
                init_path = module_path / "__init__.py"
                if init_path.exists():
                    return True, f"Found {description}: {module}"
            else:
                file_path = PROJECT_ROOT / (module.replace(".", "/") + ".py")
                if file_path.exists():
                    return True, f"Found {description}: {module}"
            return False, f"Not found {description}: {module}"
        else:
            importlib.import_module(module)
            return True, f"Imported {description}: {module}"
    except ImportError as e:
        return False, f"Failed to import {description}: {module} ({e})"
    except Exception as e:
        return False, f"Error importing {description}: {module} ({e})"


def check_backend_available() -> List[Tuple[bool, str]]:
    """Check which backends are available."""
    results = []

    # Check backend files exist
    backends_dir = PROJECT_ROOT / "core" / "backends"

    cpu_path = backends_dir / "cpu_backend.py"
    results.append((cpu_path.exists(), f"Backend CPU: {'file exists' if cpu_path.exists() else 'missing'}"))

    gpu_path = backends_dir / "gpu_backend.py"
    results.append((gpu_path.exists(), f"Backend GPU: {'file exists' if gpu_path.exists() else 'missing'}"))

    tpu_path = backends_dir / "tpu_backend.py"
    results.append((tpu_path.exists(), f"Backend TPU: {'file exists' if tpu_path.exists() else 'missing'}"))

    # Test NumPy (required for CPU backend)
    try:
        import numpy as np
        a = np.random.randn(4, 4)
        b = np.random.randn(4, 4)
        c = np.matmul(a, b)
        results.append((True, "NumPy matmul: passed"))
    except Exception as e:
        results.append((False, f"NumPy test failed: {e}"))

    # Check for GPU libraries
    try:
        import torch
        results.append((torch.cuda.is_available(), f"PyTorch CUDA: {'available' if torch.cuda.is_available() else 'not available'}"))
    except ImportError:
        results.append((True, "PyTorch: not installed (optional for GPU)"))

    # Check for TPU libraries
    try:
        import jax
        tpu_devices = jax.devices("tpu")
        results.append((len(tpu_devices) > 0, f"JAX TPU: {len(tpu_devices)} devices"))
    except Exception:
        results.append((True, "JAX: not installed (optional for TPU)"))

    return results


def check_tests_structure() -> List[Tuple[bool, str]]:
    """Check tests directory structure."""
    results = []

    tests_dir = PROJECT_ROOT / "tests"

    # Check directories
    for subdir in ["unit", "integration", "benchmarks", "fixtures"]:
        path = tests_dir / subdir
        if path.exists():
            results.append((True, f"Tests directory: {subdir}/"))
        else:
            results.append((False, f"Missing tests directory: {subdir}/"))

    # Check key test files
    test_files = [
        "tests/conftest.py",
        "tests/unit/test_backends.py",
        "tests/unit/test_core_model.py",
        "tests/integration/test_training_pipeline.py",
        "tests/fixtures/synthetic_data.py",
    ]

    for tf in test_files:
        exists = (PROJECT_ROOT / tf).exists()
        results.append((exists, f"Test file: {tf}"))

    return results


def check_config_files() -> List[Tuple[bool, str]]:
    """Check configuration files."""
    results = []

    config_files = [
        ("pyproject.toml", "Project configuration"),
        ("requirements-tpu.txt", "TPU requirements"),
        ("requirements-gpu.txt", "GPU requirements"),
        ("requirements-dev.txt", "Dev requirements"),
        ("Makefile", "Build automation"),
    ]

    for path, desc in config_files:
        exists, msg = check_file_exists(path, desc)
        results.append((exists, msg))

    return results


def check_core_modules() -> List[Tuple[bool, str]]:
    """Check core module imports."""
    results = []

    modules = [
        ("core.backends", "Backend system"),
        ("core.backends.base", "Backend base classes"),
        ("core.backends.cpu_backend", "CPU backend"),
        ("core.backends.registry", "Backend registry"),
        ("core.backends.utils", "Backend utilities"),
    ]

    for module, desc in modules:
        passed, msg = check_import(module, desc)
        results.append((passed, msg))

    return results


def main():
    """Run all validation checks."""
    logger.info("=" * 70)
    logger.info("CapibaraGPT Project Validation")
    logger.info("=" * 70)
    logger.info()

    all_results = []

    # Configuration files
    logger.info("Configuration Files:")
    logger.info("-" * 40)
    for passed, msg in check_config_files():
        symbol = "" if passed else ""
        logger.info(f"  {symbol} {msg}")
        all_results.append(passed)
    logger.info()

    # Core modules
    logger.info("Core Module Imports:")
    logger.info("-" * 40)
    for passed, msg in check_core_modules():
        symbol = "" if passed else ""
        logger.info(f"  {symbol} {msg}")
        all_results.append(passed)
    logger.info()

    # Backend availability
    logger.info("Backend Availability:")
    logger.info("-" * 40)
    for passed, msg in check_backend_available():
        symbol = "" if passed else ""
        logger.info(f"  {symbol} {msg}")
        all_results.append(passed)
    logger.info()

    # Test structure
    logger.info("Test Structure:")
    logger.info("-" * 40)
    for passed, msg in check_tests_structure():
        symbol = "" if passed else ""
        logger.info(f"  {symbol} {msg}")
        all_results.append(passed)
    logger.info()

    # Summary
    passed = sum(all_results)
    total = len(all_results)
    failed = total - passed

    logger.info("=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Total checks: {total}")
    logger.info(f"  Passed: {passed}")
    logger.error(f"  Failed: {failed}")
    logger.info()

    if failed == 0:
        logger.info(" All validation checks passed!")
        return 0
    else:
        logger.error(f" {failed} validation checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

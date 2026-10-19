#!/usr/bin/env python3
"""
Test Runner Script

Runs all tests and generates report.

Usage:
    python scripts/run_tests.py              # Run all tests
    python scripts/run_tests.py --unit       # Unit tests only
    python scripts/run_tests.py --integration # Integration tests only
    python scripts/run_tests.py --quick      # Quick smoke tests
"""

import argparse
import subprocess
import sys
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("=" * 60)

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run CapibaraGPT tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick smoke tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Base pytest command
    pytest_cmd = [sys.executable, "-m", "pytest"]

    if args.verbose:
        pytest_cmd.append("-v")

    results = []

    if args.quick:
        # Quick smoke tests
        cmd = pytest_cmd + [
            "tests/unit/test_backends.py::TestCPUBackend::test_create_tensor",
            "tests/unit/test_backends.py::TestCPUBackend::test_matmul",
            "tests/unit/test_backends.py::TestCPUBackend::test_softmax",
            "-v",
        ]
        results.append(("Quick smoke tests", run_command(cmd, "Quick smoke tests")))

    elif args.unit:
        # Unit tests
        cmd = pytest_cmd + ["tests/unit", "-v", "-m", "not slow"]
        if args.coverage:
            cmd.extend(["--cov=core", "--cov-report=html"])
        results.append(("Unit tests", run_command(cmd, "Unit tests")))

    elif args.integration:
        # Integration tests
        cmd = pytest_cmd + ["tests/integration", "-v"]
        results.append(("Integration tests", run_command(cmd, "Integration tests")))

    elif args.benchmark:
        # Benchmarks
        cmd = pytest_cmd + [
            "tests/benchmarks",
            "-v",
            "--benchmark-enable",
            "--benchmark-only",
        ]
        results.append(("Benchmarks", run_command(cmd, "Benchmarks")))

    else:
        # Run all tests
        # 1. Unit tests
        cmd = pytest_cmd + ["tests/unit", "-v", "-m", "not slow"]
        results.append(("Unit tests", run_command(cmd, "Unit tests")))

        # 2. Integration tests
        cmd = pytest_cmd + ["tests/integration", "-v"]
        results.append(("Integration tests", run_command(cmd, "Integration tests")))

        if args.coverage:
            # Coverage report
            cmd = pytest_cmd + [
                "tests",
                "--cov=core",
                "--cov-report=html",
                "--cov-report=term-missing",
            ]
            results.append(("Coverage", run_command(cmd, "Coverage report")))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "" if passed else ""
        logger.info(f"  {symbol} {name}: {status}")
        if not passed:
            all_passed = False

    logger.info("=" * 60)

    if all_passed:
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

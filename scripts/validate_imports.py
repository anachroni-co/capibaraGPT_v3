#!/usr/bin/env python3
"""Validate that all Python modules can be compiled.

This script is used in CI to catch syntax errors in Python files.
Import validation is optional since it depends on the environment.

Usage:
    python scripts/validate_imports.py
"""

import sys
import py_compile
from pathlib import Path
from typing import List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def find_python_files(root: Path, exclude_patterns: List[str] = None) -> List[Path]:
    """Find all Python files in the project."""
    exclude_patterns = exclude_patterns or [
        '__pycache__', '.git', 'build', '.egg', 'venv', '.venv', 'node_modules'
    ]

    python_files = []
    for path in root.rglob('*.py'):
        if not any(pattern in str(path) for pattern in exclude_patterns):
            python_files.append(path)

    return python_files


def validate_syntax(files: List[Path]) -> Tuple[int, List[str]]:
    """Validate Python syntax for all files."""
    errors = []

    for filepath in files:
        try:
            py_compile.compile(str(filepath), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(f"Syntax error in {filepath.relative_to(PROJECT_ROOT)}: {e.msg}")

    return len(files), errors


def main() -> int:
    """Main entry point."""
    print("=" * 60)
    print("Validating Python syntax...")
    print("=" * 60)

    # Find all Python files
    python_files = find_python_files(PROJECT_ROOT)
    print(f"\nFound {len(python_files)} Python files")

    # Validate syntax
    print("\nChecking syntax...")
    total, syntax_errors = validate_syntax(python_files)

    if syntax_errors:
        print(f"\n Found {len(syntax_errors)} syntax errors:\n")
        for error in syntax_errors:
            print(f"  - {error}")
        print("\n" + "=" * 60)
        print(f" FAILED: {len(syntax_errors)} syntax errors found")
        return 1
    else:
        print(f" All {total} files have valid syntax")
        print("\n" + "=" * 60)
        print(" PASSED: All validations successful")
        return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Script to check documentation coverage in Python files.

Author: Skydesk International Dev Team.

This script scans Python files and reports:
- Files missing module-level docstrings
- Classes missing docstrings
- Functions missing docstrings
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DocIssue:
    """Represents a documentation issue."""
    file_path: str
    issue_type: str  # 'module', 'class', 'function'
    name: str
    line: int


def has_module_docstring(source: str) -> bool:
    """Check if source code has a module-level docstring."""
    # Remove leading comments and whitespace
    lines = source.split('\n')
    first_code_line = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            first_code_line = i
            break

    # Check if first non-comment line is a docstring
    remaining = '\n'.join(lines[first_code_line:])
    remaining = remaining.lstrip()

    return remaining.startswith('"""') or remaining.startswith("'''")


def extract_doc_issues(file_path: Path, source: str) -> list[DocIssue]:
    """Extract all documentation issues from a Python file."""
    issues = []

    # Check module docstring
    if not has_module_docstring(source):
        issues.append(DocIssue(
            file_path=str(file_path),
            issue_type='module',
            name=file_path.stem,
            line=1
        ))

    # Parse AST for classes and functions
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return issues

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check if class has docstring
            docstring = ast.get_docstring(node)
            if not docstring:
                issues.append(DocIssue(
                    file_path=str(file_path),
                    issue_type='class',
                    name=node.name,
                    line=node.lineno
                ))

        elif isinstance(node, ast.FunctionDef):
            # Skip private methods (starting with _)
            if node.name.startswith('_') and not node.name.startswith('__'):
                continue
            # Skip dunder methods
            if node.name.startswith('__') and node.name.endswith('__'):
                continue

            docstring = ast.get_docstring(node)
            if not docstring:
                issues.append(DocIssue(
                    file_path=str(file_path),
                    issue_type='function',
                    name=node.name,
                    line=node.lineno
                ))

    return issues


def scan_directory(directory: Path, exclude_dirs: set[str] = None) -> list[DocIssue]:
    """Scan a directory for Python files with documentation issues."""
    if exclude_dirs is None:
        exclude_dirs = {'.git', '.venv', 'venv', '__pycache__', 'node_modules', '.eggs', 'build', 'dist'}

    all_issues = []

    for py_file in directory.rglob('*.py'):
        # Skip excluded directories
        if any(part in exclude_dirs for part in py_file.parts):
            continue

        try:
            source = py_file.read_text(encoding='utf-8')
            issues = extract_doc_issues(py_file, source)
            all_issues.extend(issues)
        except Exception as e:
            print(f"Error reading {py_file}: {e}")

    return all_issues


def generate_report(issues: list[DocIssue], repo_root: Path) -> str:
    """Generate a markdown report of documentation issues."""
    # Group by directory
    by_dir: dict[str, list[DocIssue]] = {}

    for issue in issues:
        rel_path = Path(issue.file_path).relative_to(repo_root)
        dir_name = str(rel_path.parent).split('/')[0] if '/' in str(rel_path.parent) else str(rel_path.parent)

        if dir_name not in by_dir:
            by_dir[dir_name] = []
        by_dir[dir_name].append(issue)

    # Count by type
    by_type = {'module': 0, 'class': 0, 'function': 0}
    for issue in issues:
        by_type[issue.issue_type] += 1

    # Build report
    lines = [
        "# Documentation Coverage Report",
        "",
        f"**Total Issues**: {len(issues)}",
        "",
        "## Summary by Type",
        "",
        f"| Type | Count |",
        f"|------|-------|",
        f"| Module docstrings missing | {by_type['module']} |",
        f"| Class docstrings missing | {by_type['class']} |",
        f"| Function docstrings missing | {by_type['function']} |",
        "",
        "## Issues by Directory",
        "",
    ]

    for dir_name in sorted(by_dir.keys()):
        dir_issues = by_dir[dir_name]
        module_issues = [i for i in dir_issues if i.issue_type == 'module']

        lines.append(f"### {dir_name}/")
        lines.append("")
        lines.append(f"- Total issues: {len(dir_issues)}")
        lines.append(f"- Files without module docstring: {len(module_issues)}")
        lines.append("")

        if module_issues:
            lines.append("**Files missing module docstring:**")
            lines.append("")
            for issue in module_issues[:10]:  # Limit to 10
                rel_path = Path(issue.file_path).relative_to(repo_root)
                lines.append(f"- `{rel_path}`")
            if len(module_issues) > 10:
                lines.append(f"- ... and {len(module_issues) - 10} more")
            lines.append("")

    return '\n'.join(lines)


def print_summary(issues: list[DocIssue], repo_root: Path) -> None:
    """Print a summary to console."""
    # Group by directory
    by_dir: dict[str, dict[str, int]] = {}

    for issue in issues:
        rel_path = Path(issue.file_path).relative_to(repo_root)
        dir_name = str(rel_path.parent).split('\\')[0] if '\\' in str(rel_path.parent) else str(rel_path.parent)

        if dir_name not in by_dir:
            by_dir[dir_name] = {'module': 0, 'class': 0, 'function': 0}
        by_dir[dir_name][issue.issue_type] += 1

    print("\n" + "=" * 60)
    print("DOCUMENTATION COVERAGE REPORT")
    print("=" * 60)

    # Sort by module issues (worst first)
    sorted_dirs = sorted(by_dir.items(), key=lambda x: x[1]['module'], reverse=True)

    print(f"\n{'Directory':<20} {'Module':<10} {'Class':<10} {'Function':<10}")
    print("-" * 60)

    for dir_name, counts in sorted_dirs[:20]:
        print(f"{dir_name:<20} {counts['module']:<10} {counts['class']:<10} {counts['function']:<10}")

    total_modules = sum(c['module'] for c in by_dir.values())
    total_classes = sum(c['class'] for c in by_dir.values())
    total_functions = sum(c['function'] for c in by_dir.values())

    print("-" * 60)
    print(f"{'TOTAL':<20} {total_modules:<10} {total_classes:<10} {total_functions:<10}")
    print("=" * 60)


def main() -> None:
    """Main entry point."""
    repo_root = Path(__file__).parent.parent

    print(f"Scanning {repo_root} for documentation issues...")

    issues = scan_directory(repo_root)

    print_summary(issues, repo_root)

    # Generate markdown report
    report = generate_report(issues, repo_root)
    report_path = repo_root / "tmp" / "docs_coverage_report.md"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(report, encoding='utf-8')

    print(f"\nDetailed report saved to: {report_path}")

    # Return exit code based on issues found
    return 0


if __name__ == "__main__":
    exit(main())

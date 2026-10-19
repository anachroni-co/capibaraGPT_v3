#!/usr/bin/env python3
"""
Create GitHub issues from pending items in TODOs_PRIORITIZED.md.

Usage:
  python scripts/create_all_todo_issues.py --repo anachroni-co/capibaraGPT_v3 --dry-run
  python scripts/create_all_todo_issues.py --repo anachroni-co/capibaraGPT_v3 --max-create 50
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class TodoIssue:
    title: str
    path: str
    priority: str
    source: str
    raw_line: str
    key: str

    @property
    def body(self) -> str:
        lines = []
        if self.path:
            lines.append(f"- Path: `{self.path}`")
        lines.append(f"- Priority: `{self.priority}`")
        if self.source:
            lines.append(f"- Source: `{self.source}`")
        lines.append(f"- TODO_KEY: `{self.key}`")
        lines.append("")
        lines.append("Extracted from `TODOs_PRIORITIZED.md`.")
        lines.append("")
        lines.append("Raw TODO line:")
        lines.append(f"> {self.raw_line}")
        return "\n".join(lines)


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_path(value: str) -> str:
    path = value.strip().replace("\\", "/").lower()
    path = re.sub(r"^[a-z]:", "", path)
    path = re.sub(r"/+", "/", path)
    return path


def parse_todos(todos_path: Path) -> list[TodoIssue]:
    lines = todos_path.read_text(encoding="utf-8", errors="replace").splitlines()
    priority = "Unspecified"
    parsed: list[TodoIssue] = []

    for line in lines:
        if line.startswith("## "):
            match = re.match(r"##\s+([A-Za-z]+)", line)
            if match:
                priority = match.group(1)
            continue

        if not line.startswith("- [ ] "):
            continue

        source = ""
        main = line
        if "| source:" in line:
            main, source = line.split("| source:", 1)
            source = source.strip()

        path = ""
        path_match = re.search(r"\s-\s`([^`]+)`\s*$", main)
        if path_match:
            path = path_match.group(1).strip()
            main = main[: path_match.start()].rstrip()

        item_match = re.match(r"- \[ \]\s+\d+:\s*(.*)$", main)
        if not item_match:
            continue

        desc = item_match.group(1).strip()
        prev = None
        while desc != prev:
            prev = desc
            desc = re.sub(r"^- \[ \]\s*", "", desc).strip()
            desc = re.sub(r"^\d+:\s*", "", desc).strip()

        if not desc:
            continue

        title = normalize_whitespace(desc)
        if len(title) > 240:
            title = title[:237] + "..."

        key_material = f"{title}|{path}|{priority}|{source}"
        key = hashlib.sha1(key_material.encode("utf-8")).hexdigest()[:12]
        parsed.append(
            TodoIssue(
                title=title,
                path=path,
                priority=priority,
                source=source,
                raw_line=normalize_whitespace(line),
                key=key,
            )
        )

    return parsed


def dedupe_todos(items: Iterable[TodoIssue]) -> list[TodoIssue]:
    seen: set[tuple[str, str]] = set()
    unique: list[TodoIssue] = []
    for item in items:
        dedupe_key = (item.title.casefold(), item.path.casefold())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        unique.append(item)
    return unique


def run_gh(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True)


def get_open_issues(repo: str) -> list[dict]:
    cp = run_gh(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            "1000",
            "--json",
            "number,title,body",
        ]
    )
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr.strip() or cp.stdout.strip() or "gh issue list failed")
    return json.loads(cp.stdout)


def issue_exists(item: TodoIssue, open_issues: list[dict]) -> bool:
    title_norm = item.title.casefold()
    key_token = f"TODO_KEY: `{item.key}`"
    for issue in open_issues:
        issue_title = normalize_whitespace(issue.get("title", "")).casefold()
        issue_body = issue.get("body", "") or ""

        if key_token in issue_body:
            return True

        # Backward-compatible matching for manually-created issues.
        if issue_title == title_norm:
            if not item.path:
                return True

            issue_body_path_norm = normalize_path(issue_body)
            item_path_norm = normalize_path(item.path)

            if item_path_norm and item_path_norm in issue_body_path_norm:
                return True

            repo_tail = item_path_norm
            if "/capibaragpt_v3/" in item_path_norm:
                repo_tail = item_path_norm.split("/capibaragpt_v3/", 1)[1]
            if repo_tail and repo_tail in issue_body_path_norm:
                return True

    return False


def create_issue(repo: str, item: TodoIssue) -> tuple[bool, str]:
    cmd = [
        "gh",
        "issue",
        "create",
        "--repo",
        repo,
        "--title",
        item.title,
        "--body",
        item.body,
    ]

    for attempt in range(1, 6):
        cp = run_gh(cmd)
        if cp.returncode == 0:
            return True, cp.stdout.strip()

        error = (cp.stderr or cp.stdout).strip()
        error_lower = error.lower()
        if "secondary rate limit" in error_lower or "api rate limit exceeded" in error_lower:
            time.sleep(10 * attempt)
            continue
        if "was submitted too quickly" in error_lower:
            time.sleep(5 * attempt)
            continue
        return False, error

    return False, "API rate limit retries exhausted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create issues from pending TODOs.")
    parser.add_argument(
        "--repo",
        required=True,
        help="GitHub repository in owner/name format",
    )
    parser.add_argument(
        "--todos-file",
        default="TODOs_PRIORITIZED.md",
        help="Path to prioritized TODO markdown file",
    )
    parser.add_argument(
        "--max-create",
        type=int,
        default=0,
        help="Maximum number of issues to create (0 = no limit)",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.25,
        help="Pause between issue creations",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without creating issues",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    todos_path = Path(args.todos_file)
    if not todos_path.exists():
        print(f"ERROR: TODO file not found: {todos_path}")
        return 2

    parsed = parse_todos(todos_path)
    unique = dedupe_todos(parsed)

    try:
        open_issues = get_open_issues(args.repo)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 2

    pending_all = [item for item in unique if not issue_exists(item, open_issues)]
    to_create = pending_all
    if args.max_create and args.max_create > 0:
        to_create = to_create[: args.max_create]

    print(f"Parsed pending TODO rows: {len(parsed)}")
    print(f"Unique TODOs (title+path): {len(unique)}")
    print(f"Open issues currently: {len(open_issues)}")
    print(f"Missing TODO issues total: {len(pending_all)}")
    print(f"Issues to create now: {len(to_create)}")

    if args.dry_run:
        print("")
        print("Dry-run sample:")
        for item in to_create[:10]:
            print(f"- {item.title}")
        return 0

    created = 0
    failed = 0
    for index, item in enumerate(to_create, start=1):
        ok, output = create_issue(args.repo, item)
        if ok:
            created += 1
        else:
            failed += 1
            err_lower = output.lower()
            if "resource not accessible by personal access token" in err_lower:
                print("ERROR: token lacks permission to create issues (createIssue).")
                print("Grant issues:write (or repo scope) and rerun.")
                print(f"Stopped at item {index}/{len(to_create)}.")
                print(f"Last error: {output}")
                return 3
            print(f"FAILED [{index}/{len(to_create)}]: {item.title}")
            print(output)

        if index % 25 == 0:
            print(f"Progress: {index}/{len(to_create)} | created={created} failed={failed}")
        time.sleep(args.sleep_seconds)

    print("")
    print(f"Done. Created: {created} | Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

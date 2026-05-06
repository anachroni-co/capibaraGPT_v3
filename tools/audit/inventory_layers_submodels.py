"""Inventory of ``layers/`` and ``sub_models/`` for BACKLOG-016.

Walks every Python file under ``layers/``, ``sub_models/`` and
``capibara/sub_models/``, classifies it as ``alive`` / ``referenced`` / ``dead``
based on who imports it across the rest of the repository, and emits a manifest
in JSON + Markdown.

Usage::

    # Regenerate manifest in-place (writes docs/sub_models_inventory.{json,md}).
    python tools/audit/inventory_layers_submodels.py

    # Drift check (used by tests/unit/test_inventory_consistency.py and CI).
    python tools/audit/inventory_layers_submodels.py --check

The script is intentionally pure stdlib (``ast`` + ``re`` + ``pathlib``) so it
runs in <30 s on the full repo without any extra dependency. It must NOT import
any module from the audited packages — that would taint classification by
side-effect.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

# Repo root resolves to capibaraGPT_v3/ — the directory two levels above this file.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Directories whose Python files are the subject of the audit.
AUDIT_DIRS: tuple[str, ...] = ("layers", "sub_models", "capibara/sub_models")

# Directories that count as "productive" importers (mark a file ``alive``).
PRODUCTIVE_DIRS: tuple[str, ...] = ("core", "scripts", "training")

# Directories that count as "test" importers (contribute to ``referenced``).
TEST_DIRS: tuple[str, ...] = ("tests",)

# Known misleading-name traps. Format: ``{stem_substring: required_substring_in_source}``.
# A file whose stem matches but whose source does not contain the required
# substring is flagged ``misleading-name:<stem>``.
MISLEADING_NAMES: dict[str, str] = {
    "mixture_of_rookies": "2202.04990",
}

# Stem-based duplicate map. If a file's repo-relative path appears here, it is
# flagged ``duplicate-of:<other_path>``. Hand-curated for now; BACKLOG-017 will
# resolve the actual physical duplicates.
KNOWN_DUPLICATES: dict[str, list[str]] = {
    "ssm_tpu": [
        "capibara/ssm/ssm_tpu.py",
        "sub_models/SSM_TPU.py",
    ],
    "spike_ssm": [
        "capibara/ssm/spike_ssm.py",
        "sub_models/experimental/spike_ssm.py",
    ],
}

# Pattern catching obvious PyTorch syntax (``.unsqueeze(``) which would fail at
# runtime under JAX. Used for the ``broken`` flag.
PYTORCH_ONLY_RE = re.compile(r"\b\.unsqueeze\(")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class FileEntry:
    path: str  # repo-relative, POSIX-style
    dotted: str  # e.g. ``layers.sparsity.mixture_of_rookies``
    defined_symbols: list[str] = field(default_factory=list)
    external_importers: list[str] = field(default_factory=list)
    classification: str = "unknown"  # alive | referenced | dead
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _posix(p: Path) -> str:
    return str(p).replace("\\", "/")


def _rel(p: Path) -> str:
    return _posix(p.relative_to(REPO_ROOT))


def to_dotted(p: Path) -> str:
    """Convert a repo-relative path to its dotted import path.

    ``capibara/ssm/__init__.py`` → ``capibara.ssm``
    ``layers/sparsity/mixture_of_rookies.py`` → ``layers.sparsity.mixture_of_rookies``
    """
    rel = p.relative_to(REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def importer_package(p: Path) -> str:
    """Dotted package of the file that *contains* the import.

    Used to resolve relative imports. For ``capibara/ssm/spike_ssm.py``
    returns ``capibara.ssm``; for ``capibara/ssm/__init__.py`` also returns
    ``capibara.ssm``.
    """
    rel = p.relative_to(REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    else:
        parts = parts[:-1] if len(parts) > 1 else []
    return ".".join(parts)


def iter_audited_files(root: Path) -> Iterable[Path]:
    for d in AUDIT_DIRS:
        base = root / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            yield p


EXCLUDE_DIR_NAMES = {
    "__pycache__",
    "venv",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "site-packages",
}

EXCLUDE_DIR_SUFFIXES = (".egg-info",)


def _is_excluded(p: Path) -> bool:
    for part in p.parts:
        if part.startswith("."):
            return True
        if part in EXCLUDE_DIR_NAMES:
            return True
        if any(part.endswith(suf) for suf in EXCLUDE_DIR_SUFFIXES):
            return True
    return False


def iter_repo_py_files(root: Path) -> Iterable[Path]:
    """All first-party Python files in the repo.

    Excludes virtual envs, build artefacts, hidden dirs and ``__pycache__``.
    """
    for p in sorted(root.rglob("*.py")):
        if _is_excluded(p):
            continue
        yield p


def _safe_read(p: Path) -> str | None:
    for enc in ("utf-8", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


def extract_top_level_symbols(tree: ast.Module) -> list[str]:
    """Public top-level classes, functions and UPPER_CASE constants."""
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                names.append(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id.isupper() and not tgt.id.startswith("_"):
                    names.append(tgt.id)
    return names


# Regex-based importer scan used for Pass 2 (the whole repo).  AST parsing
# 500+ files over a slow filesystem pushes us past the workspace bash budget;
# regex over already-loaded text is ~50× faster and correct enough for the
# dotted-prefix matching we need.  We deliberately accept a small false-positive
# rate (e.g. an ``import`` token inside a triple-quoted docstring) since the
# cost of a wrong attribution is at most a too-permissive ``referenced``
# classification, never a wrongly silent ``dead`` one.
_FROM_RE = re.compile(
    r"^[ \t]*from[ \t]+(\.+)?([\w.]*)[ \t]+import[ \t]+(.+?)(?:[ \t]*#.*)?$",
    re.MULTILINE,
)
_IMPORT_RE = re.compile(
    r"^[ \t]*import[ \t]+([\w.,\s]+?)(?:[ \t]*#.*)?$",
    re.MULTILINE,
)
_BARE_NAME_RE = re.compile(r"^([\w]+)(?:[ \t]+as[ \t]+\w+)?$")
_DOTTED_NAME_RE = re.compile(r"^([\w.]+)(?:[ \t]+as[ \t]+\w+)?$")


def collect_imports_text(
    src: str,
    importer_pkg: str,
    reexports: dict[tuple[str, str], str] | None = None,
) -> set[str]:
    """Drop-in replacement for the AST-based importer extractor.

    ``importer_pkg`` is the dotted package containing the importing file;
    used to resolve relative imports (``from . import x``).
    ``reexports``: optional ``(parent_pkg, exported_name) -> source_dotted`` map
    so that ``from layers import SelfAttention`` is also attributed to
    ``layers.self_attention`` (which is what actually defines ``SelfAttention``).
    """
    imports: set[str] = set()

    for m in _FROM_RE.finditer(src):
        dots, mod, names_blob = m.groups()
        if dots:
            pkg_parts = importer_pkg.split(".") if importer_pkg else []
            trim = len(dots) - 1
            base_parts = pkg_parts[: len(pkg_parts) - trim] if trim > 0 else pkg_parts
            if mod:
                base_parts = base_parts + mod.split(".")
            base = ".".join(base_parts)
        else:
            base = mod or ""
        if not base:
            continue
        imports.add(base)
        names_blob = names_blob.replace("(", " ").replace(")", " ").replace("\\", " ")
        for raw in names_blob.split(","):
            raw = raw.strip()
            if not raw or raw == "*":
                continue
            mt = _BARE_NAME_RE.match(raw)
            if mt:
                imports.add(f"{base}.{mt.group(1)}")
                if reexports is not None:
                    src_mod = reexports.get((base, mt.group(1)))
                    if src_mod:
                        imports.add(src_mod)

    for m in _IMPORT_RE.finditer(src):
        for raw in m.group(1).split(","):
            raw = raw.strip()
            if not raw:
                continue
            mt = _DOTTED_NAME_RE.match(raw)
            if mt:
                imports.add(mt.group(1))

    return imports


def build_reexport_map(root: Path) -> dict[tuple[str, str], str]:
    """Scan every ``__init__.py`` for ``from .child import Name`` patterns and
    return a ``(parent_pkg_dotted, exported_name) -> child_module_dotted`` map.

    This lets ``from layers import SelfAttention`` count as an import of
    ``layers.self_attention`` (the actual definer), so re-exported modules
    do not look ``dead`` in the classification.
    """
    reexports: dict[tuple[str, str], str] = {}
    for p in iter_repo_py_files(root):
        if p.name != "__init__.py":
            continue
        src = _safe_read(p)
        if src is None:
            continue
        pkg = importer_package(p)
        for m in _FROM_RE.finditer(src):
            dots, mod, names_blob = m.groups()
            if not dots:
                continue  # only relative re-exports count
            pkg_parts = pkg.split(".") if pkg else []
            trim = len(dots) - 1
            base_parts = pkg_parts[: len(pkg_parts) - trim] if trim > 0 else pkg_parts
            if mod:
                base_parts = base_parts + mod.split(".")
            target = ".".join(base_parts)
            if not target:
                continue
            names_blob = names_blob.replace("(", " ").replace(")", " ").replace("\\", " ")
            for raw in names_blob.split(","):
                raw = raw.strip()
                if not raw or raw == "*":
                    continue
                mt = _BARE_NAME_RE.match(raw)
                if mt:
                    reexports[(pkg, mt.group(1))] = target
    return reexports


_AVAILABILITY_LIBS = {"jax", "flax", "torch", "tensorflow", "tf", "tpu_v6"}


def _try_imports_only_availability_libs(try_body: list[ast.stmt]) -> bool:
    """True iff every import in the try block is for a known availability lib
    (jax, flax, torch, …). The availability-shim pattern is distinct from
    a silent-fallback for an in-repo optional feature."""
    seen_any = False
    for n in try_body:
        if isinstance(n, ast.Import):
            for alias in n.names:
                seen_any = True
                top = alias.name.split(".")[0]
                if top not in _AVAILABILITY_LIBS:
                    return False
        elif isinstance(n, ast.ImportFrom):
            seen_any = True
            top = (n.module or "").split(".")[0]
            if top not in _AVAILABILITY_LIBS:
                return False
    return seen_any


def detect_notes(p: Path, src: str, tree: ast.Module) -> list[str]:
    notes: list[str] = []

    # ``try: import X / except: X = None`` — bad if X is an in-repo optional
    # feature, **acceptable** if X is jax / flax / torch and the except handler
    # sets an availability flag. We distinguish the two so BACKLOG-017's exit
    # criteria do not penalise the legitimate JAX-availability shim pattern.
    has_silent = False
    has_shim = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        try_has_import = any(
            isinstance(n, (ast.Import, ast.ImportFrom)) for n in node.body
        )
        if not try_has_import:
            continue
        triggers = False
        for handler in node.handlers:
            for stmt in handler.body:
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant):
                    if stmt.value.value in (None, False):
                        triggers = True
                        break
            if triggers:
                break
        if not triggers:
            continue
        if _try_imports_only_availability_libs(node.body):
            has_shim = True
        else:
            has_silent = True

    if has_silent:
        notes.append("silent-fallback")
    if has_shim:
        notes.append("availability-shim")

    # broken: PyTorch-only syntax inside a JAX repo.
    if PYTORCH_ONLY_RE.search(src):
        notes.append("broken:unsqueeze")

    # misleading-name: file stem matches a known trap but the source doesn't
    # cite the expected paper / reference.
    stem = p.stem.lower()
    src_lower = src.lower()
    for trap, expected in MISLEADING_NAMES.items():
        if trap in stem and expected.lower() not in src_lower:
            notes.append(f"misleading-name:{trap}")

    # duplicate-of: hand-curated stem map.
    rel = _rel(p)
    for entries in KNOWN_DUPLICATES.values():
        if rel in entries:
            others = [e for e in entries if e != rel]
            if others:
                notes.append(f"duplicate-of:{others[0]}")

    return sorted(set(notes))


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

def _starts_with_any(path: str, prefixes: Iterable[str]) -> bool:
    return any(path == pre or path.startswith(pre + "/") for pre in prefixes)


def _is_own_package_init(importer_path: str, dotted: str) -> bool:
    """True if ``importer_path`` is an ``__init__.py`` of an ancestor package
    of ``dotted`` (or of ``dotted`` itself).

    This filter prevents a package re-exporting one of its own submodules from
    counting as an external importer.
    """
    if not importer_path.endswith("/__init__.py"):
        return False
    importer_pkg = importer_path[: -len("/__init__.py")].replace("/", ".")
    return dotted == importer_pkg or dotted.startswith(importer_pkg + ".")


def classify(entry: FileEntry, importer_index: dict[str, set[str]]) -> None:
    candidates: set[str] = set()
    for imp, importers in importer_index.items():
        if imp == entry.dotted or imp.startswith(entry.dotted + "."):
            candidates.update(importers)
    candidates.discard(entry.path)

    external = sorted(p for p in candidates if not _is_own_package_init(p, entry.dotted))
    entry.external_importers = external

    is_alive = any(_starts_with_any(p, PRODUCTIVE_DIRS) for p in external)
    is_test_referenced = any(_starts_with_any(p, TEST_DIRS) for p in external)
    is_audit_referenced = any(_starts_with_any(p, AUDIT_DIRS) for p in external)

    if is_alive:
        entry.classification = "alive"
    elif is_test_referenced or is_audit_referenced:
        entry.classification = "referenced"
    else:
        entry.classification = "dead"


# --------------------------------------------------------------------------- #
# Manifest rendering
# --------------------------------------------------------------------------- #

def render_markdown(entries: list[FileEntry]) -> str:
    lines: list[str] = [
        "# Inventory: `layers/` and `sub_models/`",
        "",
        "Generated by `tools/audit/inventory_layers_submodels.py` (BACKLOG-016).",
        "",
        "Do not edit by hand. Re-run the script after any change to the audited",
        "directories. CI runs the same script with `--check` and fails on drift.",
        "",
        "## Classification rules",
        "",
        "- `alive` — imported from `core/`, `scripts/`, or `training/` (productive paths).",
        "- `referenced` — imported only from `tests/` or by other audited modules under `layers/` / `sub_models/` / `capibara/sub_models/`.",
        "- `dead` — no importer outside its own package `__init__.py`.",
        "",
        "## Note tokens",
        "",
        "- `broken:unsqueeze` — uses PyTorch `.unsqueeze(` inside a JAX repo (would fail at runtime).",
        "- `silent-fallback` — `try: import X / except: X = None` pattern for an in-repo optional feature (violates `CONTRIBUTING.md` §1).",
        "- `availability-shim` — accepted variant of the above where X is an external library (`jax` / `flax` / `torch`); the except handler sets an `*_AVAILABLE` flag that callers branch on. Not a silent failure.",
        "- `misleading-name:<trap>` — file stem suggests a paper / concept that the source does not cite.",
        "- `duplicate-of:<path>` — hand-curated duplicate of another file in the repo.",
        "- `syntax-error:<msg>` — file does not parse with `ast`.",
        "",
    ]

    counts: dict[str, int] = {"alive": 0, "referenced": 0, "dead": 0, "unknown": 0}
    for e in entries:
        counts[e.classification] = counts.get(e.classification, 0) + 1

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total audited: **{len(entries)}**")
    lines.append(f"- alive: **{counts['alive']}**")
    lines.append(f"- referenced: **{counts['referenced']}**")
    lines.append(f"- dead: **{counts['dead']}**")
    if counts["unknown"]:
        lines.append(f"- unknown: **{counts['unknown']}**")
    lines.append("")

    for status in ("alive", "referenced", "dead", "unknown"):
        bucket = [e for e in entries if e.classification == status]
        if not bucket:
            continue
        lines.append(f"## {status.title()} ({len(bucket)})")
        lines.append("")
        lines.append("| Path | Symbols | External importers | Notes |")
        lines.append("|------|---------|--------------------|-------|")
        for e in sorted(bucket, key=lambda x: x.path):
            symbols = ", ".join(e.defined_symbols[:5]) or "—"
            if len(e.defined_symbols) > 5:
                symbols += f" (+{len(e.defined_symbols) - 5})"
            importers = ", ".join(f"`{p}`" for p in e.external_importers[:3]) or "—"
            if len(e.external_importers) > 3:
                importers += f" (+{len(e.external_importers) - 3})"
            notes = ", ".join(f"`{n}`" for n in e.notes) or "—"
            lines.append(f"| `{e.path}` | {symbols} | {importers} | {notes} |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def build_manifest() -> tuple[list[FileEntry], dict[str, set[str]]]:
    audited: dict[str, FileEntry] = {}

    # Pass 1: index every audited file (defined symbols + notes).
    for p in iter_audited_files(REPO_ROOT):
        rel = _rel(p)
        src = _safe_read(p)
        if src is None:
            audited[rel] = FileEntry(path=rel, dotted=to_dotted(p), notes=["unreadable"])
            continue
        try:
            tree = ast.parse(src, filename=rel)
        except SyntaxError as exc:
            audited[rel] = FileEntry(
                path=rel, dotted=to_dotted(p), notes=[f"syntax-error:{exc.msg}"]
            )
            continue
        entry = FileEntry(path=rel, dotted=to_dotted(p))
        entry.defined_symbols = extract_top_level_symbols(tree)
        entry.notes = detect_notes(p, src, tree)
        audited[rel] = entry

    # Pass 2: importer index across the whole repo.
    # Uses regex-based scanning (``collect_imports_text``) instead of ``ast.parse``
    # because parsing 500+ files over a slow Windows-mount filesystem blows past
    # the 30-s budget BACKLOG-016 sets for this script.
    # We pre-compute a re-export map so ``from layers import SelfAttention``
    # is also attributed to ``layers.self_attention`` (the module that
    # actually defines the class), avoiding false ``dead`` classifications
    # for files only reached via package-level re-exports.
    reexports = build_reexport_map(REPO_ROOT)
    importer_index: dict[str, set[str]] = {}
    for p in iter_repo_py_files(REPO_ROOT):
        rel = _rel(p)
        src = _safe_read(p)
        if src is None:
            continue
        pkg = importer_package(p)
        for imp in collect_imports_text(src, pkg, reexports):
            importer_index.setdefault(imp, set()).add(rel)

    # Pass 3: classify each audited file.
    for entry in audited.values():
        classify(entry, importer_index)

    return sorted(audited.values(), key=lambda e: e.path), importer_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the on-disk manifest differs from the freshly built one.",
    )
    parser.add_argument("--out-json", default="docs/sub_models_inventory.json")
    parser.add_argument("--out-md", default="docs/sub_models_inventory.md")
    args = parser.parse_args()

    entries, _ = build_manifest()

    manifest = {
        "generated_by": "tools/audit/inventory_layers_submodels.py",
        "audit_dirs": list(AUDIT_DIRS),
        "productive_dirs": list(PRODUCTIVE_DIRS),
        "test_dirs": list(TEST_DIRS),
        "entries": [asdict(e) for e in entries],
    }
    json_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    md_text = render_markdown(entries)

    json_path = REPO_ROOT / args.out_json
    md_path = REPO_ROOT / args.out_md

    if args.check:
        existing_json = json_path.read_text(encoding="utf-8") if json_path.exists() else ""
        existing_md = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
        if existing_json != json_text:
            print(
                f"DRIFT: {json_path.relative_to(REPO_ROOT)} out of date. "
                "Re-run `python tools/audit/inventory_layers_submodels.py`.",
                file=sys.stderr,
            )
            return 2
        if existing_md != md_text:
            print(
                f"DRIFT: {md_path.relative_to(REPO_ROOT)} out of date. "
                "Re-run `python tools/audit/inventory_layers_submodels.py`.",
                file=sys.stderr,
            )
            return 2
        print("OK: inventory manifests are in sync.")
        return 0

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")

    counts: dict[str, int] = {"alive": 0, "referenced": 0, "dead": 0, "unknown": 0}
    for e in entries:
        counts[e.classification] = counts.get(e.classification, 0) + 1
    print(f"Inventoried {len(entries)} files. {counts}")
    print(f"  -> {json_path.relative_to(REPO_ROOT)}")
    print(f"  -> {md_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

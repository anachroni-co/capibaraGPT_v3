"""Naming sentinels for the BACKLOG-017 cleanup.

These tests fail CI if a misleading name reappears in the tree without an
explicit reference to the paper it claims to implement. They are the
"named after a paper but isn't" trap detector required by the BACKLOG-017
exit criteria.

Per CONTRIBUTING.md: a filename or class name that quotes a paper must
either implement that paper, or carry a reference to the paper's arXiv id
in the same file (so reviewers can verify intent).
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _all_python_files() -> list[Path]:
    """All tracked-style .py files under layers/ and sub_models/."""
    out: list[Path] = []
    for top in ("layers", "sub_models"):
        root = REPO_ROOT / top
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            # skip caches and quarantine (broken-by-design)
            if "__pycache__" in p.parts:
                continue
            out.append(p)
    return out


def test_mixture_of_rookies_filename_trap():
    """No file under layers/ or sub_models/ may be named ``mixture_of_rookies``.

    The historical ``layers/sparsity/mixture_of_rookies.py`` was a vanilla
    top-k MoE that did NOT implement Pinto/Arnau/Gonzalez (arXiv 2202.04990).
    BACKLOG-017 deleted it. If anyone reintroduces the file, fail loudly.
    """
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for p in _all_python_files()
        if "mixture_of_rookies" in p.name
    ]
    assert not offenders, (
        "mixture_of_rookies filename reintroduced: "
        f"{offenders}. Either implement arXiv 2202.04990 verbatim or pick "
        "a name that describes what the file actually does."
    )


def test_mixture_of_rookies_classname_trap():
    """A class literally named ``MixtureOfRookies`` requires the arXiv id.

    Walks the source text (cheap substring check, not AST) for any file that
    declares ``class MixtureOfRookies`` and asserts the same file mentions
    ``2202.04990`` or ``arXiv:2202.04990``. If you legitimately implement
    the paper, the citation costs nothing; if you don't, pick a different
    name.
    """
    offenders: list[str] = []
    for p in _all_python_files():
        try:
            src = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "class MixtureOfRookies" not in src:
            continue
        if "2202.04990" not in src:
            offenders.append(str(p.relative_to(REPO_ROOT)))
    assert not offenders, (
        "class MixtureOfRookies declared without arXiv 2202.04990 reference: "
        f"{offenders}. Cite the paper in the same file or rename the class."
    )

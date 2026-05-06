"""Drift detector for the ``layers/`` + ``sub_models/`` inventory.

Re-runs ``tools/audit/inventory_layers_submodels.py`` in-process and asserts the
on-disk manifests under ``docs/`` are byte-identical to what the script would
produce now. Fails CI on drift, so a future PR cannot quietly add a new dead
module or change a notes flag without regenerating the manifest.

Per BACKLOG-016 exit criteria (CI gate against drift).
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools" / "audit"

# Make the tools/ package importable as a flat module.
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def inventory_module():
    from tools.audit import inventory_layers_submodels as mod  # type: ignore

    return mod


def _expected_json(mod) -> str:
    entries, _ = mod.build_manifest()
    manifest = {
        "generated_by": "tools/audit/inventory_layers_submodels.py",
        "audit_dirs": list(mod.AUDIT_DIRS),
        "productive_dirs": list(mod.PRODUCTIVE_DIRS),
        "test_dirs": list(mod.TEST_DIRS),
        "entries": [asdict(e) for e in entries],
    }
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def _expected_md(mod) -> str:
    entries, _ = mod.build_manifest()
    return mod.render_markdown(entries)


def test_inventory_json_matches_disk(inventory_module):
    expected = _expected_json(inventory_module)
    on_disk = (REPO_ROOT / "docs" / "sub_models_inventory.json").read_text(encoding="utf-8")
    assert on_disk == expected, (
        "docs/sub_models_inventory.json is out of date. "
        "Re-run `python tools/audit/inventory_layers_submodels.py`."
    )


def test_inventory_md_matches_disk(inventory_module):
    expected = _expected_md(inventory_module)
    on_disk = (REPO_ROOT / "docs" / "sub_models_inventory.md").read_text(encoding="utf-8")
    assert on_disk == expected, (
        "docs/sub_models_inventory.md is out of date. "
        "Re-run `python tools/audit/inventory_layers_submodels.py`."
    )


def test_inventory_classifications_are_complete(inventory_module):
    """No entry should remain ``unknown``; every audited file must be classified."""
    entries, _ = inventory_module.build_manifest()
    unknowns = [e.path for e in entries if e.classification == "unknown"]
    assert not unknowns, f"Unclassified entries: {unknowns}"


def test_inventory_check_mode_is_clean(inventory_module, monkeypatch, capsys):
    """``--check`` must exit 0 when manifest on disk matches a fresh build."""
    monkeypatch.setattr(sys, "argv", ["inventory_layers_submodels.py", "--check"])
    rc = inventory_module.main()
    captured = capsys.readouterr()
    assert rc == 0, (
        f"--check reported drift (rc={rc}). "
        f"stdout: {captured.out} stderr: {captured.err}"
    )

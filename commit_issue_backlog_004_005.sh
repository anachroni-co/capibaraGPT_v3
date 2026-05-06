#!/usr/bin/env bash
#
# Combined commit plan for BACKLOG-004 + BACKLOG-005.
#
# Creates TWO independent branches off origin/main:
#
#   1. fix/issue-backlog-004-inference-real-params
#        - inference/hybrid_inference_engine.py
#        - inference/engines/advanced_quantized_engine.py
#        - tests/integration/test_engines_no_mock_params.py  (new)
#
#   2. fix/issue-backlog-005-data-lineage-isolated
#        - training/data_lineage/inference_safe_parameter_controller.py
#        - training/data_lineage/demo_traceability_system.py
#        - tests/integration/test_data_lineage_demo_isolated.py  (new)
#
# Pre-req: your working tree holds all 6 edited files from the ongoing
# session.
#
# Run from the repo root on your machine, after:
#   sed -i 's/\r$//' commit_issue_backlog_004_005.sh

set -euo pipefail

FILES_004=(
    "inference/hybrid_inference_engine.py"
    "inference/engines/advanced_quantized_engine.py"
    "tests/integration/test_engines_no_mock_params.py"
)

FILES_005=(
    "training/data_lineage/inference_safe_parameter_controller.py"
    "training/data_lineage/demo_traceability_system.py"
    "tests/integration/test_data_lineage_demo_isolated.py"
)

echo ">>> current branch: $(git rev-parse --abbrev-ref HEAD)"
git status --short | head -20

# ------------------------------------------------------------------
# 0. Safety stash - all 6 files (tracked + new tests)
# ------------------------------------------------------------------
git stash push --include-untracked -m "pre-backlog-004-005" -- \
    "${FILES_004[@]}" \
    "${FILES_005[@]}"

git fetch origin

# ==================================================================
# BRANCH 1: BACKLOG-004
# ==================================================================
git checkout -B fix/issue-backlog-004-inference-real-params origin/main

# Apply the 6 files from stash into the clean branch.
git stash pop

# Sanity check the 3 BACKLOG-004 files.
python - <<'PY'
import ast
paths = [
    "inference/hybrid_inference_engine.py",
    "inference/engines/advanced_quantized_engine.py",
    "tests/integration/test_engines_no_mock_params.py",
]
for p in paths:
    ast.parse(open(p, encoding="utf-8").read())
print("BACKLOG-004: all 3 files parse OK")
PY

python -m pytest tests/integration/test_engines_no_mock_params.py -q || {
    echo "BACKLOG-004 tests failed - aborting"; exit 1;
}

# Stage ONLY the BACKLOG-004 files.
git add "${FILES_004[@]}"

git status --short
echo "--- review above, press Enter to commit BACKLOG-004, Ctrl+C to abort ---"
read -r _

git commit -m "fix(BACKLOG-004): real parameter loading in inference engines

ISSUE-004 (BACKLOG.md): the two production inference engines still shipped
hard-coded placeholder parameters on the main path:

- inference/hybrid_inference_engine.TPUInferenceEngine._load_model_params
  silently fell back to {\"dummy\": jnp.array([1.0])} whenever the pickle
  checkpoint was missing, masking missing-weights bugs.
- inference/hybrid_inference_engine.TPUInferenceEngine._compile_generation_function
  returned logits built from jnp.ones((B, S, 32000)), so generate() was
  producing tokens from a synthetic distribution instead of the real model.
- inference/engines/advanced_quantized_engine.QuantizedInferenceEngine
  ._load_model_params built a full 12-layer transformer from
  np.random.randn(...) arrays regardless of model_path, which meant
  quantization calibration ran on pure noise.
- QuantizedInferenceEngine._get_memory_usage returned a constant 512.0 MB
  so every performance report was wrong.

Changes:

- _load_model_params (hybrid) probes in order: checkpoint.pkl,
  flax msgpack (checkpoint.msgpack / params.msgpack) and an orbax
  checkpoint/ directory, then raises FileNotFoundError when nothing is
  usable. No synthetic fallback.
- _compile_generation_function now consults an optional
  self.model_module (a real Flax nn.Module). If one is attached it JITs
  a real forward pass; if not, it leaves compiled_generate = None so
  generate() fails fast with a BACKLOG-004 message instead of producing
  fake logits.
- _load_model_params (advanced) loads .npz / .safetensors / .pkl (opt-in
  by extension) and rejects every other shape with FileNotFoundError or
  ValueError. A new _unflatten_param_tree helper turns flat checkpoint
  dicts into the nested {embedding, transformer_layer_N, output} shape
  consumed by the quantizer.
- _get_memory_usage measures the current process RSS via psutil and
  returns -1.0 when psutil is unavailable so callers can tell placeholder
  readings from real measurements.

Validation:

- 0 syntax errors across the three touched files.
- New tests/integration/test_engines_no_mock_params.py (8 tests, all
  passing) asserts via AST + source inspection that the placeholders are
  gone and the real load / compile / mem-usage branches are present."

git push -u origin fix/issue-backlog-004-inference-real-params

# ------------------------------------------------------------------
# Preserve the remaining 3 BACKLOG-005 files before switching branch.
# ------------------------------------------------------------------
git stash push --include-untracked -m "carry-backlog-005" -- \
    "${FILES_005[@]}"

# ==================================================================
# BRANCH 2: BACKLOG-005 (independent, also from origin/main)
# ==================================================================
git checkout -B fix/issue-backlog-005-data-lineage-isolated origin/main

git stash pop

python - <<'PY'
import ast
paths = [
    "training/data_lineage/inference_safe_parameter_controller.py",
    "training/data_lineage/demo_traceability_system.py",
    "tests/integration/test_data_lineage_demo_isolated.py",
]
for p in paths:
    ast.parse(open(p, encoding="utf-8").read())
print("BACKLOG-005: all 3 files parse OK")
PY

python -m pytest tests/integration/test_data_lineage_demo_isolated.py -q || {
    echo "BACKLOG-005 tests failed - aborting"; exit 1;
}

git add "${FILES_005[@]}"

git status --short
echo "--- review above, press Enter to commit BACKLOG-005, Ctrl+C to abort ---"
read -r _

git commit -m "fix(BACKLOG-005): isolate demo + remove runtime mock lineage in data_lineage

ISSUE-005 (BACKLOG.md): training/data_lineage still mixed demo-only code
paths and runtime mocks into the production package:

- InferenceSafeParameterController.create_dataset_mask_safe silently
  fabricated a lineage of the first 1/3 of base_parameters whenever a
  dataset had no registered lineage (comment: 'Create mock lineage for
  testing'). Callers then got scale_factors computed from this fake
  lineage and thought the mask was real.
- demo_traceability_system.py called logging.basicConfig(level=INFO) at
  module import time. Merely importing training.data_lineage (which
  transitively touches the demo in some test helpers) silently
  reconfigured the root logger of the whole process.
- demo_traceability_system.py exposed a top-level public 'MockModel'
  class that looked importable as production code.
- The demo had no opt-in guard, so running the module as a script ran
  the full mock pipeline without the user realising it is demo-only.

Changes:

- create_dataset_mask_safe now emits a logger.warning for unknown
  datasets and returns an empty dataset_params list - no fabricated
  scale_factors. Comment explicitly references BACKLOG-005 so the
  regression cannot silently come back.
- demo_traceability_system.py moves logging.basicConfig into a new
  _configure_demo_logging() helper that is only invoked from __main__.
  The module-level scope no longer mutates the root logger on import.
- __main__ now requires CAPIBARA_DATA_LINEAGE_DEMO=1 to run; otherwise
  it prints an instructional message and exits with status 2.
- The public 'MockModel' class is renamed to the private
  '_DemoMockModel'. __init__.py did not re-export it, and the only
  in-repo reference was internal to the demo, so this rename is safe.

Validation:

- 0 syntax errors across the three touched files.
- New tests/integration/test_data_lineage_demo_isolated.py (7 tests,
  all passing) asserts via AST + source inspection that: the mock
  lineage fallback is gone, no module-level logging.basicConfig, the
  CLI is env-gated, and only the private _DemoMockModel class exists."

git push -u origin fix/issue-backlog-005-data-lineage-isolated

echo
echo "Done. Open PRs at:"
echo "  BACKLOG-004: https://github.com/anachroni-co/capibaraGPT_v3/pull/new/fix/issue-backlog-004-inference-real-params"
echo "  BACKLOG-005: https://github.com/anachroni-co/capibaraGPT_v3/pull/new/fix/issue-backlog-005-data-lineage-isolated"

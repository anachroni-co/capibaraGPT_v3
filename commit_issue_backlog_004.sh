#!/usr/bin/env bash
#
# Commit plan for BACKLOG ISSUE-004 (real parameter loading in inference).
#
# Pre-req: your working tree currently holds the three edited files from
# the ongoing session:
#   - inference/hybrid_inference_engine.py
#   - inference/engines/advanced_quantized_engine.py
#   - tests/integration/test_engines_no_mock_params.py  (new)
#
# Run from the repo root on your machine, after:
#   sed -i 's/\r$//' commit_issue_backlog_004.sh

set -euo pipefail

echo ">>> current branch: $(git rev-parse --abbrev-ref HEAD)"
git status --short | head -10

# ------------------------------------------------------------------
# 0. Safety stash
# ------------------------------------------------------------------
git stash push --include-untracked -m "pre-backlog-004-inference" -- \
    inference/hybrid_inference_engine.py \
    inference/engines/advanced_quantized_engine.py \
    tests/integration/test_engines_no_mock_params.py

# ------------------------------------------------------------------
# 1. Fresh branch from origin/main
# ------------------------------------------------------------------
git fetch origin
git checkout -B fix/issue-backlog-004-inference-real-params origin/main

# ------------------------------------------------------------------
# 2. Re-apply the three edited files (tracked + new test)
# ------------------------------------------------------------------
git stash pop

# ------------------------------------------------------------------
# 3. Sanity check
# ------------------------------------------------------------------
python - <<'PY'
import ast
paths = [
    "inference/hybrid_inference_engine.py",
    "inference/engines/advanced_quantized_engine.py",
    "tests/integration/test_engines_no_mock_params.py",
]
for p in paths:
    ast.parse(open(p, encoding="utf-8").read())
print("All 3 files parse OK")
PY

# Run the non-regression tests
python -m pytest tests/integration/test_engines_no_mock_params.py -q || {
    echo "Tests failed - aborting"; exit 1;
}

# ------------------------------------------------------------------
# 4. Commit + push
# ------------------------------------------------------------------
git add \
    inference/hybrid_inference_engine.py \
    inference/engines/advanced_quantized_engine.py \
    tests/integration/test_engines_no_mock_params.py

git status --short
echo "--- review above and press Enter to commit, Ctrl+C to abort ---"
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

echo
echo "Done. Open PR at:"
echo "  https://github.com/anachroni-co/capibaraGPT_v3/pull/new/fix/issue-backlog-004-inference-real-params"

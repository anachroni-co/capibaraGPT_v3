#!/usr/bin/env bash
#
# Commit plan for BACKLOG-008: model factory MoE + configs 1B/3B.
#
# Uses git worktree to create an independent branch off origin/main so your
# current working tree is never touched. Run from the repo root, after:
#   sed -i 's/\r$//' commit_issue_backlog_008.sh
#
# Env vars:
#   SKIP_TESTS=1   run AST + TOML checks only, skip pytest (useful when JAX
#                  is not installed locally; CI will run the smoke tests).

set -euo pipefail

test -d .git || { echo "Run from repo root (no .git directory here)"; exit 1; }

REPO_ROOT="$(pwd)"
BRANCH="fix/issue-backlog-008-model-factory-moe"

echo ">>> repo root: $REPO_ROOT"
echo ">>> current branch: $(git rev-parse --abbrev-ref HEAD)"
echo ">>> target branch: $BRANCH"

git fetch origin

FILES=(
    "core/model_factory/__init__.py"
    "core/model_factory/config.py"
    "core/model_factory/blocks.py"
    "core/model_factory/model.py"
    "configs/1b.toml"
    "configs/3b.toml"
    "tests/unit/test_model_factory.py"
)

for f in "${FILES[@]}"; do
    test -f "$f" || { echo "missing file in working tree: $f"; exit 1; }
done

SNAP_DIR="$(mktemp -d)"
echo ">>> snapshot dir: $SNAP_DIR"
for f in "${FILES[@]}"; do
    mkdir -p "$SNAP_DIR/$(dirname "$f")"
    cp "$f" "$SNAP_DIR/$f"
done

MSG='feat(BACKLOG-008): model factory (Mixtral-style MoE) + 1B / 3B presets

New module core/model_factory/ (no capibara/* dependency, only jax + flax):
- config.py    : ModelConfig frozen dataclass with __post_init__ validation
                 (GQA divisibility, head_dim*n_heads == d_model, top_k <=
                 n_experts). active_params_estimate() / total_params_estimate()
                 give analytic param counts. Presets: preset_1b (~1.10B
                 active, ~2.01B total), preset_3b (~2.93B active, ~5.43B
                 total), preset_smoke (~150K, used for CI). from_toml()
                 loads configs/*.toml; PRESETS dict + get_preset(key).
- blocks.py    : Flax modules. RMSNorm (Llama-style, fp32 reduce). RoPE
                 cos/sin tables + apply_rotary. GroupedQueryAttention with
                 K/V repeat to match n_heads. SwiGLUExpert (gate/up/down).
                 TopKRouter -> jax.lax.top_k -> renormalized gates +
                 onehot mask -> Switch-Transformer load-balance loss
                 n_experts * sum(P_e * f_e). SparseMoEBlock stacks experts
                 as (n_experts, D, ff_dim) parameters and dispatches per
                 top_k slot. TransformerBlock pre-norm pattern returns
                 (x_new, aux_loss). make_causal_mask shape (1,1,T,T).
- model.py     : CapibaraMoEModel @nn.compact assembles embed -> N
                 TransformerBlocks -> RMSNorm -> tied lm_head (or
                 separate Dense). Returns ModelOutput(logits, aux_loss).
                 Validates input length <= cfg.max_seq_len. build_model,
                 init_params, count_params helpers exposed.
- __init__.py  : minimal public surface (ModelConfig, PRESETS, get_preset,
                 CapibaraMoEModel, ModelOutput, build_model, init_params,
                 count_params).

New training recipes:
- configs/1b.toml : 1.10B active / 2.01B total (d_model=1536, 24 layers,
                    GQA 24:8, ff_dim=4096, 4 experts top-2). batch=1024,
                    seq=4096, 100k steps, lr_peak=3.0e-4, cosine to 10%.
- configs/3b.toml : 2.93B active / 5.43B total (d_model=2048, 36 layers,
                    GQA 32:8, ff_dim=5632, 4 experts top-2). batch=1024,
                    grad_accum=2, 200k steps, lr_peak=1.5e-4.
Both use bf16 compute_dtype + fp32 param_dtype, RoPE theta=500000, tied
embeddings, load_balance_weight=0.01.

Tests (tests/unit/test_model_factory.py, 18 tests):
- Layer 1 (no JAX needed): preset_1b/3b active params within target band,
  total > active, smoke preset constructs, get_preset known/unknown,
  GQA divisibility / head_dim / top_k validation raises ValueError,
  TOML round-trip equals in-code preset (configs/1b.toml and 3b.toml
  must stay in sync with code).
- Layer 2 (skips gracefully without JAX): smoke preset init produces
  finite params, runtime param count matches analytic within 1%, forward
  shape == (B, T, vocab_size), aux_loss is finite scalar >= 0, causal
  mask is real (changing token at position 5 leaves logits[:, :5] bit-
  identical), seq_len > max_seq_len raises.

Bootstrap detail in the test file: we mount core/model_factory as the
synthetic parent package "_mfu_pkg" via importlib so test imports do not
trigger core/__init__.py and do not need an editable install.

Validation: 18/18 unit tests pass locally with jax 0.6.2 + flax 0.10.7
on CPU. Smoke forward (~155K params) returns finite logits and a finite
aux_loss; analytic-vs-runtime param diff is 320 (=5 RMSNorm scale vectors
of d_model=64).'

wt_dir="$(mktemp -d)"
echo
echo "========================================================="
echo " BRANCH: $BRANCH"
echo " worktree: $wt_dir"
echo "========================================================="

git worktree add -B "$BRANCH" "$wt_dir" origin/main >/dev/null

for f in "${FILES[@]}"; do
    mkdir -p "$wt_dir/$(dirname "$f")"
    cp "$SNAP_DIR/$f" "$wt_dir/$f"
done

(
    cd "$wt_dir"

    # AST parse check on the Python files only.
    python3 - "${FILES[@]}" <<'PY'
import ast, sys
checked = 0
for p in sys.argv[1:]:
    if p.endswith(".py"):
        ast.parse(open(p, encoding="utf-8").read())
        checked += 1
print(f"AST OK for {checked} Python files")
PY

    # TOML validation for the two recipes.
    python3 - "${FILES[@]}" <<'PY'
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
checked = 0
for p in sys.argv[1:]:
    if p.endswith(".toml"):
        tomllib.loads(open(p, encoding="utf-8").read())
        checked += 1
print(f"TOML OK for {checked} TOML files")
PY

    # Tests: run the model_factory unit tests. Layer-2 (JAX) tests skip
    # gracefully if JAX/Flax are not importable.
    TESTS=(
        "tests/unit/test_model_factory.py"
    )

    if [ "${SKIP_TESTS:-0}" = "1" ]; then
        echo "SKIP_TESTS=1 - skipping pytest for $BRANCH (AST + TOML checks already passed)"
    elif command -v pytest >/dev/null 2>&1; then
        pytest "${TESTS[@]}" -q || { echo "tests failed in $BRANCH - aborting"; exit 1; }
    elif python3 -c "import pytest" 2>/dev/null; then
        python3 -m pytest "${TESTS[@]}" -q || { echo "tests failed in $BRANCH - aborting"; exit 1; }
    else
        echo ""
        echo "!!! pytest not found in PATH nor in python3's site-packages."
        echo "!!! Options:"
        echo "!!!   a) pip install --user pytest  (then re-run)"
        echo "!!!   b) source your venv where pytest lives (then re-run)"
        echo "!!!   c) SKIP_TESTS=1 bash commit_issue_backlog_008.sh"
        echo "!!!      (relies only on AST + TOML checks)"
        exit 1
    fi

    git add "${FILES[@]}"
    git status --short

    echo "--- press Enter to commit $BRANCH, Ctrl+C to abort ---"
    read -r _

    git commit -m "$MSG"
    git push -u origin "$BRANCH"
)

git worktree remove "$wt_dir"
rm -rf "$SNAP_DIR"

echo
echo "Done. Open the PR at:"
echo "  https://github.com/anachroni-co/capibaraGPT_v3/pull/new/$BRANCH"
echo
echo "Your original branch is unchanged."

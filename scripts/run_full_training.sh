#!/usr/bin/env bash
# ============================================================
# Capibara Legal — Full training pipeline
# Google Axion c4a-standard-32 (32 vCPU · 128 GB RAM)
#
# Usage:
#   bash scripts/run_full_training.sh            # all phases
#   bash scripts/run_full_training.sh 3          # from phase 3
#   bash scripts/run_full_training.sh 3 5        # phases 3 to 5
#
# Phases:
#   0  Download + tokenize legal corpus
#   1  Small  34M  phase 2  (resume from soup, 10k steps, ~14 h)
#   2  Medium 114M          (10k steps, ~29 h)
#   3  Large  474M phase 1  (general corpus, 35k steps, ~6 d)
#   4  Large  474M phase 2  (legal DAPT, 10k steps, ~2 d)
#   5  Soup all models
# ============================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

VENV="${HOME}/.venv/capibara"
PYTHON="${VENV}/bin/python"

DATA_DIR="data/tokenized"
LEGAL_RAW="data/raw/legal"
LEGAL_TOK="data/tokenized/legal"

CKPT_SMALL1="checkpoints/axion_mixed"           # small phase 1 (done)
CKPT_SMALL2="checkpoints/axion_small_phase2"
CKPT_MEDIUM="checkpoints/axion_medium_bf16"
CKPT_LARGE1="checkpoints/axion_large_phase1"
CKPT_LARGE2="checkpoints/axion_large_legal"

THREADS=32

# Optimisation flags (all optimisations on)
OPT_FLAGS="--dtype bf16 --grad-checkpoint --compile-cache cache/jax_compile --n-devices 4"

START_PHASE=${1:-0}
END_PHASE=${2:-5}

# ── Helpers ───────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
skip() { echo -e "${YELLOW}[SKIP]${NC} $*"; }

run_phase() {
    local phase=$1
    [[ $phase -ge $START_PHASE && $phase -le $END_PHASE ]]
}

latest_ckpt() {
    # Returns path of latest ckpt_step_*.pkl in a directory, or ""
    ls -1 "${1}/ckpt_step_"*.pkl 2>/dev/null | sort | tail -1 || true
}

soup_exists() { [[ -f "${1}/soup_uniform.pkl" ]]; }

# ── Phase 0: Legal corpus ─────────────────────────────────────────────────────

if run_phase 0; then
    log "═══ Phase 0: Download + tokenise legal corpus ═══"

    if [[ -d "$LEGAL_TOK" && $(find "$LEGAL_TOK" -name "*.npy" 2>/dev/null | wc -l) -gt 0 ]]; then
        skip "Legal corpus already tokenised in $LEGAL_TOK"
    else
        log "Downloading legal sources → $LEGAL_RAW"
        "$PYTHON" scripts/download_legal_corpus.py \
            --source all \
            --output "$LEGAL_RAW" \
            --year-from 2000          # TC sentencias from 2000 (~1 h vs 4 h)

        log "Tokenising legal corpus → $LEGAL_TOK"
        "$PYTHON" scripts/prepare_corpus.py \
            --input  "$LEGAL_RAW" \
            --output "$LEGAL_TOK" \
            --extensions .md .txt .adoc
    fi
    ok "Phase 0 done"
fi

# ── Phase 1: Small 34M phase 2 ────────────────────────────────────────────────

if run_phase 1; then
    log "═══ Phase 1: Small 34M — phase 2 (resume + refine) ═══"

    SOUP1="${CKPT_SMALL1}/soup_uniform.pkl"
    if [[ ! -f "$SOUP1" ]]; then
        log "Running soup on small phase 1 checkpoints…"
        "$PYTHON" scripts/soup_checkpoints.py "$CKPT_SMALL1" --n 3
    fi

    if soup_exists "$CKPT_SMALL2"; then
        skip "Small phase 2 soup already exists — skipping"
    else
        RESUME_FROM="$SOUP1"
        LAST=$(latest_ckpt "$CKPT_SMALL2")
        [[ -n "$LAST" ]] && RESUME_FROM="$LAST" && log "Resuming from $LAST"

        log "Training small phase 2 → $CKPT_SMALL2"
        "$PYTHON" scripts/launch_axion_training.py \
            --data-dir "$DATA_DIR" \
            --preset small \
            --batch-size 32 --grad-accum 8 \
            --lr 1e-4 --steps 10000 \
            --threads $THREADS \
            $OPT_FLAGS \
            --output "$CKPT_SMALL2" \
            --resume "$RESUME_FROM"

        log "Soupe small phase 2…"
        "$PYTHON" scripts/soup_checkpoints.py "$CKPT_SMALL2" --n 3
    fi
    ok "Phase 1 done"
fi

# ── Phase 2: Medium 114M ──────────────────────────────────────────────────────

if run_phase 2; then
    log "═══ Phase 2: Medium 114M ═══"

    if soup_exists "$CKPT_MEDIUM"; then
        skip "Medium soup already exists — skipping"
    else
        LAST=$(latest_ckpt "$CKPT_MEDIUM")
        RESUME_ARG=""
        [[ -n "$LAST" ]] && RESUME_ARG="--resume $LAST" && log "Resuming from $LAST"

        log "Training medium → $CKPT_MEDIUM"
        "$PYTHON" scripts/launch_axion_training.py \
            --data-dir "$DATA_DIR" \
            --preset medium \
            --batch-size 16 --grad-accum 8 \
            --steps 10000 \
            --threads $THREADS \
            $OPT_FLAGS \
            --output "$CKPT_MEDIUM" \
            $RESUME_ARG

        log "Soup medium…"
        "$PYTHON" scripts/soup_checkpoints.py "$CKPT_MEDIUM" --n 3
    fi
    ok "Phase 2 done"
fi

# ── Phase 3: Large 474M — general pre-training ────────────────────────────────

if run_phase 3; then
    log "═══ Phase 3: Large 474M — general pre-training (~6 d) ═══"

    if soup_exists "$CKPT_LARGE1"; then
        skip "Large phase 1 soup already exists — skipping"
    else
        LAST=$(latest_ckpt "$CKPT_LARGE1")
        RESUME_ARG=""
        [[ -n "$LAST" ]] && RESUME_ARG="--resume $LAST" && log "Resuming from $LAST"

        log "Training large phase 1 → $CKPT_LARGE1"
        # batch=16 with --grad-checkpoint allows 2x batch vs batch=8
        # --n-devices 4: 4 virtual JAX devices × 8 cores each
        "$PYTHON" scripts/launch_axion_training.py \
            --data-dir "$DATA_DIR" \
            --preset large \
            --batch-size 16 --grad-accum 16 \
            --steps 35000 \
            --threads $THREADS \
            $OPT_FLAGS \
            --output "$CKPT_LARGE1" \
            $RESUME_ARG

        log "Soup large phase 1…"
        "$PYTHON" scripts/soup_checkpoints.py "$CKPT_LARGE1" --n 3
    fi
    ok "Phase 3 done"
fi

# ── Phase 4: Large 474M — legal DAPT ─────────────────────────────────────────

if run_phase 4; then
    log "═══ Phase 4: Large 474M — legal DAPT (~2 d) ═══"

    SOUP_LARGE1="${CKPT_LARGE1}/soup_uniform.pkl"
    if [[ ! -f "$SOUP_LARGE1" ]]; then
        log "No large phase 1 soup found at $SOUP_LARGE1 — running soup first"
        "$PYTHON" scripts/soup_checkpoints.py "$CKPT_LARGE1" --n 3
    fi

    if soup_exists "$CKPT_LARGE2"; then
        skip "Large legal soup already exists — skipping"
    else
        LAST=$(latest_ckpt "$CKPT_LARGE2")
        RESUME_FROM="$SOUP_LARGE1"
        [[ -n "$LAST" ]] && RESUME_FROM="$LAST" && log "Resuming from $LAST"

        if [[ ! -d "$LEGAL_TOK" || $(find "$LEGAL_TOK" -name "*.npy" | wc -l) -eq 0 ]]; then
            log "Legal corpus not tokenised — run phase 0 first"
            exit 1
        fi

        log "Training large legal DAPT → $CKPT_LARGE2"
        "$PYTHON" scripts/launch_axion_training.py \
            --data-dir "$LEGAL_TOK" \
            --preset large \
            --batch-size 16 --grad-accum 16 \
            --lr 5e-5 --steps 10000 \
            --threads $THREADS \
            $OPT_FLAGS \
            --output "$CKPT_LARGE2" \
            --resume "$RESUME_FROM"

        log "Soup large legal…"
        "$PYTHON" scripts/soup_checkpoints.py "$CKPT_LARGE2" --n 3
    fi
    ok "Phase 4 done"
fi

# ── Phase 5: Summary ──────────────────────────────────────────────────────────

if run_phase 5; then
    log "═══ Phase 5: Final model inventory ═══"
    echo ""
    echo "  Model                    Params  Checkpoint"
    echo "  ─────────────────────────────────────────────────────────"
    for entry in \
        "Small-34M-phase1-soup|34M|${CKPT_SMALL1}/soup_uniform.pkl" \
        "Small-34M-phase2-soup|34M|${CKPT_SMALL2}/soup_uniform.pkl" \
        "Medium-114M-soup|114M|${CKPT_MEDIUM}/soup_uniform.pkl" \
        "Large-474M-general-soup|474M|${CKPT_LARGE1}/soup_uniform.pkl" \
        "Large-474M-legal-soup|474M|${CKPT_LARGE2}/soup_uniform.pkl"
    do
        name=$(echo "$entry" | cut -d'|' -f1)
        params=$(echo "$entry" | cut -d'|' -f2)
        path=$(echo "$entry" | cut -d'|' -f3)
        if [[ -f "$path" ]]; then
            size=$(du -sh "$path" 2>/dev/null | cut -f1)
            ok "  %-28s %-6s %s (%s)" "$name" "$params" "$path" "$size"
        else
            echo -e "  ${YELLOW}MISSING${NC}  $name → $path"
        fi
    done
    echo ""
    log "Next steps:"
    echo "  • Distillation large→medium→small: scripts/distil.py (pending)"
    echo "  • LoRA fine-tuning per legal specialty (penal/civil/laboral/…)"
    echo "  • Instruction SFT + DPO alignment"
fi

log "Pipeline complete."

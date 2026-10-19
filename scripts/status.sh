#!/usr/bin/env bash
# Capibara — estado de todos los procesos y artefactos
# Uso: bash scripts/status.sh

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'

echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Capibara Legal — Estado  $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"

# ── Sesiones tmux ─────────────────────────────────────────────────────────────
echo -e "\n${CYAN}[ Sesiones tmux ]${NC}"
for SESSION in small medium large legal distil lora; do
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        LAST=$(tmux capture-pane -t "$SESSION" -p 2>/dev/null | grep -v '^$' | tail -1)
        printf "  ${GREEN}●${NC} %-8s %s\n" "$SESSION" "$LAST"
    else
        printf "  ${RED}○${NC} %-8s (sin sesión)\n" "$SESSION"
    fi
done

# ── Entrenamientos — último checkpoint ────────────────────────────────────────
echo -e "\n${CYAN}[ Checkpoints de entrenamiento ]${NC}"
for entry in \
    "Small-ph2|checkpoints/axion_small_phase2" \
    "Medium   |checkpoints/axion_medium_bf16" \
    "Large-ph1|checkpoints/axion_large_phase1" \
    "Large-leg|checkpoints/axion_large_legal" \
    "Dist-Med |checkpoints/distil_medium_legal" \
    "Dist-Sml |checkpoints/distil_small_legal" \
    "Cerebro  |checkpoints/distil_cerebro"
do
    NAME=$(echo "$entry" | cut -d'|' -f1)
    DIR=$(echo "$entry"  | cut -d'|' -f2)
    SOUP="${DIR}/soup_uniform.pkl"
    LAST=$(ls -1 "${DIR}/ckpt_step_"*.pkl 2>/dev/null | sort | tail -1)
    STEP=$(basename "$LAST" 2>/dev/null | grep -o '[0-9]*' | head -1 | sed 's/^0*//')

    if [[ -f "$SOUP" ]]; then
        SIZE=$(du -sh "$SOUP" 2>/dev/null | cut -f1)
        printf "  ${GREEN}✓${NC} %-10s soup listo (%s)\n" "$NAME" "$SIZE"
    elif [[ -n "$LAST" ]]; then
        printf "  ${YELLOW}⏳${NC} %-10s en curso — último paso %s\n" "$NAME" "${STEP:-?}"
    else
        printf "  ${RED}·${NC} %-10s pendiente\n" "$NAME"
    fi
done

# ── LoRA adapters ──────────────────────────────────────────────────────────────
echo -e "\n${CYAN}[ LoRA adapters ]${NC}"
TOTAL=0; LISTO=0
for ADAPTER in penal civil laboral constitucional administrativo mercantil \
               resumen herramientas instruccion qa extraccion \
               redaccion dialogo razonamiento traduccion; do
    TOTAL=$((TOTAL + 1))
    FILE="checkpoints/lora/large_${ADAPTER}/lora_final.pkl"
    if [[ -f "$FILE" ]]; then
        SIZE=$(du -sh "$FILE" 2>/dev/null | cut -f1)
        printf "  ${GREEN}✓${NC} %-20s (%s)\n" "$ADAPTER" "$SIZE"
        LISTO=$((LISTO + 1))
    else
        printf "  ${RED}·${NC} %-20s pendiente\n" "$ADAPTER"
    fi
done
echo -e "  → ${LISTO}/${TOTAL} adaptadores listos"

# ── Datos ──────────────────────────────────────────────────────────────────────
echo -e "\n${CYAN}[ Datos ]${NC}"
for entry in \
    "Corpus legal tok|data/tokenized/legal" \
    "RAG index       |data/rag_index/meta.json" \
    "FT summarization|data/finetune/summarization.jsonl" \
    "FT instruccion  |data/finetune/instruccion.jsonl" \
    "FT herramientas |data/finetune/herramientas.jsonl"
do
    NAME=$(echo "$entry" | cut -d'|' -f1)
    PATH_=$(echo "$entry" | cut -d'|' -f2)
    if [[ -f "$PATH_" || -d "$PATH_" ]]; then
        SIZE=$(du -sh "$PATH_" 2>/dev/null | cut -f1)
        printf "  ${GREEN}✓${NC} %-20s (%s)\n" "$NAME" "$SIZE"
    else
        printf "  ${RED}·${NC} %-20s pendiente\n" "$NAME"
    fi
done

# ── Disco ──────────────────────────────────────────────────────────────────────
echo -e "\n${CYAN}[ Disco ]${NC}"
df -h . | awk 'NR==2 {printf "  Usado: %s / %s  (%s libre)\n", $3, $2, $4}'

echo -e "\n${BOLD}═══════════════════════════════════════════════════════${NC}"

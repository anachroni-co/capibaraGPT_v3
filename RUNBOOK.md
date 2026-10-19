# Capibara Legal — Runbook de ejecución

Guía de comandos ordenada por fase. Ejecutar desde la raíz del repo en la VM
(`/home/elect/live/capibara-slim`) con el venv activado:

```bash
source ~/.venv/capibara/bin/activate
cd /home/elect/live/capibara-slim
```

---

## 0 · Preparación inicial (una sola vez)

```bash
# Instalar dependencias del sistema
sudo apt-get update && sudo apt-get install -y tmux git

# Crear venv e instalar paquetes Python
python3 -m venv ~/.venv/capibara
source ~/.venv/capibara/bin/activate
pip install jax[cpu] flax optax datasets huggingface_hub \
            sentence-transformers faiss-cpu duckduckgo-search \
            pdfminer.six pypdf

# Crear sesiones tmux (una por entrenamiento largo)
tmux new-session -d -s small   "bash"
tmux new-session -d -s medium  "bash"
tmux new-session -d -s large   "bash"
tmux new-session -d -s legal   "bash"
tmux new-session -d -s distil  "bash"
tmux new-session -d -s lora    "bash"
```

---

## 1 · Corpus legal (datos de entrenamiento)

### 1a — Descargar corpus legal
```bash
# ← ya hecho en sesión 'legal'
# Si necesitas relanzar:
tmux send-keys -t legal "python scripts/download_legal_corpus.py \
    --source all \
    --output data/raw/legal/ \
    --year-from 2000" Enter
```

### 1b — Tokenizar corpus legal  ← EJECUTAR AHORA
```bash
tmux send-keys -t legal "python scripts/prepare_corpus.py \
    --input  data/raw/legal/ \
    --output data/tokenized/legal/ \
    --extensions .md .txt .adoc" Enter
```

### 1c — Construir índice RAG  (después de 1b)
```bash
tmux send-keys -t legal "python scripts/rag_indexer.py \
    --input-dir data/raw/legal/ \
    --output    data/rag_index/ \
    --chunk-size 300 \
    --chunk-overlap 50" Enter
```

---

## 2 · Datos de fine-tuning (LoRA)

```bash
# Resúmenes (DACSA, MLSUM, arXiv, PQAI)
python scripts/download_summarization_data.py \
    --source all \
    --output data/raw/summarization/ \
    --finetune-dir data/finetune/ \
    --merge

# Habilidades básicas (instruccion, qa, extraccion, redaccion,
#                      dialogo, razonamiento, traduccion)
python scripts/download_instruction_data.py \
    --source all \
    --finetune-dir data/finetune/

# Uso de herramientas (sintético, ~1000 ejemplos)
python scripts/download_tool_data.py \
    --output data/finetune/herramientas.jsonl \
    --count 1000
```

---

## 3 · Entrenamiento base

### 3a — Small 34M fase 2  (reanuda desde soup de fase 1)
```bash
tmux send-keys -t small "python scripts/launch_axion_training.py \
    --data-dir data/tokenized/ \
    --preset   small \
    --batch-size 32 --grad-accum 8 \
    --lr 1e-4 --steps 10000 \
    --threads 32 \
    --dtype bf16 --grad-checkpoint \
    --compile-cache cache/jax_compile \
    --n-devices 4 \
    --output checkpoints/axion_small_phase2 \
    --resume checkpoints/axion_mixed/soup_uniform.pkl" Enter
```

Cuando termine:
```bash
python scripts/soup_checkpoints.py checkpoints/axion_small_phase2 --n 3
```

### 3b — Medium 114M
```bash
tmux send-keys -t medium "python scripts/launch_axion_training.py \
    --data-dir data/tokenized/ \
    --preset   medium \
    --batch-size 16 --grad-accum 8 \
    --steps 10000 \
    --threads 32 \
    --dtype bf16 --grad-checkpoint \
    --compile-cache cache/jax_compile \
    --n-devices 4 \
    --output checkpoints/axion_medium_bf16" Enter
```

Cuando termine:
```bash
python scripts/soup_checkpoints.py checkpoints/axion_medium_bf16 --n 3
```

### 3c — Large 474M fase 1 (preentrenamiento general, ~6 días)
```bash
tmux send-keys -t large "python scripts/launch_axion_training.py \
    --data-dir data/tokenized/ \
    --preset   large \
    --batch-size 16 --grad-accum 16 \
    --steps 35000 \
    --threads 32 \
    --dtype bf16 --grad-checkpoint \
    --compile-cache cache/jax_compile \
    --n-devices 4 \
    --output checkpoints/axion_large_phase1" Enter
```

Cuando termine:
```bash
python scripts/soup_checkpoints.py checkpoints/axion_large_phase1 --n 3
```

### 3d — Large 474M fase 2 — DAPT legal (~2 días)
```bash
tmux send-keys -t large "python scripts/launch_axion_training.py \
    --data-dir data/tokenized/legal/ \
    --preset   large \
    --batch-size 16 --grad-accum 16 \
    --lr 5e-5 --steps 10000 \
    --threads 32 \
    --dtype bf16 --grad-checkpoint \
    --compile-cache cache/jax_compile \
    --n-devices 4 \
    --output checkpoints/axion_large_legal \
    --resume checkpoints/axion_large_phase1/soup_uniform.pkl" Enter
```

Cuando termine:
```bash
python scripts/soup_checkpoints.py checkpoints/axion_large_legal --n 3
```

---

## 4 · Destilación

### 4a — Large → Medium  (~10 h)
```bash
tmux send-keys -t distil "python scripts/distil.py \
    --teacher         checkpoints/axion_large_legal/soup_uniform.pkl \
    --teacher-preset  large \
    --student-preset  medium \
    --student-resume  checkpoints/axion_medium_bf16/soup_uniform.pkl \
    --data-dir        data/tokenized/legal/ \
    --output          checkpoints/distil_medium_legal \
    --steps 10000 --batch-size 16 --grad-accum 8 \
    --temperature 4.0 --alpha 0.7 \
    --dtype bf16 --threads 32" Enter

# Cuando termine:
python scripts/soup_checkpoints.py checkpoints/distil_medium_legal --n 3
```

### 4b — Medium destilado → Small (cadena)  (~6 h)
```bash
python scripts/distil.py \
    --teacher         checkpoints/distil_medium_legal/soup_uniform.pkl \
    --teacher-preset  medium \
    --student-preset  small \
    --student-resume  checkpoints/axion_small_phase2/soup_uniform.pkl \
    --data-dir        data/tokenized/legal/ \
    --output          checkpoints/distil_small_legal \
    --steps 10000 --batch-size 32 --grad-accum 8 \
    --temperature 4.0 --alpha 0.7 \
    --dtype bf16 --threads 32

python scripts/soup_checkpoints.py checkpoints/distil_small_legal --n 3
```

### 4c — Large → Small directo / Cerebro  (~8 h, mejor alineación)
```bash
python scripts/distil.py \
    --teacher         checkpoints/axion_large_legal/soup_uniform.pkl \
    --teacher-preset  large \
    --student-preset  small \
    --student-resume  checkpoints/axion_small_phase2/soup_uniform.pkl \
    --data-dir        data/tokenized/legal/ \
    --output          checkpoints/distil_cerebro \
    --steps 10000 --batch-size 32 --grad-accum 8 \
    --temperature 4.0 --alpha 0.7 \
    --dtype bf16 --threads 32

python scripts/soup_checkpoints.py checkpoints/distil_cerebro --n 3
```

---

## 5 · LoRA — Adaptadores de especialidad legal

```bash
BASE="checkpoints/axion_large_legal/soup_uniform.pkl"

for SPECIALTY in penal civil laboral constitucional administrativo mercantil; do
    tmux send-keys -t lora "python scripts/lora_finetune.py \
        --base-ckpt  $BASE \
        --preset     large \
        --data       data/finetune/legal_qa.jsonl \
        --specialty  $SPECIALTY \
        --output     checkpoints/lora/large_${SPECIALTY} \
        --steps 2000 --batch-size 4 \
        --rank 16 --lora-alpha 32 \
        --dtype bf16 --threads 32" Enter
    # Esperar a que termine antes de lanzar el siguiente
done
```

---

## 6 · LoRA — Habilidades básicas

```bash
BASE="checkpoints/axion_large_legal/soup_uniform.pkl"

# Resumen
python scripts/lora_finetune.py \
    --base-ckpt $BASE --preset large \
    --data      data/finetune/summarization.jsonl \
    --specialty resumen \
    --output    checkpoints/lora/large_resumen \
    --steps 2000 --batch-size 4 --rank 16 --lora-alpha 32 \
    --dtype bf16 --threads 32

# Herramientas (tool use)
python scripts/lora_finetune.py \
    --base-ckpt $BASE --preset large \
    --data      data/finetune/herramientas.jsonl \
    --specialty herramientas \
    --output    checkpoints/lora/large_herramientas \
    --steps 2000 --batch-size 4 --rank 16 --lora-alpha 32 \
    --dtype bf16 --threads 32

# Resto de habilidades (instruccion, qa, extraccion, redaccion,
#                       dialogo, razonamiento, traduccion)
for SKILL in instruccion qa extraccion redaccion dialogo razonamiento traduccion; do
    python scripts/lora_finetune.py \
        --base-ckpt $BASE --preset large \
        --data      data/finetune/${SKILL}.jsonl \
        --specialty $SKILL \
        --output    checkpoints/lora/large_${SKILL} \
        --steps 2000 --batch-size 4 --rank 16 --lora-alpha 32 \
        --dtype bf16 --threads 32
done
```

---

## 7 · Pipeline completo automático (alternativa)

En lugar de los pasos manuales anteriores, puedes lanzar todo de golpe:

```bash
# Todo desde fase 0 hasta fase 7
bash scripts/run_full_training.sh

# Desde una fase concreta (ej: empezar en destilación)
bash scripts/run_full_training.sh 6

# Rango de fases (ej: solo LoRA)
bash scripts/run_full_training.sh 7 7
```

---

## 8 · Inferencia

### Interactivo (CLI)
```bash
python scripts/speculative_inference.py \
    --cerebro  checkpoints/distil_cerebro/soup_uniform.pkl \
    --medium   checkpoints/distil_medium_legal/soup_uniform.pkl \
    --large    checkpoints/axion_large_legal/soup_uniform.pkl \
    --lora-dir checkpoints/lora/ \
    --rag-index data/rag_index/ \
    --tools \
    --dtype bf16 --threads 32
```

### Servidor HTTP
```bash
python scripts/speculative_inference.py \
    --cerebro  checkpoints/distil_cerebro/soup_uniform.pkl \
    --medium   checkpoints/distil_medium_legal/soup_uniform.pkl \
    --large    checkpoints/axion_large_legal/soup_uniform.pkl \
    --lora-dir checkpoints/lora/ \
    --rag-index data/rag_index/ \
    --tools \
    --serve --port 8080 \
    --dtype bf16 --threads 32 &

# Test
curl -s -X POST http://localhost:8080/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "¿Qué dice el artículo 248 del Código Penal?", "max_tokens": 200}'

# Health check
curl http://localhost:8080/health
```

### Servidor MCP (para Claude Code)
```bash
# Arrancar el servidor MCP
python scripts/mcp_server.py \
    --rag-index data/rag_index/

# El servidor imprime en stderr el snippet para .claude/settings.json:
# {
#   "mcpServers": {
#     "capibara-legal": {
#       "command": "python",
#       "args": ["scripts/mcp_server.py", "--rag-index", "data/rag_index/"]
#     }
#   }
# }
```

### Benchmark de rendimiento
```bash
python scripts/speculative_inference.py \
    --cerebro  checkpoints/distil_cerebro/soup_uniform.pkl \
    --medium   checkpoints/distil_medium_legal/soup_uniform.pkl \
    --large    checkpoints/axion_large_legal/soup_uniform.pkl \
    --lora-dir checkpoints/lora/ \
    --benchmark \
    --bench-tokens 128 --bench-runs 3
```

---

## 9 · Monitoreo de sesiones tmux

```bash
# Ver todas las sesiones activas
tmux ls

# Entrar a una sesión
tmux attach -t small     # entrenamiento small
tmux attach -t medium    # entrenamiento medium
tmux attach -t large     # entrenamiento large
tmux attach -t legal     # corpus / rag
tmux attach -t distil    # destilación
tmux attach -t lora      # LoRA

# Ver las últimas líneas sin entrar (útil para supervisar rápido)
tmux capture-pane -t small  -p | tail -5
tmux capture-pane -t medium -p | tail -5
tmux capture-pane -t large  -p | tail -5
tmux capture-pane -t legal  -p | tail -5

# Ver todas de golpe
for s in small medium large legal distil lora; do
    echo "=== $s ==="; tmux capture-pane -t $s -p 2>/dev/null | tail -3; done

# Salir de una sesión sin matarla: Ctrl+B, luego D
```

---

## 10 · Inventario de artefactos

```bash
# Ver qué hay listo (tamaño en disco)
for f in \
    "checkpoints/axion_small_phase2/soup_uniform.pkl" \
    "checkpoints/axion_medium_bf16/soup_uniform.pkl" \
    "checkpoints/axion_large_phase1/soup_uniform.pkl" \
    "checkpoints/axion_large_legal/soup_uniform.pkl" \
    "checkpoints/distil_medium_legal/soup_uniform.pkl" \
    "checkpoints/distil_small_legal/soup_uniform.pkl" \
    "checkpoints/distil_cerebro/soup_uniform.pkl" \
    "checkpoints/lora/large_penal/lora_final.pkl" \
    "checkpoints/lora/large_civil/lora_final.pkl" \
    "checkpoints/lora/large_laboral/lora_final.pkl" \
    "checkpoints/lora/large_constitucional/lora_final.pkl" \
    "checkpoints/lora/large_administrativo/lora_final.pkl" \
    "checkpoints/lora/large_mercantil/lora_final.pkl" \
    "checkpoints/lora/large_resumen/lora_final.pkl" \
    "checkpoints/lora/large_herramientas/lora_final.pkl" \
    "checkpoints/lora/large_instruccion/lora_final.pkl" \
    "checkpoints/lora/large_qa/lora_final.pkl" \
    "checkpoints/lora/large_extraccion/lora_final.pkl" \
    "checkpoints/lora/large_redaccion/lora_final.pkl" \
    "data/rag_index/meta.json"; do
    if [ -f "$f" ]; then
        size=$(du -sh "$f" | cut -f1)
        echo "  ✓  $f  ($size)"
    else
        echo "  ·  $f  (pendiente)"
    fi
done
```

---

## 11 · Estado actual (2025-05-08)

| Fase | Estado | Detalle |
|------|--------|---------|
| Corpus legal — descarga | ✅ Hecho | `data/raw/legal/` |
| Corpus legal — tokenización | ✅ Hecho | 24.511 ficheros · 1.684B tokens · 166 shards |
| Índice RAG | 🔄 En curso | `rag_indexer.py` corriendo en sesión `legal` |
| Datos fine-tuning (summ/instruct/tools) | ⏳ Pendiente | §2 cuando RAG termine |
| Small 34M fase 2 | 🔄 En curso | ~8.040/15.000 pasos |
| Medium 114M | 🔄 En curso | ~1.010/10.000 pasos |
| Large 474M fase 1 | ⏳ Pendiente | Tras medium |
| Large 474M DAPT legal | ⏳ Pendiente | Tras Large fase 1 |
| Destilación (×3) | ⏳ Pendiente | Tras Large DAPT |
| LoRA adapters (×15) | ⏳ Pendiente | Tras destilación |

### Corpus legal — números clave
```
Ficheros procesados : 24.511
Tokens totales      : 1.684.578.451  (1.684B)
Shards              : 166
Ejemplos (seq=1024) : ~1.644.120
Cobertura DAPT      : ~1.56× el corpus con 10k pasos · batch efectivo 256
```

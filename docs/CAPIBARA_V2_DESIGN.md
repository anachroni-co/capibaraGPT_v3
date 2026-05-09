# Capibara Legal V2 — Diseño y Plan de Implementación

**Estado**: Documento de planificación. Ejecutar cuando V1 haya terminado.  
**Prerequisito**: V1 en producción con benchmarks de referencia documentados.  
**Estimación total V2**: ~16–18 días de entrenamiento + ~2 días de integración.

---

## Resumen ejecutivo

V2 introduce seis mejoras que, combinadas, producen un salto cualitativo
significativo respecto a V1:

| Mejora | Impacto principal | Coste |
|--------|-------------------|-------|
| BPE tokenizer 32k | 4× más contexto útil por ventana | Reentrenar todo |
| Think-anywhere tags | Razonamiento estructurado interno | Datos sintéticos nuevos |
| seq\_len 2048 | Documentos legales completos en contexto | ~2× memoria por step |
| CrewAI multi-agent | Flujos de trabajo legales complejos | Wrapper + config |
| LoRA merging | Adapters combinados sin reentrenar | Script nuevo |
| Multi-seed soup en destilación | Student más robusto y generalizable | 3× tiempo destilación |

Las mejoras 1 y 3 están acopladas (BPE habilita seq\_len largo con sentido).
Las mejoras 2, 5 y 6 son independientes y pueden hacerse en paralelo.
La mejora 4 requiere que 1 y 3 estén terminadas.

---

## Mejora 1 — Tokenizador BPE 32k

### Por qué

Con el tokenizador byte-level de V1 (vocab=512), cada carácter español ocupa 2 bytes
en UTF-8, por lo que 1024 tokens ≈ 200–250 palabras. Con BPE entrenado sobre el corpus
legal español, 1024 tokens ≈ 700–900 palabras — casi 4× más texto legal en la misma
ventana, sin cambiar la arquitectura del modelo.

Términos legales largos como "contencioso-administrativo" pasan de ~25 tokens a 2–3.
Esto reduce la fragmentación y mejora la representación de terminología especializada.

### Estado

`scripts/train_tokenizer.py` ya existe y está preparado.
Entrenamiento en background recomendado durante V1 (4 threads, sin impacto):

```bash
# Lanzar ahora (si no está corriendo ya)
tmux new-session -d -s tokenizer "bash"
tmux send-keys -t tokenizer "
  source ~/.venv/capibara/bin/activate && \
  python scripts/train_tokenizer.py train \
      --input-dir data/raw/legal/ \
      --output    tokenizer/ \
      --vocab-size 32000 \
      --max-lines  5000000 \
      --num-threads 4
" Enter

# Verificar cuando termine (1-3 horas)
python scripts/train_tokenizer.py test tokenizer/
```

### Integración en el pipeline V2

`prepare_corpus.py` y `launch_axion_training.py` ya aceptan `--tokenizer`.
Los demás scripts (`distil.py`, `lora_finetune.py`) necesitan el mismo parámetro añadido
siguiendo el mismo patrón (ver commits de V1).

Re-tokenizar corpus con BPE:

```bash
# General (mixed)
python scripts/prepare_corpus.py \
    --input     data/raw/ \
    --output    data/tokenized_bpe/ \
    --tokenizer tokenizer/capibara_legal.model

# Legal DAPT
python scripts/prepare_corpus.py \
    --input     data/raw/legal/ \
    --output    data/tokenized_bpe/legal/ \
    --tokenizer tokenizer/capibara_legal.model
```

### Cambios en arquitectura del modelo

`vocab_size`: 512 → 32 000. El embedding table crece ~62×, pero representa
una fracción pequeña del total de parámetros en modelos ≥34M:

| Modelo | Emb V1 | Emb V2 | Params total V2 |
|--------|--------|--------|-----------------|
| Small  | 0.1M   | 8.2M   | ~42M (+8M)      |
| Medium | 0.4M   | 24.6M  | ~138M (+24M)    |
| Large  | 0.7M   | 40.9M  | ~515M (+40M)    |

El aumento en parámetros de embedding se compensa con mejor calidad de representación.

---

## Mejora 2 — Think-Anywhere Tags

### Por qué fue descartado en V1

Con byte-level, `<think>` ocupa 7 tokens (bytes literales del string UTF-8).
Dentro de un contexto de 1024 tokens, ese overhead hace que los bloques de
razonamiento consuman demasiado espacio y rompan la eficiencia del speculative decoding.

### Por qué funciona en V2

Con BPE y tokens especiales reservados:
- `<think>` → ID 8 (1 token)
- `</think>` → ID 9 (1 token)

Un bloque de razonamiento de 50 tokens cuesta 52 tokens totales — overhead del 4%.
El modelo aprende a usar `<think>` para razonamiento interno y `</think>` para volver
al output final visible al usuario.

### Arquitectura

```
Query: "¿Puede un menor de 16 años firmar un contrato?"
                    ↓
[MODELO GENERA]
<think>
El artículo 1263 CC establece que los menores no emancipados no pueden
prestar consentimiento. La emancipación se produce a los 16 años por
varios supuestos (art. 314 CC). Debo distinguir si está emancipado.
</think>
Los menores de 16 años no emancipados no pueden firmar contratos válidos
según el artículo 1263 del Código Civil...
                    ↓
[INFERENCE STRIPPING]
Output visible: "Los menores de 16 años no emancipados..."
```

El bloque `<think>...</think>` se genera normalmente (Cerebro drafta, Medium/Large
verifican) pero se elimina del output final antes de devolver al usuario.

### Datos de entrenamiento

Se necesitan ejemplos de fine-tuning con think traces para las especialidades
que más se benefician:

**razonamiento** (adaptador existente, extender con think traces):
```jsonl
{"prompt": "Razona paso a paso: ...", "completion": "<think>...\n</think>\n..."}
```

**penal**, **civil**, **laboral** (casos legales complejos con cadena de razonamiento):
```jsonl
{"prompt": "Analiza si existe delito en el siguiente caso: ...",
 "completion": "<think>Elementos del tipo: ...\nHechos: ...\nSubsunción: ...</think>\nSí, concurren los elementos..."}
```

Script nuevo a crear: `scripts/download_think_data.py`
- Genera ~2000 ejemplos sintéticos con think traces por especialidad legal
- Usa plantillas de razonamiento jurídico (análisis de hechos → norma → subsunción)
- Ejemplos de IRAC (Issue, Rule, Application, Conclusion) en español

### Cambios en `speculative_inference.py`

```python
# En _strip_markers: eliminar bloques <think>...</think> del output
import re
def _strip_think_blocks(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

# En generate(): aplicar antes de devolver
output = self._strip_think_blocks(self._strip_markers(raw_output))
```

El speculative decoding no necesita cambios — los tokens `<think>`/`</think>` se
tratan exactamente igual que cualquier otro token durante el draft/verify loop.

---

## Mejora 3 — seq\_len 2048

### Por qué

Con BPE, seq\_len=2048 cubre ~1400–1800 palabras — suficiente para:
- Un artículo legal completo con su exposición de motivos
- Una sentencia corta completa
- Un contrato de 2–3 páginas
- Contexto RAG expandido: 5–8 artículos relacionados en lugar de 3 fragmentos

### Implicaciones de memoria y velocidad

La atención es O(n²) en secuencia: pasar de 1024 a 2048 tokens cuadruplica
la memoria de activaciones de atención.

| Preset | seq V1 | seq V2 | Δ memoria | Batch V2 | Throughput est. |
|--------|--------|--------|-----------|----------|-----------------|
| Small  | 512    | 1024   | ×4        | 16       | ~2000 tok/s     |
| Medium | 1024   | 2048   | ×4        | 8        | ~700 tok/s      |
| Large  | 1024   | 2048   | ×4        | 4        | ~150 tok/s      |

Gradient checkpointing obligatorio en Medium y Large con seq=2048.

Estimación de tiempo de entrenamiento V2 con estos parámetros:

| Fase | Steps | Tiempo est. |
|------|-------|-------------|
| Small 42M fase 1    | 15 000 | ~21 h  |
| Medium 138M         | 10 000 | ~40 h  |
| Large 515M fase 1   | 35 000 | ~14 d  |
| Large 515M DAPT     | 10 000 | ~4 d   |
| Distilación ×3      | 10 000 | ~3 d   |
| LoRA ×15            | 2 000  | ~1 d   |

Large fase 1 pasa de ~6 días a ~14 días con seq=2048. Si el tiempo es un
constraint, se puede mantener seq=1024 solo para Large y usar 2048 en Small/Medium.

### Comandos V2 con seq 2048

```bash
# Medium V2
python scripts/launch_axion_training.py \
    --data-dir  data/tokenized_bpe/ \
    --tokenizer tokenizer/capibara_legal.model \
    --preset    medium \
    --seq-len   2048 \
    --batch-size 8 --grad-accum 16 \
    --steps 10000 --dtype bf16 --grad-checkpoint \
    --output checkpoints/v2/axion_medium

# Large V2 fase 1
python scripts/launch_axion_training.py \
    --data-dir  data/tokenized_bpe/ \
    --tokenizer tokenizer/capibara_legal.model \
    --preset    large \
    --seq-len   2048 \
    --batch-size 4 --grad-accum 32 \
    --steps 35000 --dtype bf16 --grad-checkpoint \
    --output checkpoints/v2/axion_large_phase1
```

### RAG con contexto expandido

Con 2048 tokens disponibles y ~600 dedicados a query+respuesta,
el contexto RAG puede crecer hasta ~1400 tokens ≈ 4–5 artículos completos:

```python
# En rag_retriever.py — parámetros V2
RAGRetriever(index_dir, top_k=5, max_context_bytes=1400)
```

---

## Mejora 4 — CrewAI Multi-Agent

### Prerequisito

CrewAI requiere una API compatible con OpenAI (`/v1/chat/completions`).
El servidor HTTP de V1 expone `POST /generate` con formato propio.
Solución: wrapper de traducción de protocolo.

### Script nuevo: `scripts/openai_wrapper.py`

Servidor Flask/Werkzeug que traduce el formato OpenAI → formato Capibara:

```
POST /v1/chat/completions
  {"model": "capibara-legal", "messages": [...], "max_tokens": 256}
        ↓ wrapper
POST http://localhost:8080/generate
  {"prompt": "<bos>Sistema: ...\nUsuario: ...\nAsistente:", "max_tokens": 256}
        ↓ respuesta
{"choices": [{"message": {"role": "assistant", "content": "..."}}], ...}
```

El wrapper también:
- Convierte historial de mensajes a formato de prompt Capibara
- Mapea `stop` sequences
- Expone `GET /v1/models` con `{"data": [{"id": "capibara-legal"}]}`

### Arquitectura CrewAI

Tres agentes especializados para flujos de trabajo legales complejos:

```python
from crewai import Agent, Task, Crew, LLM

capibara = LLM(
    model="capibara-legal",
    base_url="http://localhost:8081/v1",   # openai_wrapper.py
    api_key="none",
)

investigador = Agent(
    role="Investigador Legal",
    goal="Buscar jurisprudencia y legislación relevante",
    backstory="Especialista en búsqueda legal con acceso a BOE y CENDOJ",
    llm=capibara,
    tools=[search_boe_tool, search_cendoj_tool],
)

analista = Agent(
    role="Analista Jurídico",
    goal="Analizar aplicabilidad de normas al caso concreto",
    backstory="Jurista experto en interpretación normativa",
    llm=capibara,
)

redactor = Agent(
    role="Redactor Legal",
    goal="Redactar documentos jurídicos precisos y completos",
    backstory="Especialista en redacción de escritos judiciales",
    llm=capibara,
    tools=[calculate_plazo_tool],
)

crew = Crew(
    agents=[investigador, analista, redactor],
    tasks=[task_investigar, task_analizar, task_redactar],
    verbose=True,
)
```

Casos de uso iniciales:
1. **Análisis de viabilidad** — investigador busca jurisprudencia, analista valora el caso, redactor genera informe
2. **Redacción de escritos** — investigador busca normativa, redactor genera demanda/recurso con citas reales
3. **Due diligence** — múltiples investigadores en paralelo por área (penal, civil, mercantil)

### Archivos a crear

- `scripts/openai_wrapper.py` — servidor de traducción de protocolo
- `scripts/crewai_legal.py` — definición de agentes y tareas ejemplo
- `examples/crew_analisis_caso.py` — ejemplo de análisis de caso completo
- `examples/crew_redaccion_demanda.py` — ejemplo de redacción automatizada

---

## Mejora 5 — LoRA Merging

### Qué es y por qué

Los 15 adapters LoRA de V1 son independientes: en cada query el router activa
exactamente uno. LoRA merging permite combinar varios adapters en uno solo
mediante media ponderada de las matrices A y B, creando adapters multi-habilidad
sin reentrenar.

**Casos de uso:**

| Adapter merged | Composición | Uso |
|----------------|-------------|-----|
| `legal_completo` | penal + civil + laboral + constitucional + administrativo + mercantil (α=1/6 cada uno) | Adapter genérico legal — backup cuando el router no tiene señal clara |
| `razonamiento_legal` | razonamiento + penal/civil/laboral (α=0.5 / α=0.5/3) | Análisis legal con cadena de pensamiento |
| `asistente_legal` | instruccion + dialogo + redaccion (α=1/3 cada uno) | Interfaz conversacional general |
| `investigacion` | herramientas + qa + extraccion (α=1/3 cada uno) | Búsqueda + extracción de información |

### Script: `scripts/merge_loras.py`

```python
def merge_loras(
    adapters: list[tuple[str, float]],  # [(path, weight), ...]
    output: str,
) -> None:
    """
    Weighted average of LoRA A and B matrices across adapters.
    Weights are normalized to sum to 1.0.

    Works because LoRA reparametrizes W' = W + α/r · BA:
    merged BA = Σ wᵢ · αᵢ/rᵢ · BᵢAᵢ  (weighted sum of low-rank updates)
    """
    ...
```

Uso:
```bash
# Crear adapter legal completo
python scripts/merge_loras.py \
    --adapters \
        checkpoints/lora/large_penal/lora_final.pkl:1 \
        checkpoints/lora/large_civil/lora_final.pkl:1 \
        checkpoints/lora/large_laboral/lora_final.pkl:1 \
        checkpoints/lora/large_constitucional/lora_final.pkl:1 \
        checkpoints/lora/large_administrativo/lora_final.pkl:1 \
        checkpoints/lora/large_mercantil/lora_final.pkl:1 \
    --output checkpoints/lora/large_legal_completo/lora_final.pkl

# Crear adapter razonamiento legal (pesos manuales)
python scripts/merge_loras.py \
    --adapters \
        checkpoints/lora/large_razonamiento/lora_final.pkl:3 \
        checkpoints/lora/large_penal/lora_final.pkl:1 \
        checkpoints/lora/large_civil/lora_final.pkl:1 \
    --output checkpoints/lora/large_razonamiento_legal/lora_final.pkl
```

### Evaluación de merging

Antes de producción, evaluar los adapters merged contra los individuales
en un conjunto de test por especialidad. El merging puede degradar ligeramente
la especialización — es un tradeoff calidad vs versatilidad.

---

## Mejora 6 — Multi-seed soup en destilación

### Por qué funciona especialmente bien en destilación

V1 ya aplica `soup_checkpoints.py` al final de cada destilación (promedia los
últimos 3 checkpoints del mismo run). Eso es **intra-run soup**. La mejora de V2
añade **inter-run soup**: lanzar la destilación 3 veces con seeds distintos y
promediar los students finales.

La loss de destilación es KL-divergence con temperatura T=4.0 — una función
significativamente más suave que la cross-entropy de preentrenamiento. Esto produce
un paisaje de loss más plano donde runs distintos convergen a mínimos locales
cercanos pero no idénticos. El promedio de esos tres mínimos cae en una región
de mayor generalización que cualquiera individual.

```
Run seed=42  → student_42  (mínimo local A)
Run seed=123 → student_123 (mínimo local B)  ←  todos en la misma cuenca
Run seed=777 → student_777 (mínimo local C)
              ↓
         soup_multiseed    (centro del triángulo A-B-C)
              ↓
    mejor generalización que A, B o C por separado
```

Resultado esperado: mejora de ~2–4% en perplexity sobre el test set legal
respecto al soup intra-run solo. El gain es mayor que en preentrenamiento
precisamente porque T=4.0 suaviza el paisaje.

### Comandos V2

Se aplica a las tres destilaciones (Large→Medium, Medium→Small, Large→Small/Cerebro):

```bash
# Ejemplo: destilación Large → Medium con 3 seeds
for SEED in 42 123 777; do
    tmux send-keys -t distil "python scripts/distil.py \
        --teacher         checkpoints/v2/axion_large_legal/soup_uniform.pkl \
        --teacher-preset  large \
        --student-preset  medium \
        --student-resume  checkpoints/v2/axion_medium/soup_uniform.pkl \
        --data-dir        data/tokenized_bpe/legal/ \
        --tokenizer       tokenizer/capibara_legal.model \
        --output          checkpoints/v2/distil_medium_s${SEED} \
        --seed            ${SEED} \
        --steps 10000 --batch-size 8 --grad-accum 8 \
        --temperature 4.0 --alpha 0.7 \
        --dtype bf16" Enter
    # Esperar a que termine antes del siguiente seed
done

# Intra-run soup de cada seed (los 3 últimos checkpoints de cada run)
for SEED in 42 123 777; do
    python scripts/soup_checkpoints.py checkpoints/v2/distil_medium_s${SEED} --n 3
done

# Inter-run soup: promediar los tres students finales
python scripts/soup_checkpoints.py \
    checkpoints/v2/distil_medium_s42/soup_uniform.pkl \
    checkpoints/v2/distil_medium_s123/soup_uniform.pkl \
    checkpoints/v2/distil_medium_s777/soup_uniform.pkl \
    --output checkpoints/v2/distil_medium_legal/soup_uniform.pkl
```

### Coste vs beneficio

| Variante | Tiempo | Gain esperado |
|----------|--------|---------------|
| V1: intra-run soup (3 ckpts) | 1× destilación | baseline |
| V2: intra-run × 3 seeds | 3× destilación | +2–4% perplexity |
| V2 opcional: 5 seeds | 5× destilación | +3–5% (rendimientos decrecientes) |

Con 3 seeds el ratio coste/beneficio es óptimo. Con 5 seeds el gain adicional
sobre 3 seeds es pequeño (~1%) y el coste es 67% mayor.

### Nota sobre `soup_checkpoints.py`

El script actual toma un directorio y promedia los últimos N checkpoints del mismo run.
Para el inter-run soup necesita aceptar paths explícitos. Añadir el flag `--files`:

```bash
# Forma actual (intra-run):
python scripts/soup_checkpoints.py checkpoints/distil_medium_s42 --n 3

# Forma nueva (inter-run, --files):
python scripts/soup_checkpoints.py \
    --files \
        checkpoints/v2/distil_medium_s42/soup_uniform.pkl \
        checkpoints/v2/distil_medium_s123/soup_uniform.pkl \
        checkpoints/v2/distil_medium_s777/soup_uniform.pkl \
    --output checkpoints/v2/distil_medium_legal/soup_uniform.pkl
```

---

## Plan de ejecución V2

### Fase 0 — Preparación (durante V1, paralelo)

```
[ ] Tokenizador BPE entrenando en sesión tmux 'tokenizer' (4 threads)
[ ] Diseñar y generar datos de think traces (scripts/download_think_data.py)
[ ] Implementar scripts/openai_wrapper.py
[ ] Implementar scripts/merge_loras.py
[ ] Añadir --tokenizer a distil.py y lora_finetune.py
[ ] Añadir flag --files a soup_checkpoints.py para inter-run soup
```

### Fase 1 — Corpus (Día 0–1 tras V1)

```bash
# Verificar tokenizador
python scripts/train_tokenizer.py test tokenizer/

# Re-tokenizar corpus general
python scripts/prepare_corpus.py \
    --input data/raw/ --output data/tokenized_bpe/ \
    --tokenizer tokenizer/capibara_legal.model

# Re-tokenizar corpus legal (DAPT)
python scripts/prepare_corpus.py \
    --input data/raw/legal/ --output data/tokenized_bpe/legal/ \
    --tokenizer tokenizer/capibara_legal.model

# Generar datos de fine-tuning con think traces
python scripts/download_think_data.py \
    --output data/finetune/think/ \
    --specialties penal civil laboral razonamiento
```

### Fase 2 — Entrenamiento base (Día 1–3)

```bash
# Small V2 (~21 h)
tmux send-keys -t small "python scripts/launch_axion_training.py \
    --data-dir data/tokenized_bpe/ \
    --tokenizer tokenizer/capibara_legal.model \
    --preset small --seq-len 1024 \
    --batch-size 16 --grad-accum 8 --steps 15000 \
    --dtype bf16 --grad-checkpoint \
    --output checkpoints/v2/axion_small" Enter

# Medium V2 (~40 h)
tmux send-keys -t medium "python scripts/launch_axion_training.py \
    --data-dir data/tokenized_bpe/ \
    --tokenizer tokenizer/capibara_legal.model \
    --preset medium --seq-len 2048 \
    --batch-size 8 --grad-accum 16 --steps 10000 \
    --dtype bf16 --grad-checkpoint \
    --output checkpoints/v2/axion_medium" Enter
```

### Fase 3 — Large V2 (Día 3–17)

```bash
# Large fase 1 general (~14 d con seq=2048 — evaluar si reducir a seq=1024 por tiempo)
tmux send-keys -t large "python scripts/launch_axion_training.py \
    --data-dir data/tokenized_bpe/ \
    --tokenizer tokenizer/capibara_legal.model \
    --preset large --seq-len 2048 \
    --batch-size 4 --grad-accum 32 --steps 35000 \
    --dtype bf16 --grad-checkpoint \
    --output checkpoints/v2/axion_large_phase1" Enter

# Large DAPT legal (~4 d)
# ... (tras soup de fase 1)
```

### Fase 4 — Destilación V2 con multi-seed soup (Día 17–26)

3 runs por cada destilación (seeds 42, 123, 777) + intra-run soup + inter-run soup.
Ver sección Mejora 6 para los comandos completos.

Orden de ejecución (secuencial por dependencias de teacher/student):
1. Large → Medium ×3 seeds → inter-run soup (~9 días)
2. Medium destilado → Small ×3 seeds → inter-run soup (~6 días, paralelo con Large→Small)
3. Large → Small/Cerebro ×3 seeds → inter-run soup (~6 días, paralelo con Medium→Small)

### Fase 5 — LoRA V2 con think traces (Día 20–21)

```bash
# Adapters legales con datos augmentados (pares normales + think traces)
for SPECIALTY in penal civil laboral constitucional administrativo mercantil; do
    python scripts/lora_finetune.py \
        --base-ckpt checkpoints/v2/axion_large_legal/soup_uniform.pkl \
        --tokenizer tokenizer/capibara_legal.model \
        --preset large \
        --data data/finetune/legal_qa.jsonl \
        --data-extra data/finetune/think/${SPECIALTY}.jsonl \
        --specialty $SPECIALTY \
        --output checkpoints/v2/lora/large_${SPECIALTY} \
        --steps 2000 --batch-size 4 --rank 16 --lora-alpha 32 \
        --dtype bf16
done
```

### Fase 6 — LoRA Merging y CrewAI (Día 21)

```bash
# Generar adapters merged
python scripts/merge_loras.py ...  # ver sección Mejora 5

# Arrancar wrapper OpenAI + servidor Capibara
python scripts/speculative_inference.py --serve --port 8080 ... &
python scripts/openai_wrapper.py --backend http://localhost:8080 --port 8081 &

# Test CrewAI
python examples/crew_analisis_caso.py
```

### Fase 7 — Benchmark V1 vs V2

Métricas a comparar:

| Métrica | V1 baseline | V2 objetivo |
|---------|-------------|-------------|
| Perplexity legal (test set) | — | < V1 |
| Tokens/s (Large+LoRA) | ~350 tok/s | ≥350 tok/s |
| Acceptance rate Medium | ~87% | ≥87% |
| Acceptance rate Large | ~99% | ≥99% |
| BLEU summarization | — | > V1 |
| Exact match QA legal | — | > V1 |
| Palabras en contexto (seq=2048) | ~220 | ~1500 |

---

## Decisión de trade-off: Large con seq=1024 vs 2048

Si 14 días de Large V2 es demasiado tiempo, opción intermedia:

- **Large V2 con seq=1024** (~6–7 días, igual que V1) + BPE tokenizer
  → Recupera 4× contexto útil sin coste adicional de tiempo
  → El seq largo (2048) solo para Small y Medium

- **Large V2 con seq=2048** (~14 días)
  → Máximo beneficio para documentos largos
  → Solo si el servidor puede estar ocupado ese tiempo

Recomendación: decidir en base a los benchmarks de V1 y la carga operativa
en el momento de iniciar V2.

---

## Archivos a crear para V2 (checklist)

```
[ ] scripts/download_think_data.py     — datos think traces para fine-tuning
[ ] scripts/merge_loras.py             — merging ponderado de adapters LoRA
[ ] scripts/openai_wrapper.py          — traducción protocolo OpenAI → Capibara
[ ] scripts/crewai_legal.py            — definición agentes CrewAI
[ ] examples/crew_analisis_caso.py     — ejemplo flujo multi-agent
[ ] examples/crew_redaccion_demanda.py — ejemplo redacción automática
```

Modificaciones a scripts existentes:
```
[ ] scripts/distil.py              — añadir --tokenizer, --seed
[ ] scripts/lora_finetune.py       — añadir --tokenizer, --data-extra
[ ] scripts/soup_checkpoints.py    — añadir --files para inter-run soup
[ ] scripts/speculative_inference.py — BPETokenizer en lugar de byte-level,
                                       _strip_think_blocks(), tool tokens por ID
[ ] scripts/run_full_training.sh   — pipeline V2 completo con 3 seeds por destilación
[ ] RUNBOOK.md                     — sección V2
```

---

## Notas de compatibilidad V1 → V2

- Los checkpoints V1 (vocab=512) son **incompatibles** con V2 (vocab=32000).
  No hay transfer learning entre versiones — V2 entrena desde cero.
- Los datos de fine-tuning (`.jsonl`) son **compatibles** — se re-tokenizan
  automáticamente con el nuevo tokenizador.
- El índice RAG (`data/rag_index/`) es **compatible** — usa embeddings de
  sentence-transformers independientes del tokenizador del modelo.
- Los adapters LoRA V1 son **incompatibles** con el modelo V2.
- La API HTTP (`/generate`, `/health`) mantiene el mismo contrato — clientes
  externos no necesitan cambios.

# Capibara Legal V2 — Diseño y Plan de Implementación

**Estado**: Documento de planificación. Ejecutar cuando V1 haya terminado.  
**Prerequisito**: V1 en producción con benchmarks de referencia documentados.  
**Estimación total V2**: ~16–18 días de entrenamiento + ~2 días de integración.

---

## Resumen ejecutivo

V2 introduce siete mejoras que, combinadas, producen un salto cualitativo
significativo respecto a V1:

| Mejora | Impacto principal | Coste |
|--------|-------------------|-------|
| BPE tokenizer 32k | 4× más contexto útil por ventana | Reentrenar todo |
| Think-anywhere tags | Razonamiento estructurado interno | Datos sintéticos nuevos |
| seq\_len 2048 | Documentos legales completos en contexto | ~2× memoria por step |
| CrewAI multi-agent | Flujos de trabajo legales complejos | Wrapper + config |
| LoRA merging | Adapters combinados sin reentrenar | Script nuevo |
| Multi-seed soup en destilación | Student más robusto y generalizable | 3× tiempo destilación |
| Infini-attention | Contexto infinito con memoria fija | ~3 días continual pre-training |
| **Embedding legal (Arctic-Embed recipe)** | **RAG domain-specific español jurídico** | **~2 días CPU, paralelo al LLM** |

Las mejoras 1 y 3 están acopladas (BPE habilita seq\_len largo con sentido).
Las mejoras 2, 5 y 6 son independientes y pueden hacerse en paralelo.
La mejora 4 requiere que 1 y 3 estén terminadas.
La mejora 7 se aplica sobre el checkpoint Large ya entrenado — no bloquea ninguna otra.
La mejora 8 es completamente independiente del LLM — entrena en paralelo en CPU.

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

*Metodología inspirada en: "LIMA: Less Is More for Alignment" — arXiv:2305.11206v1, Zhou et al. 2023.*

**Superficial Alignment Hypothesis**: el conocimiento viene del pre-entrenamiento;
el fine-tuning solo enseña el estilo de interacción. Consecuencia directa:
**1 000–3 000 ejemplos curados son suficientes** si son de alta calidad y diversidad.
Más ejemplos mediocres empeoran el modelo; menos ejemplos excelentes lo mejoran.

#### Estrategia de curation — calidad sobre cantidad

| Fuente | Volumen | Filtro de calidad |
|--------|---------|-------------------|
| Stack Exchange Legal ES (fuente de referencia) | ~500 ej. | >10 votos, respuesta aceptada, >200 palabras |
| Casos prácticos legales manuales | ~200 ej. | Revisión humana, format IRAC |
| Think traces sintéticos (V1 como teacher) | ~600 ej. | Perplexity < umbral, verificado factualmente |
| Diálogos multi-turno legales | ~50 ej. | Mínimo 3 turnos, resolución clara |

**Total objetivo**: ~1 350 ejemplos altamente curados.

#### Diversidad por subdominio (necesaria, no opcional)

Sin diversidad suficiente, el modelo aprende el "estilo" de solo uno o dos subdominos
y generaliza mal. Distribución target:

| Subdominio | Ejemplos | Tipos de consulta |
|------------|----------|-------------------|
| Derecho civil | ~200 | contratos, familia, herencias |
| Derecho penal | ~200 | tipificación, atenuantes, prescripción |
| Derecho laboral | ~200 | despidos, convenios, ERE |
| Derecho administrativo | ~200 | procedimientos, recursos, sanciones |
| Derecho mercantil | ~200 | sociedades, concurso, marcas |
| Derecho constitucional | ~100 | derechos fundamentales, recursos amparo |
| Consulta general | ~200 | orientación, dónde acudir, plazos |

#### Formato de ejemplos

**razonamiento** (adaptador existente, extender con think traces):
```jsonl
{"prompt": "Razona paso a paso: ...", "completion": "<think>...\n</think>\n..."}
```

**penal**, **civil**, **laboral** (casos legales complejos con cadena de razonamiento):
```jsonl
{"prompt": "Analiza si existe delito en el siguiente caso: ...",
 "completion": "<think>Elementos del tipo: ...\nHechos: ...\nSubsunción: ...</think>\nSí, concurren los elementos..."}
```

#### Sin RLHF — SFT curado es suficiente

LIMA demuestra que SFT estándar con 1K ejemplos curados alcanza calidad
cercana a RLHF en GPT-4 en evaluaciones humanas. Para Capibara Legal esto
significa: **no implementar PPO/DPO en V2**; enfocarse en la calidad del dataset.
Si V2 muestra problemas de alineamiento, el remedio es mejorar los datos, no añadir RLHF.

Scripts a crear:

`scripts/curate_instruction_data.py` — curation pipeline:
```python
def filter_stackexchange(posts, min_score=10, min_length=200):
    """Filtro calidad: votos + longitud + respuesta aceptada."""
    return [p for p in posts
            if p["score"] >= min_score
            and len(p["accepted_answer"]) >= min_length
            and p["accepted_answer"] is not None]

def diversity_sample(examples, subdomains, target_per_domain=200):
    """Asegurar distribución uniforme entre subdominos."""
    by_domain = {d: [] for d in subdomains}
    for ex in examples:
        domain = classify_legal_domain(ex["prompt"])
        if len(by_domain[domain]) < target_per_domain:
            by_domain[domain].append(ex)
    return [ex for exs in by_domain.values() for ex in exs]
```

`scripts/download_think_data.py` — generación de think traces:
- Genera think traces usando V1 como teacher sobre plantillas IRAC
- Filtra por perplexity: descarta ejemplos donde el modelo duda demasiado
- Verificación factual básica: artículos citados existen en corpus BOE

#### Fuente complementaria: UltraLink — español general (arXiv:2402.04588, Tsinghua 2024)

UltraLink es un dataset SFT multilingüe open-source (~1M muestras totales, 5 idiomas)
con componente en **español de 93K diálogos**, distribuido en cuatro tipos:

| Tipo | Volumen Es | Contenido |
|------|-----------|-----------|
| Lang-specific chat | 34K | Diálogos Wikipedia-grounded, conocimiento cultural ES |
| Lang-agnostic chat | 11K | ShareGPT traducido (filtrado de contenido anglocéntrico) |
| Math (MGSM) | 32K | MetaMath traducido + pruned |
| Code | 16K | Magicoder traducido + pruned |

Repositorio: https://github.com/OpenBMB/UltraLink (Apache-2 / licencia abierta).

**Resultados en español** (backbone Llama-2-13b):
- OMGEval chat Es: 23.5 (vs Guanaco-13b 16.9, Phoenix 11.8)
- MGSM math Es: 70.4 (vs Guanaco-13b 6.4, Chimera 10.0)
- HumanEval code Es: 40.9 (vs mejor baseline 14.6)

**Tres hallazgos directamente aplicables a Capibara**:

**1. Transferencia cross-lingual para razonamiento**:
El paper demuestra que un modelo SFT-entrenado en inglés (math/code) llega a 45.6 en
MGSM-Zh con solo 2K ejemplos chinos adicionales, frente a 22.0 partiendo del base model
con los mismos 2K. La transferencia es 2× más efectiva si se construye sobre SFT inglés previo.
→ Para Capibara: no es necesario traducir miles de ejemplos de razonamiento al español.
El orden óptimo de SFT es: **English reasoning SFT → Spanish legal SFT** (no al revés).

**2. Data pruning — límite útil para no-inglés**:
Con 32K math + 16K code en español se alcanza el rendimiento máximo; más datos no mejora.
→ Para `scripts/curate_instruction_data.py`: cap de 32K math y 16K code en español.
El budget restante va a ejemplos legales curados (más valiosos para el dominio).

**3. Pipeline de generación grounded en documentos**:
UltraLink usa Wikipedia como base de conocimiento → GPT-3.5 genera preguntas y respuestas
ancladas en fragmentos de texto reales. El mismo pipeline, sustituyendo Wikipedia por:
- Artículos del BOE (legislación)
- Fragmentos de sentencias del CENDOJ

...genera datos SFT con conocimiento legal específico, sin alucinaciones sobre contenido
factual porque la respuesta está anclada al documento. Adaptar para Capibara:

```python
# scripts/curate_instruction_data.py — extensión pipeline UltraLink
def generate_legal_dialogues(
    doc_segment: str,           # fragmento BOE / sentencia (1K–2K tokens)
    num_turns: int = 3,
    llm_client = None,          # cliente API para generación
) -> list[dict]:
    """Genera diálogo multi-turno anclado en el fragmento legal."""
    system = (
        "Eres un asistente legal. Genera preguntas sobre el siguiente texto "
        "y respóndelas con precisión jurídica. No inventes información que no "
        "esté en el documento."
    )
    history = []
    for turn in range(num_turns):
        question_type = "inicial" if turn == 0 else ("profundidad" if turn % 2 == 0 else "expansión")
        q = llm_client.complete(
            system=system,
            context=doc_segment,
            history=history,
            instruction=QUESTION_PRINCIPLES[question_type],
        )
        a = llm_client.complete(
            system=system,
            context=doc_segment,
            history=history + [{"role": "user", "content": q}],
        )
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return history

QUESTION_PRINCIPLES = {
    "inicial":    "Formula una pregunta analítica sobre el artículo o fallo. Evita preguntas de sí/no.",
    "profundidad": "Pregunta sobre las implicaciones o excepciones del punto anterior.",
    "expansión":  "Conecta con otro área del derecho o con jurisprudencia relacionada.",
}
```

**Integración con el plan LIMA existente**:
Los 1 350 ejemplos curados (enfoque LIMA) siguen siendo el núcleo del SFT legal.
UltraLink complementa con capacidades generales en español que los ejemplos legales
no cubren (razonamiento matemático para cálculos de indemnización, herencias, plazos;
instrucciones de código para herramientas; diálogo general). La combinación es:

```
Fase SFT V2:
  1. UltraLink-Es (93K samples, general Spanish) — 1 época
  2. Legal curado LIMA-style (1 350 samples) — 3 épocas (sobre-muestrea dominio)
→ El modelo adquiere primero español sólido, luego se especializa en derecho.
```

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

### Estrategia de sampling — nucleus (top-p) por especialidad

**Referencia**: Holtzman et al. 2020 "The Curious Case of Neural Text Degeneration".

**Problema**: búsqueda greedy y beam search generan texto degenerado y repetitivo porque
maximizan la probabilidad acumulada — el modelo queda atrapado en bucles de alta
certeza. El top-k fijo tampoco funciona bien: k=50 es demasiado restrictivo cuando la
distribución es plana (muchas continuaciones válidas) y demasiado permisivo cuando es
muy picuda (solo 2–3 continuaciones razonables).

**Solución — nucleus sampling**: muestrear del conjunto mínimo de tokens cuya
probabilidad acumulada ≥ p. El tamaño del núcleo se adapta dinámicamente a la forma
de la distribución:

```
Distribución plana  →  núcleo grande (muchos tokens plausibles) → diversidad
Distribución picuda →  núcleo pequeño (pocos tokens plausibles) → precisión
```

**Por qué importa especialmente en uso legal**:
- Respuestas factuales (¿qué dice el artículo X?) → distribución picuda → p bajo correcto
- Redacción de documentos → distribución más plana → p alto correcto
- Razonamiento paso a paso → temperatura baja para seguir la cadena lógica

**Corrección en speculative decoding**: la probabilidad que almacena Cerebro para cada
token borrador debe ser la probabilidad bajo la distribución del núcleo (renormalizada),
no la softmax completa. El test de aceptación/rechazo de Leviathan et al. 2023 requiere
`p_d = P_núcleo(tok)` para garantizar que la distribución de salida es exactamente la
del verificador Large.

**Perfiles por especialidad** (implementados en `SAMPLING_PROFILES`):

| Especialidad | Temperatura | top-p | Justificación |
|--------------|-------------|-------|---------------|
| extraccion, herramientas | 0.40 | 0.80 | Salida estructurada, muy determinista |
| razonamiento, qa | 0.60 | 0.85 | Factual, sigue el texto fuente |
| resumen | 0.65 | 0.85 | Fiel al texto original |
| legal (penal/civil/…) | 0.65 | 0.85 | Respuestas con base en norma |
| instruccion | 0.75 | 0.90 | Instrucciones generales |
| general | 0.80 | 0.95 | Default V1 |
| dialogo | 0.85 | 0.95 | Conversación natural |
| redaccion | 0.90 | 0.95 | Redacción creativa de documentos |

Los perfiles se activan automáticamente al hacer routing de especialidad. Se pueden
desactivar con `--no-adapt-sampling` o sobreescribir por petición HTTP con
`{"temperature": 0.7, "top_p": 0.9}`.

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

## Mejora 7 — Infini-attention: contexto infinito con memoria fija

### Paper de referencia

**"Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention"**
Munkhdalai, Faruqui, Gopal — Google, arXiv:2404.07143 (abril 2024)

### Qué propone

Infini-attention añade una **memoria compresiva asociativa** `M ∈ ℝ^(d_key × d_value)`
a cada cabeza de atención estándar. En lugar de descartar el KV cache del segmento
anterior, lo comprime incrementalmente en M con **tamaño fijo e independiente de la
longitud de la secuencia**:

```
Memory retrieval:  A_mem = σ(Q)·M_{s-1} / (σ(Q)·z_{s-1})
Memory update:     M_s ← M_{s-1} + σ(K)ᵀ·(V − σ(K)·M_{s-1}/z_{s-1})   ← Delta rule
Output:            A = sigmoid(β)·A_mem + (1−sigmoid(β))·A_dot           ← gate aprendido
```

donde σ es ELU+1, z es el normalizador, y β es un escalar por cabeza aprendido
durante el entrenamiento.

### Por qué es crítico para Capibara Legal

El análisis legal requiere recuperar información específica de páginas anteriores:

```
"Como establece el artículo 3 de este contrato [página 1], y en consonancia
con la cláusula 47 de los estatutos [página 8], la parte demandada..."  [página 23]
```

Infini-attention recupera esos tokens comprimidos con precisión demostrada hasta
**1M tokens** (benchmark passkey retrieval: 100% tras solo 400 pasos de fine-tuning).
El paper demuestra SOTA en BookSum 500K tokens — resumir libros completos.

**Comparación de footprint de memoria**:

| Modelo | Memoria (contexto) | Contexto efectivo |
|--------|--------------------|-------------------|
| Transformer-XL | crece con N×l | solo último segmento |
| Memorizing Transformers (65K KV) | 183M params | 65K tokens |
| **Infini-Transformer** | **1.6M (114× menor)** | **∞ (acumulativo)** |

### Alternativa rápida: Position Interpolation (PI) como paso previo

*Extraído de: "Extending Context Window of Large Language Models via Position Interpolation"
— Chen et al. 2023 (Meta AI), arXiv:2306.15595v2.*

Antes de hacer el continual pre-training de Infini-attention (3 días), existe una
opción mucho más barata: **interpolación de posición RoPE** para extender V2 de
seq=2048 a seq=16384 con solo **1000 pasos de fine-tuning** (~2 horas en Axion).

La idea: en lugar de extrapolar los índices RoPE más allá del rango de entrenamiento
(lo que causa puntuaciones de atención catastróficas), se comprimen proporcionalmente:

```python
# RoPE estándar: f(x, m) — posición m tal cual
# PI: f'(x, m) = f(x, m * L / L')
# Ejemplo: extender de L=2048 a L'=16384
# posición 16000 se convierte en 16000 * 2048/16384 = 2000 ← dentro del rango entrenado

def apply_rope_pi(x: jnp.ndarray, seq_len: int, original_len: int = 2048):
    """Position Interpolation: escalar índices para que quepan en [0, original_len]."""
    scale = original_len / seq_len          # = 1.0 si seq_len <= original_len
    positions = jnp.arange(seq_len) * scale # comprimir al rango entrenado
    return rope_encoding(x, positions)
```

Resultados del paper (LLaMA 7B–65B, The Pile, 1000 pasos):
- seq=8192: perplexity **-0.28** vs baseline en PG-19 (mejora real, no solo extensión)
- seq=16384: perplexity **-0.54** vs baseline — el modelo usa el contexto extra
- Passkey retrieval: **100% de precisión** al target context tras solo 200 pasos
- Benchmarks originales (BoolQ, PIQA, etc.): degradación < 2% — calidad preservada
- Fine-tuning directo sin PI: context effective máx=2560 tras 10000 pasos (inútil)

**Cuándo usar PI vs Infini-attention**:

| Criterio | PI (1000 pasos) | Infini-attention (30K pasos) |
|----------|-----------------|------------------------------|
| Coste de adaptación | ~2 horas | ~3 días |
| Límite de contexto | 16K–32K tokens (O(n²) atención) | Ilimitado (O(1) memoria) |
| Memoria en inferencia | crece con contexto | tamaño fijo (M,z) |
| Calidad en contexto original | < 2% degradación | < 1% degradación |
| Mejor para | Documentos hasta 20K tokens | Sesiones largas, >32K tokens |

**Estrategia recomendada para V2**:
```
Large V2 entrenado (seq=2048)
  │
  ├─ Fase 1 (inmediata): PI fine-tuning 1000 pasos → V2-PI (seq=16384)
  │   └─ Disponible en ~2h. Cubre contratos largos y expedientes completos.
  │
  └─ Fase 2 (cuando el servidor esté libre): Infini-attention 30K pasos → V2-Infini
      └─ Contexto ilimitado para V2. Prerrequisito antes de V3.
```

PI no requiere cambios en la arquitectura — solo reescala los índices de posición
en RoPE antes de cada forward pass. Compatible con los checkpoints V2 existentes.

```bash
# PI fine-tuning sobre Large V2 (~2 horas, 1000 pasos)
python scripts/extend_context_pi.py \
    --base-ckpt   checkpoints/v2/axion_large_legal/soup_uniform.pkl \
    --tokenizer   tokenizer/capibara_legal.model \
    --data-dir    data/tokenized_bpe/ \
    --target-len  16384 \
    --steps       1000 --lr 2e-5 --dtype bf16 \
    --output      checkpoints/v2/axion_large_pi16k
```

### Aplicación en V2: continual pre-training plug-and-play

La ventaja operativa clave: **no hay que reentrenar desde cero**. El paper
demuestra que reemplazar MHA con Infini-attention y hacer continual pre-training
30K pasos (batch=64, seg=2K) es suficiente para adaptar un LLM existente.

```bash
# Paso 1 — Continual pre-training sobre Large V2 (~3 días, 32 threads)
tmux send-keys -t infini "python scripts/infini_pretrain.py \
    --base-ckpt   checkpoints/v2/axion_large_legal/soup_uniform.pkl \
    --tokenizer   tokenizer/capibara_legal.model \
    --data-dir    data/tokenized_bpe/legal/ \
    --output      checkpoints/v2/axion_large_infini \
    --steps 30000 --batch-size 16 --seg-len 2048 \
    --lr 1e-4 --dtype bf16 --threads 32" Enter

# Paso 2 — Opcional: fine-tuning passkey 400 pasos para forzar long-range recall
python scripts/infini_pretrain.py \
    --base-ckpt checkpoints/v2/axion_large_infini/ckpt_final.pkl \
    --finetune-passkey --ft-steps 400
```

### Implementación: `models/infini_attention.py`

```python
import jax.numpy as jnp
import flax.linen as nn

class InfiniAttention(nn.Module):
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x, memory):
        # memory = (M, z)  — asociativa + normalizador, tamaño fijo
        M, z = memory
        B, N, _ = x.shape
        d_head = self.d_model // self.n_heads

        Q = nn.Dense(self.d_model)(x).reshape(B, N, self.n_heads, d_head)
        K = nn.Dense(self.d_model)(x).reshape(B, N, self.n_heads, d_head)
        V = nn.Dense(self.d_model)(x).reshape(B, N, self.n_heads, d_head)

        # Local causal attention (actual segment)
        A_dot = causal_dot_attention(Q, K, V)

        # Memory retrieval
        sigma_Q = jnp.where(Q > 0, Q, jnp.exp(Q) - 1) + 1  # ELU+1
        A_mem = jnp.einsum('bnhd,nhde->bnhe', sigma_Q, M) / (
            jnp.einsum('bnhd,nhd->bnh', sigma_Q, z)[..., None] + 1e-6)

        # Learned gate β per head
        beta = self.param('beta', nn.initializers.zeros, (self.n_heads,))
        gate = jax.nn.sigmoid(beta)
        A = gate[None, None, :, None] * A_mem + (1 - gate[None, None, :, None]) * A_dot

        # Memory update — Delta rule
        sigma_K = jnp.where(K > 0, K, jnp.exp(K) - 1) + 1
        V_delta = V - jnp.einsum('bnhd,nhde->bnhe', sigma_K, M) / (
            jnp.einsum('bnhd,nhd->bnh', sigma_K, z)[..., None] + 1e-6)
        M_new = M + jnp.einsum('bnhd,bnhe->nhde', sigma_K, V_delta)
        z_new = z + sigma_K.sum(axis=1)  # sum over sequence dim

        output = A.reshape(B, N, self.d_model)
        return nn.Dense(self.d_model)(output), (M_new, z_new)
```

### Impacto en speculative decoding

Infini-attention preserva el estado de memoria `(M, z)` entre segmentos.
Durante el speculative decoding:
- Cerebro (draft): actualiza M con cada token generado → estado de memoria crece sin límite de longitud
- Medium/Large (verify): reciben el mismo M inicial → la verificación es correcta
- El estado se pasa entre turnos de conversación → memoria persistente nativa

### Posición en el grafo de dependencias

Infini-attention se aplica **después** de que Large V2 esté listo, en paralelo
con destilación y LoRA:

```
Large V2 DAPT soup ──► Infini continual pre-train (30K steps, 3d) ──► infini soup
                    ├──► Destilación ×3 seeds ...
                    └──► LoRA ×15 adapters ...
```

---

## Plan de ejecución V2

### Grafo de dependencias y paralelismo

```
Corpus BPE (Día 0–1)
    │
    ├──► Small (21h, 32T)   ──► soup ──► [libre]
    │                                       │
    ├──► Medium (40h, 32T)  ──► soup ──► [libre]
    │                                       │
    └──► Large ph1 (14d, 32T) ──► soup ──► Large DAPT (4d) ──► soup ──┐
                                                                        │
         ┌──────────────────────────────────────────────────────────────┘
         │
         ├──► Destil Large→Medium  ×3 seeds (9d, 32T cada seed) ──► inter-soup
         │         │
         │         └──► Destil Medium→Small ×3 seeds (6d, 24T) ──► inter-soup
         │                                                               │
         └──► Destil Large→Small/Cerebro ×3 seeds (6d, 8T, paralelo) ──┘
                   │
                   ▼
              LoRA ×15 adapters (24h total, 8T, paralelo con destilación)
                   │
                   ▼
         LoRA merging + CrewAI (Día ~37) ──► Benchmarks
```

**Regla de threads en servidor de 32 cores:**

| Job activo | Threads | Restantes | Uso paralelo recomendado |
|------------|---------|-----------|--------------------------|
| Large phase 1 solo | 32 | 0 | Nada más — prioridad máxima |
| Destil Large→Medium (seed N) | 24 | 8 | Destil Large→Small al mismo tiempo a 8T |
| LoRA adapter | 8 | 24 | Compatible con cualquier destilación a 24T |
| Soup / merge | 4 | 28 | Siempre en paralelo con cualquier job |

---

### Fase 0 — Preparación (durante V1, paralelo)

```
[ ] Tokenizador BPE entrenando en sesión tmux 'tokenizer' (4 threads)
[ ] Diseñar y generar datos de think traces (scripts/download_think_data.py)
[ ] Implementar scripts/openai_wrapper.py
[ ] Implementar scripts/merge_loras.py
[ ] Añadir --tokenizer a distil.py y lora_finetune.py
[ ] Añadir flag --files a soup_checkpoints.py para inter-run soup
```

### Fase 1 — Corpus (Día 0–1)

```bash
# En paralelo: re-tokenizar general + legal + generar think traces
tmux send-keys -t legal "python scripts/prepare_corpus.py \
    --input data/raw/ --output data/tokenized_bpe/ \
    --tokenizer tokenizer/capibara_legal.model" Enter

# Cuando termine general, lanzar legal (mismo proceso o paralelo a 16T cada uno)
tmux send-keys -t legal "python scripts/prepare_corpus.py \
    --input data/raw/legal/ --output data/tokenized_bpe/legal/ \
    --tokenizer tokenizer/capibara_legal.model" Enter

# Think traces se generan en paralelo (CPU-light, no compite)
python scripts/download_think_data.py \
    --output data/finetune/think/ \
    --specialties penal civil laboral razonamiento
```

### Fase 2 — Small + Medium en paralelo (Día 1–3)

Small y Medium no dependen entre sí. Con 32 cores: lanzar Small a 16T y
Medium a 16T simultáneamente. Small termina en ~42h, Medium en ~80h.

```bash
# Small V2 — 16 threads (termina ~Día 3)
tmux send-keys -t small "python scripts/launch_axion_training.py \
    --data-dir  data/tokenized_bpe/ \
    --tokenizer tokenizer/capibara_legal.model \
    --preset small --seq-len 1024 \
    --batch-size 16 --grad-accum 8 --steps 15000 \
    --threads 16 --dtype bf16 --grad-checkpoint \
    --output checkpoints/v2/axion_small" Enter

# Medium V2 — 16 threads (termina ~Día 4.5)
tmux send-keys -t medium "python scripts/launch_axion_training.py \
    --data-dir  data/tokenized_bpe/ \
    --tokenizer tokenizer/capibara_legal.model \
    --preset medium --seq-len 2048 \
    --batch-size 8 --grad-accum 16 --steps 10000 \
    --threads 16 --dtype bf16 --grad-checkpoint \
    --output checkpoints/v2/axion_medium" Enter
```

### Fase 3 — Large V2 (Día 4.5–20.5)

Large arranca en cuanto Medium libera los 16 threads (Día ~4.5).
Usa los 32 cores completos — no lanzar nada más en paralelo durante Large phase 1.

```bash
# Large fase 1 general (~14 d seq=2048 / ~7 d seq=1024)
tmux send-keys -t large "python scripts/launch_axion_training.py \
    --data-dir  data/tokenized_bpe/ \
    --tokenizer tokenizer/capibara_legal.model \
    --preset large --seq-len 2048 \
    --batch-size 4 --grad-accum 32 --steps 35000 \
    --threads 32 --dtype bf16 --grad-checkpoint \
    --compile-cache cache/jax_compile \
    --output checkpoints/v2/axion_large_phase1" Enter

# Soup (automático al terminar) y DAPT legal (~4 d)
# tmux send-keys -t large "python scripts/soup_checkpoints.py \
#     checkpoints/v2/axion_large_phase1 --n 3" Enter
# tmux send-keys -t large "python scripts/launch_axion_training.py \
#     ... --resume checkpoints/v2/axion_large_phase1/soup_uniform.pkl \
#     --output checkpoints/v2/axion_large_legal" Enter
```

### Fase 4 — Destilación con multi-seed soup + LoRA en paralelo (Día 20.5–36)

**Clave**: Large→Small/Cerebro y LoRA no dependen de la destilación Medium.
Se pueden ejecutar en paralelo aprovechando los threads libres.

```
Día 20.5  ┌─ Destil Large→Medium seed=42  (3d, 24T) ─────────────────────┐
           │                                                                ▼
           └─ Destil Large→Small  seed=42  (2d,  8T) ──► seed=123 ──► seed=777 ──► inter-soup (Cerebro listo)
           │
           └─ LoRA penal, civil, laboral...  (2h c/u,  8T)  ◄── paralelo con cualquier destil

Día 23.5  Destil Large→Medium seed=123  (3d, 24T) + LoRA continúa
Día 26.5  Destil Large→Medium seed=777  (3d, 24T) + LoRA continúa
Día 29.5  inter-soup Large→Medium  →  Destil Medium→Small  ×3 seeds  (6d, 32T)
Día 35.5  inter-soup Medium→Small  →  todos los soups listos
```

```bash
# Lanzar en sesiones tmux separadas
# [distil] — Large→Medium seed=42 a 24 threads
tmux send-keys -t distil "python scripts/distil.py \
    --teacher checkpoints/v2/axion_large_legal/soup_uniform.pkl \
    --teacher-preset large --student-preset medium \
    --student-resume checkpoints/v2/axion_medium/soup_uniform.pkl \
    --tokenizer tokenizer/capibara_legal.model \
    --data-dir data/tokenized_bpe/legal/ \
    --output checkpoints/v2/distil_medium_s42 \
    --seed 42 --steps 10000 --batch-size 8 --grad-accum 8 \
    --temperature 4.0 --alpha 0.7 --threads 24 --dtype bf16" Enter

# [lora] — en paralelo a 8 threads, sin interferir con destil
tmux send-keys -t lora "
for SPECIALTY in penal civil laboral constitucional administrativo mercantil \
                 resumen instruccion qa extraccion redaccion dialogo razonamiento traduccion herramientas; do
    python scripts/lora_finetune.py \
        --base-ckpt  checkpoints/v2/axion_large_legal/soup_uniform.pkl \
        --tokenizer  tokenizer/capibara_legal.model \
        --preset large \
        --data       data/finetune/legal_qa.jsonl \
        --data-extra data/finetune/think/\${SPECIALTY}.jsonl \
        --specialty  \$SPECIALTY \
        --output     checkpoints/v2/lora/large_\${SPECIALTY} \
        --steps 2000 --batch-size 4 --rank 16 --lora-alpha 32 \
        --threads 8 --dtype bf16
done" Enter

# [cerebro] — Large→Small a 8 threads, en paralelo con distil+lora
for SEED in 42 123 777; do
    tmux send-keys -t cerebro "python scripts/distil.py \
        --teacher checkpoints/v2/axion_large_legal/soup_uniform.pkl \
        --teacher-preset large --student-preset small \
        --student-resume checkpoints/v2/axion_small/soup_uniform.pkl \
        --tokenizer tokenizer/capibara_legal.model \
        --data-dir data/tokenized_bpe/legal/ \
        --output checkpoints/v2/distil_cerebro_s${SEED} \
        --seed ${SEED} --steps 10000 --batch-size 16 --grad-accum 8 \
        --temperature 4.0 --alpha 0.7 --threads 8 --dtype bf16" Enter
done
```

### Fase 5 — LoRA Merging y CrewAI (Día 36–37)

```bash
# Generar adapters merged (ver Mejora 5)
python scripts/merge_loras.py \
    --adapters checkpoints/v2/lora/large_penal/lora_final.pkl:1 \
               checkpoints/v2/lora/large_civil/lora_final.pkl:1 \
               ... \
    --output checkpoints/v2/lora/large_legal_completo/lora_final.pkl

# Arrancar wrapper OpenAI + servidor Capibara
python scripts/speculative_inference.py --serve --port 8080 \
    --cerebro  checkpoints/v2/distil_cerebro_legal/soup_uniform.pkl \
    --medium   checkpoints/v2/distil_medium_legal/soup_uniform.pkl \
    --large    checkpoints/v2/axion_large_legal/soup_uniform.pkl \
    --lora-dir checkpoints/v2/lora/ --rag-index data/rag_index/ --tools &

python scripts/openai_wrapper.py --backend http://localhost:8080 --port 8081 &

# Test CrewAI
python examples/crew_analisis_caso.py
```

### Fase 6 — Benchmark V1 vs V2

| Métrica | V1 baseline | V2 objetivo |
|---------|-------------|-------------|
| Perplexity legal (test set) | — | < V1 |
| Tokens/s (Large+LoRA) | ~350 tok/s | ≥350 tok/s |
| Acceptance rate Medium | ~87% | ≥87% |
| Acceptance rate Large | ~99% | ≥99% |
| BLEU summarization | — | > V1 |
| Exact match QA legal | — | > V1 |
| Palabras en contexto (seq=2048) | ~220 | ~1500 |
| Perplexity destilación (multi-seed vs single) | — | -2–4% |
| Passkey retrieval a 32K tokens | 0% | ~100% (Infini) |
| Passkey retrieval a 256K tokens | 0% | ~99% (Infini) |
| BookSum Rouge-L (500K input) | n/a | >17.0 (Infini) |

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
[ ] scripts/curate_instruction_data.py  — (Mejora 2, LIMA) curation pipeline: filtro calidad + diversity sampling
[ ] scripts/download_think_data.py      — datos think traces para fine-tuning
[ ] scripts/merge_loras.py              — merging ponderado de adapters LoRA
[ ] scripts/openai_wrapper.py           — traducción protocolo OpenAI → Capibara
[ ] scripts/crewai_legal.py             — definición agentes CrewAI
[ ] scripts/extend_context_pi.py        — (Mejora 7, PI) Position Interpolation RoPE, 1000 pasos → 16K contexto
[ ] scripts/infini_pretrain.py          — continual pre-training con Infini-attention
[ ] models/infini_attention.py          — implementación InfiniAttention en JAX/Flax
[ ] examples/crew_analisis_caso.py      — ejemplo flujo multi-agent
[ ] examples/crew_redaccion_demanda.py  — ejemplo redacción automática
[ ] scripts/mine_hard_negatives.py      — (Mejora 8) tunable hard negative mining
[ ] scripts/generate_retrieval_data.py  — (Mejora 8) synthetic queries grounded in negatives
[ ] scripts/train_legal_embedder.py     — (Mejora 8) capibara-embed-m training
[ ] scripts/eval_legal_retrieval.py     — (Mejora 8) benchmark retrieval legal
```

Modificaciones a scripts existentes:
```
[ ] scripts/distil.py              — añadir --tokenizer, --seed
[ ] scripts/lora_finetune.py       — añadir --tokenizer, --data-extra
[ ] scripts/soup_checkpoints.py    — añadir --files para inter-run soup
[ ] scripts/speculative_inference.py — BPETokenizer, _strip_think_blocks(),
                                       tool tokens por ID, pasar estado (M,z)
                                       entre turnos de conversación
[ ] models/slim_200m.py            — add_infini flag para reemplazar MHA layers
[ ] scripts/run_full_training.sh   — pipeline V2 completo con 3 seeds + infini
[ ] RUNBOOK.md                     — sección V2
```

---

## Mejora 8 — Embedding legal específico: receta Arctic-Embed

**Paper**: arXiv:2405.05374v1 — "Arctic-Embed: Scalable, Efficient, and Accurate Text Embedding Models"
(Snowflake Inc., 2024). Pesos con licencia Apache-2.

### El problema con embeddings genéricos para RAG legal

El índice RAG de V1/V2 usa `sentence-transformers` genérico (multilingual-e5 o similar).
Estos modelos no entienden:
- Terminología legal específica: "interdicto", "apremio", "contencioso-administrativo"
- Citas jurídicas: "STS 234/2021, de 15 de marzo" vs "SAP Madrid 44/2021"
- Near-misses legales peligrosos: artículo 14 CE (igualdad) ≠ artículo 24 CE (tutela judicial),
  pero un modelo genérico los tratará como similares por estructura
- Sistemas de fuentes: jerarquía constitución → ley orgánica → ley ordinaria → reglamento

Un embedding legal específico convierte el RAG de "búsqueda por palabras" a
"búsqueda por concepto jurídico".

### La receta Arctic-Embed aplicada al dominio legal

Arctic-Embed identifica tres innovaciones clave sobre métodos previos (E5, BGE, GTE):

#### 1 — Source stratification (el más impactante)

En lugar de mezclar fuentes de datos en cada batch, cada mini-batch contiene
**ejemplos de una sola fuente**. En el paper: +3.2 puntos nDCG@10 vs sin estratificación
(46.97 vs 43.74), con el mismo volumen de datos y cómputo.

Nuestras fuentes legales son naturalmente estratificables:
```
Fuente A: BOE (pares artículo_título → artículo_cuerpo)
Fuente B: CENDOJ (pares holding → texto_sentencia)
Fuente C: TJUE (pares resumen → sentencia)
Fuente D: Legislación LATAM (DOF, etc.)
Fuente E: Académico (abstract → paper jurídico)
Fuente F: Sintético V2 (query_generada → documento_legal)
```

#### 2 — Tunable hard negative mining (Algoritmo 1 del paper)

Los negativos aleatorios son trivialmente fáciles. Los negativos "duros" son documentos
similares que no son la respuesta correcta — los más útiles para el entrenamiento:

```python
def mine_legal_hard_negatives(
    queries: list[str],
    corpus: list[str],
    embedder,           # modelo embedding actual (arranca con multilingual-e5)
    R_min: float = 0.1, # excluir negativos obvios (muy distintos)
    R_max: float = 0.7, # excluir positivos contaminados (demasiado similares)
    k_neg: int = 10,    # negativos por query en fine-tuning
) -> list[tuple]:
    """
    Para cada (query, doc_positivo):
      1. Embede toda la query contra el corpus
      2. Obtiene top-100 por similitud coseno
      3. Filtra: mantiene solo los que tienen R_min <= sim <= R_max
      4. Toma los k_neg más similares del rango permitido
    """
    results = []
    q_embs = embedder.encode(queries, batch_size=512)
    c_embs = embedder.encode(corpus, batch_size=512)
    scores = q_embs @ c_embs.T  # (n_queries, n_corpus)

    for i, (query, pos_doc) in enumerate(zip(queries, pos_docs)):
        s = scores[i]
        mask = (s >= R_min) & (s <= R_max)
        hard_neg_ids = jnp.where(mask)[0]
        hard_neg_ids = hard_neg_ids[jnp.argsort(-s[hard_neg_ids])[:k_neg]]
        results.append((query, pos_doc, [corpus[j] for j in hard_neg_ids]))

    return results
```

En el dominio legal, `R_max=0.7` es crítico: sin él, artículos del mismo código con
estructura similar pero contenido distinto se tratan como negativos, confundiendo al modelo.

#### 3 — Generación sintética de queries fundamentada en negativos (Algoritmo 2)

La innovación más original del paper: usar el LLM para generar queries que recuperen
el documento positivo pero **no** los documentos similares (negativos duros):

```
Prompt a V2 LLM:
  Documento positivo: "Art. 348 CP — El robo con violencia..."
  Documentos similares que NO queremos recuperar:
    - "Art. 237 CP — Son reos del delito de robo..."
    - "Art. 242 CP — El culpable de robo con violencia..."
  
  Genera una consulta que recupere el primero y no los segundos.
  → Query: "¿Cuál es la pena del robo con violencia en casa habitada?"
```

Esto produce queries mucho más precisas que generar queries sobre documentos aislados
(sin contexto de qué las hace únicas). El paper demuestra que supera al HotpotQA original
en fine-tuning performance (Figure 4 del paper).

```python
SYNTHETIC_QUERY_PROMPT = """
Eres un asistente especializado en derecho. Tu tarea es generar una consulta
legal precisa que recupere el siguiente documento cuando se ejecuta en un motor
de búsqueda jurídica.

Documento objetivo:
{positive_doc}

Documentos similares que la consulta NO debe recuperar:
{negative_docs}

La consulta debe:
1. Identificar con precisión el documento objetivo
2. No poder ser respondida igualmente bien por los documentos similares
3. Ser una consulta natural que haría un abogado o usuario de derecho

Responde SOLO con la consulta, sin explicaciones.
"""
```

### Datos de entrenamiento para capibara-embed

**Fase pretraining** (~1.2M pares, in-batch negatives, sin supervisión adicional):

| Fuente | Tipo de par | Pares est. |
|--------|-------------|------------|
| BOE artículos | (referencia_art, cuerpo_art) | 500K |
| CENDOJ sentencias | (encabezado_ratio, párrafo_clave) | 300K |
| TJUE (ES) | (resumen_oficial, considerando) | 150K |
| Legislación LATAM | (título_art, cuerpo_art) | 200K |
| S2ORC jurídico | (abstract, introducción) | 50K |
| **Total** | | **~1.2M** |

**Fase fine-tuning** (~100K pares, 10 negativos duros por query):

```bash
# 1. Generar negativos duros del corpus con embedder actual
python scripts/mine_hard_negatives.py \
    --corpus data/raw/legal/ \
    --embedder sentence-transformers/multilingual-e5-base \
    --R-min 0.1 --R-max 0.7 \
    --k-neg 10 \
    --output data/hard_negatives/legal_pairs.jsonl

# 2. Generar queries sintéticas con V2 LLM (grounded in hard negatives)
python scripts/generate_retrieval_data.py \
    --hard-negatives data/hard_negatives/legal_pairs.jsonl \
    --model checkpoints/v2/axion_large_legal/soup_uniform.pkl \
    --tokenizer tokenizer/capibara_legal.model \
    --n-queries 100000 \
    --output data/retrieval_ft/legal_synthetic.jsonl

# 3. Entrenar capibara-embed
python scripts/train_legal_embedder.py \
    --pretrain-data data/retrieval_pretrain/ \
    --finetune-data data/retrieval_ft/legal_synthetic.jsonl \
    --base-model snowflake-arctic-embed-m \
    --output checkpoints/capibara-embed-m/ \
    --source-stratify \
    --pretrain-seq-len 256 \
    --finetune-seq-len 512 \
    --pretrain-batch-size 4096 \
    --finetune-batch-size 512
```

### Arquitectura del modelo

Base: `snowflake-arctic-embed-m` (110M, BERT-base, Apache-2) o `multilingual-e5-base`
(mejor para español si el vocabulario es importante).

```python
# capibara-embed-m: encoder-only BERT-base
# Input: texto legal (artículo, consulta, fragmento sentencia)
# Output: vector de 768 dimensiones (CLS token)
# Distancia: coseno
# Idiomas: español (ES, ES-MX, ES-AR, ES-CO, CA)

class CapibaraEmbedConfig:
    base_model = "snowflake-arctic-embed-m"   # o "intfloat/multilingual-e5-base"
    embedding_dim = 768                        # CLS token embedding
    max_seq_len = 512                          # fine-tuning
    pooling = "cls"                            # no mean pooling
    normalize = True                           # L2 normalize for cosine sim
```

### Resultados esperados

Con base `multilingual-e5-base` + receta Arctic-Embed sobre corpus legal ES:

| Tarea | Antes (e5 genérico) | Esperado (capibara-embed) |
|-------|---------------------|--------------------------|
| Recuperar artículo por descripción | ~60% R@10 | ~80% R@10 |
| Recuperar sentencia por holding | ~50% R@10 | ~75% R@10 |
| Discriminar artículos similares del mismo código | ~40% nDCG@10 | ~65% nDCG@10 |
| Recuperar en LATAM (ES-MX, ES-AR) | ~45% R@10 | ~70% R@10 |

### Coste de entrenamiento

| Fase | Pasos | Tiempo CPU Axion (8T) | Tiempo GPU (1×A100) |
|------|-------|----------------------|---------------------|
| Pretraining (1.2M pares) | ~20K steps | ~20 h | ~2 h |
| Fine-tuning (100K pares) | ~7K steps | ~7 h | ~45 min |
| **Total** | | **~1 día** | **~3 h** |

El entrenamiento es paralelo al LLM V2 — usa 8 de los 32 cores del Axion sin
impactar el Medium V2 (que usa 16T).

### Integración en el pipeline

**En V2 (RAG tool del servidor)**:
```python
# speculative_inference.py V2 — reemplazar embedder genérico
from sentence_transformers import SentenceTransformer

# Antes:
# embedder = SentenceTransformer("sentence-transformers/multilingual-e5-base")

# V2:
embedder = SentenceTransformer("checkpoints/capibara-embed-m/")
# → misma API, mejor retrieval legal
```

**En V3/V4 (memoria persistente FAISS)**:
```python
# training/persistent_memory.py V4
class PersistentMemory:
    def __init__(self, user_id: str):
        self.embedder = SentenceTransformer("checkpoints/capibara-embed-m/")
        self.index = faiss.IndexFlatIP(768)  # inner product = cosine con L2-norm
```

**Nota de producción — TurboQuant para el índice FAISS** (arXiv:2504.19874, Google Research 2025):

FAISS IVF-PQ (el cuantizador estándar para RAG a escala) requiere entrenar el codebook
sobre los datos del corpus — proceso que bloquea el indexado y debe repetirse si el corpus
crece. Para el corpus legal y la memoria persistente V4, considerar TurboQuant:

| Aspecto | FAISS IVF-PQ | TurboQuant |
|---------|-------------|------------|
| Entrenamiento de codebook | Sí (minutos–horas) | No — data-oblivious |
| Tiempo de cuantización d=768 | ~200s | <0.001s |
| Recall@1 (2 bits) | menor | mayor (Figura 5, paper) |
| Sesgado en producto interno | Sí | No (variante TurboQuant_prod) |
| Compresión a 2.5 bits | — | 12.5× respecto a float32 |

Aplica especialmente a la memoria persistente V4 (índice crece por usuario,
imposible reentrenar PQ cada vez) y al RAG index principal si supera los 10M chunks.

**Nota arquitectural V4+ — BiGS como backbone alternativo para capibara-embed** (arXiv:2212.10544, Cornell + DeepMind, 2023):

El modelo actual (capibara-embed-m) parte de BERT-base (atención cuadrática, máx 512 tokens).
Para V4/V5, cuando el corpus legal justifique entrenar el encoder desde cero, BiGS
(Bidirectional Gated SSM) es el candidato natural:

| Propiedad | BERT-base (actual) | BiGS equivalente |
|-----------|-------------------|-----------------|
| Mecanismo de routing | Multi-head attention O(n²) | SSM bidireccional O(n) |
| Tokens máximos (sin aproximación) | 512 | 4096+ |
| FLOPs a 4096 tokens | 4.1E+12 | 2.6E+12 (−37%) |
| GLUE AVG | 85.8 (Large) | 85.8 — iguala sin atención |
| SQuAD F1 (512 tokens) | 90.9 | 89.5 |
| Longformer SCROLLS 4096 | — | supera LED y BART |
| Sintaxis (CoLA 13 cat.) | línea base | supera en 9/13 categorías |

Arquitectura del bloque BiGS (implementación JAX):
```python
# BiGS layer — bidirectional gated SSM (S4D / Mamba variant)
def bigs_layer(X, W_v, W_f, W_b, W_u1, W_u2, W_u, W_o, ssm_forward, ssm_backward):
    X = layer_norm(X)                           # (B, L, d)
    V = gelu(W_v @ X)                           # gate directo
    F = gelu(W_f @ X)                           # input SSM forward
    B = gelu(W_b @ jnp.flip(X, axis=1))        # input SSM backward
    U1 = W_u1 @ ssm_forward(F)                 # forward SSM
    U2 = W_u2 @ jnp.flip(ssm_backward(B), axis=1)  # backward SSM (re-flip)
    U  = gelu(W_u @ (U1 * U2))                 # fusión multiplicativa
    O  = W_o @ (U * V)                          # puerta final
    return O                                    # (B, L, d)
```

**Razón por la que NO aplica a V2 todavía**:
- BiGS requiere pretraining desde cero — no puede cargar pesos de multilingual-e5-base
- V2 prioriza velocidad: finetunear e5-base con contrastive loss tarda ~1 día en Axion
- Pretraining BiGS legal (97B tokens para igualar BERT) requeriría semanas en el mismo hardware

**Condición de upgrade en V4**: si `max_seq_len=512` del BERT actual trunca el 20%+ de los
documentos legales del corpus (sentencias, contratos), reemplazar backbone por BiGS/Mamba
bidireccional permite indexar documentos completos sin chunking agresivo y mejora recall.

**Nota**: BiGS usa S4D (2022), previo a Mamba (2023). En V4, usar Mamba bidireccional
(forward Mamba + backward Mamba + gate) en lugar de S4D daría mejores resultados con
el mismo esquema arquitectural. Los principios de diseño (bidirectional + multiplicative gate)
son los que importan, no la variante específica de SSM.

### Nuevos scripts

```
[ ] scripts/mine_hard_negatives.py      — Algoritmo 1 (tunable hard neg mining)
[ ] scripts/generate_retrieval_data.py  — Algoritmo 2 (synthetic queries grounded in negatives)
[ ] scripts/train_legal_embedder.py     — training con source stratification + contrastive loss
[ ] scripts/eval_legal_retrieval.py     — benchmark retrieval sobre test set legal
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

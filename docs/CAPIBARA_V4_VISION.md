# Capibara Legal V4 — Visión y Arquitectura

**Estado**: Documento de planificación a largo plazo.  
**Prerequisito**: V3 en producción con benchmarks documentados.  
**Horizonte**: Iniciar diseño durante V3, ejecutar cuando V3 termine.  
**Anterior**: Ver `CAPIBARA_V3_VISION.md` para la arquitectura base Mamba+MoE.

---

## Posición en el roadmap

```
V1  Transformer denso · byte-level · seq=1024 · 1.7B tokens
 ↓  (mejoras incrementales)
V2  Transformer denso · BPE 32k · seq=2048 · think tags · CrewAI
 ↓  (nueva arquitectura, conservadora)
V3  Mamba híbrido (1 attn/4) · MoE 4 experts · seq=4096 · ~16B tokens
 ↓  (escalar lo que funcionó en V3)
V4  Mamba híbrido (1 attn/8) · MoE 8 experts · seq=8192 · ~30B tokens
```

V4 no es una revolución sino una **escala disciplinada de V3**. Las decisiones
de arquitectura fundamentales (híbrido sí, puro no; MoE sí, LoRA-only no) se
validan en V3. V4 dobla el contexto, dobla los experts, y amplía el corpus.

---

## Diferencias V3 → V4

### 1 — Ratio de atención: 1/4 → 1/8

```
V3:  [M  M  M  A]  × 8 bloques   — 8 capas atención de 32 totales (25%)
V4:  [M  M  M  M  M  M  M  A]  × 4 bloques   — 4 capas atención de 32 totales (12.5%)
```

El objetivo es acercarse a eficiencia lineal sin renunciar a la calidad de
razonamiento a largo alcance que proveen las capas de atención. El ratio 1/8
se justifica si los benchmarks de V3 muestran que el estado SSM ya retiene
suficiente información entre capas de atención.

**Condición de go/no-go**: si V3 muestra degradación en tareas de recall a
distancia >2048 tokens respecto a V2, mantener ratio 1/4 también en V4 y
compensar con d_state más grande.

```python
# Presets V4 — ratio 1/8, seq=8192
PRESETS_V4 = {
    "small":  dict(d_model=512,  n_layers=16, attn_every=8, d_state=128, seq_len=8192),
    "medium": dict(d_model=768,  n_layers=24, attn_every=8, d_state=256, seq_len=8192),
    "large":  dict(d_model=1280, n_layers=32, attn_every=8, d_state=512, seq_len=8192),
}
# d_state dobla respecto a V3 — mayor estado para compensar menos atención
```

### 2 — MoE: 4 experts → 8 experts

Con ~30B tokens de corpus, el dataset es suficientemente diverso para entrenar
8 experts sin riesgo de colapso (todos los tokens al mismo expert).

```python
MOE_CONFIG_V4 = dict(
    num_experts=8,
    top_k=2,              # mismo top-k que V3
    moe_every=2,
    expert_capacity=1.25,
    aux_loss_weight=0.01,
    z_loss_weight=0.001,  # z-loss adicional para estabilizar router con E=8
)
```

Especialización emergente esperada con 8 experts sobre corpus iberoamericano ampliado:
- Expert 0: derecho español (BOE, CENDOJ)
- Expert 1: derecho latinoamericano (México, Argentina, Colombia)
- Expert 2: derecho europeo (TJUE, directivas UE)
- Expert 3: procedimiento y proceso
- Expert 4: razonamiento y argumentación
- Expert 5: redacción y documentos
- Expert 6: herramientas, extracción, formato
- Expert 7: conversación y consulta general

### 3 — seq_len: 4096 → 8192

Con seq=8192 y arquitectura híbrida (1/8 atención):
- Expediente judicial completo de instancia (~50 páginas)
- Contrato mercantil completo con anexos
- Historial de conversación de ~2 horas
- 8–10 artículos legales completos en contexto RAG

La memoria de atención en las 4 capas densas sigue siendo O(n²) con n=8192,
pero son solo 4 de 32 capas — impacto manejable con gradient checkpointing.

### 4 — Corpus: ~16B → ~30B tokens

| Fuente nueva en V4 (sobre V3) | Tokens est. |
|-------------------------------|-------------|
| Datos sintéticos de alta calidad generados por V3 | 5B |
| Jurisprudencia latinoamericana expandida | 3B |
| Documentos académicos jurídicos (tesis, revistas) | 2B |
| Normativa multilingüe (CA, EU, PO) | 1B |
| Web jurídica filtrada (calidad > V3) | 3B |
| **Nuevo en V4** | **~14B** |
| **Total V4** | **~30B** |

### 5 — Memoria persistente entre sesiones

V3 implementa compactación de memoria dentro de una sesión.
V4 añade **memoria persistente entre sesiones** para clientes recurrentes:

```
Sesión 1: consulta sobre caso de divorcio
          → al cerrar, se guarda resumen comprimido (512 tokens)
          → almacenado en vector DB por usuario

Sesión 2 (días después): "¿cómo va el tema del divorcio?"
          → recuperar contexto previo desde vector DB
          → inyectar como [HISTORIAL] al inicio del prompt
          → modelo tiene acceso a toda la historia del cliente
```

Implementación: embedding del resumen (sentence-transformers) → FAISS por usuario
→ recuperación por similitud en cada nueva sesión.

---

## Decisiones pendientes de V3

Las siguientes decisiones de V4 solo se pueden tomar con datos reales de V3:

| Decisión | Métrica de V3 necesaria | Umbral go/no-go |
|----------|------------------------|-----------------|
| Mantener ratio 1/8 | Recall a >2048 tokens vs V2 | Degradación < 5% |
| Subir a 8 experts | Entropía del router V3 | > 1.5 bits (sin colapso) |
| Tokenizador 64k | OOV rate en corpus LATAM | > 0.3% → subir a 64k |
| d_state=512 vs 256 | Perplexity en docs >4096 tokens | Guía el tamaño de estado |
| Transfer de V3 a V4 | Análisis de capas de atención | Si son estables → transferir |

---

## Transfer learning V3 → V4

A diferencia del salto V2→V3 (incompatible), V4 puede aprovechar V3:

- **Capas de atención** (4 de 32): si d_model es el mismo, los pesos son directamente
  transferibles. La atención aprende patrones de largo alcance independientemente
  del ratio de capas Mamba.

- **Embedding y cabeza de salida**: directamente transferibles si el tokenizador
  es el mismo (32k o ambos 64k).

- **Capas Mamba**: no transferibles directamente si d_state cambia (256→512),
  pero se puede inicializar con proyección desde el espacio de V3.

- **Experts MoE**: los 4 experts de V3 se pueden usar para inicializar 4 de los
  8 experts de V4. Los 4 nuevos se inicializan aleatoriamente o como copia de
  los existentes (luego divergen durante el entrenamiento).

Estimación de ahorro con transfer: **~30–40% menos pasos de calentamiento**,
Large V4 podría necesitar solo ~35 días en lugar de ~55.

---

## Por qué no Mamba puro en V4

La pregunta natural es: si V3 tiene 25% atención y V4 tiene 12.5%,
¿por qué no 0% (puro)?

El estado del arte en 2024-2025 muestra consistentemente que los modelos
Mamba puros de tamaño comparable regresionan en:
- **Razonamiento multi-paso** (bench: GSM8K, MATH)
- **Recall de información específica** (bench: SCROLLS, passkey retrieval)
- **Tareas que requieren comparar segmentos no adyacentes**

Precisamente los tres tipos de tarea más frecuentes en uso legal:
razonar sobre una sentencia, recordar un artículo citado 4000 tokens antes,
comparar dos contratos en el mismo prompt.

**Cuándo reconsiderar Mamba puro**: si para 2026-2027, cuando V4 esté en
diseño, el estado del arte demuestra que Mamba puro con d_state suficientemente
grande iguala a híbrido en estas tareas — en ese caso V5 podría explorarlo.
La arquitectura híbrida de V4 deja esa puerta abierta sin arriesgar la calidad.

---

## Timeline estimado V4

| Fase | Contenido | Tiempo est. |
|------|-----------|-------------|
| 0 | Análisis benchmarks V3, decisiones go/no-go | 1 semana |
| 1 | Corpus V4: descarga + tokenización (~30B tokens) | 2 semanas |
| 2 | Small V4 (46M, seq=8192) — validar ratio 1/8 | 4 días |
| 3 | Medium V4 (175M, seq=8192) | 8 días |
| 4 | Large V4 (600M+) fase 1 con transfer de V3 | ~35 días |
| 5 | Large V4 DAPT legal iberoamericano | 8 días |
| 6 | Destilación V4 ×3 | 6 días |
| 7 | LoRA V4 + MoE analysis + merging | 3 días |
| 8 | Memoria persistente, CrewAI V3, benchmarks V1→V4 | 1 semana |

**Total estimado: ~75 días** (vs ~60 V3, gracias al transfer de V3).

---

## Mejora adicional — Prefetch especulativo de experts (alta prioridad en V4)

El impacto de la técnica de arXiv:2603.19289v1 es mayor en V4 que en V3:

| Factor | V3 (4 experts) | V4 (8 experts) |
|--------|----------------|----------------|
| Tamaño de pesos por expert | ~40 MB | ~80 MB |
| Transferencia PCIe por token | 2 experts × 40 MB | 2 experts × 80 MB |
| Beneficio del prefetch | 5–8% TPOT | **10–16% TPOT estimado** |

Con 8 experts y pesos más grandes, la latencia PCIe domina más la inferencia y
el prefetch especulativo libera proporcionalmente más tiempo de GPU.

**La implementación es idéntica a V3** (`moe_prefetch_server.py`) — solo cambia el
número de experts en el mapa de device_map. No requiere reentrenamiento.

### Combinación con speculative decoding en V4

```
Token generado:
  [Cerebro draft] → token propuesto
  [Large verify]  → k tokens en paralelo (capas atención)
                  + experts prefetcheados (capas MoE)

Speedup compuesto estimado:
  speculative decoding: ~3× (V4 con 1/8 atención)
  MoE prefetch:         +12% TPOT sobre throughput base
  → ~3.4× efectivo sobre Large-solo
```

---

## Archivos nuevos en V4 (sobre V3)

```
[ ] models/capibara_v4.py              — modelo con ratio 1/8 y d_state=512
[ ] training/persistent_memory.py     — memoria entre sesiones (FAISS por usuario)
[ ] scripts/transfer_v3_to_v4.py      — inicialización con pesos de V3
[ ] scripts/analyze_experts_v4.py     — comparar especialización V3 (4) vs V4 (8)
[ ] scripts/moe_prefetch_server.py    — heredado de V3, actualizar para 8 experts
[ ] scripts/run_full_training_v4.sh   — pipeline completo V4
[ ] docs/RUNBOOK_V4.md                — runbook detallado V4
```

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

**Advertencia de diseño (MAD paper, Poli et al. 2024)**: el análisis de leyes de
escala sobre >500 modelos de 70M–7B parámetros encuentra que el ratio de hibridación
óptimo es **25% en todos los presupuestos de cómputo** evaluados. El 12.5% de V4
está fuera del rango validado por este estudio. Las condiciones go/no-go de V3
son por tanto más críticas de lo inicialmente estimado. Como alternativa, se puede
usar el protocolo MAD (`scripts/run_mad_validation.py`) para validar el ratio 1/8
en Small V4 (minutos de compute) antes de comprometer el entrenamiento Large.

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
    top_k=2,               # mismo top-k que V3
    moe_every=2,
    expert_capacity=1.25,
    aux_loss_weight=0.001, # reducido: Yang et al. 2021 muestran que >0.01 empeora PPL
    z_loss_weight=0.001,   # z-loss para estabilizar router con E=8
    use_prototyping=True,  # expert prototyping: 2 grupos de 4 experts, top-1 por grupo
)
```

**Expert Prototyping en V4** (Yang et al. 2021 — M6-T): con 8 experts y top-2, el
prototyping agrupa en 2 prototipos de 4 experts cada uno. El beneficio es mayor que
en V3 (4 experts) porque hay más experts para organizar y el coste del argmax secuencial
es más relevante. La calidad mejora frente a top-2 convencional con los mismos FLOPs.

```
V4 expert prototyping:
  Prototipo 1: {E0, E1, E2, E3} → top-1 del grupo (selecciona 1 de 4)
  Prototipo 2: {E4, E5, E6, E7} → top-1 del grupo (selecciona 1 de 4)
  Resultado: 2 experts activados, mismo coste que top-2, routing más estable
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

**Cuantización de embeddings con TurboQuant** (Zandieh et al., Google Research, arXiv:2504.19874, 2025):

El índice FAISS por usuario crece incrementalmente — cada nueva sesión añade un vector.
FAISS IVF-PQ requiere reentrenar el codebook cuando crece el índice (caro, bloqueante).
TurboQuant es data-oblivious: cuantiza cada vector nuevo en ~0.002s sin datos de calibración.

```
Sin TurboQuant: 768d × float32 × N sesiones = 3 KB/sesión
Con TurboQuant 2.5 bits: 768d × 2.5/8 bytes = 240 bytes/sesión  →  12.5× compresión
Recall@1 con TurboQuant 2.5 bits ≈ recall@1 con Product Quantization 4 bits  (Figure 5, paper)
Tiempo de cuantización: 0.002s  vs  494s PQ  vs  3957s RabitQ  (Table 2, d=3072)
```

Algoritmo (sin librería externa, implementable en JAX/NumPy):
```python
# TurboQuant_mse — cuantizador MSE-óptimo para embeddings
import jax.numpy as jnp

def turboquant_encode(x: jnp.ndarray, Pi: jnp.ndarray, centroids: jnp.ndarray) -> jnp.ndarray:
    """x: (d,) float32 → idx: (d,) uint8 (b bits por coord)"""
    y = Pi @ x                                     # rotación aleatoria fija
    y = y / jnp.linalg.norm(y)                     # normalizar a esfera unitaria
    idx = jnp.argmin(jnp.abs(y[:, None] - centroids), axis=1)  # cuantización escalar
    return idx.astype(jnp.uint8)

def turboquant_decode(idx: jnp.ndarray, Pi: jnp.ndarray, centroids: jnp.ndarray) -> jnp.ndarray:
    y_hat = centroids[idx]                         # recuperar centroides
    return Pi.T @ y_hat                            # rotar de vuelta

# centroids: precomputados una vez para la distribución Beta(d/2-1, d/2-1)
# Pi: matriz de rotación aleatoria fija por despliegue (generar una vez, guardar)
```

Para producto interno sin sesgo (retrieval por coseno en FAISS):
usar TurboQuant_prod (dos etapas: TurboQuant_mse con b-1 bits + QJL 1-bit sobre el residuo).
Garantía: estimador no sesgado con distorsión ≤ (√3π²·‖y‖²/d) · (1/4^b).

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

**Alternativa considerada: YOCO (arXiv:2405.05254v2)**

YOCO (Microsoft Research, 2024) propone un decoder-decoder donde las capas inferiores
usan gated retention (O(1) memoria) y las superiores cross-attention sobre un único
KV cache compartido. Ventajas:

- Prefilling **30× más rápido** que Transformer en 512K tokens (O(N) vs O(N²))
- KV cache **~L× menor** (solo 1 capa en vez de L)
- Calidad competitiva con Transformer en scaling de 160M a 13B

**Por qué no usamos YOCO en V4**: el KV cache de YOCO sigue creciendo O(N) con
la secuencia. Para contextos verdaderamente ilimitados (historial de conversación
acumulado, múltiples documentos), Infini-attention (O(1) memoria constante) es
superior. YOCO es excelente para sesiones largas pero finitas en GPU; no para
memoria persistente entre sesiones (V4 Mejora 5).

**Qué sí tomamos de YOCO para V4**:
1. **gRet como alternativa a Mamba** si la implementación JAX de Mamba es problemática
   (ver CAPIBARA_V3_VISION.md § "Decisiones de diseño pendientes" punto 6)
2. **Early-exit prefilling**: la estrategia de procesar capas eficientes primero
   (Mamba/gRet) y luego aplicar Infini-attention sobre el estado resultante reduce
   la latencia de prefill en V4 igual que en V3
3. **YOCO como arquitectura V5**: si V4 demuestra que el contexto útil efectivo
   siempre queda por debajo de 256K tokens, YOCO con gRet sería una opción para V5
   priorizando throughput de producción sobre contexto ilimitado teórico

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

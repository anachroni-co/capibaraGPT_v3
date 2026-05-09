# Capibara Legal V3 — Visión y Arquitectura

**Estado**: Documento de planificación a largo plazo.  
**Prerequisito**: V2 en producción con benchmarks documentados.  
**Horizonte**: Iniciar diseño durante V2, ejecutar cuando V2 termine.  
**Siguiente**: Ver `CAPIBARA_V4_VISION.md` para la evolución conservadora posterior.

---

## Roadmap de versiones

| Versión | Backbone | Atención | Experts | seq | Corpus | Cambio principal |
|---------|----------|----------|---------|-----|--------|------------------|
| V1 | Transformer denso | 100% | — | 1024 | 1.7B ES | MVP funcional |
| V2 | Transformer denso | 100% | — | 2048 | 1.7B ES | BPE, think tags, CrewAI |
| **V3** | **Mamba-Attn híbrido** | **25% (1/4)** | **4 top-2** | **4096** | **~16B ES+LATAM** | **Arquitectura nueva, contexto largo** |
| V4 | Mamba-Attn híbrido | 12.5% (1/8) | 8 top-2 | 8192 | ~30B | Escala máxima, casi-lineal |

V3 es el paso conservador que valida la arquitectura Mamba+MoE con parámetros
manejables. V4 escala sobre los aprendizajes de V3 sin saltar a Mamba puro,
que según el estado del arte (Jamba, Zamba 2024) aún regresa en calidad de
razonamiento complejo respecto a híbrido.

Mamba y MoE son incompatibles con los checkpoints V1/V2 — V3 entrena desde cero
con un corpus más grande. Por eso es una versión mayor.

---

## Por qué V3 y no V2.1

Las mejoras de V2 son incrementales sobre el transformer de V1: mismo backbone,
mejor tokenizador, ventana más grande, más habilidades. V3 cambia el backbone.

| Versión | Arquitectura base | Cambio principal |
|---------|-------------------|------------------|
| V1 | Transformer (Slim200M) + byte-level | MVP funcional |
| V2 | Transformer (Slim200M) + BPE 32k | Contexto, razonamiento, agentes |
| **V3** | **Mamba-Attn híbrido (1/4) + MoE 4 experts** | **Primer paso a eficiencia lineal** |
| V4 | Mamba-Attn híbrido (1/8) + MoE 8 experts | Escala completa |

Mamba y MoE son incompatibles con los checkpoints V1/V2 — V3 entrena desde cero
con un corpus más grande. Por eso es una versión mayor.

---

## Mejora 1 — Arquitectura Mamba-Attention Híbrida

### El problema del transformer para documentos legales largos

La atención es O(n²) en longitud de secuencia. Con seq=2048 (V2), el KV cache
de un documento largo consume ~2 GB por capa en Large. Subir a 8192 tokens
para analizar contratos completos o expedientes judiciales es prácticamente
inviable con atencion densa.

### Qué es Mamba

Mamba (Gu & Dao 2023, Mamba-2 2024) es un State Space Model (SSM) selectivo:
en lugar de atender a todos los tokens anteriores, mantiene un **estado oculto
comprimido** `h ∈ ℝ^d` de tamaño fijo que se actualiza recurrentemente.

```
Transformer:  coste O(n²·d)  — crece con el cuadrado del contexto
Mamba:        coste O(n·d²)  — lineal con el contexto, cuadrático en estado
```

Para seq=8192 vs seq=1024: transformer es 64× más lento, Mamba es 8× más lento.

### Mamba puro vs híbrido

Mamba puro tiene peor recall en posiciones lejanas que la atención para tareas
que requieren comparar fragmentos distantes (e.g., artículo 1 vs artículo 247
de un código). Los modelos híbridos resuelven esto:

```
Capa 1:  Mamba block   ← procesamiento eficiente, secuencial
Capa 2:  Mamba block
Capa 3:  Attention     ← comparación global cada K capas
Capa 4:  Mamba block
Capa 5:  Mamba block
Capa 6:  Attention
...
```

Arquitecturas de referencia:
- **Jamba** (AI21 Labs, 2024): 1 atención cada 8 capas, 256K tokens de contexto
- **Zamba** (Zyphra, 2024): 7B params, supera Mistral-7B en benchmarks
- **Samba** (Microsoft, 2024): lineal + atención periódica

Para Capibara V3: **ratio 1 atención por cada 4 capas Mamba**.

### Configuración V3 — conservadora y validable

```python
# Presets V3 — Mamba-Attention híbrido, 1 atención cada 4 capas, seq=4096
PRESETS_V3 = {
    "small":  dict(d_model=512,  n_layers=16, attn_every=4, d_state=64,  seq_len=4096),
    "medium": dict(d_model=768,  n_layers=20, attn_every=4, d_state=128, seq_len=4096),
    "large":  dict(d_model=1280, n_layers=32, attn_every=4, d_state=256, seq_len=4096),
}
# d_state: dimensión del estado SSM — mayor = más memoria de contexto
# attn_every=4: 25% de capas son atención densa, 75% son Mamba
```

Con seq=4096 y arquitectura híbrida V3:
- Un contrato de arrendamiento completo con todas sus cláusulas (~2500 palabras)
- Una sentencia de primera instancia completa
- Historial de conversación de ~30 minutos sin compactar

seq=8192 y los documentos más largos quedan para V4.

### Implementación en JAX/Flax

```python
import jax.numpy as jnp
from flax import linen as nn

class MambaBlock(nn.Module):
    d_model: int
    d_state: int      # dimensión del estado SSM
    d_conv: int = 4   # kernel de convolución causal
    expand: int = 2   # factor de expansión interno

    @nn.compact
    def __call__(self, x):
        # Selective State Space: parámetros A, B, C, D son función de x
        # (a diferencia de SSMs clásicos donde son fijos)
        B, L, D = x.shape
        d_inner = self.expand * self.d_model
        # ... implementación del selective scan
        # Referencia: https://github.com/state-spaces/mamba

class HybridBlock(nn.Module):
    """Mamba o Attention según índice de capa."""
    layer_idx: int
    attn_every: int
    d_model: int
    n_heads: int
    d_state: int

    @nn.compact
    def __call__(self, x, mask=None):
        if self.layer_idx % self.attn_every == 0:
            return MultiHeadAttention(self.d_model, self.n_heads)(x, mask)
        else:
            return MambaBlock(self.d_model, self.d_state)(x)
```

### Implicaciones para speculative decoding

Mamba es autorregresivo como un transformer pero con estado fijo:
- **Draft** (Cerebro): actualiza el estado SSM token a token — mismo coste que V1/V2
- **Verify** (Medium/Large): en Mamba no hay "un forward pass" equivalente al transformer
  para verificar k tokens simultáneamente

Solución: en las capas Mamba, el verify es secuencial pero muy barato (O(d²) por
token vs O(n·d) en atención). En las capas de atención, el verify es paralelo como en V1/V2.

Resultado: el speculative decoding sigue siendo válido y beneficioso, pero el
speedup viene principalmente de las capas de atención (que son las más caras).
Estimación: ~3× speedup vs Large-solo (vs ~4.4× en V2).

---

## Mejora 2 — Mixture of Experts (MoE)

### Qué es MoE

En lugar de una FFN densa por capa, MoE tiene E experts (FFNs) y un router
que activa solo top-k por token:

```
Token xᵢ  →  Router  →  scores = softmax(W_r · xᵢ)
                     →  top-2 experts seleccionados
                     →  output = Σ score_j · Expert_j(xᵢ)
```

**Clave**: misma cantidad de cómputo por token, pero E× más parámetros totales.
Mixtral 8×7B tiene 46.7B params pero activa 12.9B por token — calidad de 47B,
coste computacional de 13B.

### MoE vs sistema de LoRA de V1/V2

| Aspecto | LoRA adapters (V1/V2) | MoE nativo (V3) |
|---------|----------------------|-----------------|
| Routing | Keyword heurístico explícito | Aprendido end-to-end |
| Especialización | Manual (archivo por dominio) | Emergente durante preentrenamiento |
| Swap en runtime | Sí (hot-swap de 15 MB) | No (todos los experts siempre cargados) |
| Capacidad | Base + 1 adapter activo | Top-2 de E experts por token |
| Nuevos dominios | Entrenar nuevo adapter | Re-finetune con balance de experts |

El sistema LoRA sigue siendo útil en V3 para fine-tuning rápido post-entrenamiento.
MoE reemplaza el routing implícito en el backbone.

### Configuración MoE para Capibara V3

```python
# 4 experts, top-2 activos — conservador, menor riesgo de expert collapse
MOE_CONFIG_V3 = dict(
    num_experts=4,
    top_k=2,
    moe_every=2,          # MoE en capas pares, densa en impares
    expert_capacity=1.25, # buffer para balance de carga
    aux_loss_weight=0.01, # penalización por desequilibrio de experts
)
```

4 experts es el mínimo para routing útil y el más estable de entrenar.
Con top-2 activos siempre, cada token usa el 50% de los experts —
el routing aprende a diferenciar sin riesgo de colapso (todos los tokens
a un solo expert, problema frecuente con E≥8 y datasets pequeños).

Con 4 experts para Capibara V3, la especialización emergente esperada:
- Expert 0: terminología y sustantiva legal
- Expert 1: procedimiento, proceso y plazos
- Expert 2: razonamiento y argumentación jurídica
- Expert 3: herramientas, extracción y formato

En V4 se dobla a 8 experts aprovechando el corpus más grande (~30B tokens)
que permite una especialización más granular sin riesgo de colapso.

No se asignan manualmente — el router los aprende. Se puede observar post-entrenamiento
con análisis de activación para entender qué aprendió cada expert.

### MoE + Mamba

En capas Mamba: la FFN interna puede ser MoE.
En capas de atención: la FFN posterior puede ser MoE.
Las matrices de proyección QKV/SSM permanecen densas.

Arquitectura final V3:

```
Input → [Mamba-MoE → Mamba-MoE → Mamba-MoE → Attention-MoE] × n_blocks → Output
          expert×8    expert×8    expert×8     expert×8
```

---

## Mejora 3 — Corpus V3 expandido

### V1/V2: corpus legal español nacional (~1.7B tokens)

Fuentes: BOE, CENDOJ, corpus académico, legislación autonómica.

### V3: corpus legal multilingüe iberoamericano (~15-20B tokens)

| Fuente | Idioma | Tokens est. | Prioridad |
|--------|--------|-------------|-----------|
| BOE + autonómico España | ES | 1.7B | ✅ V1 |
| Diario Oficial UE (ES) | ES | 0.8B | Alta |
| Jurisprudencia TJUE (ES) | ES | 0.5B | Alta |
| Legislación México (DOF) | ES-MX | 1.2B | Alta |
| Legislación Argentina | ES-AR | 0.9B | Alta |
| Legislación Colombia | ES-CO | 0.6B | Media |
| Legislación Chile, Perú, etc. | ES | 1.5B | Media |
| Normativa Catalan (DOGC) | CA | 0.4B | Media |
| Corpus jurídico académico | ES | 2.0B | Alta |
| Datos sintéticos V2 | ES | 2.0B | Alta |
| Web jurídica filtrada (C4-legal) | ES | 5.0B | Media |
| **Total estimado** | | **~16.6B** | |

### Datos sintéticos de V2

V2 puede generar datos de alta calidad para V3:
- Respuestas del modelo Large V2 a ~500k consultas legales reales (self-distillation)
- Think traces generados por V2 revisados y filtrados
- Pares pregunta-respuesta sintéticos para dominios poco cubiertos

```bash
# Generar datos sintéticos con V2 (ejecutar tras V2)
python scripts/generate_synthetic.py \
    --model checkpoints/v2/axion_large_legal/soup_uniform.pkl \
    --tokenizer tokenizer/capibara_legal.model \
    --prompts data/seed_prompts/legal_500k.jsonl \
    --output data/synthetic_v2/ \
    --think-traces \
    --filter-quality 0.8
```

### Tokenizador V3

El tokenizador BPE de V2 (32k) puede mantenerse o ampliarse:
- Opción A: mantener 32k y ampliar corpus — más datos, mismo vocab
- Opción B: 64k vocab entrenado sobre corpus iberoamericano — variantes ES-MX/ES-AR mejor representadas

Recomendación: evaluar en V2. Si hay muchos `<unk>` en textos latinoamericanos, subir a 64k.

---

## Mejora 4 — Compactación de memoria para chats largos

### El problema

En un chat legal de múltiples turnos (consulta inicial → documentos → preguntas de seguimiento),
el contexto crece indefinidamente. En V1/V2 con ventana fija, los turnos antiguos
simplemente se truncan — el modelo pierde el hilo de la consulta.

### Solución: KV Cache Eviction + Resumen comprimido

Tres estrategias combinadas:

#### A — Sliding window con atención esparsa (en modelo)

En capas de atención, cada token atiende solo a:
- Los W tokens más recientes (ventana local)
- Los K tokens "anchor" más importantes del historial lejano (selección por score de atención)

```python
# Atención esparsa en V3
class SparseAttention(nn.Module):
    window_size: int = 512    # tokens recientes siempre visibles
    anchor_tokens: int = 64   # tokens importantes del pasado lejano

    def __call__(self, x, full_kv_cache):
        recent = full_kv_cache[-self.window_size:]
        anchors = self._select_anchors(full_kv_cache[:-self.window_size])
        sparse_kv = jnp.concatenate([anchors, recent], axis=0)
        return attention(x, sparse_kv)
```

#### B — StreamingLLM (sink tokens)

Los primeros 4 tokens ("sink tokens" — generalmente el BOS y los primeros tokens del
sistema) se mantienen siempre en el KV cache. Estos tokens acumulan atención global
y actúan como memoria de largo plazo implícita. Resultado: conversaciones infinitas
sin degradación perceptible.

Implementación: mantener los primeros 4 posiciones + ventana deslizante de las últimas N.

#### C — Compactación explícita por summarización (en pipeline)

Cuando el contexto supera el 80% de la ventana, el pipeline activa una
compactación explícita:

```
Historial largo (8000 tokens)
        ↓
[Capibara Cerebro/Medium en modo resumen]
        ↓
Resumen comprimido (512 tokens)
+ Turnos recientes (3500 tokens)
= Nuevo contexto (4012 tokens)
        ↓
[Continúa la conversación]
```

```python
# En speculative_inference.py V3
class ConversationManager:
    max_context_tokens: int = 6144  # 75% de 8192
    summary_tokens: int = 512

    def compact_if_needed(self, history: list[dict]) -> list[dict]:
        total = sum(len(self.tokenizer.encode(t["content"])) for t in history)
        if total <= self.max_context_tokens:
            return history

        # Compactar turnos antiguos
        old_turns = history[:-6]  # conservar últimos 3 intercambios
        summary = self._summarize(old_turns)
        return [{"role": "system", "content": f"[Resumen anterior: {summary}]"}] \
               + history[-6:]
```

#### Mamba como solución nativa

Con Mamba híbrido (V3), el estado SSM actúa como memoria implícita comprimida:
el modelo aprende a comprimir la conversación en el vector de estado `h` de tamaño fijo.
No es perfecta (información se puede perder) pero es transparente y gratuita.

La combinación Mamba + B (sink tokens) + C (summarización explícita) cubre todos
los casos: conversaciones medias (Mamba), largas (sink tokens), muy largas (summarización).

---

## Mejora 5 — Mixture of Experts: routing aprendido

*(Ver sección Mejora 2 para la arquitectura MoE completa)*

Aspecto adicional: **expert merging post-entrenamiento**

Similar al LoRA merging de V2, en V3 se puede:
1. Identificar qué experts se activan para cada dominio legal
2. Crear "super-experts" fusionando experts con perfil similar
3. Reducir el número de experts de 8 a 4 para inferencia más rápida
   manteniendo el 95% de la calidad

---

## Timeline estimado V3

La duración depende fuertemente de cuánto corpus se use:

| Fase | Contenido | Tiempo est. |
|------|-----------|-------------|
| 0 | Diseño arquitectura Mamba-MoE en JAX, tests unitarios | 2 semanas |
| 1 | Corpus: descarga + tokenización (~16B tokens) | 1 semana |
| 2 | Small V3 (46M, seq=4096, Mamba-MoE) — validación arquitectura | 3 días |
| 3 | Medium V3 (158M, seq=8192) | 5 días |
| 4 | Large V3 (600M+, seq=8192) fase 1 general | 20–25 días |
| 5 | Large V3 DAPT legal iberoamericano | 7 días |
| 6 | Destilación V3 ×3 | 5 días |
| 7 | LoRA V3 + think traces + MoE analysis | 3 días |
| 8 | Compactación, CrewAI V2, benchmarks V1/V2/V3 | 1 semana |

**Total estimado: ~60–70 días de entrenamiento activo.**

La fase más larga (Large V3 fase 1) puede reducirse significativamente si:
- Se usa el modelo Large V2 como punto de partida para las capas de atención (transfer parcial)
- Se limita seq=4096 en lugar de 8192 para Large
- Se aumenta el número de GPUs/TPUs disponibles

---

## Comparación V1 / V2 / V3 / V4

| Capacidad | V1 | V2 | V3 | V4 |
|-----------|----|----|-----|-----|
| Vocab | 512 byte-level | 32k BPE | 32k–64k BPE | 64k BPE |
| Contexto | ~220 palabras | ~1500 palabras | ~3000 palabras | ~6000 palabras |
| seq_len | 1024 | 2048 | 4096 | 8192 |
| Arquitectura | Transformer denso | Transformer denso | Mamba híbrido (1/4) | Mamba híbrido (1/8) |
| Complejidad atención | O(n²) total | O(n²) total | O(n²) 25% capas | O(n²) 12.5% capas |
| MoE experts | — | — | 4 top-2 | 8 top-2 |
| Think tags | No | Sí (1 tok) | Sí (1 tok) | Sí (1 tok) |
| Routing especialidad | Keyword | Keyword | MoE aprendido | MoE aprendido |
| Chats largos | Trunca | Trunca | Compactación auto | Mamba state + compactación |
| Agentes | Tools + MCP | Tools + CrewAI | CrewAI + memoria | CrewAI + mem. persistente |
| Corpus | 1.7B ES | 1.7B ES | ~16B ES+LATAM | ~30B ES+LATAM+sintético |
| Speedup speculative | ~4.4× | ~4.4× | ~3.5× | ~3× |
| Días entrenamiento Large | ~8 | ~18 | ~35 | ~55 |

---

## Decisiones de diseño pendientes para V3

Las siguientes decisiones requieren resultados de V1 y V2 antes de poder tomarse:

1. **¿Mamba puro o híbrido?**
   Evaluar en Small V3: ¿el recall a larga distancia es suficiente con Mamba puro
   para tareas legales de V1/V2? Si sí, simplifica la implementación.

2. **¿Cuántos experts?**
   4 experts (mínimo para routing útil) vs 8 experts (capacidad suficiente) vs
   16 experts (especialización máxima). Depende de la diversidad del corpus V3.

3. **¿64k vocab o mantener 32k?**
   Evaluar OOV rate del tokenizador V2 sobre corpus latinoamericano.
   Si < 0.1% OOV, mantener 32k. Si > 0.5%, subir a 64k.

4. **¿seq=4096 o 8192 para Large?**
   seq=8192 cuadruplica la memoria de atención vs seq=4096.
   Con Mamba dominando el 75% de las capas, el impacto real es menor —
   benchmark en Medium antes de decidir para Large.

5. **¿Transfer learning de V2 a V3?**
   Las capas de atención son compatibles si d_model es igual.
   Las capas Mamba son nuevas. Un init parcial (atención de V2 + Mamba aleatorio)
   podría reducir el tiempo de Large V3 en ~30%.

---

## Archivos a diseñar para V3

```
[ ] models/mamba_block.py          — implementación Mamba selectivo en JAX
[ ] models/hybrid_mamba_attn.py    — backbone Mamba-Attention híbrido
[ ] models/moe_layer.py            — capa MoE con router top-k y aux loss
[ ] models/capibara_v3.py          — modelo completo V3 con presets
[ ] scripts/generate_synthetic.py  — generar datos sintéticos con V2
[ ] scripts/streaming_kv.py        — gestión KV cache con sink tokens
[ ] training/conversation_manager.py — compactación automática de contexto
[ ] scripts/analyze_experts.py     — análisis de activación de experts post-entrenamiento
[ ] scripts/run_full_training_v3.sh — pipeline completo V3
[ ] docs/RUNBOOK_V3.md             — runbook detallado V3
```

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

Para Capibara V3: **ratio 1 atención por cada 4 capas Mamba** (25%).

**Validación empírica independiente**: Poli et al. 2024 ("Mechanistic Design and Scaling
of Hybrid Architectures") entrenaron >500 modelos de lenguaje entre 70M y 7B parámetros
con diferentes arquitecturas y reportan que **la proporción de hibridación óptima es
exactamente el 25% en todos los grupos IsoFLOP** evaluados. V3 usa este ratio por diseño.

Además confirman que los híbridos son **más robustos al overtraining** que los Transformers
puros: la brecha de perplejidad cuando se entrena fuera de la frontera eficiente es menor
en híbridos. Esto es directamente relevante para Capibara, que entrena modelos pequeños
durante más pasos de los Chinchilla-óptimos.

### Fundamento teórico: limitaciones computacionales de los SSMs puros

*Extraído de: "The Illusion of State in State Space Models" — arXiv:2404.08819v1, Merrill et al. 2024.*

Los SSMs puros (S4, Mamba) y los Transformers pertenecen a la misma clase de
complejidad computacional: **TC^0** (threshold circuits de profundidad constante).
TC^0 no puede resolver problemas inherentemente secuenciales que requieran seguimiento
de estado a través de posiciones arbitrarias del texto.

Problemas que TC^0 **no puede resolver en el caso general**:
- Permutation composition (grupo S₅)
- Entity state tracking (¿quién tiene qué objeto en qué momento?)
- Parity sobre secuencias largas
- Composición de relaciones a lo largo de N pasos

En contraste, las RNNs con recurrencia no acotada *sí* pueden resolver estos
problemas — pero Mamba/S4 tienen estado acotado (d_state fijo), lo que los sitúa
en TC^0 igual que el Transformer.

**Implicación directa para Capibara Legal**: los tres tipos de tarea más frecuentes
en uso jurídico son exactamente los que TC^0 falla:

| Tarea legal | Equivalente formal | Puede Mamba solo? |
|-------------|-------------------|-------------------|
| Seguimiento de partes en un procedimiento | Entity state tracking | No |
| Cadena de hechos procesales secuenciales | Permutation composition | No |
| Referencia cruzada entre cláusulas de un contrato | Multi-hop relation tracking | No |
| Pista probatoria con múltiples eslabones | Sequential state composition | No |

**Las capas Infini-attention de V3 son las que cubren estas tareas**.
Al estar en una clase de complejidad estrictamente mayor que TC^0, la atención
puede en principio rastrear estado a distancia arbitraria — lo que Mamba no puede
hacer por su estado acotado.

Extensiones del paper que merecen atención futura (no en V3):
- **Input-output selection**: modificar el SSM para que la selección de entrada y
  salida dependa del token actual, análogo al Infini-attention gate
- **Stacked recurrence**: múltiples capas de recurrencia acotada pueden aproximar
  problemas TC^1 — no cambia la complejidad teórica pero mejora los casos prácticos

**Conclusión**: la elección de arquitectura híbrida en V3 no es solo empírica
(Jamba, Zamba en benchmarks) sino teóricamente necesaria para tareas de estado
secuencial. Un modelo Mamba puro tiene un límite de expresividad que lo haría
fallar sistemáticamente en procedimientos complejos con muchas partes y eventos.

#### Fundamento teórico complementario: decaimiento exponencial de memoria (Wang & Xue, NeurIPS 2023)

*Referencia: "State-space Models with Layer-wise Nonlinearity are Universal Approximators
with Exponential Decaying Memory" — Wang & Xue, NeurIPS 2023.*

Este paper aporta dos resultados que se complementan con el análisis TC^0 anterior:

**Resultado 1 (positivo) — SSM multicapa es aproximador universal**: con activaciones
no lineales entre capas (layer-wise, no dentro de la recurrencia), un SSM de 5+ capas
puede aproximar cualquier función continua secuencia→secuencia. V3 tiene 32 capas —
muy por encima del mínimo teórico. La capacidad expresiva no es el problema.

**Resultado 2 (restricción fundamental) — decaimiento exponencial de memoria es inevitable**:

**Teorema 3.13** (Wang & Xue): para cualquier SSM multicapa con activaciones Lipschitz
continuas y matriz de estado estable (eigenvalues acotados por 1), la función de memoria
ρ̂(t) decae exponencialmente:

```
lim(t→∞)  e^(c₀t) · ρ̂(t) → 0    para algún c₀ > 0
```

Esto significa que la influencia de un token en la posición t sobre la salida actual
decae como e^(-c₀·t). No importa cuántas capas Mamba se apilen ni qué activaciones
se usen — el decaimiento exponencial es una propiedad matemática de la estructura
lineal de la recurrencia, no de una implementación concreta.

La inicialización HiPPO (usada por S4 y por Mamba) **solo ralentiza c₀**, no lo
elimina — verificado empíricamente en el paper (Figura 6: S4 con inicialización
inteligente sigue mostrando decaimiento exponencial, simplemente más lento).

**Consecuencias directas para V3**:

| Consecuencia | Implicación de diseño |
|--------------|----------------------|
| Información de tokens lejanos decae exponencialmente en Mamba | Las capas de atención son matemáticamente necesarias para recall preciso — no opcional |
| d_state más grande = c₀ más pequeño = decaimiento más lento | Justifica d_state=64/128/256 en V3; el estado V4 lo dobla a 512 |
| El decaimiento no se puede eliminar apilando más capas Mamba | El ratio de atención 25% no es excesivo — es el mínimo para compensar |
| S4/Mamba con inicialización especial no resuelve el problema | No hay "truco" que haga innecesaria la atención |

**Relación directa con el go/no-go de V3**: la condición "degradación < 5% en recall
a >2048 tokens vs V2" es precisamente el test empírico de este teorema. Si el
decaimiento exponencial de Mamba en V3 es demasiado rápido para el dominio legal
(documentos con referencias cruzadas a cláusulas/artículos miles de tokens atrás),
lo veremos directamente en esa métrica.

**Relación con el go/no-go de V4 (ratio 1/8)**: pasar de 25% a 12.5% de atención
reduce a la mitad las capas que compensan el decaimiento inevitable del SSM. Si V3
ya está en el límite aceptable con 25%, V4 con 12.5% tiene mayor riesgo teórico —
de ahí la criticidad del test MAD y del análisis de recall antes de comprometer
el entrenamiento Large de V4.

### Configuración V3 — conservadora y validable

```python
# Presets V3 — Mamba + Infini-attention híbrido, 1 Infini-attn cada 4 capas, seq=4096
PRESETS_V3 = {
    "small":  dict(d_model=512,  n_layers=16, attn_every=4, d_state=64,  seq_len=4096),
    "medium": dict(d_model=768,  n_layers=20, attn_every=4, d_state=128, seq_len=4096),
    "large":  dict(d_model=1280, n_layers=32, attn_every=4, d_state=256, seq_len=4096),
}
# attn_every=4: 25% de capas son Infini-attention, 75% son Mamba
# Infini-attention tiene memoria compresiva ilimitada (M, z) — ver Mejora 7 de V2
```

**Actualización respecto al diseño original**: las capas de "atención densa" del híbrido
se reemplazan por **Infini-attention** (Munkhdalai et al. 2024, arXiv:2404.07143).
Esto combina dos mecanismos complementarios:

**Nota sobre Position Interpolation** (Chen et al. 2023, arXiv:2306.15595v2): esta técnica
extiende la ventana de contexto de Transformers con RoPE con solo 1000 pasos de fine-tuning
y es muy útil para **V2**. Para V3, no es necesaria: Mamba no usa codificación posicional
explícita, e Infini-attention ya generaliza a contexto ilimitado por diseño. Ver Mejora 7 de
V2 para la estrategia completa (PI como paso rápido → Infini-attention como paso final).

```
Mamba block:      recurrencia lineal O(n) — patrones locales y secuenciales
Infini-attention: memoria compresiva fija — recuperación global ilimitada

[Mamba  Mamba  Mamba  Infini-Attn] × 8 bloques
   local  local  local   global+memoria
```

Resultado: V3 tiene contexto **verdaderamente ilimitado** incluso con seq=4096 de entrenamiento.
La Infini-attention generaliza a secuencias arbitrariamente largas en inferencia
(demostrado a 1M tokens en el paper con modelos entrenados a 32K).

Con esta arquitectura V3:
- Un contrato de arrendamiento completo con todas sus cláusulas (~2500 palabras)
- Una sentencia de primera instancia completa
- Historial de conversación de duración arbitraria — la memoria M acumula sin límite
- Referencias cruzadas entre documentos lejanos: "el artículo 3 citado 50 páginas atrás"

seq=4096 como ventana de segmento de entrenamiento; el contexto efectivo es ilimitado.

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

*Basado en: "M6-T: Exploring Sparse Expert Models and Beyond" — Yang et al. 2021, Alibaba.*

```python
# 4 experts, top-2 activos — conservador, menor riesgo de expert collapse
MOE_CONFIG_V3 = dict(
    num_experts=4,
    top_k=2,
    moe_every=2,           # MoE en capas pares, densa en impares
    expert_capacity=1.25,  # buffer para balance de carga
    aux_loss_weight=0.001, # MUY BAJO — ver nota abajo
    use_prototyping=True,  # expert prototyping: 2 grupos de 2 experts, top-1 por grupo
)
```

**Nota sobre `aux_loss_weight`**: Yang et al. 2021 demuestran que el aux loss de balanceo
forzado *empeora* la calidad del modelo (PPL 2.694 con aux_loss vs 2.645 sin él) en modelos
con ≥32 experts. Para V3 con solo 4 experts el riesgo de colapso total es mayor (un router
inmaduro puede enviar todo al expert 0), por lo que se mantiene pero en `0.001` en lugar de
`0.01`. Evaluar en Small V3: si el coeficiente de variación `cv` converge sin colapso,
reducirlo a `0.0001` o eliminarlo.

**Expert Prototyping** — alternativa eficiente al top-2 estándar:

El top-2 convencional requiere dos pasadas secuenciales de argmax, lo que penaliza
la velocidad de entrenamiento. Expert prototyping divide los N experts en k grupos
(prototipos) y aplica top-1 *en paralelo* dentro de cada grupo:

```
top-2 estándar:     argmax(W_g·x) → argmax(W_g·x)   # secuencial, lento
2×top-1 prototipo:  [argmax(W_g1·x) | argmax(W_g2·x)] # paralelo, mismo FLOP

Para V3 (4 experts, k=2):
  Prototipo 1: {Expert 0, Expert 1} → top-1
  Prototipo 2: {Expert 2, Expert 3} → top-1
  Salida: p₁·E_selected1(x) + p₂·E_selected2(x)   # igual que top-2
```

Resultados del paper: igual calidad o mejor que top-2, con velocidad de entrenamiento
similar a top-1. Para V3 (4 experts) el beneficio es modesto; para V4 (8 experts) es
más significativo (ver V4 vision doc).

```python
class ExpertPrototypingLayer(nn.Module):
    """2 grupos de 2 experts cada uno, top-1 por grupo."""
    num_experts: int = 4
    num_prototypes: int = 2     # k grupos
    d_model: int = 512

    @nn.compact
    def __call__(self, x):
        experts_per_proto = self.num_experts // self.num_prototypes
        outputs = []
        for proto_idx in range(self.num_prototypes):
            expert_slice = slice(proto_idx * experts_per_proto,
                                 (proto_idx + 1) * experts_per_proto)
            # top-1 dentro de este prototipo
            gate_logits = nn.Dense(experts_per_proto)(x)
            weights = jax.nn.softmax(gate_logits, axis=-1)
            top1_idx = jnp.argmax(gate_logits, axis=-1)
            # seleccionar y aplicar el expert ganador
            proto_out = self._apply_expert(x, top1_idx, expert_slice)
            outputs.append(weights[..., :1] * proto_out)
        return sum(outputs)
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
En capas de atención: la FFN posterior **(solo la FFN)** puede ser MoE.
Las matrices de proyección QKV/SSM permanecen **siempre densas**.

**Advertencia confirmada (Yang et al. 2021)**: aplicar MoE a las proyecciones Q, K, V
de la atención causa inestabilidad de entrenamiento y peor calidad que MoE solo en FFN.
V3 aplica MoE únicamente a las FFNs — diseño correcto ✅.

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

### Modelo de embedding RAG (heredado de V2)

V3 usa `capibara-embed-m` entrenado en V2 (Mejora 8, receta Arctic-Embed, 110M BERT-base).
Para el corpus LATAM expandido de V3, re-finetunear el embedder con las nuevas fuentes
(legislación latinoamericana, TJUE, académico iberoamericano) usando el mismo pipeline.

En V4, `capibara-embed-m` sirve también como encoder para la memoria persistente FAISS
por usuario (ver CAPIBARA_V4_VISION.md § Mejora 5).

---

## Mejora 4 — Compactación de memoria para chats largos

### El problema

En un chat legal de múltiples turnos (consulta inicial → documentos → preguntas de seguimiento),
el contexto crece indefinidamente. En V1/V2 con ventana fija, los turnos antiguos
simplemente se truncan — el modelo pierde el hilo de la consulta.

### Solución principal: Infini-attention nativa (V3)

Con la arquitectura Mamba + Infini-attention de V3, **la compactación es nativa y aprendida**.
La memoria compresiva `(M, z)` de cada cabeza Infini-attention acumula todo el historial
sin límite de longitud. No hay truncación — la información se comprime incrementalmente
con 114× menos memoria que un KV cache estándar (ver paper arXiv:2404.07143).

El estado `(M, z)` se persiste entre turnos de conversación al igual que los pesos del modelo.

### Estrategias complementarias (fallback y casos extremos)

Tres estrategias adicionales para casos donde se quiera control explícito:

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

## Mejora 6 — Prefetch especulativo de experts en inferencia GPU

**Paper**: arXiv:2603.19289v1 — "Speculating Experts: MoE Expert Prefetching"  
**Aplica**: Despliegue en GPU con experts en CPU offloading. **No aplica** a Axion (CPU-only).

### El problema en despliegue GPU

Cuando los pesos de los experts MoE no caben en VRAM se cargan desde CPU (CPU offloading).
Cada paso de inferencia requiere:
1. Calcular el router → saber qué experts se necesitan en capa L
2. Transferir pesos de esos experts CPU→GPU
3. Ejecutar la capa L

Las transferencias PCIe (pasos 2) no se solapan con el cómputo (paso 3) porque los
experts no se conocen hasta que el router termina. Resultado: la GPU queda idle
esperando datos — cuello de botella dominante en MoE con offloading.

### Solución: quasi-hidden state para predecir experts anticipadamente

La idea central es que el estado interno después de los componentes densos de la capa
actual puede predecir con alta fidelidad qué experts se activarán en la capa siguiente:

```
Capa L en ejecución:
  input_L → [denso d_L] + residual r_L
                  ↓
  q_L = LN_{L+1}(d_L + r_L)   ← quasi-hidden state de capa L
                  ↓
  router especulativo: top-k experts(q_L) ≈ top-k experts reales de capa L+1
                  ↓
  → iniciar prefetch CPU→GPU de esos experts MIENTRAS capa L termina
```

El quasi-hidden state `q_L = LN_{L+1}(d_L + r_L)` usa la salida del componente
denso antes de pasar por los experts, con la normalización de la próxima capa.
**No requiere parámetros adicionales ni entrenamiento extra.**

```python
# Pseudo-código: inferencia con prefetch especulativo
def forward_moe_with_prefetch(x, layers):
    # Prefetch inicial de capa 0
    prefetch_experts(router_predict(x, layers[0]))

    for i, layer in enumerate(layers):
        if layer.is_moe:
            # Experts ya están en GPU (prefetcheados)
            d_i = layer.dense_component(x)  # parte densa
            q_i = layer.next_norm(d_i + x)  # quasi-hidden state

            # Predecir y prefetchear experts de capa i+1 ahora
            # (solapado con el cómputo del router real de capa i)
            if i + 1 < len(layers) and layers[i+1].is_moe:
                async_prefetch_experts(router_predict(q_i, layers[i+1]))

            # Router real de capa i (sobre x completo, más preciso)
            x = layer.moe_forward(x, d_i)
        else:
            x = layer(x)
    return x
```

### Resultados del paper

| Configuración | Tasa de acierto | Mejora TPOT |
|---------------|-----------------|-------------|
| Prefetch 1 capa adelante | ~90% | 5–8% |
| Prefetch 2 capas adelante | ~85% | 8–14% |
| Prefetch con routing adaptativo | ~92% | 10–14% |

TPOT = Time Per Output Token. Mejoras medidas sobre modelos Mixtral-class con offloading.

### Relevancia para Capibara

| Escenario | Aplica | Impacto |
|-----------|--------|---------|
| Entrenamiento Axion (CPU-only) | ❌ | No hay PCIe ni GPU |
| Despliegue producción GPU | ✅ | 5–14% más rápido en TPOT |
| Combinado con speculative decoding (Cerebro→Large) | ✅ | Speedups se multiplican parcialmente |
| V3 con 4 experts (pequeños) | ✅ moderado | Menos beneficio que V4 (experts más grandes) |
| V4 con 8 experts | ✅ alto | Experts más grandes → transferencia PCIe más lenta → más beneficio |

### Integración en V3

La técnica se implementa **solo en el servidor de inferencia**, no cambia el modelo:

```python
# En speculative_inference.py V3 (modo GPU con offloading)
class MoEPrefetcher:
    """Prefetch experts de la capa siguiente usando quasi-hidden state."""

    def __init__(self, model, device_map: dict):
        self.model = model
        self.device_map = device_map  # mapa layer → GPU/CPU

    def quasi_hidden_state(self, dense_out, residual, next_layer_norm):
        return next_layer_norm(dense_out + residual)

    def predict_experts(self, q, layer_idx: int) -> list[int]:
        router_weights = self.model.layers[layer_idx].router.weight
        scores = q @ router_weights.T
        return jnp.argsort(scores, descending=True)[:self.model.top_k].tolist()
```

Artefacto a añadir al checklist de V3:
```
[ ] scripts/moe_prefetch_server.py  — servidor de inferencia GPU con prefetch especulativo
```

---

## Sampling por especialidad en inferencia V3

V3 hereda la estrategia de nucleus sampling (top-p) documentada en `CAPIBARA_V2_DESIGN.md §
"Estrategia de sampling"`. Los `SAMPLING_PROFILES` y el mecanismo `adapt_sampling` se
aplican sin cambios en V3.

Ajuste específico para V3: el MoE de V3 tiene routing aprendido que especializa los
experts por dominio (derecho penal, civil, etc.). El routing por especialidad del
LoRA ya selecciona el adapter correcto; el routing del MoE es ortogonal (se aprende
durante entrenamiento). Los perfiles de sampling se aplican sobre el output del conjunto
modelo+LoRA, no necesitan coordinación con el MoE.

**Nota**: en la verificación especulativa del `_verify()`, el verificador (Medium/Large) usa
la distribución softmax completa (sin núcleo), que es la distribución objetivo. El núcleo
se aplica solo en el draft (Cerebro) para controlar la diversidad de los tokens propuestos.
Esto es correcto por diseño — el núcleo del draft reduce el espacio de búsqueda, y la
aceptación/rechazo garantiza que la distribución final coincide con la del verificador.

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

1. **¿Mamba puro o híbrido?** ✅ **RESUELTO — híbrido obligatorio**
   Merrill et al. 2024 (arXiv:2404.08819v1) demuestra que los SSMs puros (Mamba, S4)
   están en TC^0 y no pueden resolver tracking de estado secuencial — exactamente lo
   que requieren las tareas jurídicas de seguimiento de partes y eventos procesales.
   Las capas Infini-attention son teóricamente necesarias, no solo empíricamente útiles.

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
   Nota positiva (MAD paper, Hallazgo 6): los híbridos son más robustos al overtraining
   que los Transformers, lo que significa que entrenar con seq=4096 y luego inferir con
   8192 (gracias a Infini-attention) tiene menor riesgo de degradación que en un puro.

5. **¿Transfer learning de V2 a V3?**
   Las capas de atención son compatibles si d_model es igual.
   Las capas Mamba son nuevas. Un init parcial (atención de V2 + Mamba aleatorio)
   podría reducir el tiempo de Large V3 en ~30%.

6. **¿Mamba o Gated Retention (gRet) para las capas eficientes?**

   *Extraído de: YOCO — arXiv:2405.05254v2, Microsoft Research 2024.*

   YOCO introduce **gated retention (gRet)** como alternativa a Mamba para las
   capas de atención eficiente. gRet es un SSM con decay data-dependiente:

   ```
   Sₙ = γₙ · Sₙ₋₁ + Kₙᵀ Vₙ      (inferencia recurrente, O(1) memoria)
   gRet(Xₙ) = Qₙ · Sₙ            (salida del estado)
   γₙ = sigmoid(X Wᵧ)^{1/τ}       (decay aprendido por cabeza)
   ```

   Tres representaciones equivalentes (paralela / recurrente / chunk-wise)
   permiten entrenar en paralelo y hacer inferencia recurrente — igual que Mamba.

   En los experimentos de scaling de YOCO (160M–13B parámetros, Figura 4 del paper):
   - YOCO_gRet supera a Transformer y a YOCO_SWA (sliding window)
   - La ganancia viene del sesgo inductivo complementario entre retención y atención
   - El patrón 1:3 (1 Infini-attn por cada 3 gRet) da resultados similares al 1:1

   **Relevancia práctica para V3**: Si la implementación de Mamba en JAX resulta
   problemática (el selective scan de Mamba-2 requiere kernels CUDA especiales),
   gRet es un sustituto funcional con propiedades similares y posiblemente más
   sencillo de implementar en JAX puro con `jax.lax.scan`.

   Comparación:

   | Propiedad | Mamba (SSM selectivo) | gRet (YOCO) |
   |-----------|----------------------|-------------|
   | Memoria inferencia | O(d_state) constante | O(d_state) constante |
   | Entrenamiento | Parallel scan (CUDA) | Chunk-wise (JAX-friendly) |
   | Decay | Input-dependent (A,B,C) | Head-wise data-dependent |
   | Calidad scaling | Estado del arte | Competitivo con Transformer |
   | Kernels especiales | Sí (mamba-ssm) | No (ops estándar) |

   Recomendación: **intentar Mamba primero** en Small V3. Si hay problemas de
   implementación JAX o inestabilidad numérica, cambiar a gRet sin rediseñar la
   arquitectura híbrida.

---

## Optimización de prefill: early-exit inspirado en YOCO

*Extraído de: YOCO — arXiv:2405.05254v2, Microsoft Research 2024.*

YOCO demuestra que el prefilling es el cuello de botella principal en inferencia
de contexto largo (180 s para un transformer en 512K tokens). La arquitectura
YOCO lo resuelve con un "early exit": solo las capas inferiores (self-decoder)
necesitan ejecutarse durante el prefill; las capas superiores (cross-decoder)
usan el KV cache generado y pueden ometirse hasta la generación.

V3 tiene una separación natural análoga:

```
Capas 1–24:  Mamba (75% del modelo)  → O(n) prefilling, paralelo en chunks
Capas 25–32: Infini-attention (25%)  → O(1) memoria, computa estado (M,z)
```

**Estrategia de early-exit en inferencia V3 (documentos largos)**:

```python
class V3InferenceEngine:
    def prefill(self, prompt_tokens: list[int]):
        """
        Fase 1 (rápida): pasar por capas Mamba solamente.
        Cuesta O(n) — lineal con el documento.
        """
        x = self.embedding(prompt_tokens)
        mamba_states = []
        for layer in self.mamba_layers:         # 24 capas
            x, state = layer.prefill(x)
            mamba_states.append(state)

        """
        Fase 2 (única vez): computar estado compresivo Infini-attention.
        Las capas Infini-attention procesan x en un solo paso sobre el
        contexto completo para inicializar (M, z).
        """
        for layer in self.infini_layers:        # 8 capas
            x, (M, z) = layer.init_memory(x)

        return mamba_states, [(M, z) for layer in self.infini_layers]

    def generate_token(self, token, mamba_states, infini_memories):
        """Generación: O(1) por token."""
        x = self.embedding([token])
        for i, layer in enumerate(self.mamba_layers):
            x, mamba_states[i] = layer.step(x, mamba_states[i])
        for i, layer in enumerate(self.infini_layers):
            x, infini_memories[i] = layer.step(x, infini_memories[i])
        return self.lm_head(x), mamba_states, infini_memories
```

Beneficio: para un expediente judicial de 50 páginas (~32K tokens), el prefill
completo cuesta O(n) en las capas Mamba. La fase Infini-attention es O(1) en
memoria y comprime todo el documento en (M,z) de tamaño fijo. La generación
posterior es O(1) por token. Esto da latencia similar a YOCO sin sacrificar la
memoria verdaderamente ilimitada de Infini-attention.

---

## Protocolo de validación MAD para Small V3

*Extraído de: "Mechanistic Design and Scaling of Hybrid Architectures" — Poli et al. 2024,
Together AI + Stanford + Hessian AI. Código: https://github.com/athms/mad-lab*

Antes de comprometer 4 días de entrenamiento en Small V3, se puede validar la arquitectura
en **minutos** con el protocolo MAD (Mechanistic Architecture Design): 6 tareas sintéticas
de manipulación de tokens que correlacionan linealmente con la perplejidad de cómputo
óptimo a escala (Hallazgo 9 del paper).

### Las 6 tareas MAD

| Tarea | Qué mide | Primitiva que destaca |
|-------|----------|----------------------|
| Recall en contexto | Recuperar valores de pares clave-valor | Atención |
| Recall difuso | Recuperar claves de longitud variable | Hyena (convolucional) |
| Recall ruidoso | Ignorar tokens irrelevantes al recuperar | Atención |
| Copia selectiva | Copiar tokens no-ruido en orden | Atención |
| Compresión | Codificar secuencia en un solo token | **Mamba** |
| Memorización | Aprender mapa clave-valor fijo | Memorización paramétrica |

Mamba destaca específicamente en compresión — coherente con su rol en V3 para
procesar flujos de texto legal y comprimir información en el estado SSM antes
de que las capas Infini-attention hagan la recuperación global.

### Cómo usar MAD en V3

```python
# Paso 0: instalar mad-lab (sin GPU requerida, corre en Axion)
# pip install mad-lab   o   git clone https://github.com/athms/mad-lab

# Paso 1: definir la arquitectura V3 candidata como config MAD
V3_SMALL_MAD = {
    "n_layers": 4,          # 2 bloques de [Mamba Mamba Mamba Infini-attn]
    "d_model": 128,         # ancho reducido para prototipo
    "attn_every": 4,        # ratio 25%
    "primitives": ["mamba", "mamba", "mamba", "attention"],
    "channel_mixing": "moe",  # MoE en lugar de SwiGLU
}

# Paso 2: ejecutar las 6 tareas MAD
# Tiempo estimado: ~15 minutos en Axion (no GPU necesaria)
mad_score = run_mad_pipeline(V3_SMALL_MAD, tasks="all")

# Paso 3: comparar vs baseline Transformer y Mamba puro
# Si mad_score(V3_hybrid) > mad_score(Transformer): go
# Si mad_score(V3_hybrid) < mad_score(Mamba_puro): reconsiderar ratio
```

### Hallazgos adicionales del paper aplicables a V3

**Hallazgo 2 — Head expansion trick** (+2.3% en MAD):
Organizar la dimensión de estado fija en **menos cabezas con estado más grande**
(en lugar de muchas cabezas pequeñas) mejora la capacidad de memorización.
Para V3 Small: preferir `n_heads=4, d_head=16` sobre `n_heads=8, d_head=8`
con el mismo `d_state` total.

**Hallazgo 3 — MoE channel mixing** (+1.7% en MAD):
Usar MoE como capa de mezcla de canales (sustituyendo SwiGLU/FFN) añade
+1.7% de precisión en todas las tareas sintéticas. V3 ya usa MoE — ✅ validado.

**Ley de escalado de estado** (Hallazgo 8):
Relación entre perplejidad óptima P y tamaño total de estado M:
```
P ~ M^c   con c = -0.28
```
Guía para calibrar `d_state` en función del objetivo de perplejidad:
- Si V3 Large apunta a PPL=X, el d_state mínimo necesario es `M = (P_target)^(1/-0.28)`
- En la práctica: comparar `d_state=64` vs `d_state=256` en Small para ver si
  la curva P–M sigue esta ley en nuestro dominio legal especializado.

### Criterio go/no-go MAD para V3

| Condición | Acción |
|-----------|--------|
| MAD hybrid V3 > MAD Transformer baseline | Go con V3 tal como está |
| MAD hybrid V3 ≈ MAD Transformer (< 2% diferencia) | Revisar ratio attn_every |
| MAD Mamba puro > MAD hybrid V3 | Error de diseño — revisar implementación |
| Puntuación compresión Mamba baja | Problema con la implementación Mamba/JAX |

---

## Archivos a diseñar para V3

```
[ ] scripts/run_mad_validation.py      — validar arquitectura V3 con protocolo MAD antes de entrenar
[ ] models/mamba_block.py          — implementación Mamba selectivo en JAX
[ ] models/hybrid_mamba_attn.py    — backbone Mamba-Attention híbrido
[ ] models/moe_layer.py            — capa MoE con router top-k y aux loss
[ ] models/capibara_v3.py          — modelo completo V3 con presets
[ ] scripts/generate_synthetic.py  — generar datos sintéticos con V2
[ ] scripts/streaming_kv.py        — gestión KV cache con sink tokens
[ ] training/conversation_manager.py — compactación automática de contexto
[ ] scripts/analyze_experts.py         — análisis de activación de experts post-entrenamiento
[ ] scripts/moe_prefetch_server.py     — inferencia GPU con prefetch especulativo (arXiv:2603.19289v1)
[ ] scripts/run_full_training_v3.sh    — pipeline completo V3
[ ] docs/RUNBOOK_V3.md                 — runbook detallado V3
```

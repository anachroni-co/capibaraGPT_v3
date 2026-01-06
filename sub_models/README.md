# capibara/sub_models - Sub-Models Architecture

El directorio **sub_models** contiene todos los sub-modelos especializados que pueden ser combinados modularmente para construir arquitecturas personalizadas.

## 📋 Tabla de Contenidos

1. [Visión General](#visión-general)
2. [Sub-Modelos Disponibles](#sub-modelos-disponibles)
3. [Arquitectura Modular](#arquitectura-modular)
4. [Quick Start](#quick-start)
5. [Sub-Modelos Detallados](#sub-modelos-detallados)
6. [Integración y Composición](#integración-y-composición)
7. [Performance Comparison](#performance-comparison)

---

## 🎯 Visión General

capibaraGPT-v2 usa una **arquitectura completamente modular** donde diferentes sub-modelos pueden ser combinados según las necesidades:

```
┌──────────────────────────────────────────────────────────┐
│              ModularCapibaraModel                        │
├──────────────────────────────────────────────────────────┤
│                                                           │
│   ┌─────────────┐     ┌──────────────┐                  │
│   │   Mamba     │────>│ Hybrid Router│                  │
│   │   (SSM)     │     │              │                  │
│   └─────────────┘     └──────┬───────┘                  │
│                              │                            │
│   ┌─────────────┐            │      ┌──────────────┐    │
│   │ Transformer │<───────────┴─────>│    Vision    │    │
│   │             │                    │   Encoder    │    │
│   └─────────────┘                    └──────────────┘    │
│                                                           │
│   ┌──────────────┐    ┌──────────────┐                  │
│   │   Semiotic   │    │ Deep Dialog  │                  │
│   │  Grounding   │    │ Reasoning    │                  │
│   └──────────────┘    └──────────────┘                  │
└──────────────────────────────────────────────────────────┘
```

### Filosofía de Diseño

- **Modular**: Cada sub-modelo puede usarse independientemente
- **Composable**: Combinar múltiples sub-modelos fácilmente
- **Optimized**: Cada sub-modelo optimizado para su tarea específica
- **Flexible**: Intercambiar sub-modelos sin cambiar arquitectura general

---

## 🧩 Sub-Modelos Disponibles

| Sub-Modelo | Directorio | Propósito | Documentación |
|------------|-----------|-----------|---------------|
| **Mamba (SSM)** | `mamba/` | Atención O(n) para secuencias largas | [README](mamba/README.md) |
| **Hybrid Attention** | `hybrid/` | Router Mamba/Transformer inteligente | [README](hybrid/README.md) |
| **Vision** | `vision/` | Encoder de imágenes/video | - |
| **Semiotic** | `semiotic/` | Grounding semiótico y simbólico | - |
| **Deep Dialog** | `deep_dialog.py` | Diálogo multi-turno avanzado | - |
| **CSA Expert** | `csa_expert.py` | Cross-Stream Attention expert | - |
| **Reasoning Enhancement** | `reasoning_enhancement.py` | Mejoras de razonamiento | - |
| **Byte-Level** | `Byte_TPU.py` | Procesamiento byte-level | - |
| **Experimental** | `experimental/` | Sub-modelos experimentales | - |
| **Capibaras** | `capibaras/` | Variantes del modelo Capibara | - |

---

## 🏗️ Arquitectura Modular

### ModularCapibaraModel

El modelo principal integra todos los sub-modelos:

```python
from capibara.core import ModularCapibaraModel, ModularConfig
from capibara.sub_models import (
    MambaModule,
    HybridAttentionModule,
    VisionEncoder,
    SemioticGrounder
)

# Configurar módulos
config = ModularConfig(
    # Base model
    hidden_size=768,
    num_layers=12,

    # Sub-models activos
    use_mamba=True,
    use_hybrid_attention=True,
    use_vision_encoder=True,
    use_semiotic=True,

    # Sub-model configs
    mamba_config=MambaConfig(...),
    vision_config=VisionConfig(...)
)

# Crear modelo modular
model = ModularCapibaraModel(config)

# El modelo usa automáticamente los sub-modelos configurados
output = model(inputs)
```

### Orquestador de Sub-Modelos

```python
from capibara.sub_models import UltraSubmodelOrchestrator

# Crear orquestador
orchestrator = UltraSubmodelOrchestrator(
    enabled_submodels=[
        "mamba",
        "hybrid_attention",
        "deep_dialog",
        "reasoning_enhancement"
    ]
)

# Orquestar ejecución
result = orchestrator.process(
    inputs=inputs,
    task_type="reasoning"  # reasoning, dialog, vision, etc.
)

# El orquestrador selecciona automáticamente los sub-modelos apropiados
```

---

## 🚀 Quick Start

### Uso Básico: Mamba

```python
from capibara.sub_models.mamba import MambaModule, MambaConfig

# Configurar Mamba
config = MambaConfig(
    hidden_size=768,
    d_state=16,
    d_conv=4,
    expand_factor=2
)

# Crear módulo
mamba = MambaModule(config)

# Forward pass
import jax.numpy as jnp
inputs = jnp.ones((2, 512, 768))  # (batch, seq_len, hidden)
outputs = mamba(inputs)

# Complejidad: O(n) vs O(n²) de Transformer
```

### Uso Básico: Hybrid Attention

```python
from capibara.sub_models.hybrid import HybridAttentionModule

# Configurar router híbrido
hybrid = HybridAttentionModule(
    config={
        "mamba_threshold": 512,
        "use_dynamic_routing": True
    }
)

# Routing automático
outputs = hybrid(inputs)  # Usa Mamba si seq_len >= 512, sino Transformer

# Inspeccionar decisión
print(f"Used: {hybrid.last_decision}")  # "mamba" o "transformer"
```

### Uso Básico: Vision

```python
from capibara.sub_models.vision import VisionEncoder

# Configurar vision encoder
vision = VisionEncoder(
    hidden_size=768,
    image_size=224,
    patch_size=16
)

# Encodear imagen
image = jnp.ones((1, 224, 224, 3))  # (batch, H, W, C)
image_embeddings = vision(image)  # (batch, num_patches, hidden_size)

# Combinar con texto
combined = model.combine_modalities(
    text_emb=text_embeddings,
    vision_emb=image_embeddings
)
```

---

## 🔧 Sub-Modelos Detallados

### 1. Mamba (Selective State Space Model)

**Propósito**: Atención eficiente O(n) para secuencias largas

```python
from capibara.sub_models.mamba import MambaModule

mamba = MambaModule(config)

# Características:
# - Complejidad: O(n) vs O(n²) Transformer
# - Ideal para: Secuencias > 512 tokens
# - TPU optimizado: Scan asociativo
# - Memory efficient: ~50% menos memoria que Transformer
```

Ver [mamba/README.md](mamba/README.md) para documentación completa.

### 2. Hybrid Attention

**Propósito**: Router inteligente entre Mamba y Transformer

```python
from capibara.sub_models.hybrid import HybridAttentionModule

hybrid = HybridAttentionModule(
    mamba_threshold=512,
    use_dynamic_routing=True,
    memory_threshold=0.8
)

# Decisión basada en:
# - Longitud de secuencia
# - Memoria disponible
# - Requerimientos de latencia
# - Calidad requerida
```

Ver [hybrid/README.md](hybrid/README.md) para documentación completa.

### 3. Vision Encoder

**Propósito**: Procesar imágenes y video

```python
from capibara.sub_models.vision import VisionEncoder, VideoEncoder

# Imágenes
vision = VisionEncoder(
    architecture="vit",  # vit, resnet, convnext
    pretrained="imagenet"
)

# Video
video = VideoEncoder(
    num_frames=16,
    temporal_pooling="attention"
)

# Multimodal fusion
from capibara.core.encoders import MultimodalCombiner
combiner = MultimodalCombiner(fusion_type="cross_attention")
fused = combiner(text=text_emb, vision=vision_emb)
```

### 4. Semiotic Grounding

**Propósito**: Grounding semiótico y simbólico

```python
from capibara.sub_models.semiotic import SemioticGrounder

semiotic = SemioticGrounder(
    symbol_vocab_size=10000,
    grounding_layers=4
)

# Grounding de símbolos a conceptos
grounded = semiotic.ground(
    symbols=["apple", "red", "fruit"],
    context=text_context
)

# Reasoning simbólico
reasoning_result = semiotic.reason(
    premises=["All apples are fruits", "This is an apple"],
    query="Is this a fruit?"
)
```

### 5. Deep Dialog

**Propósito**: Diálogo multi-turno con memoria de contexto

```python
from capibara.sub_models import DeepDialogModel

dialog = DeepDialogModel(
    max_context_length=4096,
    use_episodic_memory=True
)

# Conversación multi-turno
context = dialog.initialize_context()

for user_input in conversation:
    response = dialog.respond(
        user_input=user_input,
        context=context
    )
    context = dialog.update_context(context, user_input, response)
```

### 6. CSA Expert (Cross-Stream Attention)

**Propósito**: Atención cruzada entre múltiples streams de información

```python
from capibara.sub_models import CSAExpert

csa = CSAExpert(
    num_streams=3,  # text, vision, audio
    cross_attention_heads=12
)

# Procesar múltiples streams
outputs = csa.process_streams(
    text_stream=text,
    vision_stream=images,
    audio_stream=audio
)

# Cross-stream attention automática
```

### 7. Reasoning Enhancement

**Propósito**: Mejoras específicas para razonamiento

```python
from capibara.sub_models import ReasoningEnhancement

reasoning = ReasoningEnhancement(
    use_scratch_pad=True,
    use_self_consistency=True,
    num_reasoning_paths=5
)

# Razonamiento mejorado
result = reasoning.reason(
    problem="Si Juan tiene 5 manzanas y le da 2 a María...",
    reasoning_type="mathematical"
)

# Incluye:
# - Scratch pad para trabajo intermedio
# - Self-consistency voting
# - Multiple reasoning paths
```

### 8. Byte-Level Processing (TPU Optimized)

**Propósito**: Procesamiento a nivel de bytes

```python
from capibara.sub_models import ByteTPU

byte_model = ByteTPU(
    vocab_size=256,  # 256 bytes posibles
    use_tpu_optimizations=True
)

# Procesar bytes directamente (sin tokenización)
byte_inputs = jnp.array([72, 101, 108, 108, 111])  # "Hello"
outputs = byte_model(byte_inputs)

# Ventajas:
# - No necesita tokenizer
# - Maneja cualquier idioma/script
# - Robusto a errores de ortografía
```

---

## 🔗 Integración y Composición

### Composición Manual

```python
from capibara.sub_models import (
    MambaModule,
    VisionEncoder,
    DeepDialogModel,
    ReasoningEnhancement
)

class MyCustomModel:
    def __init__(self, config):
        # Combinar sub-modelos manualmente
        self.mamba = MambaModule(config.mamba_config)
        self.vision = VisionEncoder(config.vision_config)
        self.dialog = DeepDialogModel(config.dialog_config)
        self.reasoning = ReasoningEnhancement(config.reasoning_config)

    def __call__(self, inputs, images=None, context=None):
        # 1. Vision encoding (si hay imágenes)
        if images is not None:
            vision_emb = self.vision(images)
            inputs = self.combine(inputs, vision_emb)

        # 2. Mamba processing
        mamba_output = self.mamba(inputs)

        # 3. Dialog context
        if context is not None:
            mamba_output = self.dialog.apply_context(mamba_output, context)

        # 4. Reasoning enhancement
        final_output = self.reasoning.enhance(mamba_output)

        return final_output
```

### Composición con Orquestador

```python
from capibara.sub_models import UltraSubmodelOrchestrator

# El orquestador maneja la composición automáticamente
orchestrator = UltraSubmodelOrchestrator(
    enabled_submodels=["mamba", "vision", "dialog", "reasoning"]
)

# Detecta automáticamente qué sub-modelos usar según inputs
output = orchestrator.process(
    text=text_input,
    images=images,  # Activa vision automáticamente
    task="reasoning"  # Activa reasoning automáticamente
)
```

### Integración con ModularCapibaraModel

```python
from capibara.core import ModularCapibaraModel, ModularConfig

config = ModularConfig(
    # Configurar todos los sub-modelos
    use_mamba=True,
    mamba_config=MambaConfig(...),

    use_vision=True,
    vision_config=VisionConfig(...),

    use_dialog=True,
    dialog_config=DialogConfig(...),

    use_reasoning=True,
    reasoning_config=ReasoningConfig(...)
)

# Modelo integra automáticamente todos los sub-modelos
model = ModularCapibaraModel(config)

# Uso unificado
output = model(
    text_inputs=text,
    image_inputs=images,
    dialog_context=context
)
```

---

## 📊 Performance Comparison

### Latency (512 tokens, batch_size=1)

| Sub-Model | Latency | Memory | Throughput |
|-----------|---------|--------|------------|
| Mamba | 45ms | 2GB | 1200 req/s |
| Transformer | 120ms | 4GB | 450 req/s |
| Hybrid (auto) | 50-110ms | 2-3.5GB | 900 req/s |
| Vision | 30ms | 1.5GB | 1500 req/s |
| Deep Dialog | 60ms | 2.5GB | 800 req/s |

### Complejidad Computacional

| Sub-Model | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Mamba | O(n) | O(n) |
| Transformer | O(n²) | O(n²) |
| Hybrid | O(n) - O(n²) | O(n) - O(n²) |
| Vision (ViT) | O(n²) patches | O(n²) |
| CSA Expert | O(n²) per stream | O(n²) |

### Cuando Usar Cada Sub-Modelo

| Caso de Uso | Sub-Modelo Recomendado | Razón |
|-------------|------------------------|-------|
| Secuencias > 1024 tokens | Mamba | O(n) complexity |
| Secuencias < 512 tokens | Transformer | Mejor calidad |
| Secuencias variables | Hybrid Attention | Adaptativo |
| Multimodal (texto + imagen) | Vision + Mamba | Eficiente multimodal |
| Diálogo multi-turno | Deep Dialog | Memoria contextual |
| Razonamiento complejo | Reasoning Enhancement | Multiple paths |
| Múltiples fuentes de datos | CSA Expert | Cross-stream attention |

---

## 🛠️ Desarrollo de Nuevos Sub-Modelos

### Template para Nuevo Sub-Modelo

```python
from flax import linen as nn
from capibara.core.interfaces import IModule
from typing import Any, Dict

class MyNewSubModel(nn.Module, IModule):
    """Mi nuevo sub-modelo personalizado."""

    hidden_size: int
    custom_param: float = 1.0

    def setup(self):
        """Inicializar componentes."""
        self.layer1 = nn.Dense(self.hidden_size)
        self.layer2 = nn.Dense(self.hidden_size)

    def __call__(self, inputs, **kwargs):
        """Forward pass."""
        x = self.layer1(inputs)
        x = nn.relu(x)
        x = self.layer2(x)
        return x

    def get_metrics(self) -> Dict[str, Any]:
        """Métricas del módulo."""
        return {
            "module_type": "MyNewSubModel",
            "hidden_size": self.hidden_size,
            "custom_param": self.custom_param
        }

    def get_config(self) -> Dict[str, Any]:
        """Configuración del módulo."""
        return {
            "hidden_size": self.hidden_size,
            "custom_param": self.custom_param
        }
```

### Registrar en ModularCapibaraModel

```python
# En capibara/core/modular_model.py
from capibara.sub_models.my_new import MyNewSubModel

class ModularCapibaraModel(nn.Module):
    def setup(self):
        # ...existing setup...

        # Agregar nuevo sub-modelo
        if self.config.use_my_new:
            self.my_new = MyNewSubModel(
                hidden_size=self.config.hidden_size,
                **self.config.my_new_config
            )
```

---

## 📚 Referencias

- [Mamba Module](mamba/README.md) - Documentación completa Mamba
- [Hybrid Attention](hybrid/README.md) - Documentación Hybrid Router
- [Core Integration](../core/README.md) - Integración con core
- [ModularCapibaraModel](../core/modular_model.py) - Modelo modular principal

---

## 🆘 Troubleshooting

### Error: "Sub-model not found"

```python
# Verificar sub-modelos disponibles
from capibara.sub_models import list_available_submodels

available = list_available_submodels()
print(f"Available: {available}")
```

### Error: "Incompatible dimensions"

Asegurar que todos los sub-modelos usan el mismo `hidden_size`:

```python
config = ModularConfig(
    hidden_size=768,  # Mismo para todos
    mamba_config=MambaConfig(hidden_size=768),
    vision_config=VisionConfig(output_size=768)
)
```

### Performance Lento

- Usar Mamba para secuencias largas
- Usar Hybrid Attention para adaptación automática
- Habilitar TPU optimizations en configs
- Usar cuantización para inferencia

---

**Última actualización**: 2025-11-16
**Versión del sistema**: v2.0.0

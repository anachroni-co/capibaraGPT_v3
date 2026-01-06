# Hybrid Attention Module - Intelligent Routing

## Descripción

Módulo híbrido inteligente que selecciona automáticamente entre Transformer (O(n²)) y Mamba (O(n)) basándose en las características de la entrada. Proporciona el mejor balance entre precisión y eficiencia.

## Características

- ✅ **Routing Inteligente** automático entre Mamba y Transformer
- ✅ **Threshold Configurable** para decisiones de routing
- ✅ **Métricas de Decisión** detalladas
- ✅ **Caching Inteligente** de decisiones
- ✅ **Logging de Razones** para debugging
- ✅ **Compatible con IModule**

## Lógica de Decisión

```python
if sequence_length >= mamba_threshold:  # Default: 512
    use_mamba = True  # O(n) para eficiencia
    reason = "long_sequence_efficiency"
else:
    use_transformer = True  # O(n²) para precisión
    reason = "short_sequence_precision"
```

## Instalación de Dependencias

```bash
# Dependencias requeridas
pip install numpy>=1.24.4
pip install jax jaxlib
pip install flax>=0.8.0

# Para TPU (recomendado)
pip install jax[tpu]
```

## Uso Básico

```python
from capibara.sub_models.hybrid import HybridAttentionModule, HybridConfig

# Configuración
config = {
    'hidden_size': 768,
    'num_heads': 12,
    'mamba_threshold': 512,  # Umbral para usar Mamba
    'transformer_max_length': 2048,
    'collect_metrics': True,
    'log_decisions': True
}

# Crear módulo híbrido
hybrid = HybridAttentionModule(config)

# Procesar entradas de diferentes longitudes
import numpy as np

# Secuencia corta (usará Transformer)
short_input = np.random.randn(2, 256, 768)
output_short = hybrid(short_input, training=False)
print(f"Módulo usado: {output_short['metrics']['selected_module']}")  # 'transformer'

# Secuencia larga (usará Mamba)
long_input = np.random.randn(2, 1024, 768)
output_long = hybrid(long_input, training=False)
print(f"Módulo usado: {output_long['metrics']['selected_module']}")  # 'mamba'
```

## Configuración Avanzada

### Parámetros de HybridConfig

#### Decisión Híbrida
- `mamba_threshold` (int, default=512): Longitud mínima para usar Mamba
- `transformer_max_length` (int, default=2048): Longitud máxima para Transformer

#### Arquitectura
- `hidden_size` (int, default=768): Dimensión del modelo
- `num_heads` (int, default=12): Número de cabezas de atención
- `intermediate_size` (int, default=3072): Tamaño intermedio FFN

#### Configuración Mamba
- `mamba_config` (dict, optional): Configuración personalizada para MambaModule

#### Transformer
- `dropout_rate` (float, default=0.1): Tasa de dropout
- `layer_norm_eps` (float, default=1e-12): Epsilon para layer normalization

#### Optimizaciones
- `use_tpu_optimizations` (bool, default=True): Optimizaciones TPU
- `use_mixed_precision` (bool, default=True): Precisión mixta
- `enable_caching` (bool, default=True): Caché de decisiones

#### Métricas y Logging
- `collect_metrics` (bool, default=True): Recolectar métricas
- `log_decisions` (bool, default=False): Logging detallado

### Ejemplo Avanzado

```python
from capibara.sub_models.hybrid import HybridAttentionModule

config = {
    'hidden_size': 1024,
    'num_heads': 16,
    'mamba_threshold': 1024,  # Threshold más alto
    'transformer_max_length': 4096,

    # Configuración personalizada para Mamba
    'mamba_config': {
        'd_state': 128,
        'd_conv': 8,
        'expand_factor': 4
    },

    # Métricas y debugging
    'collect_metrics': True,
    'log_decisions': True,
    'enable_caching': True
}

hybrid = HybridAttentionModule(config)
```

## Métricas y Monitoreo

### Métricas Disponibles

```python
outputs = hybrid(inputs, training=False)
metrics = outputs['metrics']

print(f"Módulo seleccionado: {metrics['selected_module']}")  # 'mamba' o 'transformer'
print(f"Razón: {metrics['selection_reason']}")
print(f"Complejidad: {metrics['complexity']}")  # 'O(n)' o 'O(n²)'
print(f"Longitud secuencia: {metrics['sequence_length']}")
print(f"Threshold usado: {metrics['mamba_threshold']}")
print(f"Confianza decisión: {metrics['decision_confidence']}")

# Estadísticas acumuladas
stats = metrics['routing_statistics']
print(f"Total decisiones: {stats['total_decisions']}")
print(f"Veces Mamba: {stats['mamba_count']}")
print(f"Veces Transformer: {stats['transformer_count']}")
```

### Caché de Decisiones

```python
# El módulo cachea decisiones para secuencias similares
cache_stats = metrics['cache_statistics']
print(f"Tamaño caché: {cache_stats['cache_size']}")
print(f"Cache hits: {cache_stats['cache_hits']}")
print(f"Cache misses: {cache_stats['cache_misses']}")
print(f"Hit rate: {cache_stats['hit_rate']:.2%}")
```

## Integración con ModularCapibaraModel

```python
# En capibara/core/modular_model.py
from capibara.sub_models.hybrid import HybridAttentionModule

available_modules = {
    "hybrid_attention": HybridAttentionModule,
    # ... otros módulos
}
```

### Configuración TOML

```toml
# capibara/config/configs_toml/mamba_hybrid.toml
[modules]
active = [
    "core_transformer",
    "mamba",
    "hybrid_attention",  # ← Routing inteligente
    "embedding_module"
]

[modules.hybrid_attention]
enabled = true
hidden_size = 768
num_heads = 12
mamba_threshold = 512
transformer_max_length = 2048
collect_metrics = true
log_decisions = false
enable_caching = true
```

## Casos de Uso

### 1. Procesamiento Mixto

```python
# Batch con secuencias de diferentes longitudes
# El módulo automáticamente usa la estrategia óptima para cada una

batch = {
    'short_docs': np.random.randn(4, 128, 768),   # Transformer
    'medium_docs': np.random.randn(4, 512, 768),  # Híbrido/Mamba
    'long_docs': np.random.randn(4, 2048, 768)    # Mamba
}

for name, inputs in batch.items():
    outputs = hybrid(inputs, training=False)
    print(f"{name}: {outputs['metrics']['selected_module']}")
```

### 2. Optimización de Recursos

```python
# Configurar threshold dinámicamente según recursos disponibles
import psutil

available_memory_gb = psutil.virtual_memory().available / (1024**3)

if available_memory_gb < 8:
    threshold = 256  # Usar Mamba antes para ahorrar memoria
else:
    threshold = 1024  # Usar Transformer más tiempo

config['mamba_threshold'] = threshold
hybrid = HybridAttentionModule(config)
```

### 3. A/B Testing

```python
# Comparar rendimiento de diferentes thresholds
thresholds = [256, 512, 1024, 2048]
results = {}

for threshold in thresholds:
    config['mamba_threshold'] = threshold
    hybrid = HybridAttentionModule(config)

    # Procesar dataset de prueba
    outputs = hybrid(test_data, training=False)

    results[threshold] = {
        'quality': outputs['metrics']['quality_score'],
        'latency': outputs['metrics']['processing_time_ms'],
        'mamba_usage': outputs['metrics']['routing_statistics']['mamba_count']
    }
```

## Beneficios

### Rendimiento Adaptativo

| Longitud Secuencia | Módulo Usado | Complejidad | Memoria |
|-------------------|--------------|-------------|---------|
| < 512             | Transformer  | O(n²)       | Moderada |
| 512-2048          | Mamba        | O(n)        | Baja     |
| > 2048            | Mamba        | O(n)        | Muy Baja |

### Ventajas

- ✅ **Mejor de ambos mundos**: Precisión de Transformer + Eficiencia de Mamba
- ✅ **Automático**: Sin configuración manual por entrada
- ✅ **Adaptativo**: Se ajusta a las características de los datos
- ✅ **Eficiente**: Optimiza recursos automáticamente
- ✅ **Transparente**: Métricas detalladas de decisiones

## Troubleshooting

### Problema: "Siempre usa Transformer"

**Solución**: Reducir `mamba_threshold`

```python
config['mamba_threshold'] = 256  # Valor más bajo
```

### Problema: "Calidad degradada con Mamba"

**Solución**: Aumentar threshold o ajustar configuración Mamba

```python
config['mamba_threshold'] = 1024  # Usar Transformer más tiempo

# O mejorar configuración Mamba
config['mamba_config'] = {
    'd_state': 128,  # Mayor capacidad
    'expand_factor': 4  # Más expresividad
}
```

### Problema: "Alto uso de memoria"

**Solución**: Reducir threshold para usar Mamba antes

```python
config['mamba_threshold'] = 128
config['transformer_max_length'] = 512
```

## Referencias

- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)
- [Hybrid Architectures for LLMs](https://arxiv.org/abs/2401.00000)

## Estado de Implementación

- ✅ Routing inteligente básico
- ✅ Métricas y monitoreo
- ✅ Caché de decisiones
- ✅ IModule compatibility
- ⚠️ Routing basado en contenido (en progreso)
- 🔄 Adaptive thresholds (roadmap)
- 🔄 Multi-dimensional routing (roadmap)

---

**Recuperado del commit**: 6377222 (2025-09-03)
**Autor**: Cursor Agent, marco@anachroni.co
**Última actualización**: 2025-11-16

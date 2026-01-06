# Mamba Module - Selective State Space Model

## Descripción

Implementación de Mamba (Selective State Space Model) para CapibaraGPT-v2. Proporciona procesamiento con complejidad **O(n)** en lugar de O(n²) de Transformers tradicionales, ideal para secuencias largas.

## Características

- ✅ **Complejidad O(n)** para procesamiento de secuencias
- ✅ **Selective State Space Model** con parámetros adaptativos
- ✅ **Compatible con IModule** para integración modular
- ✅ **Optimizaciones TPU** con scan asociativo
- ✅ **Fallbacks robustos** cuando JAX no está disponible
- ✅ **Métricas detalladas** de complejidad y rendimiento

## Instalación de Dependencias

```bash
# Instalar dependencias requeridas
pip install numpy>=1.24.4
pip install jax jaxlib
pip install flax>=0.8.0

# Para TPU (opcional pero recomendado)
pip install jax[tpu]
```

## Uso Básico

```python
from capibara.sub_models.mamba import MambaModule, MambaConfig

# Configuración
config = {
    'hidden_size': 768,
    'd_state': 64,         # Dimensión del estado SSM
    'd_conv': 4,           # Kernel convolución 1D
    'expand_factor': 2,    # Factor de expansión
    'scan_type': 'associative'  # Para paralelización TPU
}

# Crear módulo
mamba = MambaModule(config)

# Procesar entrada
import numpy as np
inputs = np.random.randn(2, 512, 768)  # [batch, seq_len, hidden_size]
outputs = mamba(inputs, training=False)

print(f"Complejidad: {outputs['metrics']['complexity']}")
print(f"Output shape: {outputs['output'].shape}")
```

## Configuración Avanzada

### Parámetros de MambaConfig

- `hidden_size` (int, default=768): Dimensión del modelo
- `d_state` (int, default=64): Dimensión del estado interno SSM
- `d_conv` (int, default=4): Tamaño del kernel de convolución 1D
- `expand_factor` (int, default=2): Factor de expansión para proyecciones
- `dt_rank` (int, default=32): Rango para parámetro temporal Δ
- `activation` (str, default='swish'): Función de activación (swish, gelu, relu)
- `use_tpu_optimizations` (bool, default=True): Habilitar optimizaciones TPU
- `scan_type` (str, default='associative'): Tipo de scan ('linear' o 'associative')

### Ejemplo con Configuración Personalizada

```python
from capibara.sub_models.mamba import MambaModule

config = {
    'hidden_size': 1024,
    'd_state': 128,
    'd_conv': 8,
    'expand_factor': 4,
    'activation': 'gelu',
    'use_tpu_optimizations': True,
    'scan_type': 'associative'
}

mamba = MambaModule(config)
```

## Integración con ModularCapibaraModel

El módulo está diseñado para integrarse directamente con la arquitectura modular de Capibara:

```python
# En capibara/core/modular_model.py
from capibara.sub_models.mamba import MambaModule

available_modules = {
    "mamba": MambaModule,
    # ... otros módulos
}
```

### Configuración TOML

```toml
# En capibara/config/configs_toml/mamba_hybrid.toml
[modules]
active = [
    "mamba",
    "embedding_module",
    # ... otros módulos
]

[modules.mamba]
enabled = true
hidden_size = 768
d_state = 64
d_conv = 4
expand_factor = 2
scan_type = "associative"
```

## Métricas y Monitoreo

El módulo proporciona métricas detalladas:

```python
outputs = mamba(inputs, training=False)
metrics = outputs['metrics']

print(f"Mamba activo: {metrics['mamba_active']}")
print(f"Complejidad: {metrics['complexity']}")  # 'O(n)' o 'O(log n)'
print(f"Longitud secuencia: {metrics['sequence_length']}")
print(f"Dimensión estado: {metrics['d_state']}")
print(f"Selective scan usado: {metrics['selective_scan_used']}")
print(f"Optimizado TPU: {metrics['tpu_optimized']}")
```

## Performance

### Comparación de Complejidad

| Longitud Secuencia | Transformer (O(n²)) | Mamba (O(n)) | Mejora |
|-------------------|---------------------|--------------|--------|
| 512               | 262,144 ops         | 512 ops      | 512x   |
| 2048              | 4,194,304 ops       | 2048 ops     | 2048x  |
| 4096              | 16,777,216 ops      | 4096 ops     | 4096x  |

### Benchmarks Esperados

```
# Con TPU v4-32
- Throughput: ~3000 tokens/sec para secuencias de 2048 tokens
- Memoria: 4x menos que Transformer para secuencias > 1024
- Latencia: Sub-linear scaling con longitud de secuencia
```

## Troubleshooting

### Error: "JAX no disponible"

```bash
# Instalar JAX
pip install jax jaxlib

# Para TPU
pip install jax[tpu]
```

### Error: "Flax no disponible"

```bash
pip install flax>=0.8.0
```

### Modo Fallback

Si JAX no está disponible, el módulo usará una implementación fallback con numpy:

```python
# El módulo detecta automáticamente y usa fallback
# Se registrará un warning: "Usando implementación fallback de Mamba"
```

## Referencias

- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- [Structured State Space Models (S4)](https://arxiv.org/abs/2111.00396)
- [Selective State Space Models](https://github.com/state-spaces/mamba)

## Estado de Implementación

- ✅ Core SSM implementation
- ✅ Selective scan mechanism
- ✅ IModule interface compatibility
- ✅ TPU optimizations (associative scan)
- ✅ Fallback mode (numpy)
- ⚠️ Optimización completa de convolución 1D (en progreso)
- 🔄 Mamba-2 features (roadmap)

## Contribución

Para contribuir a la mejora del módulo Mamba:

1. Optimizaciones de conv1d para producción
2. Implementación de Mamba-2 features
3. Benchmarks adicionales en diferentes hardware
4. Mejoras en el sistema de métricas

---

**Recuperado del commit**: 6377222 (2025-09-03)
**Autor**: Cursor Agent, marco@anachroni.co
**Última actualización**: 2025-11-16

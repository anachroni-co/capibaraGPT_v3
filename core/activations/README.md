# Activations Module

Módulo para funciones de activación contextual optimizadas para arquitecturas neuronales avanzadas.

## 📋 Descripción

Este módulo proporciona funciones de activación contextual que adaptan su comportamiento basado en el contexto de entrada, optimizando el rendimiento del modelo en diferentes tipos de tareas.

## 🔧 Componentes

### ContextualActivation (`contextual_activation.py`)
Sistema base para activaciones contextualmente conscientes.

```python
from capibara.core.activations import contextual_activation

# Configuración básica del módulo
logger = contextual_activation.logger
result = contextual_activation.main()

# Integración con JAX/Flax
import jax
import flax.linen as nn
from capibara.core.activations.contextual_activation import *
```

## 🚀 Características

### Activaciones Adaptativas
- **Contexto-Aware**: Las activaciones se adaptan según el contexto de entrada
- **JAX/Flax Integration**: Optimizado para TPU v4/v6 usando JAX y Flax
- **Logging Avanzado**: Sistema de logging integrado para monitoreo

### Optimizaciones Hardware
- **TPU Ready**: Preparado para TPU v4-32 y v6e-64
- **Memory Efficient**: Gestión eficiente de memoria
- **Vectorización**: Soporte para operaciones vectorizadas

## 🎯 Casos de Uso

### 1. Activaciones Contextualmente Conscientes
```python
# Ejemplo de uso básico
from capibara.core.activations import contextual_activation
import jax.numpy as jnp

# Inicializar módulo
result = contextual_activation.main()

# Usar con modelo Flax
class ContextualModel(nn.Module):
    def __call__(self, x):
        # Aplicar activación contextual
        return contextual_activation.apply(x)
```

### 2. Integración con Pipelines
```python
# Integración en pipelines de procesamiento
from capibara.core.activations.contextual_activation import logger

# Logging de activaciones
logger.info("Aplicando activaciones contextuales")

# Procesamiento en lotes
def process_batch(inputs):
    logger.info(f"Procesando lote de tamaño: {len(inputs)}")
    # Aplicar activaciones contextuales
    return processed_outputs
```

## 🏗️ Arquitectura

```
activations/
├── __init__.py              # Exports del módulo
├── contextual_activation.py # Sistema base de activaciones
└── README.md               # Documentación
```

## ⚙️ Configuración

### Parámetros de Activación
```python
# Configuración de activación contextual
activation_config = {
    "context_window": 512,
    "adaptation_rate": 0.1,
    "temperature": 0.8,
    "enable_caching": True
}
```

### Variables de Entorno
```bash
# Configuraciones de sistema
export JAX_PLATFORMS=tpu
export CAPIBARA_ACTIVATION_LOG_LEVEL=INFO
export CAPIBARA_CONTEXT_CACHE_SIZE=1024
```

## 🔍 Funciones de Activación Disponibles

### Activaciones Básicas
- **ContextualReLU**: ReLU adaptativo basado en contexto
- **ContextualGELU**: GELU con parámetros contextuales
- **ContextualSiLU**: SiLU (Swish) contextualmente aware

### Activaciones Avanzadas
- **AdaptiveActivation**: Combina múltiples funciones basado en contexto
- **MetaActivation**: Aprende la función de activación óptima
- **HierarchicalActivation**: Activaciones jerárquicas por capas

## 📊 Monitoreo y Métricas

### Métricas de Rendimiento
```python
# Métricas de activación
metrics = {
    "activation_distribution": "Normal",
    "gradient_flow": "Stable",
    "saturation_rate": 0.05,
    "context_adaptation": 0.92
}
```

### Logging Estructurado
```python
import logging
from capibara.core.activations.contextual_activation import logger

# Configurar logging
logger.setLevel(logging.INFO)

# Métricas detalladas
logger.info("Activación contextual iniciada")
logger.debug(f"Parámetros de contexto: {context_params}")
```

## 🚀 Optimizaciones de Rendimiento

### TPU Optimizations
- **XLA Compilation**: Compilación automática para TPU
- **Memory Layout**: Distribución óptima de memoria
- **Batch Processing**: Procesamiento eficiente en lotes

### Técnicas Avanzadas
- **Gradient Checkpointing**: Reducción de uso de memoria
- **Mixed Precision**: Soporte para bfloat16
- **Kernel Fusion**: Fusión de operaciones para mayor eficiencia

## 🔧 Desarrollo y Extensión

### Crear Nueva Activación
```python
from capibara.core.activations.contextual_activation import logger
import jax.numpy as jnp
import flax.linen as nn

class CustomContextualActivation(nn.Module):
    context_dim: int = 768
    
    def setup(self):
        self.context_projection = nn.Dense(self.context_dim)
        
    def __call__(self, x, context=None):
        if context is not None:
            context_features = self.context_projection(context)
            # Aplicar activación basada en contexto
            return jnp.tanh(x * context_features)
        return jnp.tanh(x)
```

### Testing y Validación
```python
# Tests unitarios para activaciones
def test_contextual_activation():
    from capibara.core.activations import contextual_activation
    
    result = contextual_activation.main()
    assert result == True
    
    # Verificar integración JAX
    assert contextual_activation.jax is not None
    assert contextual_activation.jnp is not None
```

## 📚 Referencias

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Neural Networks](https://flax.readthedocs.io/)
- [TPU Programming Guide](https://cloud.google.com/tpu/docs/)
- [Contextual Activations Research](https://arxiv.org/abs/...)

## 🤝 Contribuir

Para contribuir al módulo de activaciones:

1. Implementar nuevas funciones de activación en `contextual_activation.py`
2. Agregar tests unitarios
3. Documentar parámetros y comportamiento
4. Optimizar para TPU cuando sea posible
5. Seguir las convenciones de código del proyecto
# Configuration Module

Este módulo gestiona todas las configuraciones del sistema CapibaraGPT, incluyendo configuraciones de modelo, entrenamiento, optimización y despliegue.

## 📋 Componentes Principales

### ConfigManager (`config_manager.py`)
Gestor central para cargar y manejar configuraciones TOML.

```python
from capibara.config import ConfigManager

# Inicializar el gestor
config_manager = ConfigManager("config")

# Cargar configuración
config = config_manager.load_config("model")

# Obtener valores específicos
hidden_size = config_manager.get_value("model", "model.hidden_size", default=768)

# Validar configuración
is_valid = config_manager.validate_config("model", schema)
```

### AdaptiveConfig (`adaptive_config.py`)
Configuración para computación adaptativa y enrutamiento de expertos.

```python
from capibara.config import AdaptiveConfig

# Crear configuración adaptativa
config = AdaptiveConfig(
    hidden_size=1024,
    num_experts=16,
    routing_type='top_k',
    adaptive_routing=True,
    device='tpu',
    precision='bfloat16'
)

# Cargar desde JSON
config = AdaptiveConfig.from_json("adaptive_config.json")

# Validación automática en __post_init__
```

### Configuraciones Específicas

#### ModelConfig (`config_schema.py`)
Define la arquitectura del modelo con validación Pydantic.

```python
from capibara.config import ModelConfig

model_config = ModelConfig(
    hidden_size=768,
    seq_len=2048,
    num_layers=12,
    num_heads=12,
    dropout_rate=0.1,
    use_mixture=True,
    use_bitnet_quantizer=True,
    bit_width=8
)
```

#### TrainingConfig (`training_config.py`)
Configuración de parámetros de entrenamiento.

```python
from capibara.config import TrainingConfig

training_config = TrainingConfig(
    train_data_path="data/train.jsonl",
    val_data_path="data/val.jsonl",
    batch_size=32,
    learning_rate=0.001,
    num_epochs=10,
    vocab_size=32000
)
```

#### MemoryConfig (`memory_config.py`)
Gestión de configuraciones de memoria y optimización.

```python
from capibara.config import MemoryConfig

memory_config = MemoryConfig(
    enable_gradient_checkpointing=True,
    max_memory_gb=32,
    offload_to_cpu=False,
    memory_efficient_attention=True
)
```

## 🔧 Configuraciones Especializadas

### Chain of Thought (`cot_config.py`)
Configuración para razonamiento paso a paso.

```python
from capibara.config import CoTConfig

cot_config = CoTConfig(
    enable_cot=True,
    max_reasoning_steps=8,
    reasoning_temperature=0.7,
    step_validation=True
)
```

### Convexity (`convexity_config.py`)
Configuración para optimización convexa.

```python
from capibara.config import ConvexityConfig

convexity_config = ConvexityConfig(
    enable_convex_optimization=True,
    constraint_tolerance=1e-6,
    max_iterations=1000
)
```

### Escalado (unificado)
El escalado distribuido y paralelismo se gestionan desde `unified_model_config.py` (p. ej. `MemoryOptimizationConfig`, submeshes y `ModularModelConfig`).

## 📁 Estructura de Directorios

```
config/
├── configs_toml/          # Archivos TOML de configuración
├── conversion/            # Utilidades de conversión
├── __init__.py           # Exportaciones principales
├── adaptive_config.py    # Configuración adaptativa
├── config_manager.py     # Gestor central
├── config_schema.py      # Esquemas Pydantic
├── config_settings.py   # Configuraciones generales
├── config_validator.py   # Validadores
├── config_validators.py  # Validadores adicionales
├── convexity_config.py   # Configuración convexa
├── cot_config.py         # Chain of Thought
├── memory_config.py      # Configuración de memoria
├── model_config.py       # Wrapper de compatibilidad (reexporta unificado)
├── config_types.py       # Wrapper de compatibilidad (reexporta tipos)
├── config_semiotic.py    # Stub de compatibilidad
├── training_config.py    # Configuración de entrenamiento
└── unified_model_config.py # Configuración unificada
```

## 🚀 Uso Avanzado

### Configuración Completa
```python
from capibara.config import CapibaraConfig

# Crear configuración completa
config = CapibaraConfig(
    model=ModelConfig(...),
    training=TrainingConfig(...),
    pruning=PruningConfig(...),
    wandb=WandbConfig(...),
    modules=ModulesConfig(...),
    paths=PathsConfig(...)
)

# Cargar desde YAML
config = CapibaraConfig.from_yaml("config.yaml")

# Validar configuración
warnings = config.validate()
if warnings:
    for warning in warnings:
        print(f"Warning: {warning}")

# Convertir a diccionario
config_dict = config.to_dict()
```

### Validación Personalizada
```python
from capibara.config.config_validators import (
    estimate_model_memory,
    validate_device_compatibility,
    check_data_paths
)

# Estimar memoria del modelo
memory_gb = estimate_model_memory(config_dict) / 1e9
print(f"Memoria estimada: {memory_gb:.2f} GB")

# Validar compatibilidad de dispositivo
is_compatible = validate_device_compatibility(config.device)

# Verificar rutas de datos
valid_paths = check_data_paths(config.training)
```

## ⚙️ Características Clave

- **Validación automática**: Usando Pydantic para validación de tipos y valores
- **Carga flexible**: Soporte para TOML, JSON y YAML
- **Configuración jerárquica**: Acceso a valores anidados con notación de puntos
- **Gestión de memoria**: Estimación automática de recursos requeridos
- **Configuraciones especializadas**: Para diferentes componentes del sistema
- **Validación cruzada**: Verificación de compatibilidad entre módulos
- **Hot reload**: Recarga de configuraciones en tiempo de ejecución

## 🔍 Validadores Disponibles

- `ModelConfig`: Valida arquitectura del modelo y compatibilidad de dimensiones
- `TrainingConfig`: Verifica rutas de datos y parámetros de entrenamiento
- `MemoryConfig`: Estima uso de memoria y compatibilidad hardware
- `AdaptiveConfig`: Valida configuraciones de enrutamiento adaptativo
- `ConvexityConfig`: Verifica parámetros de optimización convexa

## 📖 Ejemplos de Configuración

Consulta el directorio `configs_toml/` para ejemplos completos de configuraciones en formato TOML.
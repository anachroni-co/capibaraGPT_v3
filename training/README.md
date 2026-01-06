# capibara/training - Training System

El módulo **training** implementa el sistema de entrenamiento avanzado de capibaraGPT-v2, incluyendo **Meta-Consensus**, **TPU v6e optimization**, y múltiples estrategias de training distribuido.

## 📋 Tabla de Contenidos

1. [Visión General](#visión-general)
2. [Componentes Principales](#componentes-principales)
3. [Meta-Consensus System](#meta-consensus-system)
4. [Training Strategies](#training-strategies)
5. [TPU v6e Training](#tpu-v6e-training)
6. [Quick Start](#quick-start)
7. [Configuración Avanzada](#configuración-avanzada)
8. [Optimizaciones](#optimizaciones)

---

## 🎯 Visión General

El sistema de training de capibaraGPT-v2 implementa estrategias avanzadas para entrenar modelos de lenguaje de alta calidad:

### Características Principales

- ⚡ **TPU v6e Optimized**: Training ultra-rápido en Google Cloud TPU v6e-64/256
- 🧠 **Meta-Consensus**: Sistema de consenso para combinar múltiples modelos/estrategias
- 🌐 **Distributed Training**: Data parallelism + Model parallelism + Expert parallelism
- 📊 **Multiple Strategies**: Hierarchical, Convexity, Incremental Soup, HuggingFace Integration
- 🔄 **Consensus Algorithms**: Byzantine-tolerant, Convex optimization, Federated learning
- ⚙️ **Cython Kernels**: Kernels C++ optimizados para operaciones críticas
- 📈 **Advanced Monitoring**: Dashboard en tiempo real con métricas detalladas

### Arquitectura del Sistema

```
┌──────────────────────────────────────────────────────────┐
│            Meta-Consensus Training System                │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Consensus Coordinator                   │   │
│  │  - Byzantine-tolerant voting                    │   │
│  │  - Model merging strategies                     │   │
│  │  - Quality gating                               │   │
│  └───────────────┬─────────────────────────────────┘   │
│                  │                                       │
│    ┌─────────────┼─────────────┐                       │
│    ▼             ▼             ▼                       │
│  ┌─────┐     ┌─────┐       ┌──────┐                   │
│  │Node1│     │Node2│  ...  │NodeN │                   │
│  │TPU  │     │TPU  │       │ TPU  │                   │
│  └─────┘     └─────┘       └──────┘                   │
│                                                           │
│  Each Node:                                             │
│  ├─> Training Strategy (Hierarchical/Convexity/Soup)   │
│  ├─> Local Model Updates                               │
│  ├─> Gradient Synchronization                          │
│  └─> Consensus Participation                           │
└──────────────────────────────────────────────────────────┘
```

---

## 🏗️ Componentes Principales

| Componente | Archivo | Propósito |
|------------|---------|-----------|
| **Meta-Consensus System** | `meta_consensus_system.py` | Coordina training distribuido con consenso |
| **TPU v6e Trainer** | `tpu_v6e_trainer.py` | Trainer optimizado para TPU v6e |
| **TPU v6e Config** | `tpu_v6e_config.py` | Configuración TPU v6e específica |
| **Consensus Algorithms** | `advanced_consensus_algorithms.py` | Algoritmos Byzantine-tolerant |
| **Hierarchical Strategy** | `hierarchical_training_strategy.py` | Training jerárquico multi-nivel |
| **Convexity Strategy** | `convexity_training_strategy.py` | Optimización convexa para training |
| **Incremental Soup** | `incremental_soup_strategy.py` | Model soups incrementales |
| **HuggingFace Integration** | `huggingface_consensus_strategy.py` | Integración con HF Transformers |
| **Hybrid Expert Router** | `hybrid_expert_router.py` | Routing de expertos durante training |
| **Monitoring Dashboard** | `monitoring_dashboard.py` | Dashboard de métricas en vivo |
| **Cython Kernels** | `cython_kernels/` | Kernels C++ optimizados |
| **Federated Consensus** | `federated_consensus/` | Federated learning distribuido |

---

## 🧠 Meta-Consensus System

El **Meta-Consensus System** es el núcleo del training distribuido, permitiendo entrenar múltiples variantes del modelo y combinarlas mediante consenso.

### Arquitectura Meta-Consensus

```python
from capibara.training import MetaConsensusSystem, ConsensusConfig

# Configurar sistema de consenso
consensus_config = ConsensusConfig(
    num_nodes=8,              # 8 TPUs participando
    voting_strategy="byzantine_tolerant",
    merge_strategy="weighted_average",
    quality_threshold=0.85,   # Mínimo quality score
    consensus_threshold=0.75  # 75% de nodos deben acordar
)

# Crear sistema
consensus_system = MetaConsensusSystem(consensus_config)

# Entrenar con consenso
results = consensus_system.train(
    data_path="gs://capibara-data/",
    num_epochs=10,
    strategies=["hierarchical", "convexity", "incremental_soup"]
)
```

### Algoritmos de Consenso Disponibles

#### 1. Byzantine-Tolerant Voting

Tolera hasta (n-1)/3 nodos fallidos o maliciosos:

```python
from capibara.training.advanced_consensus_algorithms import ByzantineConsensus

consensus = ByzantineConsensus(
    fault_tolerance=0.33,  # Tolera hasta 33% nodos fallidos
    verification_rounds=3
)

# Votación bizantina
consensus_model = consensus.vote(models=[model1, model2, model3])
```

#### 2. Convex Optimization Consensus

Optimización convexa para merging de modelos:

```python
from capibara.training.convexity_controller import ConvexityController

controller = ConvexityController(
    lambda_reg=0.01,
    max_iterations=100
)

# Encontrar combinación óptima
optimal_weights = controller.optimize(
    models=models,
    validation_data=val_data
)

merged_model = controller.merge(models, optimal_weights)
```

#### 3. Federated Consensus

Federated learning con differential privacy:

```python
from capibara.training.federated_consensus import FederatedConsensusStrategy

federated = FederatedConsensusStrategy(
    num_clients=20,
    privacy_budget=1.0,  # Epsilon para differential privacy
    aggregation_method="fedavg"
)

# Round de federated training
global_model = federated.train_round(
    global_model=model,
    client_data=client_datasets
)
```

---

## 📚 Training Strategies

### 1. Hierarchical Training Strategy

Training multi-nivel con especialización por nivel:

```python
from capibara.training.hierarchical_strategy import HierarchicalTrainingPipeline

# Configurar pipeline jerárquico
pipeline = HierarchicalTrainingPipeline(
    levels=[
        {  # Level 1: General knowledge
            "name": "general",
            "datasets": ["wikipedia", "books"],
            "epochs": 3
        },
        {  # Level 2: Domain specialization
            "name": "specialized",
            "datasets": ["academic", "legal", "medical"],
            "epochs": 2
        },
        {  # Level 3: Fine-tuning
            "name": "finetuning",
            "datasets": ["specific_task"],
            "epochs": 1
        }
    ]
)

# Ejecutar training jerárquico
final_model = pipeline.train(base_model)
```

### 2. Convexity Training Strategy

Optimización con garantías de convexidad:

```python
from capibara.training import ConvexityTrainingStrategy

strategy = ConvexityTrainingStrategy(
    regularization=0.01,
    convex_constraint=True,
    use_proximal_gradient=True
)

# Training con convexity constraints
model = strategy.train(
    model=model,
    train_data=train_data,
    epochs=10
)
```

### 3. Incremental Soup Strategy

Model soups incrementales (combina checkpoints):

```python
from capibara.training import IncrementalSoupStrategy

soup = IncrementalSoupStrategy(
    merging_method="uniform",  # uniform, weighted, greedy
    evaluation_metric="perplexity"
)

# Agregar checkpoints al soup
soup.add_checkpoint("checkpoint_epoch_1.pkl")
soup.add_checkpoint("checkpoint_epoch_2.pkl")
soup.add_checkpoint("checkpoint_epoch_3.pkl")

# Crear modelo soup
souped_model = soup.merge(validation_data=val_data)
```

### 4. HuggingFace Consensus Strategy

Integración con HuggingFace Transformers:

```python
from capibara.training import HuggingFaceConsensusStrategy

hf_strategy = HuggingFaceConsensusStrategy(
    model_name="bert-base-uncased",
    num_consensus_models=5,
    merge_method="weighted_average"
)

# Training con HF integration
model = hf_strategy.train(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    epochs=3
)
```

---

## ⚡ TPU v6e Training

### Configuración TPU v6e-64

```python
from capibara.training import TPUv6eTrainer, TPUv6eConfig

# Configurar TPU v6e
tpu_config = TPUv6eConfig(
    tpu_type="v6e-64",
    mesh_shape=(8, 8),  # 64 chips en mesh 8x8
    batch_size=128,
    use_bf16=True,
    enable_flash_attention=True,
    xla_flags={
        "xla_tpu_enable_data_parallelism": True,
        "xla_tpu_enable_async_collective_fusion": True
    }
)

# Crear trainer
trainer = TPUv6eTrainer(tpu_config)

# Training en TPU
results = trainer.train(
    model=model,
    train_data="gs://capibara-data/training/",
    eval_data="gs://capibara-data/validation/",
    num_steps=100000,
    checkpoint_every=1000
)
```

### Optimizaciones TPU v6e

```python
from capibara.training import TPUv6ConsensusOptimizer

# Optimizer especializado para TPU v6e
optimizer = TPUv6ConsensusOptimizer(
    learning_rate=3e-4,
    use_bf16_accumulation=True,
    gradient_clipping=1.0,
    enable_xla_fusion=True
)

# Características:
# - BFloat16 accumulation nativa
# - XLA fusion automática
# - Collective operations optimizadas
# - Memory-efficient gradient checkpointing
```

---

## 🚀 Quick Start

### Training Básico (CPU/GPU)

```python
from capibara.training import UnifiedTrainer
from capibara.core import ModularCapibaraModel, ModularConfig

# 1. Configurar modelo
config = ModularConfig.from_toml("config/development/config.toml")
model = ModularCapibaraModel(config)

# 2. Crear trainer
trainer = UnifiedTrainer(
    model=model,
    learning_rate=3e-4,
    batch_size=32
)

# 3. Entrenar
trainer.train(
    train_data="data/train/",
    eval_data="data/eval/",
    num_epochs=10,
    checkpoint_dir="checkpoints/"
)
```

### Training con Meta-Consensus (TPU)

```python
from capibara.training import MetaConsensusSystem, ConsensusConfig

# 1. Configurar consenso
consensus_config = ConsensusConfig(
    num_nodes=8,  # 8 TPUs
    voting_strategy="byzantine_tolerant",
    merge_strategy="convex_optimization"
)

# 2. Crear sistema de consenso
consensus = MetaConsensusSystem(consensus_config)

# 3. Training distribuido con consenso
results = consensus.train(
    model=model,
    data_path="gs://capibara-data/",
    strategies=["hierarchical", "convexity"],
    num_epochs=20
)

# 4. Obtener modelo consensuado
final_model = results.consensus_model
```

---

## ⚙️ Configuración Avanzada

### Training con Todas las Estrategias

```python
from capibara.training import IntegratedConsensusStrategy

# Integrar múltiples estrategias
integrated = IntegratedConsensusStrategy(
    strategies={
        "hierarchical": {
            "enabled": True,
            "weight": 0.4
        },
        "convexity": {
            "enabled": True,
            "weight": 0.3
        },
        "incremental_soup": {
            "enabled": True,
            "weight": 0.3
        }
    },
    consensus_threshold=0.8
)

# Training integrado
model = integrated.train(
    base_model=model,
    train_data=train_data,
    epochs=15
)
```

### Monitoring en Tiempo Real

```python
from capibara.training import MonitoringDashboard

# Crear dashboard
dashboard = MonitoringDashboard(
    port=8080,
    update_frequency=10,  # segundos
    metrics=[
        "loss",
        "perplexity",
        "gradient_norm",
        "consensus_score",
        "tpu_utilization",
        "memory_usage"
    ]
)

# Iniciar dashboard
dashboard.start()

# Training con monitoring
trainer = UnifiedTrainer(model=model, dashboard=dashboard)
trainer.train(...)

# Ver dashboard en http://localhost:8080
```

### Data Preprocessing

```python
from capibara.training.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(
    tokenizer=tokenizer,
    max_length=2048,
    padding="max_length",
    num_workers=8
)

# Preprocesar datos
processed_data = preprocessor.process(
    input_path="data/raw/",
    output_path="data/processed/",
    cache=True
)
```

---

## 🔧 Optimizaciones

### Cython Kernels

Kernels C++ optimizados para operaciones críticas:

```python
from capibara.training.cython_kernels import (
    fast_attention,
    optimized_matmul,
    sparse_softmax
)

# Usar kernels optimizados
# 10-50x más rápido que implementaciones Python puras
attn_output = fast_attention(q, k, v)
```

### Gradient Checkpointing

Reduce memoria a costa de un poco de cómputo:

```python
from capibara.training import enable_gradient_checkpointing

# Habilitar gradient checkpointing
model = enable_gradient_checkpointing(
    model=model,
    checkpoint_every_n_layers=2
)

# Reduce memoria ~50% con ~20% más cómputo
```

### Mixed Precision Training

```python
from capibara.training import MixedPrecisionTrainer

trainer = MixedPrecisionTrainer(
    model=model,
    precision="bf16",  # bf16, fp16, or mixed
    loss_scaling=True,
    dynamic_loss_scale=True
)

# 2-3x faster training, ~50% menos memoria
```

---

## 📊 Métricas y Evaluación

### Métricas Disponibles

```python
from capibara.training import TrainingMetrics

metrics = TrainingMetrics(
    track=[
        "loss",              # Training loss
        "perplexity",        # Language model perplexity
        "gradient_norm",     # Gradient norm
        "learning_rate",     # Current LR
        "consensus_score",   # Meta-consensus agreement
        "expert_utilization",  # MoE expert usage
        "throughput",        # Samples/second
        "memory_usage"       # GPU/TPU memory
    ]
)

# Registrar durante training
for batch in dataloader:
    loss = train_step(batch)
    metrics.log(loss=loss, step=step)

# Exportar métricas
metrics.export("metrics.json")
```

### Integration con Weights & Biases

```python
import wandb
from capibara.training import WandbIntegration

# Setup W&B
wandb.init(project="capibaraGPT-v2", name="tpu-training-run-001")

# Integrar con trainer
trainer = UnifiedTrainer(
    model=model,
    wandb_project="capibaraGPT-v2"
)

# Métricas se loggean automáticamente a W&B
```

---

## 🔍 Debugging y Troubleshooting

### Error: "TPU not found"

```bash
# Verificar TPU disponible
python -c "import jax; print(jax.devices())"

# Configurar variables de entorno
export JAX_PLATFORMS=tpu
export TPU_NAME=your-tpu-name
```

### Error: "Consensus timeout"

```python
# Aumentar timeout
consensus_config = ConsensusConfig(
    consensus_timeout=600,  # 10 minutos
    max_retries=5
)
```

### Error: "Out of memory"

Soluciones:
1. Reducir `batch_size`
2. Habilitar `gradient_checkpointing`
3. Usar `accumulation_steps` para simular batch size más grande
4. Reducir `sequence_length`

```python
trainer = UnifiedTrainer(
    batch_size=16,  # Reducir de 128
    gradient_accumulation_steps=8,  # Simula batch_size=128
    use_gradient_checkpointing=True
)
```

---

## 📁 Subdirectorios

- **`cython_kernels/`**: Kernels C++/Cython optimizados
- **`data_lineage/`**: Tracking de linaje de datos
- **`data_preprocessing/`**: Preprocesamiento de datos
- **`federated_consensus/`**: Federated learning
- **`hierarchical_strategy/`**: Training jerárquico
- **`optimizations/`**: Optimizaciones de training

---

## 📚 Referencias

- [TPU v6e Trainer](tpu_v6e_trainer.py) - Trainer para TPU v6e
- [Meta-Consensus System](meta_consensus_system.py) - Sistema de consenso
- [Consensus Algorithms](advanced_consensus_algorithms.py) - Algoritmos bizantinos
- [Monitoring Dashboard](monitoring_dashboard.py) - Dashboard de métricas
- [Config Manager](config_manager.py) - Gestión de configuración

---

## 🆘 Soporte

Para problemas con training:
1. Revisa logs en `logs/training.log`
2. Verifica dashboard de monitoring
3. Consulta métricas de consenso
4. Abre issue en GitHub con logs completos

---

**Última actualización**: 2025-11-16
**Versión del sistema**: v2.0.0

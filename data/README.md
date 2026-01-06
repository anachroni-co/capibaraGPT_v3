# capibara/data - Data Pipeline & Datasets

El módulo **data** proporciona el sistema completo de manejo de datos, incluyendo loaders, processors, datasets registry, y orquestación de pipeline de datos.

## 📋 Tabla de Contenidos

1. [Visión General](#visión-general)
2. [Arquitectura del Pipeline](#arquitectura-del-pipeline)
3. [Datasets](#datasets)
4. [Loaders](#loaders)
5. [Processors](#processors)
6. [Quick Start](#quick-start)
7. [Dataset Registry](#dataset-registry)
8. [Data Orchestrator](#data-orchestrator)

---

## 🎯 Visión General

El sistema de datos de capibaraGPT-v2 maneja la carga, procesamiento y preparación de datos para entrenamiento e inferencia.

### Componentes Principales

```
capibara/data/
├── core/                    # Componentes core (loaders, processors base)
├── capibara_datasets/       # Datasets curados de Capibara
├── datasets/                # Datasets específicos por dominio
├── loaders/                 # Loaders de datos (reexports de core/)
├── processors/              # Processors de datos (reexports de core/)
├── scrapers/                # Web scrapers para datasets
├── tools/                   # Utilidades de datos
├── configs/                 # Configuraciones de datasets
├── dataset_registry.py      # Registry centralizado
└── ultra_data_orchestrator.py  # Orquestador avanzado
```

### Pipeline de Datos

```
┌─────────────────────────────────────────────────────────┐
│              Data Pipeline Architecture                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Raw Data Sources                                       │
│  ├─> Local files                                        │
│  ├─> GCS buckets (gs://...)                            │
│  ├─> HuggingFace Hub                                    │
│  ├─> Web scraping                                       │
│  └─> APIs                                               │
│                ↓                                         │
│         ┌──────────────┐                                │
│         │   Scrapers   │                                │
│         └──────┬───────┘                                │
│                ↓                                         │
│         ┌──────────────┐                                │
│         │   Loaders    │ ← Dataset Registry             │
│         └──────┬───────┘                                │
│                ↓                                         │
│         ┌──────────────┐                                │
│         │  Processors  │ (Tokenization, Chunking, etc.) │
│         └──────┬───────┘                                │
│                ↓                                         │
│         ┌──────────────┐                                │
│         │ Orchestrator │ (Batching, Shuffling, etc.)    │
│         └──────┬───────┘                                │
│                ↓                                         │
│            Training                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Datasets

capibaraGPT-v2 soporta múltiples categorías de datasets:

### capibara_datasets/ (Datasets Curados)

Datasets específicamente curados para Capibara:

```python
from capibara.data.capibara_datasets import load_dataset

# Cargar dataset curado
dataset = load_dataset("capibara/spanish-literature")

# Datasets disponibles:
# - capibara/spanish-literature
# - capibara/academic-papers
# - capibara/minority-languages
# - ... (ver capibara_datasets/README.md)
```

Ver [capibara_datasets/README.md](capibara_datasets/README.md) para lista completa.

### datasets/ (Datasets por Dominio)

Datasets organizados por dominio:

| Dominio | Directorio | Descripción |
|---------|------------|-------------|
| Academic | `datasets/academic/` | Papers científicos |
| Economics | `datasets/economics/` | Datos económicos |
| Engineering | `datasets/engineering_design/` | Diseño de ingeniería |
| Genomic | `datasets/genomic/` | Datos genómicos |
| Historical | `datasets/historical/` | Documentos históricos |
| Humor | `datasets/humor/` | Contenido humorístico |
| Legal | `datasets/legal/` | Textos legales |
| Mathematics | `datasets/mathematics/` | Problemas matemáticos |
| Physics | `datasets/physics/` | Textos de física |
| Robotics | `datasets/robotics/` | Datos de robótica |
| Spanish Community | `datasets/spanish_community/` | Comunidades hispanohablantes |
| Spanish Government | `datasets/spanish_government/` | Documentos gubernamentales |
| Vision | `datasets/vision/` | Datasets de visión |
| Multimodal | `datasets/multimodal/` | Datos multimodales |

```python
from capibara.data.datasets import academic

# Cargar dataset específico
dataset = academic.load_arxiv_papers(
    categories=["cs.AI", "cs.LG"],
    years=[2023, 2024]
)
```

---

## 📥 Loaders

Los loaders cargan datos desde diferentes fuentes:

### Core Loaders

```python
from capibara.data.loaders import (
    TextLoader,
    JSONLoader,
    CSVLoader,
    ParquetLoader,
    HuggingFaceLoader,
    GCSLoader
)

# Cargar texto
text_loader = TextLoader(
    file_path="data/text/document.txt",
    encoding="utf-8"
)
texts = text_loader.load()

# Cargar desde HuggingFace
hf_loader = HuggingFaceLoader(
    dataset_name="wikitext",
    split="train"
)
dataset = hf_loader.load()

# Cargar desde GCS
gcs_loader = GCSLoader(
    bucket="capibara-data",
    prefix="training/",
    file_pattern="*.jsonl"
)
data = gcs_loader.load()
```

### Loader Avanzado

```python
from capibara.data.loaders import DataLoader

# Loader con cache y preprocessing
loader = DataLoader(
    data_source="gs://capibara-data/training/",
    cache_dir=".cache/data/",
    num_workers=8,
    prefetch_factor=4
)

# Iterar sobre datos
for batch in loader:
    # batch ya está preprocesado y batcheado
    train_step(batch)
```

---

## ⚙️ Processors

Los processors transforman datos raw en formato de training:

### Core Processors

```python
from capibara.data.processors import (
    TokenizationProcessor,
    ChunkingProcessor,
    NormalizationProcessor,
    AugmentationProcessor
)

# Tokenization
tokenizer = TokenizationProcessor(
    tokenizer_name="gpt2",
    max_length=2048,
    padding="max_length"
)
tokens = tokenizer.process(texts)

# Chunking (para documentos largos)
chunker = ChunkingProcessor(
    chunk_size=512,
    overlap=50,
    strategy="sliding_window"  # sliding_window, semantic, sentence
)
chunks = chunker.process(long_document)

# Normalización
normalizer = NormalizationProcessor(
    lowercase=True,
    remove_accents=False,
    remove_punctuation=False
)
normalized = normalizer.process(texts)

# Augmentation (para más datos)
augmenter = AugmentationProcessor(
    methods=["synonym_replacement", "back_translation"],
    augmentation_factor=2
)
augmented = augmenter.process(texts)
```

### Pipeline de Procesamiento

```python
from capibara.data.processors import ProcessingPipeline

# Crear pipeline
pipeline = ProcessingPipeline([
    NormalizationProcessor(),
    ChunkingProcessor(chunk_size=512),
    TokenizationProcessor(tokenizer_name="gpt2"),
    AugmentationProcessor(augmentation_factor=1.5)
])

# Procesar datos
processed_data = pipeline.process(raw_data)

# Pipeline se puede guardar y cargar
pipeline.save("pipelines/my_pipeline.json")
```

---

## 🚀 Quick Start

### Cargar y Procesar Dataset

```python
from capibara.data import load_dataset, DataProcessor

# 1. Cargar dataset
dataset = load_dataset("capibara/spanish-literature")

# 2. Crear processor
processor = DataProcessor(
    tokenizer="gpt2",
    max_length=2048,
    chunk_strategy="sliding_window"
)

# 3. Procesar dataset
processed = processor.process(dataset)

# 4. Crear DataLoader para training
from torch.utils.data import DataLoader as TorchDataLoader

dataloader = TorchDataLoader(
    processed,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# 5. Training
for batch in dataloader:
    train_step(batch)
```

### Pipeline Completo End-to-End

```python
from capibara.data import UltraDataOrchestrator

# Crear orquestador
orchestrator = UltraDataOrchestrator(
    datasets=[
        "capibara/spanish-literature",
        "datasets/academic/arxiv",
        "datasets/legal/spanish-law"
    ],
    tokenizer="gpt2",
    max_length=2048,
    batch_size=32
)

# Preparar pipeline
orchestrator.prepare_pipeline(
    cache_dir=".cache/",
    num_workers=8
)

# Iterar sobre datos mixtos
for batch in orchestrator:
    train_step(batch)

# Métricas del pipeline
metrics = orchestrator.get_metrics()
print(f"Total samples: {metrics['total_samples']}")
print(f"Datasets used: {metrics['datasets_used']}")
```

---

## 📋 Dataset Registry

El registry centraliza todos los datasets disponibles:

```python
from capibara.data import DatasetRegistry

# Obtener registry
registry = DatasetRegistry()

# Listar todos los datasets
all_datasets = registry.list_all()
print(f"Datasets disponibles: {len(all_datasets)}")

# Buscar por dominio
academic_datasets = registry.find_by_domain("academic")
legal_datasets = registry.find_by_domain("legal")

# Buscar por idioma
spanish_datasets = registry.find_by_language("es")

# Buscar por tags
multilingual = registry.find_by_tags(["multilingual", "minority_languages"])

# Registrar nuevo dataset
registry.register(
    name="my-custom-dataset",
    path="data/custom/",
    domain="custom",
    language="es",
    tags=["specialized", "small"]
)

# Cargar desde registry
dataset = registry.load("my-custom-dataset")
```

### Metadata de Datasets

```python
# Obtener metadata
metadata = registry.get_metadata("capibara/spanish-literature")

print(f"Name: {metadata['name']}")
print(f"Domain: {metadata['domain']}")
print(f"Language: {metadata['language']}")
print(f"Size: {metadata['size']} samples")
print(f"License: {metadata['license']}")
print(f"Description: {metadata['description']}")
```

---

## 🎼 Data Orchestrator

El Ultra Data Orchestrator maneja pipelines complejos:

```python
from capibara.data import UltraDataOrchestrator

# Configurar orquestador avanzado
orchestrator = UltraDataOrchestrator(
    # Datasets múltiples
    datasets=[
        {"name": "capibara/spanish-literature", "weight": 0.4},
        {"name": "datasets/academic/arxiv", "weight": 0.3},
        {"name": "datasets/legal/spanish-law", "weight": 0.2},
        {"name": "datasets/spanish_community/forums", "weight": 0.1}
    ],

    # Preprocessing
    tokenizer="gpt2",
    max_length=2048,
    preprocessing_pipeline=[
        "normalization",
        "chunking",
        "tokenization"
    ],

    # Batching strategy
    batch_size=32,
    dynamic_batching=True,  # Ajusta batch size dinámicamente
    max_batch_tokens=4096,  # Límite de tokens por batch

    # Augmentation
    use_augmentation=True,
    augmentation_factor=1.5,

    # Caching
    cache_preprocessed=True,
    cache_dir=".cache/orchestrator/",

    # Performance
    num_workers=8,
    prefetch_factor=4,
    pin_memory=True
)

# Preparar y validar pipeline
orchestrator.prepare()
orchestrator.validate()  # Verifica que todo esté correcto

# Iterar
for epoch in range(num_epochs):
    for batch in orchestrator:
        loss = train_step(batch)

    # Métricas por epoch
    epoch_metrics = orchestrator.get_epoch_metrics()
    print(f"Epoch {epoch}: {epoch_metrics}")

# Estadísticas finales
stats = orchestrator.get_statistics()
print(f"Total batches: {stats['total_batches']}")
print(f"Total tokens: {stats['total_tokens']}")
print(f"Avg batch size: {stats['avg_batch_size']:.1f}")
```

### Features Avanzados del Orchestrator

```python
# Curriculum learning
orchestrator.enable_curriculum_learning(
    start_difficulty=0.3,
    end_difficulty=1.0,
    num_steps=10000
)

# Data mixing estratégico
orchestrator.set_mixing_strategy(
    strategy="temperature_based",  # uniform, weighted, temperature_based
    temperature=0.5
)

# Deduplicación automática
orchestrator.enable_deduplication(
    method="minhash",  # minhash, exact, semantic
    threshold=0.85
)

# Quality filtering
orchestrator.enable_quality_filter(
    min_length=50,
    max_length=10000,
    language_detection=True,
    profanity_filter=True
)
```

---

## 🔧 Scrapers

Scrapers para obtener datos de la web:

```python
from capibara.data.scrapers import (
    WebScraper,
    WikipediaScraper,
    ArxivScraper,
    RedditScraper
)

# Wikipedia scraper
wiki_scraper = WikipediaScraper(
    languages=["es", "ca", "gl", "eu"],  # Español, Catalán, Gallego, Euskera
    categories=["Literatura", "Historia", "Ciencia"]
)
wiki_data = wiki_scraper.scrape(max_articles=10000)

# ArXiv scraper
arxiv_scraper = ArxivScraper(
    categories=["cs.AI", "cs.LG", "cs.CL"],
    start_date="2024-01-01",
    end_date="2024-12-31"
)
papers = arxiv_scraper.scrape()

# Reddit scraper
reddit_scraper = RedditScraper(
    subreddits=["es", "spain", "argentina", "mexico"],
    post_limit=1000,
    include_comments=True
)
reddit_data = reddit_scraper.scrape()
```

---

## 🛠️ Tools

Utilidades para manejo de datos:

```python
from capibara.data.tools import (
    data_statistics,
    data_validator,
    data_cleaner,
    data_splitter
)

# Estadísticas
stats = data_statistics.compute(dataset)
print(f"Total samples: {stats['num_samples']}")
print(f"Avg length: {stats['avg_length']:.1f}")
print(f"Vocab size: {stats['vocab_size']}")

# Validación
is_valid, errors = data_validator.validate(
    dataset,
    checks=["format", "encoding", "duplicates", "quality"]
)

# Limpieza
cleaned = data_cleaner.clean(
    dataset,
    remove_duplicates=True,
    remove_empty=True,
    fix_encoding=True
)

# Splitting
train, val, test = data_splitter.split(
    dataset,
    splits=[0.8, 0.1, 0.1],
    stratify_by="domain",
    random_seed=42
)
```

---

## 📚 Configuración

### Config Files

```python
from capibara.data.configs import load_config

# Cargar configuración de dataset
config = load_config("configs/academic_dataset.yaml")

# Config incluye:
# - Source paths
# - Preprocessing parameters
# - Tokenizer settings
# - Batching configuration
```

### Ejemplo de Config

```yaml
# configs/my_dataset.yaml
name: "custom-academic-dataset"
domain: "academic"
language: "es"

source:
  type: "local"
  path: "data/academic/"
  pattern: "*.txt"

preprocessing:
  tokenizer: "gpt2"
  max_length: 2048
  chunking:
    strategy: "sliding_window"
    chunk_size: 512
    overlap: 50

batching:
  batch_size: 32
  shuffle: true
  num_workers: 8

quality:
  min_length: 50
  max_length: 10000
  remove_duplicates: true
```

---

## 🔗 Compatibilidad

El módulo mantiene compatibilidad backwards:

```python
# Imports legacy funcionan
from capibara.data.datasets import load_dataset  # ✅ Funciona
from capibara.data.loaders import DataLoader     # ✅ Funciona
from capibara.data.processors import Tokenizer   # ✅ Funciona

# Imports nuevos recomendados
from capibara.data.core import DataLoader        # ✅ Recomendado
from capibara.data.capibara_datasets import load_dataset  # ✅ Recomendado
```

---

## 📊 Performance

### Benchmarks (Loading + Processing)

| Dataset Size | Workers | Throughput | Memory |
|--------------|---------|------------|--------|
| 1M samples | 1 | 500 samples/s | 2GB |
| 1M samples | 4 | 1800 samples/s | 4GB |
| 1M samples | 8 | 3200 samples/s | 6GB |
| 10M samples | 8 | 3000 samples/s | 8GB |

### Optimization Tips

1. **Use caching**: `cache_preprocessed=True`
2. **Increase workers**: `num_workers=8`
3. **Enable prefetching**: `prefetch_factor=4`
4. **Use parquet**: Más rápido que JSON/CSV
5. **GCS parallel loading**: Múltiples archivos en paralelo

---

## 🆘 Troubleshooting

### Error: "Dataset not found"

```python
# Verificar registry
from capibara.data import DatasetRegistry
registry = DatasetRegistry()
available = registry.list_all()
print(f"Available datasets: {available}")
```

### Slow Loading

- Aumentar `num_workers`
- Usar formato Parquet en lugar de JSON
- Habilitar cache
- Usar GCS en lugar de local para datasets grandes

### Out of Memory

- Reducir `batch_size`
- Reducir `prefetch_factor`
- Usar streaming mode (no cargar todo en memoria)
- Procesar en chunks más pequeños

---

**Última actualización**: 2025-11-16
**Versión del sistema**: v2.0.0

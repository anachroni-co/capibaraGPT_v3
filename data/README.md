# capibara/data - Data Pipeline & Datasets

The **data** module provides the complete data handling system, including loaders, processors, dataset registry, and data pipeline orchestration.

##  Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Datasets](#datasets)
4. [Loaders](#loaders)
5. [Processors](#processors)
6. [Quick Start](#quick-start)
7. [Dataset Registry](#dataset-registry)
8. [Data Orchestrator](#data-orchestrator)

---

##  Overview

The CapibaraGPT v3 data system handles data loading, processing, and preparation for training and inference.

### Main Components

```
capibara/data/
├── core/                    # Core components (loaders, base processors)
├── datasets/                # Domain-specific datasets (academic, legal, etc.)
├── loaders/                 # Data loaders (re-exports from core/)
├── processors/              # Data processors (re-exports from core/)
├── scrapers/                # Web scrapers for datasets
├── tools/                   # Data utilities
├── configs/                 # Dataset configurations
├── dataset_registry.py      # Centralized registry
└── ultra_data_orchestrator.py  # Advanced orchestrator
```

### Data Pipeline

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

##  Datasets

CapibaraGPT v3 supports multiple dataset categories organized in `datasets/`:

### datasets/ (Domain Datasets)

Datasets organized by domain:

| Domain | Directory | Description |
|--------|-----------|-------------|
| Academic | `datasets/academic/` | Scientific papers |
| Economics | `datasets/economics/` | Economic data |
| Engineering | `datasets/engineering_design/` | Engineering design |
| Genomic | `datasets/genomic/` | Genomic data |
| Historical | `datasets/historical/` | Historical documents |
| Humor | `datasets/humor/` | Humorous content |
| Legal | `datasets/legal/` | Legal texts |
| Mathematics | `datasets/mathematics/` | Mathematical problems |
| Physics | `datasets/physics/` | Physics texts |
| Robotics | `datasets/robotics/` | Robotics data |
| Spanish Community | `datasets/spanish_community/` | Spanish-speaking communities |
| Spanish Government | `datasets/spanish_government/` | Government documents |
| Vision | `datasets/vision/` | Vision datasets |
| Multimodal | `datasets/multimodal/` | Multimodal data |

```python
from capibara.data.datasets import academic

# Load specific dataset
dataset = academic.load_arxiv_papers(
    categories=["cs.AI", "cs.LG"],
    years=[2023, 2024]
)
```

---

##  Loaders

Loaders load data from different sources:

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

# Load text
text_loader = TextLoader(
    file_path="data/text/document.txt",
    encoding="utf-8"
)
texts = text_loader.load()

# Load from HuggingFace
hf_loader = HuggingFaceLoader(
    dataset_name="wikitext",
    split="train"
)
dataset = hf_loader.load()

# Load from GCS
gcs_loader = GCSLoader(
    bucket="capibara-data",
    prefix="training/",
    file_pattern="*.jsonl"
)
data = gcs_loader.load()
```

### Advanced Loader

```python
from capibara.data.loaders import DataLoader

# Loader with cache and preprocessing
loader = DataLoader(
    data_source="gs://capibara-data/training/",
    cache_dir=".cache/data/",
    num_workers=8,
    prefetch_factor=4
)

# Iterate over data
for batch in loader:
    # batch is already preprocessed and batched
    train_step(batch)
```

---

## ️ Processors

Processors transform raw data into training format:

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

# Chunking (for long documents)
chunker = ChunkingProcessor(
    chunk_size=512,
    overlap=50,
    strategy="sliding_window"  # sliding_window, semantic, sentence
)
chunks = chunker.process(long_document)

# Normalization
normalizer = NormalizationProcessor(
    lowercase=True,
    remove_accents=False,
    remove_punctuation=False
)
normalized = normalizer.process(texts)

# Augmentation (for more data)
augmenter = AugmentationProcessor(
    methods=["synonym_replacement", "back_translation"],
    augmentation_factor=2
)
augmented = augmenter.process(texts)
```

### Processing Pipeline

```python
from capibara.data.processors import ProcessingPipeline

# Create pipeline
pipeline = ProcessingPipeline([
    NormalizationProcessor(),
    ChunkingProcessor(chunk_size=512),
    TokenizationProcessor(tokenizer_name="gpt2"),
    AugmentationProcessor(augmentation_factor=1.5)
])

# Process data
processed_data = pipeline.process(raw_data)

# Pipeline can be saved and loaded
pipeline.save("pipelines/my_pipeline.json")
```

---

##  Quick Start

### Load and Process Dataset

```python
from capibara.data import load_dataset, DataProcessor

# 1. Load dataset
dataset = load_dataset("capibara/spanish-literature")

# 2. Create processor
processor = DataProcessor(
    tokenizer="gpt2",
    max_length=2048,
    chunk_strategy="sliding_window"
)

# 3. Process dataset
processed = processor.process(dataset)

# 4. Create DataLoader for training
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

### Complete End-to-End Pipeline

```python
from capibara.data import UltraDataOrchestrator

# Create orchestrator
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

# Prepare pipeline
orchestrator.prepare_pipeline(
    cache_dir=".cache/",
    num_workers=8
)

# Iterate over mixed data
for batch in orchestrator:
    train_step(batch)

# Pipeline metrics
metrics = orchestrator.get_metrics()
print(f"Total samples: {metrics['total_samples']}")
print(f"Datasets used: {metrics['datasets_used']}")
```

---

##  Dataset Registry

The registry centralizes all available datasets:

```python
from capibara.data import DatasetRegistry

# Get registry
registry = DatasetRegistry()

# List all datasets
all_datasets = registry.list_all()
print(f"Available datasets: {len(all_datasets)}")

# Search by domain
academic_datasets = registry.find_by_domain("academic")
legal_datasets = registry.find_by_domain("legal")

# Search by language
spanish_datasets = registry.find_by_language("es")

# Search by tags
multilingual = registry.find_by_tags(["multilingual", "minority_languages"])

# Register new dataset
registry.register(
    name="my-custom-dataset",
    path="data/custom/",
    domain="custom",
    language="es",
    tags=["specialized", "small"]
)

# Load from registry
dataset = registry.load("my-custom-dataset")
```

### Dataset Metadata

```python
# Get metadata
metadata = registry.get_metadata("capibara/spanish-literature")

print(f"Name: {metadata['name']}")
print(f"Domain: {metadata['domain']}")
print(f"Language: {metadata['language']}")
print(f"Size: {metadata['size']} samples")
print(f"License: {metadata['license']}")
print(f"Description: {metadata['description']}")
```

---

##  Data Orchestrator

The Ultra Data Orchestrator handles complex pipelines:

```python
from capibara.data import UltraDataOrchestrator

# Configure advanced orchestrator
orchestrator = UltraDataOrchestrator(
    # Multiple datasets
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
    dynamic_batching=True,  # Dynamically adjust batch size
    max_batch_tokens=4096,  # Token limit per batch

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

# Prepare and validate pipeline
orchestrator.prepare()
orchestrator.validate()  # Verify everything is correct

# Iterate
for epoch in range(num_epochs):
    for batch in orchestrator:
        loss = train_step(batch)

    # Metrics per epoch
    epoch_metrics = orchestrator.get_epoch_metrics()
    print(f"Epoch {epoch}: {epoch_metrics}")

# Final statistics
stats = orchestrator.get_statistics()
print(f"Total batches: {stats['total_batches']}")
print(f"Total tokens: {stats['total_tokens']}")
print(f"Avg batch size: {stats['avg_batch_size']:.1f}")
```

### Advanced Orchestrator Features

```python
# Curriculum learning
orchestrator.enable_curriculum_learning(
    start_difficulty=0.3,
    end_difficulty=1.0,
    num_steps=10000
)

# Strategic data mixing
orchestrator.set_mixing_strategy(
    strategy="temperature_based",  # uniform, weighted, temperature_based
    temperature=0.5
)

# Automatic deduplication
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

##  Scrapers

Scrapers for obtaining data from the web:

```python
from capibara.data.scrapers import (
    WebScraper,
    WikipediaScraper,
    ArxivScraper,
    RedditScraper
)

# Wikipedia scraper
wiki_scraper = WikipediaScraper(
    languages=["es", "ca", "gl", "eu"],  # Spanish, Catalan, Galician, Basque
    categories=["Literature", "History", "Science"]
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

## ️ Tools

Utilities for data handling:

```python
from capibara.data.tools import (
    data_statistics,
    data_validator,
    data_cleaner,
    data_splitter
)

# Statistics
stats = data_statistics.compute(dataset)
print(f"Total samples: {stats['num_samples']}")
print(f"Avg length: {stats['avg_length']:.1f}")
print(f"Vocab size: {stats['vocab_size']}")

# Validation
is_valid, errors = data_validator.validate(
    dataset,
    checks=["format", "encoding", "duplicates", "quality"]
)

# Cleaning
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

##  Configuration

### Config Files

```python
from capibara.data.configs import load_config

# Load dataset configuration
config = load_config("configs/academic_dataset.yaml")

# Config includes:
# - Source paths
# - Preprocessing parameters
# - Tokenizer settings
# - Batching configuration
```

### Config Example

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

##  Compatibility

The module maintains backwards compatibility:

```python
# Import patterns
from capibara.data.datasets import load_dataset  #  Works
from capibara.data.loaders import DataLoader     #  Works
from capibara.data.processors import Tokenizer   #  Works
from capibara.data.core import DataLoader        #  Recommended
```

---

##  Performance

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
4. **Use parquet**: Faster than JSON/CSV
5. **GCS parallel loading**: Multiple files in parallel

---

## 🆘 Troubleshooting

### Error: "Dataset not found"

```python
# Verify registry
from capibara.data import DatasetRegistry
registry = DatasetRegistry()
available = registry.list_all()
print(f"Available datasets: {available}")
```

### Slow Loading

- Increase `num_workers`
- Use Parquet format instead of JSON
- Enable cache
- Use GCS instead of local for large datasets

### Out of Memory

- Reduce `batch_size`
- Reduce `prefetch_factor`
- Use streaming mode (don't load everything in memory)
- Process in smaller chunks

---

**Last updated**: 2025-11-16
**System version**: v3.0.0

## Example quick

Example (pseudo-flujo) para preparar datos:

```text
1) Coloca files .jsonl en data/raw/
2) Referencia esos files en tu config de entrenamiento
3) Ejecuta el pipeline de preprocesamiento
```

## Issues por hacer

- [ ] # Placeholder for real-time API loading - `data\ultra_data_orchestrator.py:625`
- [ ] return {"type": "fallback", "name": dataset_name, "data": "placeholder_data"} - `data\ultra_data_orchestrator.py:649`
- [ ] "gc_content_estimate": np.random.uniform(0.3, 0.7),  # Mock for now - `data\datasets\genomic\alphagenome_training_Generator.py:167`
- [ ] "repeat_content_estimate": np.random.uniform(0.1, 0.5)  # Mock for now - `data\datasets\genomic\alphagenome_training_Generator.py:168`
- [ ] # For now, Generate simulated data - `data\datasets\genomic\genomic_datasets.py:119`
- [ ] # Generate simulated expression matrix - `data\datasets\genomic\genomic_datasets.py:184`
- [ ] # Create placeholder and instructions - `data\datasets\multimodal\emotional_audio_datasets.py:426`
- [ ] # Placeholder for future dataset expansions - `data\datasets\multimodal\__init__.py:7`
- [ ] # Simulated experimental data - `data\datasets\physics\__init__.py:113`
- [ ] # Simulate particle collision data - `data\datasets\physics\__init__.py:117`
- [ ] # Simulated loading - in real implementation would load actual data - `data\datasets\robotics\__init__.py:80`
- [ ] 'data': f"[Simulated {dataset_name} data]", - `data\datasets\robotics\__init__.py:86`
- [ ] def simulate_download_status(self, dataset_id: str) -> Dict[str, Any]: - `data\datasets\systems\systems_logs_datasets.py:344`
- [ ] """Simulate the download status of a dataset.""" - `data\datasets\systems\systems_logs_datasets.py:345`
- [ ] logger.warning(f"Document at {path}:{line_num} missing 'text' field") - `data\processors\semantic_deduplication.py:197`
- [ ] missing_files = [] - `data\tools\validate_robotics_integration.py:30`
- [ ] missing_files.append(file) - `data\tools\validate_robotics_integration.py:34`
- [ ] if missing_files: - `data\tools\validate_robotics_integration.py:38`
- [ ] logger.error(f" ERROR: Missing files: {missing_files}") - `data\tools\validate_robotics_integration.py:39`
- [ ] missing = [] - `data\tools\validate_robotics_simple.py:119`
- [ ] missing.append(keyword) - `data\tools\validate_robotics_simple.py:122`
- [ ] if missing: - `data\tools\validate_robotics_simple.py:124`
- [ ] logger.warning(f"   ️  Palabras clave faltantes: {missing}") - `data\tools\validate_robotics_simple.py:125`
- [ ] return len(missing) == 0 - `data\tools\validate_robotics_simple.py:135`
- [ ] errors.append(f"Missing directory: {dir_path}") - `data\tools\validate_structure.py:43`
- [ ] errors.append(f"Missing file: {dir_name}/{file_name}") - `data\tools\validate_structure.py:99`
- [ ] errors.append(f"Missing __init__.py: {init_path}") - `data\tools\validate_structure.py:118`

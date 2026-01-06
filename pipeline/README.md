# Capibara Pipeline Module

**Complete data-to-training pipeline for CapibaraGPT-v2**

## 📋 Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Components](#components)
4. [Quick Start](#quick-start)
5. [Data Downloaders](#data-downloaders)
6. [Data Processors](#data-processors)
7. [Workflows](#workflows)
8. [Training Integration](#training-integration)
9. [Configuration](#configuration)
10. [Monitoring & Logging](#monitoring--logging)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

The **Capibara Pipeline Module** provides a complete, end-to-end data pipeline from raw data acquisition to model training. It orchestrates data downloading, processing, validation, and integration with the training system.

### Pipeline Stages

```
1. Download/Scrape → 2. Process/Clean → 3. Validate → 4. Train
   (downloaders/)      (processors/)    (workflows/)   (training/)
```

### Key Features

✅ **Automated Data Collection**: Web scraping, API downloads, direct file downloads
✅ **Robust Processing**: Cleaning, validation, transformation
✅ **Workflow Orchestration**: End-to-end pipeline automation
✅ **Training Integration**: Seamless integration with training system
✅ **Error Handling**: Retry logic, fallbacks, graceful degradation
✅ **Monitoring**: Real-time progress tracking and metrics
✅ **Scalable**: Parallel processing, batch operations
✅ **Resumable**: Checkpoint support for interrupted pipelines

---

## 🏗️ Pipeline Architecture

### Overall Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                   Data Sources                          │
│  (Web APIs, Datasets, Files, Web Pages)                 │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────┼───────────┬──────────────┐
        ▼           ▼           ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│   Web    │  │   API    │  │  Direct  │  │  Custom  │
│ Scraper  │  │Download  │  │Download  │  │Downloader│
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │              │             │
     └─────────────┴──────────────┴─────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │ Download Orchestrator│
         │  - Queue management  │
         │  - Retry logic       │
         │  - Rate limiting     │
         └─────────┬────────────┘
                   │
                   ▼
            data/raw/
                   │
                   ▼
         ┌─────────────────────┐
         │  Data Processor      │
         │  - Cleaning          │
         │  - Validation        │
         │  - Transformation    │
         │  - Deduplication     │
         └─────────┬────────────┘
                   │
                   ▼
          data/processed/
                   │
                   ▼
         ┌─────────────────────┐
         │  Complete Pipeline   │
         │  Workflow            │
         │  - Quality checks    │
         │  - Formatting        │
         │  - Dataset splitting │
         └─────────┬────────────┘
                   │
                   ▼
          data/training/
                   │
                   ▼
         ┌─────────────────────┐
         │  Training System     │
         │  (../training/)      │
         └─────────────────────┘
```

### Directory Structure

```
capibara/pipeline/
├── 📥 downloaders/
│   ├── __init__.py
│   ├── web_scraper.py              # Web scraping utilities
│   ├── api_downloader.py           # API-based downloads
│   ├── direct_downloader.py        # Direct file downloads
│   └── download_orchestrator.py    # Download coordination
│
├── ⚙️ processors/
│   ├── __init__.py
│   └── data_processor.py           # Data processing utilities
│
├── 🔄 workflows/
│   ├── __init__.py
│   └── complete_pipeline.py        # End-to-end workflow
│
├── 📊 data/
│   └── training/
│       └── run_training.py         # Training integration
│
├── __init__.py                     # Pipeline module exports
└── run_pipeline.py                 # Main pipeline runner
```

---

## 📦 Components

### 1. Downloaders

**Purpose**: Acquire data from various sources

**Components:**
- **WebScraper**: Scrape data from web pages
- **APIDownloader**: Download from REST APIs
- **DirectDownloader**: Download files directly (HTTP/FTP/S3)
- **DownloadOrchestrator**: Coordinate multiple downloaders

### 2. Processors

**Purpose**: Clean, validate, and transform raw data

**Components:**
- **DataProcessor**: Core processing logic
- **Validators**: Data quality validation
- **Transformers**: Data format transformation
- **Deduplicators**: Remove duplicate entries

### 3. Workflows

**Purpose**: Orchestrate complete end-to-end pipelines

**Components:**
- **CompletePipeline**: Full download → process → train workflow
- **CustomWorkflows**: User-defined pipelines
- **WorkflowScheduler**: Schedule recurring pipelines

### 4. Training Integration

**Purpose**: Bridge processed data to training system

**Components:**
- **DatasetBuilder**: Build training datasets
- **TrainingRunner**: Launch training jobs
- **CheckpointManager**: Manage training checkpoints

---

## 🚀 Quick Start

### Installation

```bash
# Install pipeline dependencies
pip install -r requirements.txt

# Install optional dependencies
pip install -r requirements-pipeline.txt
```

### Basic Pipeline

```python
from capibara.pipeline import CompletePipeline, PipelineConfig

# Create pipeline configuration
config = PipelineConfig(
    sources=[
        {
            "type": "web",
            "url": "https://example.com/data",
            "scraper": "beautifulsoup"
        },
        {
            "type": "api",
            "endpoint": "https://api.example.com/v1/data",
            "auth_token": "YOUR_TOKEN"
        }
    ],
    output_dir="data/processed",
    batch_size=1000,
    num_workers=4
)

# Create and run pipeline
pipeline = CompletePipeline(config)

# Run pipeline
result = await pipeline.run()

print(f"Downloaded: {result.downloaded_items}")
print(f"Processed: {result.processed_items}")
print(f"Valid: {result.valid_items}")
print(f"Training data ready: {result.training_ready}")
```

### Running from CLI

```bash
# Run complete pipeline
python -m capibara.pipeline.run_pipeline \
    --config config/pipeline.yaml \
    --output data/training

# Run specific stage
python -m capibara.pipeline.run_pipeline \
    --stage download \
    --config config/download.yaml

# Resume interrupted pipeline
python -m capibara.pipeline.run_pipeline \
    --resume checkpoints/pipeline_20251116.ckpt
```

---

## 📥 Data Downloaders

### WebScraper

**Purpose**: Scrape structured data from web pages

**Features:**
- BeautifulSoup/lxml parsing
- JavaScript rendering (Selenium/Playwright)
- Rate limiting and politeness delays
- Proxy support
- User-agent rotation

**Usage:**
```python
from capibara.pipeline.downloaders import WebScraper

scraper = WebScraper(config={
    "parser": "beautifulsoup",
    "rate_limit": 1.0,  # 1 request/second
    "user_agent": "CapibaraBot/2.0",
    "timeout": 30
})

# Scrape single page
data = await scraper.scrape_page(
    url="https://example.com/articles",
    selectors={
        "title": "h1.article-title",
        "content": "div.article-content",
        "author": "span.author-name"
    }
)

# Scrape multiple pages
urls = ["https://example.com/page1", "https://example.com/page2", ...]
results = await scraper.scrape_bulk(
    urls=urls,
    selectors=selectors,
    max_concurrent=5
)
```

**Advanced: JavaScript Rendering**
```python
scraper = WebScraper(config={
    "parser": "playwright",
    "headless": True,
    "javascript_enabled": True,
    "wait_for_selector": "div.dynamic-content"
})

# Scrape JavaScript-heavy page
data = await scraper.scrape_page(
    url="https://spa-example.com",
    wait_for="div.loaded"
)
```

### APIDownloader

**Purpose**: Download data from REST APIs

**Features:**
- Authentication (API keys, OAuth, JWT)
- Pagination handling
- Rate limiting
- Retry with exponential backoff
- Response caching

**Usage:**
```python
from capibara.pipeline.downloaders import APIDownloader

downloader = APIDownloader(config={
    "base_url": "https://api.example.com/v1",
    "auth": {
        "type": "bearer",
        "token": "YOUR_API_TOKEN"
    },
    "rate_limit": 10,  # 10 requests/second
    "retry_max": 3
})

# Download single endpoint
data = await downloader.download(
    endpoint="/datasets/123",
    params={"format": "json"}
)

# Download with pagination
all_data = await downloader.download_paginated(
    endpoint="/datasets",
    page_param="page",
    per_page=100,
    max_pages=50
)
```

**Advanced: OAuth Authentication**
```python
downloader = APIDownloader(config={
    "base_url": "https://api.example.com/v1",
    "auth": {
        "type": "oauth2",
        "client_id": "YOUR_CLIENT_ID",
        "client_secret": "YOUR_CLIENT_SECRET",
        "token_url": "https://auth.example.com/token"
    }
})

# Automatically handles token refresh
data = await downloader.download("/protected/resource")
```

### DirectDownloader

**Purpose**: Download files directly (HTTP, FTP, S3, GCS)

**Features:**
- Multi-protocol support (HTTP/HTTPS, FTP, S3, GCS)
- Resume capability for interrupted downloads
- Checksum verification
- Parallel chunk downloads
- Progress tracking

**Usage:**
```python
from capibara.pipeline.downloaders import DirectDownloader

downloader = DirectDownloader(config={
    "chunk_size": 1024 * 1024,  # 1MB chunks
    "max_concurrent": 5,
    "verify_checksum": True
})

# Download HTTP file
await downloader.download(
    url="https://example.com/dataset.tar.gz",
    destination="data/raw/dataset.tar.gz",
    checksum="sha256:abc123..."
)

# Download from S3
await downloader.download(
    url="s3://my-bucket/dataset.tar.gz",
    destination="data/raw/dataset.tar.gz",
    credentials={
        "aws_access_key_id": "YOUR_KEY",
        "aws_secret_access_key": "YOUR_SECRET"
    }
)

# Bulk download with progress
files = [
    {"url": "https://...", "destination": "data/raw/file1.txt"},
    {"url": "https://...", "destination": "data/raw/file2.txt"},
    ...
]

results = await downloader.download_bulk(
    files=files,
    progress_callback=lambda p: print(f"Progress: {p:.1%}")
)
```

### DownloadOrchestrator

**Purpose**: Coordinate multiple downloaders and manage queue

**Features:**
- Unified interface for all downloaders
- Priority queue
- Retry logic
- Error handling
- Progress tracking
- Resource management

**Usage:**
```python
from capibara.pipeline.downloaders import DownloadOrchestrator

orchestrator = DownloadOrchestrator(config={
    "max_concurrent": 10,
    "retry_failed": True,
    "retry_max": 3,
    "timeout": 300
})

# Add download tasks
orchestrator.add_task(
    task_type="web_scrape",
    url="https://example.com/page",
    priority=1  # High priority
)

orchestrator.add_task(
    task_type="api_download",
    endpoint="/datasets/123",
    priority=2  # Medium priority
)

# Run all downloads
results = await orchestrator.run_all()

print(f"Successful: {results.successful}")
print(f"Failed: {results.failed}")
print(f"Total time: {results.total_time_seconds}s")
```

---

## ⚙️ Data Processors

### DataProcessor

**Purpose**: Clean, validate, and transform raw data

**Features:**
- Text cleaning and normalization
- Data validation
- Format conversion
- Deduplication
- Quality filtering

**Usage:**
```python
from capibara.pipeline.processors import DataProcessor

processor = DataProcessor(config={
    "batch_size": 1000,
    "num_workers": 4,
    "validation_enabled": True,
    "cleanup_enabled": True
})

# Process single item
cleaned_data = processor.process_item(
    data=raw_data,
    operations=[
        "remove_html_tags",
        "normalize_whitespace",
        "remove_duplicates",
        "validate_schema"
    ]
)

# Batch processing
results = await processor.process_batch(
    data=raw_data_batch,
    operations=["clean", "validate", "transform"]
)

print(f"Processed: {results.processed}")
print(f"Valid: {results.valid}")
print(f"Rejected: {results.rejected}")
```

**Cleaning Operations:**
```python
# Text cleaning
cleaned = processor.clean_text(
    text=raw_text,
    operations={
        "lowercase": False,
        "remove_html": True,
        "remove_urls": False,
        "normalize_whitespace": True,
        "remove_special_chars": False,
        "strip": True
    }
)

# Data validation
is_valid = processor.validate(
    data=item,
    schema={
        "text": {"type": "string", "min_length": 10, "max_length": 10000},
        "label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "metadata": {"type": "object", "optional": True}
    }
)

# Deduplication
unique_data = processor.deduplicate(
    data=dataset,
    key_fields=["text"],
    method="exact"  # or "fuzzy", "semantic"
)
```

**Quality Filtering:**
```python
# Filter by quality metrics
high_quality = processor.filter_quality(
    data=dataset,
    criteria={
        "min_length": 50,
        "max_length": 10000,
        "language": "en",
        "min_quality_score": 0.7
    }
)
```

---

## 🔄 Workflows

### CompletePipeline

**Purpose**: End-to-end workflow from download to training

**Features:**
- Automated download → process → validate → train
- Checkpoint support
- Error recovery
- Progress monitoring
- Resource optimization

**Usage:**
```python
from capibara.pipeline.workflows import CompletePipeline

# Create complete pipeline
pipeline = CompletePipeline(config={
    "sources": [
        {"type": "web", "url": "https://..."},
        {"type": "api", "endpoint": "..."}
    ],
    "processing": {
        "operations": ["clean", "validate", "deduplicate"],
        "quality_threshold": 0.7
    },
    "training": {
        "enabled": True,
        "model_config": "config/training.toml"
    },
    "checkpoints": {
        "enabled": True,
        "interval": 1000  # Every 1000 items
    }
})

# Run pipeline
result = await pipeline.run()
```

**Custom Workflows:**
```python
from capibara.pipeline.workflows import WorkflowBuilder

# Build custom workflow
workflow = WorkflowBuilder() \
    .add_stage("download", {
        "downloader": "web_scraper",
        "urls": url_list
    }) \
    .add_stage("process", {
        "processor": "data_processor",
        "operations": ["clean", "validate"]
    }) \
    .add_stage("custom", {
        "function": my_custom_function,
        "args": {...}
    }) \
    .build()

# Execute workflow
result = await workflow.execute()
```

---

## 🎓 Training Integration

### Dataset Builder

**Purpose**: Build training-ready datasets from processed data

**Usage:**
```python
from capibara.pipeline.data.training import DatasetBuilder

builder = DatasetBuilder(config={
    "input_dir": "data/processed",
    "output_dir": "data/training",
    "split_ratios": {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1
    },
    "format": "tfrecord",  # or "jsonl", "parquet"
    "shuffle": True,
    "seed": 42
})

# Build dataset
dataset = builder.build()

print(f"Train samples: {dataset.train_size}")
print(f"Val samples: {dataset.val_size}")
print(f"Test samples: {dataset.test_size}")
```

### Training Runner

**Purpose**: Launch training jobs with processed data

**Usage:**
```python
from capibara.pipeline.data.training import TrainingRunner

runner = TrainingRunner(config={
    "dataset_path": "data/training",
    "model_config": "config/model.toml",
    "training_config": "config/training.toml",
    "output_dir": "checkpoints/"
})

# Run training
result = await runner.run_training()

print(f"Final loss: {result.final_loss}")
print(f"Best checkpoint: {result.best_checkpoint}")
```

---

## ⚙️ Configuration

### Pipeline Configuration File

**`config/pipeline.yaml`:**
```yaml
pipeline:
  version: "2.0.0"
  name: "capibara-data-pipeline"

storage:
  raw_data_path: "data/raw"
  processed_data_path: "data/processed"
  training_data_path: "data/training"
  cache_path: "data/cache"

sources:
  - type: web
    url: "https://example.com/data"
    scraper:
      parser: beautifulsoup
      rate_limit: 1.0
      selectors:
        title: "h1"
        content: "div.content"

  - type: api
    endpoint: "https://api.example.com/v1/datasets"
    auth:
      type: bearer
      token: "${API_TOKEN}"
    pagination:
      enabled: true
      page_param: "page"
      per_page: 100

processing:
  batch_size: 1000
  num_workers: 4
  operations:
    - remove_html_tags
    - normalize_whitespace
    - validate_schema
    - deduplicate
  quality_threshold: 0.7
  cleanup_enabled: true
  validation_enabled: true

training:
  enabled: true
  split_ratios:
    train: 0.8
    val: 0.1
    test: 0.1
  format: "jsonl"
  shuffle: true

monitoring:
  enabled: true
  log_level: "INFO"
  metrics_collection: true
  checkpoint_interval: 1000

error_handling:
  retry_failed: true
  retry_max: 3
  retry_delay: 5
  continue_on_error: true
```

### Environment Variables

```bash
# Pipeline configuration
CAPIBARA_PIPELINE_CONFIG=config/pipeline.yaml
CAPIBARA_DATA_DIR=data/
CAPIBARA_CACHE_DIR=data/cache/

# Download configuration
CAPIBARA_MAX_CONCURRENT_DOWNLOADS=10
CAPIBARA_DOWNLOAD_TIMEOUT=300
CAPIBARA_RETRY_MAX=3

# Processing configuration
CAPIBARA_BATCH_SIZE=1000
CAPIBARA_NUM_WORKERS=4
CAPIBARA_QUALITY_THRESHOLD=0.7

# API credentials
API_TOKEN=your_api_token_here
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

---

## 📊 Monitoring & Logging

### Progress Tracking

```python
from capibara.pipeline import CompletePipeline

pipeline = CompletePipeline(config)

# Add progress callback
def progress_callback(stage, progress):
    print(f"{stage}: {progress:.1%} complete")

pipeline.set_progress_callback(progress_callback)

# Run with live progress
result = await pipeline.run()
```

### Metrics Collection

```python
from capibara.pipeline.monitoring import MetricsCollector

metrics = MetricsCollector()

# Track pipeline metrics
metrics.track("downloads_total", len(downloaded_items))
metrics.track("processing_time_seconds", processing_time)
metrics.track("quality_score", quality_score)

# Export metrics
metrics.export_prometheus(port=9090)
metrics.export_json("metrics.json")
```

### Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

# Pipeline logs automatically
logger = logging.getLogger('capibara.pipeline')
logger.info("Pipeline started")
```

---

## 💡 Best Practices

### 1. Start Small, Scale Up

```python
# Test with small batch first
test_config = config.copy()
test_config["batch_size"] = 10
test_config["max_items"] = 100

test_result = await pipeline.run(test_config)

# If successful, scale up
if test_result.success:
    full_result = await pipeline.run(config)
```

### 2. Use Checkpoints

```python
# Enable checkpoints
config["checkpoints"] = {
    "enabled": True,
    "interval": 1000,
    "path": "checkpoints/"
}

# Resume from checkpoint if needed
if os.path.exists("checkpoints/latest.ckpt"):
    result = await pipeline.resume("checkpoints/latest.ckpt")
```

### 3. Validate Early

```python
# Validate configuration before running
errors = pipeline.validate_config(config)
if errors:
    print(f"Configuration errors: {errors}")
    exit(1)

# Dry run to test
result = await pipeline.run(dry_run=True)
print(f"Would process {result.estimated_items} items")
```

### 4. Monitor Resource Usage

```python
import psutil

def check_resources():
    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory().percent

    if cpu > 90 or memory > 90:
        logger.warning(f"High resource usage: CPU {cpu}%, Memory {memory}%")

# Check periodically
pipeline.set_resource_callback(check_resources, interval=60)
```

---

## 🔧 Troubleshooting

### Issue: Downloads Timing Out

**Solution:**
```python
# Increase timeout
config["downloader"]["timeout"] = 600  # 10 minutes

# Reduce concurrent downloads
config["max_concurrent"] = 5  # From 10

# Enable retry
config["retry_max"] = 5
```

### Issue: Out of Memory During Processing

**Solution:**
```python
# Reduce batch size
config["processing"]["batch_size"] = 500  # From 1000

# Enable disk caching
config["processing"]["use_disk_cache"] = True

# Process in streaming mode
processor = DataProcessor(config, streaming=True)
```

### Issue: Data Quality Issues

**Solution:**
```python
# Increase quality threshold
config["processing"]["quality_threshold"] = 0.8  # From 0.7

# Enable stricter validation
config["processing"]["validation"] = {
    "strict_mode": True,
    "reject_invalid": True
}

# Add custom quality filters
processor.add_quality_filter(my_custom_filter)
```

---

## 📚 References

### Related Documentation

- [Data Module](../data/README.md) - Data loading and orchestration
- [Training Module](../training/README.md) - Training system
- [Core Module](../core/README.md) - Core model architecture

### External Resources

- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/)
- [Playwright for Python](https://playwright.dev/python/)
- [Apache Airflow](https://airflow.apache.org/) - For advanced workflow orchestration

---

## 🤝 Contributing

Contributions welcome! Priority areas:

1. **New Downloaders**: Add support for more data sources
2. **Processors**: Implement additional data processing operations
3. **Workflows**: Create reusable workflow templates
4. **Performance**: Optimize download and processing speed
5. **Testing**: Add comprehensive unit and integration tests

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

---

## 📄 License

Part of the capibaraGPT-v2 project. See [LICENSE](../../LICENSE) for details.

---

**Maintained by**: Capibara ML Team
**Last Updated**: 2025-11-16

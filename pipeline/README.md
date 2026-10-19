# Pipeline

This directory contains the v3 data pipeline: download orchestration,
processing, dataset preparation, and a training handoff script.

## What Exists Today
- `downloaders/download_orchestrator.py`: queue + async orchestration that
  calls the downloaders below and writes outputs under `data/raw/`.
- `downloaders/web_scraper.py`: basic HTML scraper (stdlib, optional BS4).
- `downloaders/api_downloader.py`: basic JSON API downloader with pagination.
- `downloaders/direct_downloader.py`: direct file downloader (HTTP/HTTPS).
- `processors/data_processor.py`: real cleaning + dedup + quality scoring for
  JSONL inputs.
- `workflows/complete_pipeline.py`: runs download -> process -> dataset prep ->
  training integration.
- `run_pipeline.py`: CLI entry point (works via `capibara.pipeline` shim).

## Package Layout
```
pipeline/
├── downloaders/
│   ├── api_downloader.py
│   ├── direct_downloader.py
│   ├── download_orchestrator.py
│   └── web_scraper.py
├── processors/
│   └── data_processor.py
├── workflows/
│   └── complete_pipeline.py
├── run_pipeline.py
└── __init__.py
```

## Quickstart
```python
from pipeline.workflows import CompletePipeline

config = {
    "storage": {
        "raw_data_path": "data/raw",
        "processed_data_path": "data/processed",
        "training_data_path": "data/training",
        "cache_path": "data/cache",
    },
    "processing": {
        "max_workers": 2,
        "batch_size": 500,
        "min_quality_score": 0.6,
        "enable_deduplication": True,
    },
}

pipeline = CompletePipeline(config)
result = await pipeline.execute_complete_pipeline()
```

## Notes
- The processing stage expects JSONL files under `data/raw/`.
- Downloaders are intentionally lightweight (stdlib-based). If you need
  advanced parsing, add BeautifulSoup or other tooling in the downloader config.

## Issues por hacer

- [ ] Sin issues detectadas por patrones (TODO/FIXME/mock/simulated/missing).

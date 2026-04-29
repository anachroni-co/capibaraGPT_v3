# 🦫 Capibara Slim

A lightweight, modular AI inference and training system combining Transformers and Mamba architectures for production-grade deployment.

## 🧠 Overview

Capibara Slim is a streamlined version of the original Capibara system. It removes experimental complexity and focuses on:

- stable inference
- modular model design
- optional hybrid architectures (Transformer + Mamba)
- simple routing logic
- deployable API-first design

It is designed to be **fast, understandable, and production-ready** — not a research sandbox.

## ⚙️ Key Features

### 🧠 Hybrid Architecture (Optional)

- Transformer blocks for general reasoning
- Mamba blocks for efficient long-sequence processing
- Optional fixed hybrid stacking (no dynamic per-token routing)

### 🚀 Inference Engine

- deterministic pipeline execution
- streaming support
- optional caching layer (KV cache)

### 🧩 Modular Design

Clear separation between:

- core execution
- model layers
- functional modules
- tools (optional)

### 🔌 Tool Use (Lite MCP)

- simple function-calling interface
- external tool execution sandbox
- no deep agent framework dependency

### 🛡️ Safety Layer

- input filtering
- output filtering
- lightweight policy enforcement

### 📊 Training Pipeline

- single-strategy training loop
- reproducible configuration-based runs
- checkpointing and model export

## 🧭 Architecture

```
User
 ↓
API (services/)
 ↓
Inference Pipeline
 ↓
Core Router
 ↓
Model Stack
   ├── Transformer Blocks
   ├── Mamba Blocks
   └── Modules (reasoning/tools)
 ↓
Optional Tool Execution
 ↓
Safety Layer
 ↓
Response
```

## 📁 Project Structure

```
capibara-slim/
│
├── app/              # API layer (FastAPI or similar)
├── core/             # execution graph + router
├── inference/        # runtime pipeline
├── models/           # layers, modules, submodels
├── training/         # simplified training loop
├── data/             # datasets
├── rag/              # optional retrieval system
├── tools/            # MCP-lite tool system
├── safety/           # input/output filtering
├── config/           # global configuration
├── services/         # API + orchestration layer
├── tests/            # integration tests
├── scripts/          # dev & deployment scripts
├── docker/           # deployment setup
└── utils/            # lightweight helpers
```

## 🧠 Design Principles

### 1. Simplicity over exploration

No experimental routing systems or multi-agent frameworks.

### 2. Deterministic inference

Same input → same output (given same model state).

### 3. Controlled modularity

Modules exist, but do not introduce unnecessary abstraction layers.

### 4. Hybrid architecture by design, not by runtime chaos

Mamba and Transformer are combined in fixed, understandable patterns.

### 5. Production-first mindset

Every component must justify its existence in terms of real production value.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements-mvp.txt

# Run inference
python -m inference.pipeline --config config/default.yaml

# Run API server
python -m app.main
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## Requirements

- Python `>=3.9`
- `pip`

Optional acceleration backends:

- GPU: PyTorch + CUDA
- TPU: JAX + Flax

## License

Dual licensing (open + commercial). See `LICENSE`.

## Contact

- GitHub: `https://github.com/anachroni-co/capibaraGPT_v3`
- Website: `https://www.anachroni.co`
- Email: `info@anachroni.co`

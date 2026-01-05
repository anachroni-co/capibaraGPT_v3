# CapibaraGPT v3

An open-source conversational AI platform featuring 5 specialized language models with intelligent semantic routing, optimized for Google Cloud's ARM-Axion architecture using vLLM with NEON optimizations.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Features

- **Multi-Model Architecture**: 5 specialized AI models for different tasks
- **Semantic Router**: Intelligent query routing to the most appropriate model
- **ARM-Axion Optimized**: Custom NEON kernels for maximum CPU performance
- **Lazy Loading**: Memory-efficient on-demand model loading
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI endpoints
- **RAG Integration**: Retrieval Augmented Generation support
- **MCP Support**: Model Context Protocol for extended capabilities
- **TTS Integration**: Text-to-Speech with Kyutai

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Multi-Model Server (Port 8082)              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Semantic Router (NEON-optimized)         │  │
│  └───────────────────────────────────────────────────────┘  │
│                            │                                 │
│      ┌─────────────────────┼─────────────────────┐          │
│      │                     │                     │          │
│  ┌───▼────┐          ┌────▼─────┐         ┌────▼────┐      │
│  │  phi4  │          │ mistral  │         │  qwen   │      │
│  │  fast  │          │ balanced │         │  coder  │      │
│  └────────┘          └──────────┘         └─────────┘      │
│                                                              │
│  ┌─────────────┐              ┌──────────────────┐          │
│  │   gemma3    │              │   aya_expanse    │          │
│  │ multimodal  │              │   multilingual   │          │
│  └─────────────┘              └──────────────────┘          │
│                                                              │
│            Lazy Loading: Models loaded on-demand             │
└─────────────────────────────────────────────────────────────┘
```

## Available Models

| Model | Domain | Description | Parameters |
|-------|--------|-------------|------------|
| `phi4_fast` | General | Fast responses for simple queries | Small |
| `mistral_balanced` | Technical | Balanced for technical tasks | 7B |
| `qwen_coder` | Coding | Specialized in programming | 1.5B |
| `gemma3_multimodal` | Multimodal | Complex analysis, image support | 27B |
| `aya_expanse_multilingual` | Multilingual | 23 languages, complex reasoning | 8B |

## Quick Start

### Prerequisites

- Python 3.10+
- ARM-based CPU (recommended: Google Cloud C4A instances)
- 64GB+ RAM for full model loading

### Installation

```bash
# Clone the repository
git clone https://github.com/anachroni-co/capibaraGPT_v3.git
cd capibaraGPT_v3

# Install dependencies
pip install -r requirements.txt

# Start the server
python arm-axion-optimizations/vllm_integration/multi_model_server.py \
    --host 0.0.0.0 \
    --port 8082 \
    --config arm-axion-optimizations/vllm_integration/config.json
```

### Verify Installation

```bash
# Health check
curl http://localhost:8082/health

# List available models
curl http://localhost:8082/v1/models

# View statistics
curl http://localhost:8082/stats
```

## API Usage

### Chat Completions (OpenAI-compatible)

```bash
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen_coder",
    "messages": [{"role": "user", "content": "Write a Python function to sort a list"}],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

### Automatic Routing

Omit the model parameter to let the semantic router choose the best model:

```bash
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "temperature": 0.7
  }'
```

## Distributed Architecture

CapibaraGPT v3 is designed for distributed deployment across multiple VMs:

### Models VM (`models-europe`)
- Multi-model server on port 8082
- 5 AI models with ARM-Axion optimizations
- Semantic router

### Services VM (`services`)
- `capibara6_integrated_server.py` - Main backend
- `mcp_server.py` - Model Context Protocol (port 5003)
- `kyutai_tts_server.py` - Text-to-Speech (port 5002)
- `smart_mcp_server.py` - Alternative MCP (port 5010)

## ARM-Axion Optimizations

Performance improvements specific to ARM architecture:

| Optimization | Improvement |
|--------------|-------------|
| NEON Kernels | Vector acceleration for matrix ops |
| ARM Compute Library (ACL) | GEMM optimization |
| Flash Attention | Long sequence support (>512 tokens) |
| Optimized RMSNorm | 5x faster with vectorization |
| Fused SwiGLU | 1.5x improvement in activations |
| Vectorized RoPE | 1.4x faster |

**Overall improvement: 1.7-2.0x** (60-80% faster than non-optimized version)

## Configuration

### Main Configuration File

```bash
arm-axion-optimizations/vllm_integration/config.json
```

### Available Configurations

| Config File | Description |
|-------------|-------------|
| `config.five_models_all_working.json` | Default 5-model setup |
| `config.five_models_optimized_with_aya.json` | Optimized with Aya |
| `config.low_latency_batching.json` | Low latency mode |
| `config.optimized_kv_cache.json` | Optimized KV cache |

### Switching Configurations

```bash
cd arm-axion-optimizations/vllm_integration
ln -sf config.low_latency_batching.json config.json
# Restart the server
```

## RAG Integration

```python
from backend.rag_client import RAGClient

# Initialize client
rag = RAGClient(
    base_url="http://localhost:8001",
    enable_toon=True  # Token optimization
)

# Semantic search
results = rag.search_semantic(
    query="How does the system work?",
    n_results=5
)
```

## Monitoring

### Server Logs

```bash
# Real-time logs
tail -f /tmp/multi_model_server.log

# Recent logs
tail -100 /tmp/multi_model_server.log
```

### Process Verification

```bash
# Check server process
ps aux | grep multi_model_server

# Check listening ports
ss -tlnp | grep 8082
```

## Troubleshooting

### Server Won't Start

1. Check if port is in use:
   ```bash
   ss -tlnp | grep 8082
   ```

2. Check logs:
   ```bash
   tail -50 /tmp/multi_model_server.log
   ```

### Slow First Response

This is expected behavior due to lazy loading. First request for each model takes 20-60 seconds for loading. Subsequent requests are instant.

### Memory Issues

Adjust lazy loading settings in `config.json`:

```json
{
  "lazy_loading": {
    "max_loaded_experts": 3,
    "auto_unload_after_s": 180
  }
}
```

## Project Structure

```
capibaraGPT_v3/
├── api/                    # Vercel serverless functions
├── arm-axion-optimizations/  # ARM-specific optimizations
│   ├── kernels/            # NEON kernel implementations
│   └── vllm_integration/   # vLLM integration code
├── backend/                # Main backend services
│   ├── core/               # Core modules (RAG, CAG, router)
│   └── toon_utils/         # Token optimization utilities
├── backendModels/          # Model configurations
├── docs/                   # Documentation
├── fine-tuning/            # Fine-tuning scripts and configs
├── frontend/               # Web frontend
├── infra/                  # Infrastructure configs
├── k8s/                    # Kubernetes manifests
├── monitoring/             # Monitoring setup
├── scripts/                # Utility scripts
└── tests/                  # Test files
```

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
- [ARM Compute Library](https://github.com/ARM-software/ComputeLibrary) - ARM optimizations
- All the open-source model creators

---

**Version**: 3.0.0
**Status**: Production Ready

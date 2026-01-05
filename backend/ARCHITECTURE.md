# CapibaraGPT System Architecture

## Distributed VM Architecture

### Models VM (`models-europe`)

This VM hosts the AI inference services.

**Active Services**:
- `multi_model_server.py` - Multi-model server with semantic routing (port 8082)
- 5 specialized AI models with ARM-Axion optimizations
- Semantic router for intelligent query distribution

**Configuration**:
- VM Type: ARM Axion C4A-standard-32
- vCPUs: 32 cores
- RAM: 125 GB
- Location: europe-southwest1-b

### Services VM (`services`)

This VM hosts backend services and coordination.

**Active Services**:
- `capibara6_integrated_server.py` - Main backend server
- `mcp_server.py` - Model Context Protocol (port 5003)
- `kyutai_tts_server.py` - Text-to-Speech (port 5002)
- `smart_mcp_server.py` - Alternative MCP server (port 5010)

### RAG VM (`rag3`)

This VM hosts the complete RAG (Retrieval-Augmented Generation) system.

**Services**:
- Vector database
- Embedding generation
- Semantic search
- Document processing

## Service Communication

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│   Services  │────▶│   Models    │
│   (Vercel)  │     │     VM      │     │     VM      │
└─────────────┘     └──────┬──────┘     └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │   RAG VM    │
                   └─────────────┘
```

## Backend Servers

### `server_gptoss.py`
- Main chat server for AI models
- Port: 5001
- Handles chat requests and file uploads

### `capibara6_integrated_server.py`
- Integrated server with all features
- Supports: Chat, TTS, MCP, E2B
- Production ready

### `gateway_server.py`
- Gateway for routing requests
- Load balancing
- Health monitoring

## Configuration Files

| File | Purpose |
|------|---------|
| `config_gptoss.py` | GPT-OSS model configuration |
| `models_config.py` | Multi-model configuration |
| `production_config.py` | Production environment settings |
| `gpt_oss_optimized_config.py` | Optimized GPT-OSS settings |

## Integration Architecture

The system integrates multiple components:

1. **Chat Interface** - Web frontend for user interaction
2. **Backend API** - REST API for chat and services
3. **Model Server** - Multi-model inference with routing
4. **RAG System** - Knowledge retrieval and augmentation
5. **TTS Service** - Text-to-speech conversion
6. **MCP Server** - Extended tool capabilities

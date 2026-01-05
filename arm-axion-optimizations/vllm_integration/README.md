# vLLM Multi-Model Server - ARM Axion Optimizado

## Descripción

Servidor de inferencia multi-modelo basado en vLLM, optimizado para procesadores ARM Axion de Google Cloud. Incluye router semántico inteligente, lazy loading y optimizaciones NEON/ACL.

## Inicio Rápido

```bash
# Desde este directorio
python3 multi_model_server.py --host 0.0.0.0 --port 8082 --config config.json
```

## Arquitectura

```
multi_model_server.py          # Servidor principal FastAPI
  ├── semantic_router.py       # Router semántico con NEON
  ├── lazy_expert_manager.py   # Gestión lazy loading de modelos
  ├── embedding_cache.py       # Cache de embeddings para routing
  └── config.json              # Configuración de modelos (symlink)
```

## Archivos Principales

### Servidores

- **`multi_model_server.py`** - Servidor principal con lazy loading
  - API compatible con OpenAI
  - Lazy loading de modelos bajo demanda
  - Router semántico integrado

- **`multi_model_server_optimized.py`** - Versión con optimizaciones adicionales
  - Prefetching de modelos
  - Cache avanzado

- **`multi_model_server_stable.py`** - Versión estable sin features experimentales

- **`inference_server.py`** - Servidor usando LiveMind orchestrator
  - Orquestación avanzada
  - Generación especulativa

### Componentes Core

- **`semantic_router.py`** - Router semántico con optimizaciones NEON
  - Análisis incremental de queries
  - Selección automática de modelo experto
  - Confidence scoring

- **`lazy_expert_manager.py`** - Gestión de modelos con lazy loading
  - Carga bajo demanda
  - Auto-descarga por inactividad
  - Pool de warmup

- **`embedding_cache.py`** - Cache de embeddings para routing rápido
  - LRU cache con 20K entradas
  - Embeddings NEON-optimizados

- **`livemind_orchestrator.py`** - Orquestador LiveMind-style
  - Routing especulativo
  - Parallel RAG fetching
  - Consensus entre modelos

### Backends

- **`vllm_axion_backend.py`** - Backend vLLM con optimizaciones ARM
  - Kernels NEON personalizados
  - ACL integration
  - CPU-optimized sampling

## Configuración

### Archivo de Configuración

El servidor usa `config.json` (symlink al archivo de configuración activo):

```bash
# Ver configuración actual
ls -la config.json

# Cambiar configuración
ln -sf config.five_models_all_working.json config.json
```

### Configuraciones Disponibles

- **`config.five_models_all_working.json`** ⭐ **Actual**
  - 5 modelos: phi4, mistral, qwen, gemma3, aya_expanse
  - Optimizado para ARM Axion
  - Lazy loading habilitado

- **`config.five_models_optimized_with_aya.json`**
  - Similar a la anterior con ajustes de rendimiento

- **`config.low_latency_batching.json`**
  - Optimizado para baja latencia
  - Batching agresivo

- **`config.optimized_kv_cache.json`**
  - KV cache optimizado
  - Para contextos largos

- **`config.ultra_low_latency_kv_cache.json`**
  - Latencia mínima
  - Cache reducido

### Estructura de Configuración

```json
{
  "experts": [
    {
      "expert_id": "model_name",
      "model_path": "/path/to/model",
      "domain": "general|technical|coding|multimodal_expert|multilingual_expert",
      "priority": 1-5,
      "quantization": null,
      "enable_neon": true,
      "neon_optimizations": { ... }
    }
  ],
  "lazy_loading": {
    "enabled": true,
    "warmup_pool_size": 2,
    "max_loaded_experts": 5,
    "auto_unload_after_s": 300
  },
  "rag": {
    "enabled": true,
    "bridge_url": "http://localhost:8001"
  }
}
```

## API Endpoints

### Health Check

```bash
GET /health
```

Respuesta:
```json
{
  "status": "healthy",
  "models_loaded": 1,
  "models_available": 5
}
```

### Listar Modelos

```bash
GET /v1/models
```

### Estadísticas

```bash
GET /stats
```

### Chat Completions (OpenAI-compatible)

```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "aya_expanse_multilingual",  # Opcional, usa router si no se especifica
  "messages": [
    {"role": "user", "content": "Tu pregunta aquí"}
  ],
  "temperature": 0.7,
  "max_tokens": 100
}
```

### Completions (OpenAI-compatible)

```bash
POST /v1/completions
Content-Type: application/json

{
  "model": "phi4_fast",
  "prompt": "Tu prompt aquí",
  "max_tokens": 50
}
```

## Router Semántico

El router semántico analiza automáticamente las queries y selecciona el modelo más apropiado:

### Dominios

- **general**: Preguntas simples, saludos, chat casual → `phi4_fast`
- **technical**: Explicaciones técnicas, documentación → `mistral_balanced`
- **coding**: Programación, debugging, código → `qwen_coder`
- **multimodal_expert**: Análisis de imágenes, contexto largo → `gemma3_multimodal`
- **multilingual_expert**: Multilingüe, traducción → `aya_expanse_multilingual`

### Uso del Router

```bash
# Sin especificar modelo - usa router automático
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Escribe código Python"}]
  }'
# → Automáticamente enruta a qwen_coder
```

## Lazy Loading

Los modelos se cargan bajo demanda para optimizar memoria:

1. **Estado inicial**: Solo metadata cargada
2. **Primera solicitud**: Modelo se carga (~20-60s)
3. **Siguientes solicitudes**: Respuesta inmediata
4. **Auto-descarga**: Después de 5 min de inactividad

### Configuración de Lazy Loading

```json
"lazy_loading": {
  "enabled": true,
  "warmup_pool_size": 2,           # Modelos pre-cargados al inicio
  "max_loaded_experts": 5,         # Máximo en memoria simultáneamente
  "memory_threshold": 0.8,         # Umbral de memoria (80%)
  "auto_unload_after_s": 300       # Auto-descarga después de 5 min
}
```

## Optimizaciones ARM-Axion

### Kernels NEON

- **MatMul 8x8**: Multiplicación de matrices optimizada
- **Flash Attention**: Para seq_len > 512
- **RMSNorm**: 5x más rápido
- **RoPE**: Rotary Position Embeddings vectorizado
- **SwiGLU**: Activaciones fusionadas
- **Softmax**: Fast exponential

### ARM Compute Library (ACL)

- **GEMM**: General Matrix Multiply optimizado
- **Quantization Q4**: Cuantización eficiente

### Performance

- **Speedup global**: 1.7-2.0x vs sin optimizaciones
- **MatMul**: 1.5x más rápido
- **Attention**: 1.8x más rápido
- **RMSNorm**: 5x más rápido

## Monitoreo

### Logs

```bash
# En tiempo real
tail -f /tmp/multi_model_server.log

# Últimas 100 líneas
tail -100 /tmp/multi_model_server.log
```

### Métricas

```bash
# Estadísticas del servidor
curl http://localhost:8082/stats | jq

# Modelos cargados
curl http://localhost:8082/stats | jq '.models_loaded'

# Modelos disponibles
curl http://localhost:8082/stats | jq '.models_available'
```

## Desarrollo

### Variables de Entorno

```bash
# Forzar backend clásico (no V1)
export VLLM_USE_V1=0
export VLLM_ENABLE_V1_ENGINE=0

# Método de multiproc
export VLLM_WORKER_MULTIPROC_METHOD=fork

# Deshabilitar features incompatibles
export VLLM_USE_FLASHINFER=0
export VLLM_USE_TRITON_FLASH_ATTN=0

# Torch eager mode
export TORCH_COMPILE_BACKEND=eager
```

### Testing

```bash
# Test simple
curl http://localhost:8082/health

# Test con modelo específico
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi4_fast",
    "messages": [{"role": "user", "content": "Hola"}],
    "max_tokens": 20
  }'

# Test con router automático
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "¿Qué es Python?"}],
    "max_tokens": 50
  }'
```

## Troubleshooting

### Servidor no inicia

```bash
# 1. Verificar puerto
ss -tlnp | grep 8082

# 2. Verificar proceso
ps aux | grep multi_model_server

# 3. Ver logs
tail -50 /tmp/multi_model_server.log

# 4. Verificar configuración
cat config.json | jq '.experts | length'
```

### Modelo tarda en cargar

- Normal en primera carga (lazy loading)
- Tiempo esperado: 20-60 segundos según tamaño
- Cargas posteriores: Instantáneas

### Error de memoria

```bash
# 1. Verificar memoria disponible
free -h

# 2. Reducir modelos cargados simultáneamente
# Editar config.json:
"lazy_loading": {
  "max_loaded_experts": 3  # Reducir de 5 a 3
}

# 3. Reducir tiempo de auto-descarga
"lazy_loading": {
  "auto_unload_after_s": 180  # 3 minutos en vez de 5
}
```

## Estructura de Archivos

```
vllm_integration/
├── multi_model_server.py          # ⭐ Servidor principal
├── semantic_router.py             # Router semántico
├── lazy_expert_manager.py         # Lazy loading manager
├── embedding_cache.py             # Cache de embeddings
├── livemind_orchestrator.py       # Orquestador avanzado
├── vllm_axion_backend.py          # Backend ARM optimizado
├── rag_parallel_fetcher.py        # RAG paralelo
├── speculative_router.py          # Routing especulativo
│
├── config.json                    # ⭐ Config actual (symlink)
├── config.five_models_all_working.json  # Config de 5 modelos
├── config.*.json                  # Otras configuraciones
│
├── README.md                      # ⭐ Esta documentación
└── venv/                          # Virtual environment
```

## Referencias

- **Documentación principal**: `/home/elect/capibara6/README.md`
- **vLLM modificado**: `/home/elect/capibara6/vllm-source-modified`
- **Modelos**: `/home/elect/models/`

---

**Última actualización**: 2025-12-02
**Estado**: Producción - Operativo

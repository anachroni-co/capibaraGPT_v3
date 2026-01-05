# CapibaraGPT v3 - Sistema Multimodelo ARM-Axion con vLLM

## Descripción General

CapibaraGPT v3 es una plataforma de IA conversacional de código abierto que utiliza 5 modelos de lenguaje especializados con router semántico inteligente, optimizada para arquitectura ARM-Axion de Google Cloud. El sistema utiliza vLLM con optimizaciones NEON para máximo rendimiento en CPU.

> **Nota**: Esta es la versión 3 del proyecto, migrada y mejorada desde capibara6.

## ⚠️ Importante para Agentes - Arquitectura Distribuida

**VM models-europe (esta VM)** - Solo servicios de IA:
- ✅ `multi_model_server.py` en puerto 8082 (servidor de modelos con router semántico)
- ✅ 5 modelos de IA con optimizaciones ARM-Axion
- ❌ NO iniciar: MCP, TTS, servidores backend (corren en la VM `services`)

**VM services** - Servicios de backend y coordinación:
- ✅ `capibara6_integrated_server.py` (backend principal)
- ✅ `mcp_server.py` (Model Context Protocol en puerto 5003)
- ✅ `kyutai_tts_server.py` (Text-to-Speech en puerto 5002)
- ✅ `smart_mcp_server.py` (alternativa en puerto 5010)
- ❌ NO iniciar: Servidor de modelos vLLM (corre en models-europe)

## Estado Actual del Sistema

**Actualizado**: 2025-12-02

- **VM**: models-europe (ARM Axion C4A-standard-32)
- **vCPUs**: 32 cores ARM Axion
- **RAM**: 125 GB
- **Ubicación**: europe-southwest1-b
- **Servidor activo**: Puerto 8082
- **Modelos cargados**: 5 modelos especializados

## Modelos Disponibles

El sistema está configurado con **5 modelos de IA especializados**:

1. **phi4_fast** (general) - Modelo rápido para consultas simples
   - Path: `/home/elect/models/phi-4-mini`
   - Dominio: General
   - Prioridad: 5 (alta)

2. **mistral_balanced** (technical) - Modelo equilibrado para tareas técnicas
   - Path: `/home/elect/models/mistral-7b-instruct-v0.2`
   - Dominio: Technical
   - Prioridad: 4

3. **qwen_coder** (coding) - Especializado en programación
   - Path: `/home/elect/models/qwen2.5-coder-1.5b`
   - Dominio: Coding
   - Prioridad: 3

4. **gemma3_multimodal** (multimodal_expert) - Análisis complejo y multimodal
   - Path: `/home/elect/models/gemma-3-27b-it`
   - Dominio: Multimodal Expert
   - Prioridad: 2
   - Características: 27B params, soporte para imágenes, contexto largo (24K tokens)

5. **aya_expanse_multilingual** (multilingual_expert) - Experto multilingüe
   - Path: `/home/elect/models/aya-expanse-8b`
   - Dominio: Multilingual Expert
   - Prioridad: 2
   - Características: 8B params, 23 idiomas, razonamiento complejo

## Iniciar el Sistema

### Inicio Rápido

```bash
cd /home/elect/capibara6/arm-axion-optimizations/vllm_integration
python3 multi_model_server.py --host 0.0.0.0 --port 8082 --config config.json
```

### Verificar Estado

```bash
# Verificar salud del servidor
curl http://localhost:8082/health

# Listar modelos disponibles
curl http://localhost:8082/v1/models

# Ver estadísticas
curl http://localhost:8082/stats
```

## Endpoints API

El servidor expone una API compatible con OpenAI:

### Chat Completions

```bash
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aya_expanse_multilingual",
    "messages": [{"role": "user", "content": "Hola, ¿cómo estás?"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### Routing Automático

Si no especificas un modelo, el router semántico selecciona automáticamente el más apropiado:

```bash
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Escribe una función en Python para ordenar una lista"}],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

## Router Semántico

El sistema incluye un router semántico inteligente que:
- Analiza la consulta del usuario
- Determina el dominio (general, técnico, código, multimodal, multilingüe)
- Selecciona el modelo más apropiado automáticamente
- Usa optimizaciones NEON para análisis rápido

## Arquitectura

### Componentes Clave

```
┌─────────────────────────────────────────────────┐
│     Servidor Multi-Modelo (Puerto 8082)         │
│  ┌────────────────────────────────────────────┐ │
│  │        Router Semántico (NEON)             │ │
│  └────────────────────────────────────────────┘ │
│                      │                          │
│      ┌───────────────┼───────────────┐          │
│      │               │               │          │
│  ┌───▼───┐     ┌────▼────┐    ┌────▼────┐     │
│  │ phi4  │     │mistral  │    │  qwen   │     │
│  │ fast  │     │balanced │    │  coder  │     │
│  └───────┘     └─────────┘    └─────────┘     │
│                                                 │
│  ┌─────────────┐     ┌─────────────────┐      │
│  │   gemma3    │     │   aya_expanse   │      │
│  │ multimodal  │     │  multilingual   │      │
│  └─────────────┘     └─────────────────┘      │
│                                                 │
│  Lazy Loading: Modelos se cargan bajo demanda  │
└─────────────────────────────────────────────────┘
```

### Lazy Loading

Los modelos usan **lazy loading**:
- No se cargan todos al inicio (ahorro de memoria)
- Se cargan automáticamente cuando se solicitan por primera vez
- Primera carga: ~20-60 segundos (según tamaño del modelo)
- Cargas posteriores: Instantáneas (si el modelo sigue en memoria)
- Descarga automática: Si no se usa por 5 minutos (configurable)

## Optimizaciones ARM-Axion

El sistema incluye múltiples optimizaciones específicas para ARM:

- **Kernels NEON**: Aceleración vectorial para operaciones matriciales
- **ARM Compute Library (ACL)**: Optimización de GEMM
- **Flash Attention**: Para secuencias largas (>512 tokens)
- **RMSNorm optimizado**: 5x más rápido con vectorización
- **SwiGLU fusionado**: 1.5x mejora en activaciones
- **RoPE vectorizado**: 1.4x más rápido

Mejora global: **1.7-2.0x** (60-80% más rápido que versión sin optimizaciones)

## Configuración

### Archivo Principal

`/home/elect/capibara6/arm-axion-optimizations/vllm_integration/config.json`

Este es un enlace simbólico que apunta a:
`config.five_models_all_working.json`

### Configuraciones Disponibles

```bash
cd /home/elect/capibara6/arm-axion-optimizations/vllm_integration

# Configuración actual (5 modelos)
config.five_models_all_working.json

# Otras configuraciones disponibles
config.five_models_optimized_with_aya.json
config.low_latency_batching.json
config.optimized_kv_cache.json
```

Para cambiar configuración:

```bash
cd /home/elect/capibara6/arm-axion-optimizations/vllm_integration
ln -sf <nuevo_config.json> config.json
# Reiniciar el servidor
```

## Documentación Adicional

- **Arquitectura de Producción**: Ver `PRODUCTION_ARCHITECTURE.md`
- **Setup de Modelos**: Ver `README_MODELS_SETUP.md`
- **Confirmación Aya Expanse**: Ver `AYA_EXPANSE_MODEL_CONFIRMATION.md`
- **Docs deprecadas**: Ver `docs/deprecated/` (documentos históricos)

## Integración RAG

El sistema incluye soporte para RAG (Retrieval Augmented Generation):

```python
from backend.rag_client import RAGClient

# Inicializar cliente
rag = RAGClient(
    base_url="http://localhost:8001",  # Bridge RAG
    enable_toon=True  # Optimización de tokens
)

# Búsqueda semántica
results = rag.search_semantic(
    query="¿Cómo funciona el sistema?",
    n_results=5
)
```

**Nota**: El bridge RAG puede estar en otra VM según la arquitectura de producción.

## Monitoreo

### Logs del Servidor

```bash
# Logs en tiempo real
tail -f /tmp/multi_model_server.log

# Logs recientes
tail -100 /tmp/multi_model_server.log
```

### Verificar Procesos

```bash
# Ver proceso del servidor
ps aux | grep multi_model_server

# Ver puertos escuchando
ss -tlnp | grep 8082
```

### Verificar Recursos

```bash
# Memoria disponible
free -h

# CPU usage
top -u elect
```

## Resolución de Problemas

### Servidor no inicia

1. Verificar que el puerto 8082 no esté ocupado:
   ```bash
   ss -tlnp | grep 8082
   ```

2. Verificar logs:
   ```bash
   tail -50 /tmp/multi_model_server.log
   ```

3. Verificar configuración:
   ```bash
   cd /home/elect/capibara6/arm-axion-optimizations/vllm_integration
   cat config.json | jq '.experts | length'
   # Debe devolver 5
   ```

### Modelo tarda en responder

- **Primera vez**: Es normal, el modelo se está cargando (lazy loading)
- **Espera**: 20-60 segundos según tamaño del modelo
- **Siguientes veces**: Respuesta inmediata

### Error de memoria

1. Verificar memoria disponible:
   ```bash
   free -h
   ```

2. Ajustar configuración de lazy loading en `config.json`:
   ```json
   "lazy_loading": {
     "max_loaded_experts": 3,  // Reducir de 5 a 3
     "auto_unload_after_s": 180  // Descargar más rápido
   }
   ```

## Recursos

- **vLLM Modificado**: `/home/elect/capibara6/vllm-source-modified`
- **Modelos**: `/home/elect/models/`
- **Configuración**: `/home/elect/capibara6/arm-axion-optimizations/vllm_integration/`
- **Backend**: `/home/elect/capibara6/backend/`

## Contacto y Soporte

Para issues y soporte, revisar:
- Logs del sistema
- Documentación en `docs/`
- Configuraciones en `arm-axion-optimizations/vllm_integration/`

---

**Última actualización**: 2025-12-02
**Versión**: 1.0.0 (ARM Axion Optimized)
**Estado**: Producción - Operativo

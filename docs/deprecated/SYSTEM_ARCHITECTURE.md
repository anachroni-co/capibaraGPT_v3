# Capibara6 - Sistema de IA con vLLM y Consenso

## Resumen del Sistema

El sistema Capibara6 es una plataforma de IA conversacional con múltiples modelos que opera sobre una infraestructura distribuida en Google Cloud. El sistema ha sido migrado de Ollama a vLLM con endpoints compatibles con OpenAI.

## Arquitectura del Sistema

### Componentes Principales
- **Backend Consenso**: vm-bounty2 con sistema de múltiples modelos y consenso
- **RAG Server**: vm-rag3 con Milvus, Nebula Graph, PostgreSQL para búsqueda aumentada
- **VM Servicios**: vm-services con MCP, TTS y otros servicios especializados
- **Frontend**: Vercel con chat UI responsive y templates especializados

### Modelos Activos
1. **phi4:mini** - Modelo rápido para tareas simples (reemplazó a phi3:mini)
2. **qwen2.5-coder:1.5b** - Experto en código y tareas técnicas (reemplazó a mistral)
3. **gpt-oss-20b** - Modelo complejo para análisis profundos
4. **mixtral** - Modelo general para tareas creativas

## Migración de Ollama a vLLM

### Cambios Implementados
- **Endpoints**: `/api/generate` → `/v1/chat/completions` (compatibilidad OpenAI)
- **Formato**: de `prompt`-based a `messages`-based con `{role, content}` 
- **Autenticación**: Usando "Bearer EMPTY" para vLLM
- **Streaming**: Adaptado a eventos SSE de vLLM

### Actualización de Modelos
- **phi3:mini** → **phi4:mini** (de 3.8B a 14B parámetros)
- **mistral** → **qwen2.5-coder:1.5b** (experto en programación)

## Sistema de Consenso

### Configuración
- **Método**: Votación ponderada
- **Pesos**: phi4: 0.7, qwen2.5-coder: 0.8, gpt-oss-20b: 0.9, mixtral: 0.6
- **Mín/Max modelos**: 2/3 para consenso
- **Fallback**: phi4 como modelo de respaldo

## Integración RAG-E2B-TOON

### Sistema RAG
- **VM RAG3** (IP: 10.154.0.2): Milvus (vector), Nebula Graph (relaciones), PostgreSQL (estructurado)
- **Buscador semántico** con MiniRAG y FullRAG para diferentes niveles de profundidad
- **Integración automática** con detección de necesidad de contexto RAG

### Sistema E2B
- **Integración automática** para ejecución de código cuando se detecta necesidad
- **Detección de código** en consultas de programación
- **Entornos seguros** para ejecución de scripts

### Sistema TOON
- **Optimización de tokens** en 30-60% para contextos RAG
- **Formateo eficiente** de información para LLMs
- **Reducción de latencia** en procesamiento de contextos grandes

## Configuración de Endpoints

### Endpoints Principales
- **vLLM**: `http://34.12.166.76:8000/v1` para gpt-oss-20b
- **vLLM**: `http://34.12.166.76:8001/v1` para phi4
- **vLLM**: `http://34.12.166.76:8002/v1` para qwen2.5-coder
- **vLLM**: `http://34.12.166.76:8003/v1` para mixtral
- **RAG**: `http://10.154.0.2:8000` (interno)

## Comandos de Inicio

### Comandos de vLLM
```bash
# Iniciar servidor vLLM para modelos específicos
vllm serve microsoft/Phi-4-mini --host 0.0.0.0 --port 8001 --api-key EMPTY
vllm serve Qwen/Qwen2.5-Coder-1.5B-Instruct --host 0.0.0.0 --port 8002 --api-key EMPTY
vllm serve /home/elect/models/gpt-oss-20b --host 0.0.0.0 --port 8000 --api-key EMPTY
```

## Seguridad y Monitoreo

### Firewalls
- Puertos 8000-8003 abiertos para vLLM endpoints
- IP interna de RAG3 (10.154.0.2) configurada para comunicaciones internas
- Configuraciones específicas por VM (bounty2, rag3, services)

### Monitoreo
- **Grafana**: `http://10.154.0.2:3000` (en RAG3 para métricas del sistema)
- **Prometheus**: `http://10.154.0.2:9090` (en RAG3 para recolección de métricas)

## Carpetas Relevantes

- **/home/elect/capibara6/**: Directorio principal con configuraciones
- **/home/elect/capibara6/vm-bounty2/**: Backend de modelos y consenso
- **/home/elect/capibara6/vm-rag3/**: Sistema de búsqueda aumentada
- **/home/elect/models/**: Modelos físicos instalados (phi-4-mini, qwen2.5-coder-1.5b, etc.)

## Archivos de Configuración Importantes

- `model_config.json`: Configuración principal de modelos y endpoints
- `ollama_client.js` → `VLLMClient.js`: Cliente JS para vLLM
- `backend/ollama_client.py` → `VLLMClient.py`: Cliente Python para vLLM  
- `backend/ollama_rag_integration.py`: Integración RAG con vLLM
- `vm-bounty2/config/models_config.py`: Configuración de modelos en backend
- `vm-bounty2/servers/consensus_server.py`: Servidor de consenso
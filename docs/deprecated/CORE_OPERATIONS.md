# Capibara6 - Comandos y Configuraciones Esenciales

## Comandos de Inicio para vLLM

### Iniciar Servidores vLLM

```bash
# phi4-mini (modelo rápido)
vllm serve microsoft/Phi-4-mini --host 0.0.0.0 --port 8001 --api-key EMPTY

# qwen2.5-coder-1.5b (modelo experto en código)
vllm serve Qwen/Qwen2.5-Coder-1.5B-Instruct --host 0.0.0.0 --port 8002 --api-key EMPTY

# gpt-oss-20b (modelo complejo)
vllm serve /home/elect/models/gpt-oss-20b --host 0.0.0.0 --port 8000 --api-key EMPTY

# mixtral (modelo general/creativo)
vllm serve mistralai/Mistral-7B-Instruct-v0.2 --host 0.0.0.0 --port 8003 --api-key EMPTY

# Servidor único para múltiples modelos (alternativa)
vllm serve --model-dir /home/elect/models/ --host 0.0.0.0 --port 8000
```

## Configuraciones del Sistema

### Archivos de Configuración Principales

#### model_config.json
```json
{
  "models": {
    "fast_response": {
      "name": "phi4:mini",  // Actualizado desde phi3:mini
      "description": "Modelo más rápido para respuestas simples",
      "max_tokens": 512,
      "timeout": 8000
    },
    "balanced": {
      "name": "qwen2.5-coder-1.5b",  // Actualizado desde mistral
      "description": "Modelo experto en código y tareas técnicas",
      "max_tokens": 1024,
      "timeout": 20000
    },
    "complex": {
      "name": "gpt-oss-20b",
      "description": "Modelo más potente para tareas complejas",
      "max_tokens": 2048,
      "timeout": 240000
    }
  },
  "api_settings": {
    "vllm_endpoint": "http://34.12.166.76:8000/v1",  // vLLM endpoint
    "ollama_endpoint": "http://34.12.166.76:8000/v1",
    "default_model": "phi4:mini",
    "max_concurrent_requests": 4,
    "streaming_enabled": true
  }
}
```

### Servidores de Backend

#### vm-bounty2 Servicios Principales
```bash
# Iniciar backend principal
cd /home/elect/capibara6/vm-bounty2 && python3 servers/server_gptoss.py

# Iniciar servidor de consenso  
cd /home/elect/capibara6/vm-bounty2 && python3 servers/consensus_server.py

# Iniciar todos los servicios
cd /home/elect/capibara6/vm-bounty2 && python3 scripts/start_system.py
```

### Endpoints API

#### Endpoints Principales
- `POST /api/ai/generate` - Generar texto con selección automática de modelo
- `POST /api/ai/:modelTier/generate` - Generar texto con modelo específico  
- `POST /api/ai/classify` - Clasificar tarea sin ejecutarla
- `POST /api/consensus/query` - Consulta con sistema de consenso
- `GET /health` - Health check del sistema

## Verificación del Sistema

### Verificar Servicios
```bash
# Verificar conexión con vLLM
curl http://34.12.166.76:8000/v1/models

# Verificar conexión con backend
curl http://34.12.166.76:5001/health

# Verificar consenso
curl http://34.12.166.76:5005/health
```

### Test de Modelos
```bash
# Test vLLM endpoint
curl -X POST "http://34.12.166.76:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "phi4:mini",
    "messages": [{"role": "user", "content": "Hola, ¿cómo estás?"}],
    "max_tokens": 100
  }'
```

## Firewall y Seguridad

### Puertos Abiertos
- **8000-8003**: vLLM endpoints (TCP)
- **5001-5005**: Servicios backend (TCP)
- **3000, 9090**: Monitoreo (Grafana, Prometheus en RAG3)
- **22**: SSH (TCP, solo IPs autorizadas)

### IPs Autorizadas
- **RAG3**: 10.154.0.2 (interna, acceso a todos los servicios)
- **Frontend**: Dominio capibara6.com (acceso a endpoints API)
- **Admin**: IPs específicas para acceso SSH

## Scripts de Administración

### Scripts Disponibles en vm-bounty2
```bash
# Verificar servicios
python3 scripts/check_services.py

# Iniciar sistema completo  
python3 scripts/start_system.py

# Verificar conexión con modelos
python3 scripts/test_models.py

# Verificar RAG
python3 scripts/test_rag.py
```

## Procedimientos de Troubleshooting

### Problemas Comunes
1. **Modelo no responde**: Verificar que vLLM está corriendo y puerto está abierto
2. **Conexión RAG fallida**: Verificar conexión interna con 10.154.0.2
3. **Consensus no funciona**: Verificar que todos los modelos están accesibles
4. **E2B execution falla**: Verificar que los endpoints de código están disponibles

### Comandos de Diagnóstico
```bash
# Verificar todos los servicios
python3 check_all_services.sh

# Verificar modelos específicos
curl http://34.12.166.76:8000/v1/models

# Verificar estado del sistema
python3 verify_all_services.py
```
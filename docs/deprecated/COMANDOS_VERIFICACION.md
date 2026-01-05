# 🔍 Comandos de Verificación - Capibara6

> **Propósito:** Validar el estado actual del sistema después de la migración a vLLM  
> **Fecha:** 2025-11-24

---

## 🎯 Verificación Rápida (5 minutos)

### 1. Verificar Modelos vLLM

```bash
# Verificar GPT-OSS-20B (puerto 8000)
curl http://34.12.166.76:8000/v1/models

# Verificar Phi4-mini (puerto 8001)
curl http://34.12.166.76:8001/v1/models

# Verificar Qwen2.5-coder (puerto 8002)
curl http://34.12.166.76:8002/v1/models

# Verificar Mixtral (puerto 8003)
curl http://34.12.166.76:8003/v1/models
```

**Resultado esperado:** JSON con lista de modelos disponibles

---

### 2. Verificar Servicios Backend

```bash
# Backend principal
curl http://34.12.166.76:5001/health

# Auth server
curl http://34.12.166.76:5004/health

# Consensus server
curl http://34.12.166.76:5005/health
```

**Resultado esperado:** `{"status": "healthy"}` o similar

---

### 3. Verificar Servicios Especializados

```bash
# TTS Kyutai
curl http://34.175.136.104:5002/health

# MCP Server (puede estar deshabilitado)
curl http://34.175.136.104:5003/api/mcp/health
```

---

### 4. Verificar Sistema RAG (desde dentro de VM RAG3)

```bash
# Bridge API
curl http://10.154.0.2:8000/health

# Milvus
curl http://10.154.0.2:19530

# Nebula Graph
curl http://10.154.0.2:9669
```

---

## 🧪 Tests de Funcionalidad (15 minutos)

### Test 1: Generación de Texto con vLLM

```bash
# Test con GPT-OSS-20B
curl -X POST "http://34.12.166.76:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [
      {"role": "user", "content": "Hola, ¿cómo estás?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**Resultado esperado:** JSON con respuesta del modelo

---

### Test 2: Generación con Phi4 (modelo rápido)

```bash
curl -X POST "http://34.12.166.76:8001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "phi4:mini",
    "messages": [
      {"role": "user", "content": "¿Cuál es la capital de España?"}
    ],
    "max_tokens": 50
  }'
```

---

### Test 3: Generación con Qwen2.5-coder (código)

```bash
curl -X POST "http://34.12.166.76:8002/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "qwen2.5-coder:1.5b",
    "messages": [
      {"role": "user", "content": "Escribe una función Python para calcular factorial"}
    ],
    "max_tokens": 200
  }'
```

---

### Test 4: Streaming de Respuestas

```bash
curl -X POST "http://34.12.166.76:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [
      {"role": "user", "content": "Explica qué es la inteligencia artificial"}
    ],
    "max_tokens": 300,
    "stream": true
  }'
```

**Resultado esperado:** Stream de eventos SSE con tokens

---

### Test 5: Backend Principal (API v1)

```bash
# Test del endpoint principal
curl -X POST "http://34.12.166.76:5001/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Qué es Capibara6?",
    "model": "phi4:mini"
  }'
```

---

### Test 6: TTS (Text-to-Speech)

```bash
# Listar voces disponibles
curl http://34.175.136.104:5002/voices

# Generar audio
curl -X POST "http://34.175.136.104:5002/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hola, soy Capibara6",
    "voice": "default",
    "language": "es"
  }' \
  --output test_audio.wav
```

---

### Test 7: Sistema RAG

```bash
# Búsqueda semántica (desde VM RAG3 o con acceso interno)
curl -X POST "http://10.154.0.2:8000/api/v1/rag/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "arquitectura del sistema",
    "top_k": 5
  }'
```

---

## 📊 Verificación de Monitorización

### Grafana

```bash
# Abrir en navegador
http://10.154.0.2:3000

# Credenciales por defecto (cambiar si están configuradas):
# Usuario: admin
# Password: admin
```

**Verificar:**
- ✅ 18 dashboards disponibles
- ✅ Métricas actualizándose en tiempo real
- ✅ Sin alertas críticas activas

---

### Prometheus

```bash
# Abrir en navegador
http://10.154.0.2:9090

# Verificar métricas
# Query ejemplo: up{job="capibara6"}
```

**Verificar:**
- ✅ Todos los targets "UP"
- ✅ 30+ alertas configuradas
- ✅ Métricas de modelos disponibles

---

### Jaeger (Distributed Tracing)

```bash
# Abrir en navegador
http://10.154.0.2:16686
```

**Verificar:**
- ✅ Traces de requests recientes
- ✅ Latencias de servicios
- ✅ Dependencias entre componentes

---

## 🔧 Verificación de Configuración

### 1. Verificar archivos de configuración

```bash
# Model config
cat model_config.json

# Frontend config
cat frontend/src/config.js

# Backend requirements
cat backend/requirements.txt
```

---

### 2. Verificar servicios en ejecución (en VM Bounty2)

```bash
# Listar procesos de vLLM
ps aux | grep vllm

# Verificar puertos abiertos
netstat -tulpn | grep -E '8000|8001|8002|8003|5001|5004|5005'

# Verificar logs
tail -f /var/log/capibara6/backend.log
```

---

### 3. Verificar firewall (en cada VM)

```bash
# VM Bounty2
sudo ufw status

# Verificar reglas para puertos vLLM
sudo ufw status | grep -E '8000|8001|8002|8003'

# Verificar reglas para backend
sudo ufw status | grep -E '5001|5004|5005'
```

---

## 🚨 Troubleshooting

### Problema: Modelo no responde

```bash
# 1. Verificar que vLLM está corriendo
ps aux | grep vllm

# 2. Verificar logs
journalctl -u vllm-gpt-oss-20b -f

# 3. Reiniciar servicio
sudo systemctl restart vllm-gpt-oss-20b

# 4. Verificar puerto
curl http://localhost:8000/v1/models
```

---

### Problema: Backend no conecta con vLLM

```bash
# 1. Verificar conectividad
telnet 34.12.166.76 8000

# 2. Verificar firewall
sudo ufw status | grep 8000

# 3. Verificar logs del backend
tail -f /var/log/capibara6/backend.log

# 4. Probar endpoint directamente
curl http://34.12.166.76:8000/v1/models
```

---

### Problema: RAG no responde

```bash
# 1. Verificar servicios en VM RAG3
ssh rag3
docker ps

# 2. Verificar Milvus
curl http://10.154.0.2:19530

# 3. Verificar Nebula Graph
curl http://10.154.0.2:9669

# 4. Verificar Bridge API
curl http://10.154.0.2:8000/health
```

---

### Problema: TTS no funciona

```bash
# 1. Verificar servicio
curl http://34.175.136.104:5002/health

# 2. Listar voces
curl http://34.175.136.104:5002/voices

# 3. Verificar logs
ssh gpt-oss-20b
journalctl -u kyutai-tts -f

# 4. Reiniciar servicio
sudo systemctl restart kyutai-tts
```

---

## 📋 Checklist de Verificación Completa

### Modelos vLLM
- [ ] GPT-OSS-20B (puerto 8000) responde
- [ ] Phi4-mini (puerto 8001) responde
- [ ] Qwen2.5-coder (puerto 8002) responde
- [ ] Mixtral (puerto 8003) responde
- [ ] Streaming funciona correctamente

### Backend
- [ ] Backend principal (5001) healthy
- [ ] Auth server (5004) healthy
- [ ] Consensus server (5005) healthy
- [ ] Endpoints API v1 funcionan
- [ ] E2B execution disponible

### Servicios
- [ ] TTS Kyutai (5002) activo
- [ ] MCP Server (5003) - estado documentado
- [ ] N8N (5678) - accesible con VPN

### Sistema RAG
- [ ] Bridge API (8000) healthy
- [ ] Milvus (19530) accesible
- [ ] Nebula Graph (9669) accesible
- [ ] PostgreSQL (5432) accesible
- [ ] Redis (6379) accesible

### Monitorización
- [ ] Grafana (3000) accesible
- [ ] Prometheus (9090) recolectando métricas
- [ ] Jaeger (16686) mostrando traces
- [ ] Dashboards actualizados
- [ ] Alertas configuradas

### Frontend
- [ ] Chat carga correctamente
- [ ] Streaming de respuestas funciona
- [ ] TTS integrado funciona
- [ ] Multiidioma (ES/EN) funciona
- [ ] Historial de conversaciones funciona

---

## 📝 Registro de Verificación

**Fecha de última verificación:** _____________

**Verificado por:** _____________

**Resultados:**

| Componente | Estado | Notas |
|------------|--------|-------|
| vLLM GPT-OSS-20B | ⬜ | |
| vLLM Phi4 | ⬜ | |
| vLLM Qwen2.5 | ⬜ | |
| vLLM Mixtral | ⬜ | |
| Backend | ⬜ | |
| TTS | ⬜ | |
| RAG | ⬜ | |
| Monitorización | ⬜ | |
| Frontend | ⬜ | |

**Problemas encontrados:**
- 
- 

**Acciones tomadas:**
- 
- 

---

## 🔗 Referencias

- [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - Arquitectura del sistema
- [CORE_OPERATIONS.md](CORE_OPERATIONS.md) - Comandos de operación
- [RESUMEN_EJECUTIVO.md](RESUMEN_EJECUTIVO.md) - Resumen del estado
- [ANALISIS_ESTADO_ACTUAL.md](ANALISIS_ESTADO_ACTUAL.md) - Análisis completo

---

**Documento creado:** 2025-11-24  
**Próxima actualización:** Después de ejecutar verificaciones

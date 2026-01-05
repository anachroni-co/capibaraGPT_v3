# Hallazgos de Infraestructura - Capibara6
## Fecha: 2025-11-13

## Resumen Ejecutivo

Investigación completa de la infraestructura del proyecto Capibara6 para identificar todos los servicios, bases de datos y servidores activos o documentados.

---

## 🔍 Servicios Backend Encontrados y Verificados

### Servidor Principal: `backend/server_gptoss.py` (Puerto 5001)
**Rol:** Backend principal del chatbot con GPT-OSS-20B
**Estado:** ✅ Activo y configurado
**Endpoints:**
- `POST /api/chat` - Chat básico
- `POST /api/chat/stream` - Chat con streaming
- `GET /api/health` - Health check
- `GET /api/models` - Listar modelos
- `POST /api/save-conversation` - Guardar conversaciones

**Frontend conecta a:** `http://localhost:5001` (desarrollo) o `VM_MODELS:5001` (producción)

### Servidor FastAPI Alternativo: `backend/main.py` (Puerto 8000)
**Rol:** API alternativa con E2B integrado
**Estado:** ⚠️ Disponible pero no usado por frontend actual
**Endpoints:**
- `GET /health`
- `POST /api/v1/query` - Consulta al modelo
- `GET /api/v1/models` - Listar modelos
- `POST /api/v1/e2b/execute` - Ejecutar código en E2B

### Servidor MCP: `backend/mcp_server.py` (Puerto 5003)
**Rol:** Model Context Protocol - RAG y contexto inteligente
**Estado:** ⚠️ Opcional, deshabilitado por defecto
**Endpoints:**
- `GET /api/mcp/contexts` - Listar contextos
- `GET /api/mcp/context/<id>` - Obtener contexto específico
- `POST /api/mcp/augment` - Aumentar prompt con contexto (RAG)
- `GET /api/mcp/tools` - Listar herramientas
- `POST /api/mcp/calculate` - Calculadora
- `POST /api/mcp/verify` - Verificar hechos
- `GET /api/mcp/health` - Health check

**Configuración Frontend:** `web/config.js` → `SERVICES.MCP.enabled = false`

### Servidor TTS: `backend/kyutai_tts_server.py` (Puerto 5002)
**Rol:** Text-to-Speech con Kyutai Moshi
**Estado:** ✅ Activo en VM gpt-oss-20b (34.175.136.104:5002)
**Endpoints:**
- `POST /tts` - Síntesis de voz
- `GET /voices` - Listar voces
- `POST /clone` - Clonar voz
- `GET /health` - Health check
- `POST /preload` - Precargar modelo

**Documentación:** `SERVICES_SETUP.md`

### Servidor Auth: `backend/auth_server.py` (Puerto 5004)
**Rol:** Autenticación OAuth (GitHub y Google)
**Estado:** ✅ Configurado
**Endpoints:**
- `GET /auth/github` - Login con GitHub
- `GET /auth/google` - Login con Google
- `POST /auth/verify` - Verificar token
- `POST /auth/logout` - Cerrar sesión
- `GET /auth/callback/github` - Callback GitHub
- `GET /auth/callback/google` - Callback Google
- `GET /health` - Health check

**Cambio reciente:** Puerto cambiado de 5001 → 5004 (Fase 2)

### Servidor Consensus: `backend/consensus_server.py` (Puerto 5005)
**Rol:** Consenso multi-modelo
**Estado:** ⚠️ Deshabilitado por defecto
**Endpoints:**
- `POST /api/consensus/query` - Consulta con consenso
- `GET /api/consensus/models` - Listar modelos
- `GET /api/consensus/templates` - Templates de consenso
- `GET /api/consensus/config` - Configuración
- `GET /api/consensus/health` - Health check

**Cambio reciente:** Puerto cambiado de 5002 → 5005 (Fase 2)
**Configuración Frontend:** `web/config.js` → `SERVICES.CONSENSUS.enabled = false`

### Servidor Smart MCP Alternativo: `backend/smart_mcp_server.py` (Puerto 5010)
**Rol:** MCP alternativo con RAG selectivo simplificado
**Estado:** ⚠️ Opcional, alternativa a mcp_server.py
**Endpoints:**
- `GET /health` - Health check
- `POST /analyze` - Análisis de query
- `POST /update-date` - Actualizar fecha

**Configuración Frontend:** `web/config.js` → `SERVICES.SMART_MCP.enabled = false`

---

## 🗄️ Bases de Datos Encontradas

### PostgreSQL (Puerto 5432)
**Ubicación:** `docker-compose.yml`
**Estado:** ✅ Configurado en Docker
**Uso:** Base de datos principal para persistencia
```yaml
postgres:
  image: postgres:15
  ports:
    - "5432:5432"
  volumes:
    - postgres_data:/var/lib/postgresql/data
```

### TimescaleDB (Puerto 5433)
**Ubicación:** `docker-compose.yml`
**Estado:** ✅ Configurado en Docker
**Uso:** Time-series data (métricas, logs temporales)
```yaml
timescaledb:
  image: timescale/timescaledb:latest-pg15
  ports:
    - "5433:5432"
  volumes:
    - timescale_data:/var/lib/postgresql/data
```

### Redis (Puerto 6379)
**Ubicación:** `docker-compose.yml`
**Estado:** ✅ Configurado en Docker
**Uso:** Cache y sesiones
```yaml
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"
  volumes:
    - redis_data:/data
```

### FAISS Vector Store
**Ubicación:** `backend/config/infrastructure_config.py`
**Estado:** ✅ Configurado como vector store principal
**Uso:** Búsqueda de vectores para RAG
```python
RAG_CONFIG = {
    'vector_store': {
        'type': 'faiss',
        'index_type': 'IndexFlatIP',
        'embedding_dimension': 384
    }
}
```

### ChromaDB
**Ubicación:** `archived/backend_modules/core/rag/vector_store.py`
**Estado:** ❌ Solo en código archivado, no activo
**Nota:** Código existe pero no está en uso actualmente

---

## ✅ Servicios Encontrados en VM rag3 (ACTUALIZACIÓN 2025-11-13)

### Milvus Vector Database
**Búsqueda en repositorio:** ❌ No encontrado en código
**Búsqueda en VM rag3:** ✅ **ENCONTRADO Y ACTIVO**

**Ubicación:** VM rag3 (europe-west2-c)
**Puerto:** 19530
**Implementación:** Docker Compose en VM rag3
**Estado:** ✅ Corriendo (uptime: 3 días)
**Versión:** v2.3.10
**Contenedor:** `milvus-standalone`

**Stack Completo:**
- Milvus server (puerto 19530, 9091)
- MinIO object storage (9000-9001)
- etcd coordination (2379-2380)

**Uso:** Vector database para RAG, búsqueda semántica, embeddings

### Nebula Graph Database
**Búsqueda en repositorio:** ❌ No encontrado en código
**Búsqueda en VM rag3:** ✅ **ENCONTRADO Y ACTIVO**

**Ubicación:** VM rag3 (europe-west2-c)
**Puerto Principal:** 9669 (query service)
**Implementación:** Docker Compose en VM rag3 (cluster de 3 nodos)
**Estado:** ✅ Corriendo (uptime: 3 días)
**Versión:** v3.1.0

**Arquitectura del Cluster:**
- 3x nebula-graphd (query service - puerto 9669)
- 3x nebula-metad (metadata service - puerto 9559)
- 3x nebula-storaged (storage service - puerto 9779)
- 1x nebula-graph-studio (UI web - puerto 7001)

**Uso:** Graph database para relaciones complejas, knowledge graphs

### Servidor "Bridge" - capibara6-api
**Búsqueda en repositorio:** ❌ No encontrado explícitamente
**Búsqueda en VM rag3:** ✅ **ENCONTRADO Y ACTIVO**

**Ubicación:** VM rag3 (europe-west2-c)
**Puerto:** 8000
**Implementación:** Docker container `capibara6-api`
**Estado:** ✅ Corriendo (uptime: 2 días)
**Comando:** `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000`

**Función del Bridge:**
- ✅ API principal de integración
- ✅ Coordina Milvus (vector search)
- ✅ Coordina Nebula Graph (graph queries)
- ✅ Orquesta 3x RQ workers para procesamiento asíncrono
- ✅ Integración con PostgreSQL, TimescaleDB, Redis

**Workers Asociados:**
- capibara6-worker-1
- capibara6-worker-2
- capibara6-worker-3

**Nota sobre repositorio:**
El código de `capibara6-api` debe estar en un directorio/imagen personalizada. El frontend puede integrarse directamente con este servicio en `http://rag3:8000`

### Rol de server_gptoss.py
**`backend/server_gptoss.py`** (puerto 5001 en VM bounty2) actúa como **backend secundario** para:
- Chat directo con GPT-OSS-20B
- TTS integration
- Autenticación
- Consensus multi-modelo

Mientras que **`capibara6-api`** (puerto 8000 en VM rag3) es el **bridge principal** para:
- RAG con Milvus + Nebula
- Procesamiento asíncrono
- Integración completa del stack

**Evidencia:**
```javascript
// web/config.js
const CHATBOT_CONFIG = {
    BACKEND_URL: isLocalhost ? 'http://localhost:5001' : VM_MODELS + ':5001',
    ENDPOINTS: {
        CHAT: '/api/v1/query',
        CHAT_STREAM: '/api/v1/chat/stream',
        TTS_SPEAK: '/api/tts/speak',
        MCP_CONTEXT: '/api/v1/mcp/context',
        E2B_EXECUTE: '/api/v1/e2b/execute'
    }
}
```

---

## 📋 Servicios Externos Documentados

### N8N Workflow Automation (Puerto 5678)
**Ubicación:** VM gpt-oss-20b (34.175.136.104:5678)
**Estado:** ⚠️ Requiere VPN/túnel
**Documentación:** `SERVICES_SETUP.md`
**Configuración Frontend:** Deshabilitado en `web/config.js` → `N8N_ENABLED: false`

**Razón de deshabilitado:** No es accesible públicamente, requiere conexión VPN a la VM

### VM rag3 (europe-west2-c)
**Estado:** ✅ **VERIFICADO Y DOCUMENTADO** (2025-11-13)
**Descripción:** Sistema RAG completo con vector + graph databases
**Documentación completa:** Ver `VM_RAG3_COMPLETE_ANALYSIS.md`

**Servicios Principales:**
- ✅ **Milvus Vector Database** (puerto 19530) - Vector search para RAG
- ✅ **Nebula Graph Database** (puerto 9669) - Graph database (cluster 3 nodos)
- ✅ **capibara6-api Bridge** (puerto 8000) - API principal de integración
- ✅ **PostgreSQL** (puerto 5432) - Base de datos relacional
- ✅ **TimescaleDB** (puerto 5433) - Time-series data
- ✅ **Redis** (puerto 6379) - Cache y queue
- ✅ **N8N** (puerto 5678) - Workflow automation
- ✅ **Nginx** (puertos 80, 443) - Reverse proxy
- ✅ **Monitoring Stack** - Grafana (3000), Prometheus (9090), Jaeger (16686)

**Workers:**
- 3x RQ Workers para procesamiento asíncrono en background

---

## 🏗️ Arquitectura de VMs

### VM bounty2 (34.12.166.76)
**Servicios:**
- Backend principal (server_gptoss.py - puerto 5001)
- Auth server (puerto 5004)
- Consensus server (puerto 5005)
- Ollama (modelo local)

### VM gpt-oss-20b (34.175.136.104)
**Servicios:**
- TTS Server (puerto 5002)
- MCP Server (puerto 5003)
- Smart MCP alternativo (puerto 5010)
- N8N (puerto 5678 - VPN requerida)

### VM rag3 (dirección desconocida)
**Estado:** Mencionada en documentación pero sin detalles
**Servicios esperados:**
- Sistema RAG completo
- ¿Milvus?
- ¿Nebula Graph?
- ¿Bridge server?

---

## 🔧 Configuración de RAG

### Mini RAG
```python
'mini_rag': {
    'timeout_ms': 50,
    'max_results': 5,
    'cache_size': 1000,
    'cache_ttl_seconds': 300
}
```

### Full RAG
```python
'full_rag': {
    'max_results': 10,
    'expansion_factor': 2.0,
    'deep_search_timeout_ms': 200
}
```

### Vector Store
```python
'vector_store': {
    'type': 'faiss',
    'index_type': 'IndexFlatIP',
    'embedding_dimension': 384
}
```

---

## 📊 Resumen de Puertos - Arquitectura Completa

### VM bounty2 (34.12.166.76)
| Puerto | Servicio | Estado | Descripción |
|--------|----------|--------|-------------|
| 5001 | Backend Principal (server_gptoss.py) | ✅ Activo | Chat GPT-OSS-20B |
| 5004 | Auth Server | ✅ Configurado | OAuth GitHub/Google |
| 5005 | Consensus Server | ⚠️ Opcional | Multi-modelo |
| 8000 | FastAPI (main.py) | ⚠️ Alternativo | API E2B |

### VM gpt-oss-20b (34.175.136.104)
| Puerto | Servicio | Estado | Descripción |
|--------|----------|--------|-------------|
| 5002 | TTS Server (Kyutai) | ✅ Activo | Text-to-Speech |
| 5003 | MCP Server | ⚠️ Opcional | Context & RAG |
| 5010 | Smart MCP Alternativo | ⚠️ Opcional | RAG selectivo |
| 5678 | N8N | ⚠️ VPN requerida | Workflows |

### VM rag3 (europe-west2-c) ⭐ NUEVO
| Puerto | Servicio | Estado | Descripción |
|--------|----------|--------|-------------|
| **80** | **Nginx HTTP** | ✅ Activo | Reverse proxy |
| **443** | **Nginx HTTPS** | ✅ Activo | Reverse proxy SSL |
| **3000** | **Grafana** | ✅ Activo | Dashboards |
| **5432** | **PostgreSQL** | ✅ Activo | DB Relacional |
| **5433** | **TimescaleDB** | ✅ Activo | Time-series |
| **5678** | **N8N** | ✅ Activo | Workflows |
| **6379** | **Redis** | ✅ Activo | Cache + Queue |
| **7001** | **Nebula Studio** | ✅ Activo | Graph UI |
| **8000** | **capibara6-api (BRIDGE)** | ✅ **ACTIVO** | **API Principal** |
| **9000-9001** | **MinIO** | ✅ Activo | Object Storage |
| **9090** | **Prometheus** | ✅ Activo | Metrics |
| **9091** | **Milvus Metrics** | ✅ Activo | Milvus stats |
| **9669** | **Nebula Graph Query** | ✅ **ACTIVO** | **Graph DB** |
| **14268** | **Jaeger Collector** | ✅ Activo | Tracing |
| **16686** | **Jaeger UI** | ✅ Activo | Tracing UI |
| **19530** | **Milvus** | ✅ **ACTIVO** | **Vector DB** |

### Puertos Locales (Docker Compose)
| Puerto | Servicio | Estado | VM |
|--------|----------|--------|-----|
| 5432 | PostgreSQL | ✅ Docker | Local |
| 5433 | TimescaleDB | ✅ Docker | Local |
| 6379 | Redis | ✅ Docker | Local |

---

## 🎯 Conclusiones - ACTUALIZADO 2025-11-13

### ✅ Hallazgos Confirmados

1. **Backend principal** claramente definido (server_gptoss.py en puerto 5001 - VM bounty2)
2. **Bridge API confirmado** - capibara6-api en puerto 8000 (VM rag3)
3. **Milvus Vector Database** - ENCONTRADO y activo (puerto 19530 - VM rag3)
4. **Nebula Graph Database** - ENCONTRADO y activo (puerto 9669 - VM rag3, cluster completo)
5. **ChromaDB** - NO instalado (Milvus lo reemplaza)
6. Servicios especializados con puertos dedicados
7. Frontend correctamente configurado
8. **Stack de monitoreo completo** en VM rag3 (Grafana, Prometheus, Jaeger)

### 🏗️ Arquitectura Verificada

El sistema Capibara6 utiliza **3 VMs especializadas**:

1. **VM bounty2** - Chat y modelos
   - Backend GPT-OSS-20B
   - Auth y Consensus

2. **VM gpt-oss-20b** - Servicios especializados
   - TTS (Kyutai)
   - MCP (RAG básico)
   - N8N

3. **VM rag3** - Sistema RAG completo ⭐
   - **Milvus** (vector search)
   - **Nebula Graph** (knowledge graph)
   - **capibara6-api** (bridge/orquestador)
   - PostgreSQL + TimescaleDB + Redis
   - Stack de monitoreo
   - 3x Workers para procesamiento asíncrono

### ⚠️ Áreas Completadas
1. ✅ **VM rag3:** Completamente documentada (ver VM_RAG3_COMPLETE_ANALYSIS.md)
2. ✅ **Milvus:** Encontrado y documentado (VM rag3:19530)
3. ✅ **Nebula Graph:** Encontrado y documentado (VM rag3:9669)
4. ✅ **Bridge Server:** Identificado como capibara6-api (VM rag3:8000)

### Recomendaciones 📝

1. **Integración Frontend con VM rag3:**
   - ✅ Documentación completa creada (VM_RAG3_COMPLETE_ANALYSIS.md)
   - ⏭️ Actualizar `web/config.js` con URLs de capibara6-api
   - ⏭️ Configurar cliente para Milvus (búsqueda vectorial)
   - ⏭️ Configurar cliente para Nebula Graph (consultas de grafo)

2. **Scripts de Gestión:**
   - ⏭️ Actualizar `check-services.sh` para verificar servicios de VM rag3
   - ⏭️ Agregar healthchecks para Milvus (19530) y Nebula (9669)
   - ⏭️ Monitorear estado de workers RQ

3. **Documentación de APIs:**
   - ⏭️ Documentar endpoints de capibara6-api (puerto 8000)
   - ⏭️ Documentar esquema de Nebula Graph
   - ⏭️ Documentar colecciones de Milvus

4. **Monitoreo:**
   - ✅ Grafana ya configurado (puerto 3000)
   - ✅ Prometheus ya configurado (puerto 9090)
   - ✅ Jaeger ya configurado (puerto 16686)
   - ⏭️ Verificar alertas configuradas

5. **Seguridad:**
   - ⏭️ Verificar que puertos 19530 y 9669 no sean públicos
   - ⏭️ Configurar autenticación en Milvus y Nebula
   - ⏭️ Revisar credenciales de PostgreSQL/Redis

---

## 📁 Archivos de Referencia

### Documentación Principal
- **`VM_RAG3_COMPLETE_ANALYSIS.md`** ⭐ NUEVO - Análisis completo de VM rag3
- `INFRASTRUCTURE_FINDINGS.md` (este archivo) - Hallazgos completos de infraestructura
- `BACKEND_CONSOLIDATION_PLAN.md` - Plan de consolidación (Fases 1-4 completadas)

### Configuración
- `web/config.js` - Configuración completa de servicios frontend
- `backend/config/infrastructure_config.py` - Configuración RAG y vector store
- `docker-compose.yml` - Bases de datos locales

### Scripts de Gestión (backend/)
- `start-all-services.sh` - Iniciar servicios principales
- `start-optional-services.sh` - Iniciar servicios opcionales
- `stop-all-services.sh` - Detener todos los servicios
- `check-services.sh` - Verificar estado de servicios
- `SCRIPTS_README.md` - Documentación completa de scripts

### Herramientas de Diagnóstico
- `vm_rag3_diagnostic.sh` - Script de diagnóstico automatizado
- `VM_RAG3_INSTRUCTIONS.md` - Instrucciones para ejecutar diagnóstico
- `QUICK_VM_RAG3_CHECK.md` - Verificación rápida

### Otros
- `SERVICES_SETUP.md` - Setup de servicios en VMs
- `ARCHITECTURE_QUICK_REF.md` - Referencia rápida de arquitectura
- `FIXES_ENDPOINTS.md` - Correcciones de endpoints

---

## 🚀 Estado Actual y Próximos Pasos

### ✅ Completado (2025-11-13)
1. ✅ **VM rag3 documentada** - Análisis completo realizado
2. ✅ **Milvus encontrado** - Puerto 19530, versión v2.3.10
3. ✅ **Nebula Graph encontrado** - Puerto 9669, cluster de 3 nodos
4. ✅ **Bridge identificado** - capibara6-api en puerto 8000
5. ✅ **Fase 4 implementada** - Scripts de gestión creados y documentados
6. ✅ **Toda la infraestructura mapeada** - 3 VMs con todos sus servicios

### ⏭️ Próximos Pasos Recomendados
1. **Integrar frontend con VM rag3**
   - Actualizar `web/config.js` con capibara6-api endpoints
   - Configurar conexión a Milvus para búsqueda vectorial
   - Configurar conexión a Nebula Graph para consultas

2. **Mejorar scripts de gestión**
   - Agregar verificación de servicios remotos en check-services.sh
   - Crear scripts de conexión a Milvus y Nebula

3. **Documentación de APIs**
   - Documentar endpoints completos de capibara6-api
   - Crear guías de uso para Milvus y Nebula Graph

4. **Testing**
   - Probar integración completa frontend → bridge → databases
   - Verificar rendimiento del sistema RAG completo

# 📊 Análisis Completo del Estado Actual - Capibara6

> **Fecha de análisis:** 2025-11-24  
> **Versión del proyecto:** 3.0.0  
> **Última actualización README:** 2025-11-14  
> **Estado general:** 🟢 Sistema operativo con arquitectura distribuida en 3 VMs

---

## 🎯 Resumen Ejecutivo

**Capibara6** es una plataforma de IA conversacional distribuida en Google Cloud con arquitectura multi-modelo, sistema RAG avanzado (Retrieval Augmented Generation), y servicios especializados. El proyecto ha migrado recientemente de **Ollama a vLLM** con endpoints compatibles con OpenAI.

### Puntos Clave
- ✅ **Arquitectura distribuida** en 3 VMs de Google Cloud
- ✅ **4 modelos de IA** activos con sistema de consenso
- ✅ **Sistema RAG completo** (Milvus + Nebula Graph + PostgreSQL)
- ✅ **Servicios especializados** (TTS Kyutai, MCP, E2B, N8N)
- ✅ **Monitorización completa** (Prometheus, Grafana, Jaeger)
- ⚠️ **Migración reciente** a vLLM requiere validación
- ⚠️ **Documentación dispersa** en múltiples archivos

---

## 🏗️ Arquitectura del Sistema

### **Distribución por VMs**

```
┌─────────────────────────────────────────────────────────────┐
│                      FRONTEND (Vercel)                       │
│              Chat UI | Templates | Multiidioma              │
└────────────┬────────────────────────────────────────────────┘
             │
    ┌────────┴────────┬──────────────┬────────────────┐
    ↓                 ↓              ↓                ↓
┌─────────┐    ┌──────────┐   ┌──────────┐    ┌──────────┐
│ Bounty2 │    │ Services │   │  RAG3    │    │ Frontend │
│(Modelos)│    │(TTS/MCP) │   │(Milvus+) │    │  (Web)   │
│34.12... │    │34.175... │   │10.154... │    │  Vercel  │
│  :5001  │    │:5002/03  │   │  :8000   │    │          │
└─────────┘    └──────────┘   └──────────┘    └──────────┘
```

### **VM Bounty2** (34.12.166.76) - Modelos de IA
- **Backend Principal:** Puerto 5001
- **Auth Server:** Puerto 5004 (OAuth GitHub/Google)
- **Consensus Server:** Puerto 5005 (Multi-modelo)
- **vLLM Endpoints:** Puertos 8000-8003

**Modelos activos:**
1. **phi4:mini** (14B parámetros) - Puerto 8001 - Tareas rápidas
2. **qwen2.5-coder:1.5b** - Puerto 8002 - Experto en código
3. **gpt-oss-20b** (20.9B parámetros) - Puerto 8000 - Tareas complejas
4. **mixtral** (7B parámetros) - Puerto 8003 - Tareas creativas

### **VM Services** (34.175.136.104) - Servicios Especializados
- **TTS Kyutai:** Puerto 5002 (Text-to-Speech)
- **MCP Server:** Puerto 5003 (Model Context Protocol)
- **N8N Workflows:** Puerto 5678 (requiere VPN)

### **VM RAG3** (10.154.0.2 - IP interna) - Sistema RAG
- **Bridge API (capibara6-api):** Puerto 8000
- **Milvus Vector DB:** Puerto 19530 (v2.3.10)
- **Nebula Graph:** Puerto 9669 (v3.1.0)
- **PostgreSQL:** Puerto 5432
- **TimescaleDB:** Puerto 5433
- **Redis:** Puerto 6379
- **Prometheus:** Puerto 9090
- **Grafana:** Puerto 3000
- **Jaeger:** Puerto 16686

---

## 📂 Estructura del Proyecto

```
capibara6/
├── vm-bounty2/          # 🤖 Backend de modelos
│   ├── servers/         # Backend, Auth, Consensus
│   ├── config/          # Configuraciones de modelos
│   ├── core/            # Router semántico, E2B
│   └── deployment/      # Docker, K8s
│
├── vm-services/         # 🔧 Servicios especializados
│   ├── tts/             # Kyutai TTS
│   ├── mcp/             # Model Context Protocol
│   └── n8n/             # Workflow automation
│
├── vm-rag3/             # 🗄️ Sistema RAG
│   ├── api/             # Bridge API (capibara6-api)
│   ├── databases/       # Milvus, Nebula, PostgreSQL
│   └── monitoring/      # Prometheus, Grafana, Jaeger
│
├── frontend/            # 🌐 Aplicación Web
│   ├── public/          # HTML files
│   ├── src/             # JavaScript (clients, components)
│   └── styles/          # CSS
│
├── backend/             # 🔙 Backend consolidado
│   ├── core/            # Lógica principal
│   ├── execution/       # E2B integration
│   └── integration/     # Integraciones
│
├── archived/            # 📦 Módulos archivados (~50K líneas)
│   └── backend_modules/ # ACE, Agents, RAG, Vector Stores
│
├── docs/                # 📚 Documentación (dispersa)
├── scripts/             # 🔨 Scripts globales
└── web/                 # 🌐 Frontend alternativo
```

---

## 🔄 Cambios Recientes Importantes

### **1. Migración de Ollama a vLLM** ⚠️ CRÍTICO

**Estado:** Implementado pero requiere validación

**Cambios técnicos:**
- ✅ Endpoints actualizados: `/api/generate` → `/v1/chat/completions`
- ✅ Formato de mensajes: `prompt` → `messages` con `{role, content}`
- ✅ Autenticación: `Bearer EMPTY` para vLLM
- ✅ Streaming adaptado a eventos SSE de vLLM

**Modelos actualizados:**
- `phi3:mini` (3.8B) → `phi4:mini` (14B) ✅
- `mistral` → `qwen2.5-coder:1.5b` (experto en código) ✅

**Archivos afectados:**
- `model_config.json` - Configuración de modelos
- `frontend/src/config.js` - Endpoints frontend
- `backend/ollama_client.py` → Necesita renombrarse a `vllm_client.py`
- `vm-bounty2/config/models_config.py` - Configuración backend

### **2. Actualización de Modelos**

| Modelo Anterior | Modelo Actual | Parámetros | Puerto | Uso |
|----------------|---------------|------------|--------|-----|
| phi3:mini | **phi4:mini** | 14B | 8001 | Tareas rápidas |
| mistral | **qwen2.5-coder:1.5b** | 1.5B | 8002 | Código/técnico |
| gpt-oss-20b | **gpt-oss-20b** | 20.9B | 8000 | Tareas complejas |
| - | **mixtral** | 7B | 8003 | Tareas creativas |

### **3. Sistema de Consenso**

**Configuración actual:**
- **Método:** Votación ponderada
- **Pesos:** phi4: 0.7, qwen2.5-coder: 0.8, gpt-oss-20b: 0.9, mixtral: 0.6
- **Rango:** 2-3 modelos para consenso
- **Fallback:** phi4 como modelo de respaldo

---

## 📊 Estado de Componentes Principales

### **Frontend** 🟢 Operativo

**Ubicación:** `frontend/` y `web/`

**Archivos principales:**
- `web/index.html` (829 líneas) - Landing page ✅
- `web/chat.html` (227 líneas) - Chat principal ✅ (conflicto resuelto)
- `web/chat-app.js` (65KB) - Lógica del chat ✅
- `web/translations.js` (38KB) - Sistema multiidioma ✅
- `frontend/src/config.js` (308 líneas) - Configuración completa ✅

**Características implementadas:**
- ✅ Chat en tiempo real con streaming
- ✅ Sistema multiidioma (ES/EN)
- ✅ Renderizado Markdown + syntax highlighting
- ✅ Sistema de rating para respuestas
- ✅ Historial de conversaciones
- ✅ TTS con Kyutai
- ✅ Perfiles y plantillas de agentes
- ⚠️ MCP deshabilitado por defecto (config.js línea 54)
- ⚠️ Consensus deshabilitado por defecto (config.js línea 105)

**Pendiente:**
- ❌ Visualización de modelos activos (Fase 2 del TODO.md)
- ❌ Panel de E2B sandboxes (Fase 3 del TODO.md)
- ❌ Sistema de gemelo digital (Fases 4-6 del TODO.md)

### **Backend** 🟡 Requiere Validación

**Ubicación:** `backend/` y `vm-bounty2/`

**Archivos principales:**
- `backend/api_server.py` (19,476 bytes) - API principal
- `backend/capibara6_integrated_server.py` (26,117 bytes) - Servidor integrado
- `backend/mcp_connector.py` (40,875 bytes) - Conector MCP
- `backend/semantic_model_router.py` (13,591 bytes) - Router semántico
- `backend/ollama_client.py` (6,882 bytes) - ⚠️ Necesita actualización a vLLM

**Endpoints activos:**
- `POST /api/v1/query` - Generación con clasificación automática
- `POST /api/v1/chat/stream` - Streaming de respuestas
- `POST /api/v1/conversations/save` - Guardar conversación
- `GET /health` - Health check
- `POST /api/v1/e2b/execute` - Ejecución de código E2B
- `GET /api/v1/mcp/status` - Estado MCP
- `POST /api/tts/speak` - Text-to-Speech

**Problemas identificados:**
- ⚠️ `ollama_client.py` debe renombrarse/actualizarse a `vllm_client.py`
- ⚠️ Validar compatibilidad de todos los endpoints con vLLM
- ⚠️ Verificar que el streaming funciona correctamente

### **Sistema RAG** 🟢 Operativo

**Ubicación:** `vm-rag3/`

**Componentes:**
1. **Milvus v2.3.10** - Vector database
   - Colección: `capibara6_vectors`
   - Dimensión: 384 (all-MiniLM-L6-v2)
   - Índice: IVF_FLAT
   - Top-K: 10

2. **Nebula Graph v3.1.0** - Knowledge graph
   - Space: `capibara6_graph`
   - Cluster: 3 metad + 3 storaged + 3 graphd

3. **Bridge API (capibara6-api)** - Gateway principal
   - Puerto: 8000
   - Workers: 3 RQ workers
   - Features: Vector search, Graph queries, Async processing

**Integración:**
- ✅ TOON (Token-Oriented Object Notation) - Ahorro 30-60% tokens
- ✅ MiniRAG y FullRAG para diferentes profundidades
- ✅ Detección automática de necesidad de contexto RAG

### **Servicios Especializados** 🟡 Parcialmente Activos

**TTS Kyutai** 🟢 Activo
- Puerto: 5002
- Endpoints: `/tts`, `/voices`, `/clone`, `/health`
- Mejora: 15% menos consumo vs Coqui
- Idiomas: 8+ soportados

**MCP (Model Context Protocol)** 🔴 Deshabilitado
- Puerto: 5003
- Estado: Deshabilitado en config.js
- Razón: Requiere configuración adicional

**Smart MCP** 🔴 Deshabilitado
- Puerto: 5010
- Estado: Alternativa simplificada no activa

**N8N Workflows** 🔴 Requiere VPN
- Puerto: 5678
- Estado: Requiere VPN/túnel para acceso

**E2B Execution** 🟢 Integrado
- Integrado en backend principal (puerto 5001)
- Templates: default, data_analysis, visualization, machine_learning
- Límites: CPU, memoria, timeout configurables

### **Monitorización** 🟢 Operativo

**Stack completo en VM RAG3:**
- **Grafana** (puerto 3000) - 18 dashboards
- **Prometheus** (puerto 9090) - 30+ alertas
- **Jaeger** (puerto 16686) - Distributed tracing

---

## 📝 Documentación del Proyecto

### **Archivos de Documentación Principales**

| Archivo | Líneas | Estado | Descripción |
|---------|--------|--------|-------------|
| `README.md` | 389 | ✅ Actualizado | Documentación principal |
| `PROJECT_STATUS.md` | 334 | ⚠️ Desactualizado | Estado del proyecto (última actualización: 2025-11-09) |
| `TODO.md` | 685 | ✅ Actualizado | Roadmap de gemelo digital (18% completado) |
| `SYSTEM_ARCHITECTURE.md` | 102 | ✅ Actualizado | Arquitectura del sistema |
| `CORE_OPERATIONS.md` | 159 | ✅ Actualizado | Comandos y configuraciones |
| `MODELS_REFERENCE.md` | - | ✅ Presente | Referencia de modelos |

### **Documentación Dispersa** ⚠️

El proyecto tiene **~120 archivos .md** en el directorio raíz, lo que dificulta la navegación:

**Categorías identificadas:**
- **Configuración de VMs:** 15+ archivos
- **Solución de problemas:** 20+ archivos
- **Integraciones:** 10+ archivos (E2B, TTS, RAG, MCP, N8N)
- **Instrucciones de servicios:** 15+ archivos
- **Reportes de verificación:** 10+ archivos

**Recomendación:** Consolidar documentación en `docs/` por categorías.

---

## 🔍 Análisis de Código

### **Métricas del Proyecto**

```
Total de líneas de código:
├── Backend activo:     ~8,000 líneas
├── Frontend activo:    ~16,000 líneas
├── Módulos archivados: ~50,000 líneas
├── Fine-tuning:        ~5,000 líneas
├── K8s manifiestos:    ~1,000 líneas
└── Documentación:      ~2,000 líneas
────────────────────────────────────
TOTAL:                  ~82,000 líneas
```

### **Distribución por Lenguaje**

| Lenguaje | Líneas | Porcentaje |
|----------|--------|------------|
| Python | ~55,000 | 65% |
| JavaScript | ~18,000 | 22% |
| HTML/CSS | ~8,000 | 10% |
| YAML/JSON | ~2,500 | 3% |

### **Dependencias Backend** (`backend/requirements.txt`)

```
Flask==3.0.0
flask-cors==4.0.0
python-dotenv==1.0.0
gunicorn==21.2.0
requests==2.31.0
asyncio==3.4.32.0
openai>=1.0.0
httpx
```

**Observación:** ⚠️ `asyncio==3.4.32.0` es una versión extraña (asyncio es parte de la stdlib de Python 3.4+)

---

## 🚨 Problemas Identificados

### **Críticos** 🔴

1. **Validación de migración vLLM pendiente**
   - Los archivos aún referencian `ollama_client.py`
   - Necesita pruebas end-to-end con vLLM
   - Verificar compatibilidad de streaming

2. **Documentación dispersa**
   - 120+ archivos .md en raíz
   - Dificulta navegación y mantenimiento
   - Información duplicada/contradictoria

3. **Servicios deshabilitados sin justificación clara**
   - MCP deshabilitado (config.js línea 54)
   - Consensus deshabilitado (config.js línea 105)
   - Smart MCP no activo

### **Importantes** 🟡

4. **Dependencia asyncio incorrecta**
   - `asyncio==3.4.32.0` en requirements.txt
   - Debería eliminarse (es stdlib)

5. **Múltiples backends**
   - `backend/` y `vm-bounty2/` tienen código duplicado
   - Necesita consolidación

6. **Archivos de configuración duplicados**
   - `model_config.json` en múltiples ubicaciones
   - `.env` files dispersos

7. **Código archivado sin estrategia de reactivación**
   - 50K líneas en `archived/`
   - Módulos valiosos (E2B, ACE, Agents) sin plan de uso

### **Menores** 🟢

8. **Mix de español/inglés**
   - Código y comentarios mezclados
   - Dificulta lectura

9. **Logs y archivos temporales**
   - Múltiples archivos `.backup`
   - Scripts de verificación redundantes

10. **Falta de tests**
    - No se encontró directorio `tests/`
    - Sin cobertura de tests unitarios/integración

---

## ✅ Fortalezas del Proyecto

1. **Arquitectura distribuida bien diseñada**
   - Separación clara de responsabilidades por VM
   - Escalabilidad horizontal

2. **Sistema RAG completo y robusto**
   - Milvus + Nebula Graph + PostgreSQL
   - TOON para optimización de tokens
   - Bridge API bien estructurado

3. **Múltiples modelos con consenso**
   - 4 modelos especializados
   - Sistema de votación ponderada
   - Fallback automático

4. **Monitorización completa**
   - Prometheus + Grafana + Jaeger
   - 18 dashboards + 30+ alertas

5. **Frontend moderno y funcional**
   - Multiidioma (ES/EN)
   - Streaming en tiempo real
   - TTS integrado

6. **Integraciones avanzadas**
   - E2B para ejecución de código
   - OAuth (GitHub/Google)
   - N8N para workflows

---

## 📋 Recomendaciones Prioritarias

### **Prioridad Alta** 🔴

1. **Validar migración a vLLM**
   ```bash
   # Probar endpoints vLLM
   curl -X POST "http://34.12.166.76:8000/v1/chat/completions" \
     -H "Authorization: Bearer EMPTY" \
     -d '{"model": "gpt-oss-20b", "messages": [{"role": "user", "content": "Test"}]}'
   
   # Verificar todos los modelos
   curl http://34.12.166.76:8001/v1/models  # phi4
   curl http://34.12.166.76:8002/v1/models  # qwen2.5-coder
   curl http://34.12.166.76:8003/v1/models  # mixtral
   ```

2. **Renombrar/actualizar archivos Ollama → vLLM**
   - `backend/ollama_client.py` → `vllm_client.py`
   - `backend/ollama_rag_integration.py` → `vllm_rag_integration.py`
   - Actualizar todas las referencias

3. **Consolidar documentación**
   ```
   docs/
   ├── architecture/
   ├── deployment/
   ├── troubleshooting/
   ├── integrations/
   └── vm-guides/
   ```

### **Prioridad Media** 🟡

4. **Activar servicios deshabilitados**
   - Evaluar por qué MCP está deshabilitado
   - Documentar razones o reactivar

5. **Consolidar backends**
   - Unificar `backend/` y `vm-bounty2/`
   - Eliminar código duplicado

6. **Crear suite de tests**
   ```
   tests/
   ├── unit/
   ├── integration/
   └── e2e/
   ```

7. **Limpiar archivos temporales**
   - Eliminar `.backup` files
   - Consolidar scripts de verificación

### **Prioridad Baja** 🟢

8. **Estandarizar idioma**
   - Decidir: todo en inglés o todo en español
   - Aplicar consistentemente

9. **Implementar roadmap TODO.md**
   - Fase 2: Visualización de modelos (37.5% completado)
   - Fase 3: Panel E2B
   - Fases 4-6: Gemelo digital

10. **Evaluar módulos archivados**
    - Decidir qué reactivar
    - Eliminar lo obsoleto

---

## 🎯 Estado del Roadmap (TODO.md)

**Progreso general:** 18% completado (7/40 tareas)

### **Fase 1: Preparación** ✅ Completado (4/4)
- ✅ Análisis del proyecto
- ✅ Documentación de TODOs
- ✅ Resolución de conflictos en chat.html
- ✅ Limpieza de plantillas

### **Fase 2: Visualización de Modelos** 🔄 En progreso (3/8 - 37.5%)
- ✅ Diseño de componente de modelo activo
- ✅ Indicador de modelo por mensaje
- ✅ Panel de métricas de modelo
- ⏳ Selector manual de modelo
- ⏳ Visualización de clasificación de tarea
- ⏳ Indicador de consenso multi-modelo
- ⏳ Gráfico de uso de modelos
- ⏳ Integración con backend

### **Fases 3-8** ⏳ Pendientes
- Fase 3: Panel E2B (0/10)
- Fase 4: Importador de redes sociales (0/9)
- Fase 5: Sistema de gemelo digital (0/12)
- Fase 6: Panel avanzado (0/11)
- Fase 7: Testing (0/8)
- Fase 8: Documentación (0/6)

**Tiempo estimado total:** ~46 horas

---

## 📞 Información de Contacto

**Organización:** Anachroni s.coop  
**País:** España  
**Website:** https://www.anachroni.co  
**Email:** marco@anachroni.co  
**Producción:** https://www.capibara6.com  

---

## 🔗 Enlaces Útiles

### **Servicios en Producción**
- Backend: http://34.12.166.76:5001
- TTS: http://34.175.136.104:5002
- MCP: http://34.175.136.104:5003
- Grafana: http://10.154.0.2:3000
- Prometheus: http://10.154.0.2:9090
- Jaeger: http://10.154.0.2:16686

### **vLLM Endpoints**
- GPT-OSS-20B: http://34.12.166.76:8000/v1
- Phi4-mini: http://34.12.166.76:8001/v1
- Qwen2.5-coder: http://34.12.166.76:8002/v1
- Mixtral: http://34.12.166.76:8003/v1

---

## 📊 Conclusión

**Estado general:** 🟢 **Sistema operativo y funcional**

**Puntos fuertes:**
- Arquitectura distribuida robusta
- Sistema RAG completo
- Múltiples modelos con consenso
- Monitorización avanzada

**Áreas de mejora:**
- Validar migración vLLM
- Consolidar documentación
- Activar servicios deshabilitados
- Implementar tests

**Próximos pasos recomendados:**
1. Validar endpoints vLLM (1 hora)
2. Actualizar archivos Ollama → vLLM (2 horas)
3. Consolidar documentación (4 horas)
4. Crear suite de tests básica (8 horas)
5. Continuar con Fase 2 del roadmap (3 horas restantes)

---

**Análisis realizado por:** Antigravity AI Assistant  
**Fecha:** 2025-11-24  
**Versión del documento:** 1.0

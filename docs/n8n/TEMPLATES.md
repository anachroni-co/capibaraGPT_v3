# 📚 Guía de Plantillas n8n para Capibara6

## 📋 Índice
- [Introducción](#introducción)
- [Catálogo de Plantillas](#catálogo-de-plantillas)
- [API de Plantillas](#api-de-plantillas)
- [Uso de Plantillas](#uso-de-plantillas)
- [Plantillas Personalizadas Capibara6](#plantillas-personalizadas-capibara6)
- [Configuración](#configuración)

---

## 🎯 Introducción

Capibara6 incluye un sistema de gestión de plantillas (templates) de n8n que facilita la creación de workflows de automatización pre-configurados. Las plantillas están organizadas por categorías y casos de uso, con integración directa a los endpoints de Capibara6.

### Fuentes de Plantillas

1. **Repositorio Awesome n8n Templates**: https://github.com/enescingoz/awesome-n8n-templates
   - 100+ plantillas curadas por la comunidad
   - Categorías: AI/ML, RAG, Database, Email, PDF, etc.

2. **Plantillas Personalizadas Capibara6**
   - Workflows optimizados para Capibara6
   - Integración nativa con endpoints
   - Listas para usar inmediatamente

### Estadísticas del Catálogo

```
Total de plantillas: 24
Plantillas personalizadas: 4
Plantillas externas: 20
Plantillas recomendadas: 11
Tiempo promedio de setup: 17 minutos
```

---

## 📚 Catálogo de Plantillas

### 🤖 AI Chat & Conversación (3 plantillas)

#### 1. AI Agent Chat ⭐ RECOMENDADA
- **ID**: `ai-agent-chat`
- **Dificultad**: Media
- **Tiempo de setup**: 15 min
- **Casos de uso**: Atención al cliente, Asistente virtual, FAQ automático
- **Requiere**: OPENAI_API_KEY
- **Endpoints Capibara6**: `/api/chat`, `/api/save-conversation`

#### 2. Telegram AI Bot con LangChain ⭐ RECOMENDADA
- **ID**: `telegram-ai-bot`
- **Dificultad**: Media
- **Tiempo de setup**: 20 min
- **Casos de uso**: Soporte via Telegram, Bot conversacional
- **Requiere**: TELEGRAM_BOT_TOKEN, OPENAI_API_KEY

#### 3. Análisis de Sentimiento de Clientes ⭐ RECOMENDADA
- **ID**: `customer-sentiment-analysis`
- **Dificultad**: Fácil
- **Tiempo de setup**: 10 min
- **Casos de uso**: Feedback de clientes, Análisis de reseñas
- **Endpoints Capibara6**: `/api/save-lead`

---

### 🗄️ RAG & Base de Conocimiento (3 plantillas)

#### 1. RAG Chatbot con Pinecone ⭐ RECOMENDADA
- **ID**: `rag-chatbot-pinecone`
- **Dificultad**: Avanzada
- **Tiempo de setup**: 30 min
- **Casos de uso**: Base de conocimiento, Documentación interactiva
- **Requiere**: OPENAI_API_KEY, PINECONE_API_KEY
- **Endpoints Capibara6**: `/api/mcp/tools/call`

#### 2. Chat con PDFs usando AI ⭐ RECOMENDADA
- **ID**: `pdf-chat-ai`
- **Dificultad**: Media
- **Tiempo de setup**: 20 min
- **Casos de uso**: Análisis de contratos, Revisión de documentos
- **Requiere**: OPENAI_API_KEY

#### 3. RAG con Google Drive
- **ID**: `google-drive-rag`
- **Dificultad**: Media
- **Tiempo de setup**: 25 min
- **Casos de uso**: Knowledge base corporativa

---

### 🗃️ Automatización de Bases de Datos (3 plantillas)

#### 1. Chat con PostgreSQL ⭐ RECOMENDADA
- **ID**: `chat-postgresql`
- **Dificultad**: Media
- **Tiempo de setup**: 15 min
- **Casos de uso**: Consultas de datos en lenguaje natural
- **Requiere**: OPENAI_API_KEY, POSTGRES_CONNECTION
- **Integración**: Base de datos capibara6

#### 2. Generación de Consultas SQL con AI ⭐ RECOMENDADA
- **ID**: `sql-generation`
- **Dificultad**: Fácil
- **Tiempo de setup**: 10 min
- **Casos de uso**: Consultas dinámicas, Reportes

#### 3. Supabase - Inserción y Recuperación
- **ID**: `supabase-integration`
- **Dificultad**: Fácil
- **Tiempo de setup**: 10 min

---

### 📄 Procesamiento de Documentos (3 plantillas)

#### 1. Extracción de PDF con Claude/Gemini ⭐ RECOMENDADA
- **ID**: `pdf-extraction-claude`
- **Dificultad**: Media
- **Tiempo de setup**: 15 min
- **Casos de uso**: Digitalización de documentos, OCR inteligente
- **Requiere**: ANTHROPIC_API_KEY, GOOGLE_API_KEY

#### 2. Extracción de Facturas con LlamaParse
- **ID**: `invoice-extraction-llamaparse`
- **Dificultad**: Media
- **Tiempo de setup**: 20 min
- **Casos de uso**: Contabilidad, Procesamiento de facturas

#### 3. Parsing de CVs con Vision AI
- **ID**: `cv-parsing-vision`
- **Dificultad**: Media
- **Tiempo de setup**: 15 min
- **Casos de uso**: Reclutamiento, HR

---

### ⚡ Automatización con Webhooks (3 plantillas)

#### 1. Procesamiento Automático de Leads ⭐ RECOMENDADA - CUSTOM
- **ID**: `lead-processing`
- **Dificultad**: Fácil
- **Tiempo de setup**: 10 min
- **Casos de uso**: Marketing automation, Gestión de leads
- **Endpoints Capibara6**: `/api/save-lead`, `/webhook/lead`
- **Personalizado para Capibara6** ✨

#### 2. Logger de Conversaciones ⭐ RECOMENDADA - CUSTOM
- **ID**: `conversation-logger`
- **Dificultad**: Fácil
- **Tiempo de setup**: 10 min
- **Casos de uso**: Analytics, Auditoría, Mejora del modelo
- **Endpoints Capibara6**: `/api/save-conversation`, `/webhook/conversation`
- **Personalizado para Capibara6** ✨

#### 3. Consenso Multi-Modelo ⭐ RECOMENDADA - CUSTOM
- **ID**: `model-consensus`
- **Dificultad**: Avanzada
- **Tiempo de setup**: 25 min
- **Casos de uso**: Validación de respuestas, Alta precisión
- **Endpoints Capibara6**: `/api/chat`, `/webhook/consensus`
- **Personalizado para Capibara6** ✨
- **Funcionalidad**: Consulta a phi3, mistral y gpt-oss simultáneamente

---

### 🖥️ Monitoreo y DevOps (2 plantillas)

#### 1. Monitor de Salud del Sistema ⭐ RECOMENDADA - CUSTOM
- **ID**: `system-health-monitor`
- **Dificultad**: Fácil
- **Tiempo de setup**: 15 min
- **Casos de uso**: Monitoreo, Alertas, SRE
- **Endpoints Capibara6**: `/api/health`, `/api/mcp/status`
- **Personalizado para Capibara6** ✨

#### 2. Docker Compose Controller via Webhook
- **ID**: `docker-controller`
- **Dificultad**: Media
- **Tiempo de setup**: 20 min
- **Casos de uso**: DevOps, CI/CD

---

### 📧 Automatización de Email (2 plantillas)

#### 1. Auto-etiquetado de Emails con AI
- **ID**: `email-auto-label`
- **Dificultad**: Media
- **Tiempo de setup**: 20 min
- **Casos de uso**: Organización de inbox

#### 2. Email con RAG: Resumir y Responder
- **ID**: `email-rag-response`
- **Dificultad**: Avanzada
- **Tiempo de setup**: 30 min
- **Casos de uso**: Soporte automático

---

## 🔌 API de Plantillas

### Endpoints Disponibles

#### 1. Obtener Catálogo Completo
```bash
GET /api/n8n/templates
```

**Respuesta**:
```json
{
  "status": "success",
  "catalog": {
    "version": "1.0.0",
    "categories": [...],
    "statistics": {...}
  },
  "timestamp": "2025-11-10T06:30:00.000Z"
}
```

#### 2. Obtener Plantillas Recomendadas
```bash
GET /api/n8n/templates/recommended
```

**Respuesta**:
```json
{
  "status": "success",
  "count": 11,
  "templates": [
    {
      "id": "ai-agent-chat",
      "name": "AI Agent Chat",
      "description": "...",
      "priority": 1,
      "recommended": true
    }
  ]
}
```

#### 3. Obtener Detalles de Plantilla
```bash
GET /api/n8n/templates/{template_id}
```

**Ejemplo**:
```bash
curl http://localhost:5000/api/n8n/templates/ai-agent-chat
```

#### 4. Buscar Plantillas
```bash
GET /api/n8n/templates/search?q=chat
POST /api/n8n/templates/search
```

**Body (POST)**:
```json
{
  "query": "chat"
}
```

#### 5. Descargar JSON de Plantilla
```bash
GET /api/n8n/templates/{template_id}/download
```

**Respuesta**:
```json
{
  "status": "success",
  "template_id": "ai-agent-chat",
  "workflow": {
    "name": "AI Agent Chat",
    "nodes": [...],
    "connections": {...}
  }
}
```

#### 6. Importar Plantilla a n8n
```bash
POST /api/n8n/templates/{template_id}/import
```

**Body**:
```json
{
  "n8n_url": "http://n8n:5678",
  "api_key": "optional_api_key"
}
```

**Respuesta**:
```json
{
  "success": true,
  "workflow_id": "123",
  "message": "Workflow 'AI Agent Chat' importado exitosamente"
}
```

---

## 🚀 Uso de Plantillas

### Método 1: Via API (Recomendado)

```bash
# 1. Ver plantillas disponibles
curl http://localhost:5000/api/n8n/templates/recommended

# 2. Descargar una plantilla
curl http://localhost:5000/api/n8n/templates/ai-agent-chat/download > workflow.json

# 3. Importarla a n8n
curl -X POST http://localhost:5000/api/n8n/templates/ai-agent-chat/import \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Método 2: Import Manual en n8n

1. **Acceder a n8n**: http://localhost:5678 o https://n8n.capibara6.com
2. **Crear nuevo workflow**: Click en "+" o "New"
3. **Importar JSON**:
   - Click en menú (⋮)
   - Select "Import from File" o "Import from URL"
   - Pegar URL o seleccionar archivo

**URLs de Plantillas Externas**:
```
https://raw.githubusercontent.com/enescingoz/awesome-n8n-templates/main/[categoria]/[plantilla].json
```

**Archivos Locales**:
```
/home/elect/capibara6/backend/data/n8n/workflows/templates/capibara6-*.json
```

### Método 3: Via CLI de n8n

```bash
# Dentro del contenedor n8n
docker exec -it capibara6-n8n sh
n8n import:workflow --input=/data/n8n/templates/capibara6-lead-processing.json
```

---

## ⚙️ Configuración de Plantillas

### Requisitos Comunes

1. **API Keys**:
```bash
# En .env
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
GOOGLE_API_KEY=AIzaXXX
TELEGRAM_BOT_TOKEN=xxx
PINECONE_API_KEY=xxx
```

2. **Credenciales en n8n**:
- Ir a n8n → Settings → Credentials
- Añadir credenciales para los servicios que uses

3. **Webhooks**:
- Production URL: `https://n8n.capibara6.com/webhook/`
- Test URL: `http://localhost:5678/webhook-test/`

### Configurar Plantilla después de Importar

1. **Actualizar Credentials**:
   - Abrir cada nodo que requiera credenciales
   - Seleccionar o crear nuevas credenciales

2. **Actualizar URLs**:
   - Cambiar `http://capibara6-api:8000` si es necesario
   - Ajustar URLs de webhooks

3. **Activar Workflow**:
   - Click en botón "Active" en la esquina superior derecha

4. **Probar Workflow**:
   - Click en "Execute Workflow"
   - O enviar request al webhook

---

## 📦 Plantillas Personalizadas Capibara6

Las siguientes plantillas están optimizadas específicamente para Capibara6 y ya incluyen integración con los endpoints del backend.

### 1. Procesamiento Automático de Leads

**Archivo**: `capibara6-lead-processing.json`

**Flujo**:
```
Webhook → Validar Email → PostgreSQL → Backend API → Email → Respuesta
```

**Configuración**:
```json
{
  "webhook_url": "/webhook/lead",
  "endpoints": [
    "POST /api/save-lead",
    "PostgreSQL leads table"
  ]
}
```

**Uso**:
```bash
curl -X POST http://localhost:5678/webhook/lead \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Juan Pérez",
    "email": "juan@example.com",
    "company": "Empresa XYZ",
    "message": "Interesado en Capibara6",
    "source": "web"
  }'
```

### 2. Consenso Multi-Modelo

**Archivo**: `capibara6-model-consensus.json`

**Flujo**:
```
Webhook → Query Phi3 → Query Mistral → Query GPT-OSS → Merge → Consensus Logic → Respuesta
```

**Lógica de Consenso**:
- Consulta 3 modelos simultáneamente
- Compara similitud de respuestas
- Calcula confianza agregada
- Retorna mejor respuesta + métricas

**Uso**:
```bash
curl -X POST http://localhost:5678/webhook/consensus \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Cuál es la capital de Francia?"
  }'
```

**Respuesta**:
```json
{
  "consensus": true,
  "response": "París es la capital de Francia",
  "models_consulted": 3,
  "metrics": {
    "total_tokens": 150,
    "average_time_ms": 1200,
    "average_confidence": 0.95,
    "consensus_score": "high"
  }
}
```

---

## 🔧 Troubleshooting

### Plantilla no se importa

```bash
# Verificar formato JSON
cat workflow.json | jq .

# Verificar conexión a n8n
curl http://localhost:5678/api/v1/workflows
```

### Credenciales faltantes

1. Revisar nodos en rojo en n8n
2. Click en nodo → Select credentials
3. Añadir credenciales requeridas

### Webhook no funciona

```bash
# Verificar workflow activo
# En n8n UI: debe estar en "Active"

# Probar webhook
curl http://localhost:5678/webhook-test/lead
```

### Error de conexión a Backend

```bash
# Verificar que backend esté corriendo
curl http://localhost:5000/api/health

# Actualizar URL en nodos HTTP Request
# Docker: http://capibara6-api:8000
# Local: http://localhost:5000
```

---

## 📚 Recursos Adicionales

- **Repositorio de Plantillas**: https://github.com/enescingoz/awesome-n8n-templates
- **Documentación n8n**: https://docs.n8n.io
- **Documentación Capibara6 n8n**: `/docs/n8n/DEPLOYMENT.md`
- **API Reference**: `/docs/n8n/README.md`

---

## ✅ Checklist de Implementación

- [ ] n8n desplegado y accesible
- [ ] Credenciales configuradas en n8n
- [ ] API keys añadidas al .env
- [ ] Backend Capibara6 funcionando
- [ ] Plantillas recomendadas importadas
- [ ] Webhooks probados
- [ ] Workflows activados
- [ ] Monitoreo configurado

---

**Última actualización**: 2025-11-10
**Versión del catálogo**: 1.0.0
**Total de plantillas**: 24
**Mantenedor**: Anachroni s.coop

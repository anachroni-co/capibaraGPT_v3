# Plan de Consolidación del Backend - Capibara6

## Fecha: 2025-11-13

## 📊 Análisis de Situación Actual

### Problema Principal
El repositorio tiene **múltiples archivos backend duplicados** con configuraciones inconsistentes, causando confusión sobre cuál archivo usar y conflictos de puertos.

---

## 🗂️ Inventario de Archivos Backend

### 1. Servidores Principales (Puerto 5001)

| Archivo | Tamaño | Estado | Descripción |
|---------|--------|--------|-------------|
| `server_gptoss.py` | 18K | ✅ **USAR ESTE** | Servidor principal Flask para Ollama |
| `server_gptoss_CURRENT_WORKING.py` | 18K | 🗑️ **ELIMINAR** | Backup idéntico (0 diff) |
| `server.py` | 12K | 🗑️ **ELIMINAR** | Versión antigua/incompleta |
| `capibara6_integrated_server.py` | 16K | ⚠️ **EVALUAR** | Servidor integrado (chat+TTS+MCP) |
| `capibara6_integrated_server_ollama.py` | 18K | 🗑️ **ELIMINAR** | Duplicado del anterior |

**Conflicto:** 3 servidores intentan usar el puerto 5001.

**Decisión:** Mantener `server_gptoss.py` como principal.

---

### 2. CORS Proxy (Puerto 5000)

| Archivo | Tamaño | Estado | Problema |
|---------|--------|--------|----------|
| `cors_proxy.py` | 4.4K | ⚠️ **ACTUALIZAR** | Usa IP vieja `34.175.215.109` |

**Problemas detectados:**
```python
# ❌ IP INCORRECTA (línea 18-19)
GPTOSS_API_URL = 'http://34.175.215.109/api/chat'
GPTOSS_HEALTH_URL = 'http://34.175.215.109/health'

# ✅ IP CORRECTA (según .env.example)
# Debería ser: http://34.12.166.76:11434
```

**Decisión:** Actualizar para usar variables de entorno, no IPs hardcodeadas.

---

### 3. MCP Servers (Puertos 5003/5010)

| Archivo | Tamaño | Puerto | Estado |
|---------|--------|--------|--------|
| `mcp_server.py` | 12K | 5003 | ✅ **PRINCIPAL** |
| `smart_mcp_server.py` | 6.9K | 5010 | 🔄 **FUSIONAR** |
| `smart_mcp_completo.py` | 5.3K | 5010 | 🗑️ **ELIMINAR** |
| `smart_mcp_simple.py` | 1.9K | 5010 | 🗑️ **ELIMINAR** |
| `mcp_server_robusto.py` | 3.2K | N/A | 🗑️ **ELIMINAR** |

**Conflicto:** 3 servidores smart_mcp compiten por puerto 5010.

**Decisión:** Consolidar en `mcp_server.py` (puerto 5003), eliminar smart_mcp_*.

---

### 4. TTS Servers (Puerto 5002)

| Archivo | Tamaño | Estado |
|---------|--------|--------|
| `kyutai_tts_server.py` | 7.7K | ✅ **PRINCIPAL** |
| `kyutai_tts_server_simple.py` | 2.2K | 🗑️ **ELIMINAR** |
| `coqui_tts_server.py` | 12K | ⚠️ **ALTERNATIVO** |
| `coqui_tts_server_cloning.py` | 12K | 🗑️ **ELIMINAR** |

**Decisión:** Mantener `kyutai_tts_server.py` como principal.

---

### 5. Servidores Especializados

| Archivo | Puerto | Estado | Descripción |
|---------|--------|--------|-------------|
| `auth_server.py` | 5001 | ⚠️ **CONFLICTO** | OAuth GitHub/Google |
| `consensus_server.py` | 5002 | ⚠️ **CONFLICTO** | Consensus multi-modelo |
| `main.py` (FastAPI) | 8000 | ⚠️ **NO USADO** | API alternativa |

**Conflictos:**
- `auth_server.py` (5001) vs `server_gptoss.py` (5001)
- `consensus_server.py` (5002) vs TTS servers (5002)

---

## 🎯 Mapeo de Puertos Definitivo

### Arquitectura Propuesta

```
┌─────────────────────────────────────────────────────────┐
│                   CAPIBARA6 BACKEND                     │
└─────────────────────────────────────────────────────────┘

Puerto 5001: server_gptoss.py (Principal - Chat con Ollama)
  Endpoints:
    POST /api/chat              - Enviar mensaje
    POST /api/chat/stream       - Chat con streaming
    GET  /api/health            - Health check
    GET  /api/models            - Listar modelos
    POST /api/save-conversation - Guardar conversación

Puerto 5002: kyutai_tts_server.py (TTS - Text to Speech)
  Endpoints:
    POST /tts                   - Síntesis de voz
    GET  /health                - Health check
    POST /preload               - Precargar modelo

Puerto 5003: mcp_server.py (MCP - Context & RAG)
  Endpoints:
    GET  /api/mcp/contexts      - Listar contextos
    POST /api/mcp/augment       - Aumentar prompt (RAG)
    GET  /api/mcp/health        - Health check
    POST /api/mcp/calculate     - Calculadora
    POST /api/mcp/verify        - Verificar hechos

Puerto 5004: auth_server.py (OAuth - Authentication)
  Endpoints:
    GET  /auth/github           - Login GitHub
    GET  /auth/google           - Login Google
    POST /auth/verify           - Verificar token
    POST /auth/logout           - Cerrar sesión

Puerto 5005: consensus_server.py (Consensus - Multi-modelo)
  Endpoints:
    POST /api/consensus/query   - Query multi-modelo
    GET  /api/consensus/models  - Listar modelos
    GET  /api/consensus/health  - Health check

Puerto 8000: main.py (FastAPI - Opcional)
  Estado: No usado por frontend actual
  Acción: Mantener para futuro o eliminar

Puerto 5000: cors_proxy.py
  Estado: Deprecado (usar CORS directo)
  Acción: Eliminar
```

---

## 🔧 Correcciones Necesarias

### 1. Actualizar `cors_proxy.py` (O Eliminar)

**Opción A: Actualizar** (si se necesita proxy)
```python
# En lugar de IPs hardcodeadas:
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://34.12.166.76:11434')
GPTOSS_API_URL = f"{OLLAMA_URL}/api/chat"
```

**Opción B: Eliminar** (recomendado)
- `server_gptoss.py` ya tiene CORS habilitado
- El proxy es innecesario

---

### 2. Cambiar Puerto de `auth_server.py`

```python
# Cambiar línea 299:
# app.run(host='0.0.0.0', port=5001, debug=True)
app.run(host='0.0.0.0', port=5004, debug=True)  # Nuevo puerto
```

---

### 3. Cambiar Puerto de `consensus_server.py`

```python
# Cambiar línea 309:
# app.run(host='0.0.0.0', port=5002, debug=True)
app.run(host='0.0.0.0', port=5005, debug=True)  # Nuevo puerto
```

---

### 4. Actualizar Frontend `config.js`

```javascript
const CHATBOT_CONFIG = {
    BACKEND_URL: 'http://localhost:5001',  // server_gptoss.py

    ENDPOINTS: {
        CHAT: '/api/chat',
        CHAT_STREAM: '/api/chat/stream',
        SAVE_CONVERSATION: '/api/save-conversation',
        HEALTH: '/api/health',
        MODELS: '/api/models'
    },

    // Servicios adicionales
    SERVICES: {
        TTS: {
            enabled: true,
            url: 'http://localhost:5002',
            endpoints: {
                SPEAK: '/tts',
                HEALTH: '/health'
            }
        },
        MCP: {
            enabled: false,  // Habilitar manualmente
            url: 'http://localhost:5003',
            endpoints: {
                AUGMENT: '/api/mcp/augment',
                HEALTH: '/api/mcp/health'
            }
        },
        AUTH: {
            enabled: true,
            url: 'http://localhost:5004',
            endpoints: {
                GITHUB: '/auth/github',
                GOOGLE: '/auth/google'
            }
        },
        CONSENSUS: {
            enabled: false,  // Habilitar manualmente
            url: 'http://localhost:5005',
            endpoints: {
                QUERY: '/api/consensus/query',
                HEALTH: '/api/consensus/health'
            }
        }
    }
};
```

---

## 📋 Plan de Acción

### Fase 1: Limpieza (Sin Riesgo)

1. **Eliminar duplicados exactos:**
   ```bash
   rm backend/server_gptoss_CURRENT_WORKING.py
   rm backend/capibara6_integrated_server_ollama.py
   rm backend/smart_mcp_completo.py
   rm backend/smart_mcp_simple.py
   rm backend/mcp_server_robusto.py
   rm backend/kyutai_tts_server_simple.py
   rm backend/coqui_tts_server_cloning.py
   ```

2. **Marcar como deprecated:**
   - Crear `backend/deprecated/` folder
   - Mover: `server.py`, `cors_proxy.py`

### Fase 2: Corrección de Puertos

1. **Cambiar puerto de auth_server.py:**
   ```python
   # Línea 299
   app.run(host='0.0.0.0', port=5004, debug=True)
   ```

2. **Cambiar puerto de consensus_server.py:**
   ```python
   # Línea 309
   app.run(host='0.0.0.0', port=5005, debug=True)
   ```

3. **Actualizar documentación de puertos** en:
   - `QUICK_START.md`
   - `ARCHITECTURE_QUICK_REF.md`
   - `README.md`

### Fase 3: Actualizar Frontend

1. **Actualizar `web/config.js`:**
   - Agregar configuración de servicios detallada
   - Documentar cada puerto y endpoint

2. **Actualizar `web/smart-mcp-integration.js`:**
   - Confirmar puerto 5003
   - Endpoint `/api/mcp/augment`

3. **Actualizar `web/consensus-ui.js`:**
   - Cambiar puerto a 5005
   - Actualizar URLs de health checks

### Fase 4: Scripts de Gestión de Servicios

**Estado:** ✅ Completado (2025-11-13)

**Cambios realizados:**
- Investigación completa de infraestructura
- Creación de `INFRASTRUCTURE_FINDINGS.md` con hallazgos detallados
- Implementación de scripts de gestión

**Nota:** Ver `INFRASTRUCTURE_FINDINGS.md` para el análisis completo de servicios disponibles.

Crear `backend/start-all-services.sh`:
```bash
#!/bin/bash
# Inicia todos los servicios de Capibara6

echo "🚀 Iniciando Capibara6 Backend Services..."

# Servidor principal (Puerto 5001)
echo "▶️  Iniciando server_gptoss.py (Puerto 5001)..."
python3 server_gptoss.py &
PID_MAIN=$!

# TTS Server (Puerto 5002)
echo "▶️  Iniciando kyutai_tts_server.py (Puerto 5002)..."
python3 kyutai_tts_server.py &
PID_TTS=$!

# MCP Server (Puerto 5003) - Opcional
echo "▶️  Iniciando mcp_server.py (Puerto 5003)..."
python3 mcp_server.py &
PID_MCP=$!

# Auth Server (Puerto 5004)
echo "▶️  Iniciando auth_server.py (Puerto 5004)..."
python3 auth_server.py &
PID_AUTH=$!

# Consensus Server (Puerto 5005) - Opcional
# echo "▶️  Iniciando consensus_server.py (Puerto 5005)..."
# python3 consensus_server.py &
# PID_CONSENSUS=$!

echo "✅ Todos los servicios iniciados"
echo "📋 PIDs: Main=$PID_MAIN, TTS=$PID_TTS, MCP=$PID_MCP, Auth=$PID_AUTH"

# Esperar a que terminen
wait
```

Crear `backend/stop-all-services.sh`:
```bash
#!/bin/bash
# Detiene todos los servicios

echo "🛑 Deteniendo servicios..."
pkill -f "server_gptoss.py"
pkill -f "kyutai_tts_server.py"
pkill -f "mcp_server.py"
pkill -f "auth_server.py"
pkill -f "consensus_server.py"
echo "✅ Servicios detenidos"
```

---

## 🧪 Testing

### 1. Verificar Puertos Libres
```bash
# Antes de iniciar servicios
netstat -tuln | grep -E "500[0-5]"

# No debería mostrar nada
```

### 2. Iniciar Servicios Uno por Uno
```bash
# Terminal 1
python3 backend/server_gptoss.py

# Terminal 2
python3 backend/kyutai_tts_server.py

# Terminal 3
python3 backend/mcp_server.py

# Terminal 4
python3 backend/auth_server.py
```

### 3. Health Checks
```bash
# Main server
curl http://localhost:5001/api/health

# TTS server
curl http://localhost:5002/health

# MCP server
curl http://localhost:5003/api/mcp/health

# Auth server
curl http://localhost:5004/health
```

### 4. Verificar Frontend
1. Abrir `http://localhost:8000/chat.html`
2. Abrir DevTools (F12)
3. Verificar que no haya errores 404
4. Probar enviar un mensaje

---

## 📊 Resumen de Cambios

### Archivos a Eliminar (7)
- ✅ `server_gptoss_CURRENT_WORKING.py`
- ✅ `capibara6_integrated_server_ollama.py`
- ✅ `smart_mcp_completo.py`
- ✅ `smart_mcp_simple.py`
- ✅ `mcp_server_robusto.py`
- ✅ `kyutai_tts_server_simple.py`
- ✅ `coqui_tts_server_cloning.py`

### Archivos a Mover a Deprecated (2)
- ⚠️ `server.py`
- ⚠️ `cors_proxy.py`

### Archivos a Modificar (3)
- 🔧 `auth_server.py` - Puerto 5001 → 5004
- 🔧 `consensus_server.py` - Puerto 5002 → 5005
- 🔧 `web/config.js` - Agregar mapeo completo de servicios

### Archivos Nuevos (3)
- ✨ `backend/start-all-services.sh`
- ✨ `backend/stop-all-services.sh`
- ✨ `BACKEND_CONSOLIDATION_PLAN.md` (este archivo)

---

## ⚠️ Advertencias

1. **Antes de eliminar archivos:**
   - Hacer commit del estado actual
   - Crear tag: `git tag pre-consolidation`
   - Push: `git push origin --tags`

2. **Testing exhaustivo:**
   - Probar cada endpoint después de cambios
   - Verificar que frontend funciona correctamente

3. **Documentación:**
   - Actualizar README.md con nuevos puertos
   - Actualizar ARCHITECTURE_QUICK_REF.md

---

## ✅ Checklist de Implementación

- [ ] Hacer backup (git tag)
- [ ] Eliminar archivos duplicados
- [ ] Crear carpeta `backend/deprecated/`
- [ ] Mover archivos obsoletos
- [ ] Cambiar puerto auth_server.py → 5004
- [ ] Cambiar puerto consensus_server.py → 5005
- [ ] Actualizar web/config.js
- [ ] Actualizar web/consensus-ui.js
- [ ] Crear start-all-services.sh
- [ ] Crear stop-all-services.sh
- [ ] Hacer chmod +x en scripts
- [ ] Testing completo
- [ ] Actualizar documentación
- [ ] Commit y push

---

## 📝 Siguiente Paso

¿Quieres que proceda con la implementación del plan? Puedo:

1. **Eliminar archivos duplicados** (Fase 1)
2. **Corregir puertos** (Fase 2)
3. **Actualizar frontend** (Fase 3)
4. **Crear scripts de inicio** (Fase 4)

O prefieres revisar el plan primero y hacer ajustes?

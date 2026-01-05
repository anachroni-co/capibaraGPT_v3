# Solución: Errores 404 en Puerto 8001

## 🔴 Problema Identificado

Los errores que estás viendo en la consola del navegador:

```
GET http://localhost:8001/api/mcp/status 404 (NOT FOUND)
GET http://localhost:8001/api/n8n/templates/recommended 404 (NOT FOUND)
POST http://localhost:8001/api/ai/generate 404 (NOT FOUND)
```

**Causa Confirmada**: El navegador está cargando archivos JavaScript **desde un servidor remoto en `http://localhost:8000/`** que tiene una **versión antigua** del proyecto:

- `http://localhost:8000/chat-page.js` - ❌ NO EXISTE en el repo actual
- `http://localhost:8000/n8n-manager.js` - ❌ NO EXISTE en el repo actual

Estos archivos antiguos hacen peticiones al puerto 8001 (un proxy CORS obsoleto), causando los errores 404.

**NO es un problema de caché del navegador** - Es un problema del **servidor remoto** que está sirviendo archivos obsoletos.

## ✅ Configuración de Puertos Correcta

Los puertos actualmente en uso son:

### VM bounty2 (34.12.166.76)
- **Puerto 5001**: `server_gptoss.py` - Backend principal GPT-OSS-20B
- **Puerto 5004**: Auth server (GitHub OAuth, Google OAuth)
- **Puerto 5005**: Consensus server (multi-modelo)

### VM gpt-oss-20b (34.175.136.104)
- **Puerto 5002**: TTS server (Kyutai TTS)
- **Puerto 5003**: MCP server (Context & RAG)
- **Puerto 5678**: n8n (requiere VPN/túnel)

### VM rag3 (10.154.0.2 - IP interna)
- **Puerto 8000**: capibara6-api (Bridge para RAG)
- **Puerto 19530**: Milvus (vector database)
- **Puerto 9669**: Nebula Graph (knowledge graph)
- **Puerto 5432**: PostgreSQL
- **Puerto 5433**: TimescaleDB
- **Puerto 6379**: Redis
- **Puerto 3000**: Grafana (monitoring)
- **Puerto 9090**: Prometheus (metrics)
- **Puerto 16686**: Jaeger (tracing)

### ⚠️ Puerto 8001
**NO ESTÁ EN USO** en la configuración actual del frontend.

Solo aparece en archivos de backend antiguos como variable de entorno `MODEL_20B_ENDPOINT` pero **NO se usa en el frontend**.

## 🛠️ Solución

### Paso 1: Limpiar Caché del Navegador

**Opción A - Hard Refresh (Recomendado)**:
```
Chrome/Edge: Ctrl + Shift + R (Windows/Linux) o Cmd + Shift + R (Mac)
Firefox: Ctrl + F5 (Windows/Linux) o Cmd + Shift + R (Mac)
Safari: Cmd + Option + E, luego Cmd + R
```

**Opción B - Limpiar Caché Completo**:

1. **Chrome/Edge**:
   - Presiona `Ctrl + Shift + Delete`
   - Selecciona "Imágenes y archivos en caché"
   - Rango de tiempo: "Desde siempre"
   - Click "Borrar datos"

2. **Firefox**:
   - Presiona `Ctrl + Shift + Delete`
   - Marca "Caché"
   - Rango de tiempo: "Todo"
   - Click "Limpiar ahora"

3. **Safari**:
   - Safari > Preferencias > Avanzado
   - Marca "Mostrar menú Desarrollo"
   - Desarrollo > Vaciar cachés

**Opción C - Modo Incógnito/Privado**:
```
Chrome/Edge: Ctrl + Shift + N
Firefox: Ctrl + Shift + P
Safari: Cmd + Shift + N
```

### Paso 2: Verificar Archivos Cargados

Después de limpiar caché, abre DevTools (F12) y ve a la pestaña "Network" (Red).

**Archivos que DEBEN cargarse** (desde `web/chat.html`):
```
✅ translations.js
✅ entropy-monitor.js
✅ smart-mcp-integration.js  (puerto 5003, NO 8001)
✅ template-profiles.js
✅ rating-system.js
✅ tts-integration.js
✅ consensus-ui.js
✅ chat-app.js
✅ entropy-auto-inject.js
```

**Archivos que NO deben aparecer**:
```
❌ n8n-manager.js
❌ chat-page.js
❌ Cualquier referencia a puerto 8001 en frontend
```

### Paso 3: Verificar Configuración

Abre la consola del navegador (F12) y ejecuta:

```javascript
// Verificar configuración cargada
console.log('Configuración actual:', CHATBOT_CONFIG);

// Verificar servicios configurados
console.log('Servicios:', CHATBOT_CONFIG.SERVICES);

// El backend debe ser puerto 5001
console.log('Backend URL:', CHATBOT_CONFIG.BACKEND_URL);
// Esperado: http://localhost:5001 o http://34.12.166.76:5001

// MCP debe ser puerto 5003 (si está habilitado)
console.log('MCP URL:', CHATBOT_CONFIG.SERVICES.MCP.url);
// Esperado: http://localhost:5003 o http://34.175.136.104:5003

// Smart MCP config
console.log('Smart MCP:', SMART_MCP_CONFIG);
// serverUrl debe ser: http://localhost:5003/api/mcp/augment
```

### Paso 4: Verificar Servicios Backend Activos

Ejecuta este comando en la terminal para verificar qué puertos están activos:

```bash
# En VM bounty2
curl -s http://localhost:5001/health && echo " - Backend 5001 ✅"
curl -s http://localhost:5004/health && echo " - Auth 5004 ✅"
curl -s http://localhost:5005/health && echo " - Consensus 5005 ✅"

# En VM gpt-oss-20b
curl -s http://localhost:5002/health && echo " - TTS 5002 ✅"
curl -s http://localhost:5003/api/mcp/health && echo " - MCP 5003 ✅"

# En VM rag3
curl -s http://localhost:8000/health && echo " - Bridge 8000 ✅"
curl -s http://localhost:19530 && echo " - Milvus 19530 ✅"
```

## 📋 Verificación Final

Después de seguir los pasos, verifica en la consola del navegador:

### ✅ Correcto (sin errores):
```
✅ Smart MCP ACTIVO: healthy
✅ Conectado a backend en http://localhost:5001
✅ TTS disponible en http://localhost:5002
✅ No hay errores 404
```

### ❌ Incorrecto (si ves esto, repite los pasos):
```
❌ GET http://localhost:8001/... 404 (NOT FOUND)
❌ n8n-manager.js:52 Error
❌ chat-page.js:496 Error
```

## 🔍 Diagnóstico Adicional

Si después de limpiar caché sigues viendo errores, verifica:

### 1. ¿Qué archivo HTML estás abriendo?

Archivos correctos:
- ✅ `web/index.html` - Landing page
- ✅ `web/chat.html` - Aplicación de chat

### 2. ¿Hay archivos locales no commiteados?

```bash
cd /home/user/capibara6
git status
```

Si ves archivos como `n8n-manager.js` o `chat-page.js`, estos son antiguos y deben eliminarse:

```bash
# ⚠️ CUIDADO: Esto elimina archivos no trackeados
git clean -fd web/
```

### 3. ¿El servidor está sirviendo los archivos correctos?

Si estás usando un servidor local (como `python -m http.server`), asegúrate de que está sirviendo desde el directorio correcto:

```bash
# Correcto
cd /home/user/capibara6/web
python -m http.server 8080

# Luego abre http://localhost:8080/chat.html
```

## 📝 Resumen

El problema NO está en el código del repositorio. Los errores son causados por:

1. **Caché del navegador** con archivos JavaScript antiguos
2. **Archivos locales** que no están en el repositorio actual
3. **Configuración antigua** que referencia puerto 8001

**Solución rápida**:
1. Hard refresh (Ctrl + Shift + R)
2. O abrir en modo incógnito
3. Verificar que no hay errores 404 en la consola

## 🚀 Configuración Actual Correcta

El archivo `web/config.js` tiene la configuración correcta:

```javascript
const CHATBOT_CONFIG = {
    BACKEND_URL: isLocalhost ? 'http://localhost:5001' : 'http://34.12.166.76:5001',

    SERVICES: {
        MCP: {
            url: isLocalhost ? 'http://localhost:5003' : 'http://34.175.136.104:5003'
        },
        TTS: {
            url: isLocalhost ? 'http://localhost:5002' : 'http://34.175.136.104:5002'
        },
        RAG3_BRIDGE: {
            url: isLocalhost ? 'http://localhost:8000' : 'http://10.154.0.2:8000'
        }
    }
};
```

Y `web/smart-mcp-integration.js`:

```javascript
const SMART_MCP_CONFIG = {
    serverUrl: 'http://localhost:5003/api/mcp/augment',
    healthUrl: 'http://localhost:5003/api/mcp/health',
    enabled: false  // Deshabilitado por defecto
};
```

**NO hay referencias al puerto 8001 en el frontend actual.**

---

## 🔄 Actualización: Servidor Remoto con Archivos Antiguos

**Después de investigación adicional**, se confirmó que los errores **NO son causados por caché del navegador**, sino por un **servidor remoto** en `localhost:8000` que está sirviendo archivos JavaScript obsoletos (`chat-page.js`, `n8n-manager.js`).

**Ver guía completa**: [`ACTUALIZAR_SERVIDOR_WEB.md`](./ACTUALIZAR_SERVIDOR_WEB.md)

**Solución rápida**:
```bash
# Servir archivos actualizados desde este repositorio
cd /home/user/capibara6/web
python3 -m http.server 8000

# Limpiar caché del navegador: Ctrl + Shift + R
# Recargar: http://localhost:8000/chat.html
```

---

**Última actualización**: 2025-11-14
**Estado**: ⚠️ Servidor remoto requiere actualización de archivos
**Repositorio**: ✅ Configuración correcta

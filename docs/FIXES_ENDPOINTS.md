# Correcciones de Endpoints y Configuración

## Fecha: 2025-11-13

## Problema Reportado

Errores 404 en consola del navegador:
```
GET http://localhost:8001/api/mcp/status 404 (NOT FOUND)
POST http://localhost:8001/api/ai/generate 404 (NOT FOUND)
GET http://34.175.136.104:5678/healthz net::ERR_CONNECTION_TIMED_OUT
```

## Análisis

Los errores indican que:
1. Se estaba usando el **puerto incorrecto** para el backend (8001 en lugar de 5001 o 5003)
2. Los **endpoints no existen** en el backend actual
3. El servicio **N8N** no es accesible sin VPN/túnel

## Correcciones Realizadas

### 1. Archivo: `web/smart-mcp-integration.js`

**Cambios:**
- ✅ Corregido puerto de **5001 → 5003** (puerto real del MCP server)
- ✅ Corregido endpoint de **`/api/mcp/analyze` → `/api/mcp/augment`** (endpoint correcto)
- ✅ Agregado `healthUrl: 'http://localhost:5003/api/mcp/health'`
- ✅ Cambiado `enabled: true → false` (deshabilitado por defecto)
- ✅ Actualizado health check para usar el puerto correcto (5003 en lugar de 5010)

**Antes:**
```javascript
const SMART_MCP_CONFIG = {
    serverUrl: 'http://localhost:5001/api/mcp/analyze',
    enabled: true,
    ...
};

// Health check usaba puerto 5010
const healthUrl = 'http://localhost:5010/health';
```

**Después:**
```javascript
const SMART_MCP_CONFIG = {
    serverUrl: 'http://localhost:5003/api/mcp/augment',
    healthUrl: 'http://localhost:5003/api/mcp/health',
    enabled: false,  // Deshabilitado por defecto
    ...
};

// Health check usa la configuración correcta
const healthUrl = SMART_MCP_CONFIG.healthUrl;
```

### 2. Archivo: `web/config.js`

**Cambios:**
- ✅ Agregado `N8N_ENABLED: false` para deshabilitar el health check

**Antes:**
```javascript
SERVICES: {
    MCP_ENABLED: false,
    TTS_ENABLED: true,
    E2B_ENABLED: true
}
```

**Después:**
```javascript
SERVICES: {
    MCP_ENABLED: false,  // Requiere MCP server en puerto 5003
    TTS_ENABLED: true,   // TTS disponible en VM_SERVICES:5002
    E2B_ENABLED: true,   // E2B integrado en backend
    N8N_ENABLED: false   // N8N requiere VPN/túnel a VM_SERVICES:5678
}
```

### 3. Archivo: `web/consensus-ui.js`

**Cambios:**
- ✅ Comentado health check de N8N que causaba timeout
- ✅ Agregado checks condicionales basados en configuración
- ✅ Corregido endpoint MCP de `/health` → `/api/mcp/health`

**Antes:**
```javascript
await checkServiceStatus('mcp', 'http://34.175.136.104:5003/health');
await checkServiceStatus('n8n', 'http://34.175.136.104:5678');
```

**Después:**
```javascript
// Verificar MCP (solo si está habilitado)
if (window.CHATBOT_CONFIG?.SERVICES?.MCP_ENABLED) {
    await checkServiceStatus('mcp', 'http://34.175.136.104:5003/api/mcp/health');
}

// N8N deshabilitado por defecto - requiere VPN/túnel
// await checkServiceStatus('n8n', 'http://34.175.136.104:5678/healthz');
```

## Endpoints Correctos del Backend

Según el análisis del código backend:

### Servidor Principal: `backend/server_gptoss.py` (Puerto 5001)
- `POST /api/chat` - Enviar mensaje al chatbot
- `POST /api/chat/stream` - Chat con streaming
- `GET /api/health` - Health check del servidor
- `GET /api/models` - Listar modelos disponibles
- `POST /api/save-conversation` - Guardar conversación

### Servidor MCP: `backend/mcp_server.py` (Puerto 5003)
- `GET /api/mcp/contexts` - Listar contextos disponibles
- `GET /api/mcp/context/<id>` - Obtener contexto específico
- `POST /api/mcp/augment` - Aumentar prompt con contexto (RAG)
- `GET /api/mcp/tools` - Listar herramientas disponibles
- `POST /api/mcp/calculate` - Calculadora
- `POST /api/mcp/verify` - Verificar hechos
- `GET /api/mcp/health` - Health check del MCP

### Servidor FastAPI: `backend/main.py` (Puerto 8000)
- `GET /health` - Health check
- `POST /api/v1/query` - Consulta al modelo
- `GET /api/v1/models` - Listar modelos
- `POST /api/v1/e2b/execute` - Ejecutar código en E2B

## Nota sobre Errores de Puerto 8001

Los errores con el puerto **8001** reportados en la consola NO se encontraron en el código fuente actual. Esto sugiere:

1. **Caché del navegador:** Limpia la caché (Ctrl+Shift+R o Cmd+Shift+R)
2. **Archivos locales diferentes:** Verifica que estés usando la última versión del código
3. **Módulos cargados dinámicamente:** Algunos scripts pueden estar cargándose desde otras fuentes

## Cómo Verificar los Cambios

1. **Limpiar caché del navegador:**
   ```
   Chrome/Edge: Ctrl+Shift+Delete
   Firefox: Ctrl+Shift+Delete
   Safari: Cmd+Option+E
   ```

2. **Verificar que los servidores estén corriendo:**
   ```bash
   # Backend principal (puerto 5001)
   python3 backend/server_gptoss.py

   # MCP server (puerto 5003) - opcional
   python3 backend/mcp_server.py

   # Servidor web
   python3 -m http.server 8000 --directory web
   ```

3. **Abrir la consola del navegador** (F12) y verificar:
   - ✅ No debe haber errores 404 para MCP si está deshabilitado
   - ✅ No debe haber timeout de N8N
   - ✅ Requests deben ir a `localhost:5001` para el backend principal
   - ✅ Requests de MCP deben ir a `localhost:5003` (si está habilitado)

## Estado de los Servicios

| Servicio | Puerto | Estado | Notas |
|----------|--------|--------|-------|
| Backend Principal (Flask) | 5001 | ✅ Activo | Servidor GPT-OSS |
| MCP Server (Flask) | 5003 | ⚠️ Opcional | Deshabilitado por defecto |
| FastAPI Server | 8000 | ⚠️ Alternativo | No usado por frontend |
| TTS Server | 5002 | ✅ Activo | En VM 34.175.136.104 |
| N8N | 5678 | ❌ Requiere VPN | No accesible públicamente |

## Recomendaciones Adicionales

1. **Para habilitar MCP:**
   - Inicia el servidor: `python3 backend/mcp_server.py`
   - Cambia en `web/smart-mcp-integration.js`: `enabled: true`
   - Recarga la página con cache limpio

2. **Para acceder a N8N:**
   - Configura un túnel SSH o VPN a la VM 34.175.136.104
   - O usa N8N Cloud en lugar de la instancia self-hosted

3. **Monitoreo:**
   - Revisa regularmente los logs del backend con `tail -f backend/logs/*.log`
   - Usa el endpoint `/health` para health checks automatizados

## Archivos Modificados

- ✅ `web/smart-mcp-integration.js` (3 cambios)
- ✅ `web/config.js` (1 cambio)
- ✅ `web/consensus-ui.js` (2 cambios)

## Próximos Pasos

- [ ] Limpiar caché del navegador y verificar que los errores hayan desaparecido
- [ ] Decidir si se necesita el MCP server activo y habilitarlo si es necesario
- [ ] Configurar acceso a N8N si se requiere
- [ ] Considerar implementar los endpoints faltantes si son necesarios para el frontend

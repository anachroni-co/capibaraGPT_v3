# Actualizar Servidor Web con Archivos Antiguos

## 🔍 Problema Identificado

El navegador está cargando archivos JavaScript **antiguos** desde un servidor en `http://localhost:8000/`:

```
http://localhost:8000/chat-page.js  ❌ NO EXISTE en repo actual
http://localhost:8000/n8n-manager.js ❌ NO EXISTE en repo actual
```

Estos archivos hacen peticiones al puerto 8001 que generan errores 404.

## 🎯 Causa Raíz

Hay un **servidor externo o túnel SSH** en el puerto 8000 que está sirviendo una **versión antigua** del proyecto con archivos JavaScript obsoletos que ya no están en el repositorio actual.

El mensaje en la consola confirma esto:
```javascript
backend_target: "http://34.12.166.76:5001"
service: "capibara6-cors-proxy"
```

## ✅ Soluciones

### Opción 1: Servir Archivos Localmente (Recomendado para Desarrollo)

Si estás desarrollando localmente, sirve los archivos desde este repositorio actualizado:

```bash
# 1. Ir al directorio web
cd /home/user/capibara6/web

# 2. Iniciar servidor HTTP local en puerto 8080
python3 -m http.server 8080

# 3. Abrir en navegador
# http://localhost:8080/chat.html
# http://localhost:8080/index.html
```

**Ventajas**:
- ✅ Usas archivos actualizados del repositorio
- ✅ Los cambios se reflejan inmediatamente
- ✅ No hay archivos antiguos en caché

### Opción 2: Actualizar Servidor Remoto en VM

Si el servidor en puerto 8000 está en una VM (probablemente VM rag3), necesitas actualizar los archivos allí:

#### A. Identificar qué VM está sirviendo los archivos

```bash
# Verificar túneles SSH activos
ps aux | grep "ssh.*8000"

# O verificar si hay un servicio corriendo
curl -I http://localhost:8000/

# El header "Server:" te dirá qué tipo de servidor es
```

#### B. Actualizar archivos en la VM

Si identificas que es VM rag3 (basado en que sirve el proxy CORS):

```bash
# 1. SSH a la VM
gcloud compute ssh --zone "europe-west2-c" "rag3" --project "mamba-001"

# 2. Localizar el directorio web
find /home -name "chat-page.js" -o -name "n8n-manager.js" 2>/dev/null

# Probablemente esté en algo como:
# /home/user/capibara6-old/web/
# /var/www/capibara6/
# /opt/capibara6/web/

# 3. Hacer backup de archivos antiguos
cd /ruta/al/directorio/web
mkdir ../backup-$(date +%Y%m%d)
mv chat-page.js n8n-manager.js ../backup-$(date +%Y%m%d)/ 2>/dev/null

# 4. Copiar archivos actualizados desde este repo
# (desde tu máquina local)
gcloud compute scp --zone "europe-west2-c" \
    --recurse /home/user/capibara6/web/* \
    rag3:/ruta/al/directorio/web/

# 5. Reiniciar el servicio web (si es necesario)
# Si es un servidor Python
sudo systemctl restart capibara6-web

# Si es nginx
sudo systemctl restart nginx

# Si es un proceso manual
# Matar el proceso viejo y reiniciar
```

### Opción 3: Crear Túnel SSH al Servidor Correcto

Si quieres seguir usando `localhost:8000` pero con archivos actualizados:

```bash
# 1. Cerrar cualquier túnel SSH existente en puerto 8000
# Buscar PIDs
lsof -ti:8000

# Matar procesos si existen
kill $(lsof -ti:8000)

# 2. Iniciar servidor local con archivos actualizados
cd /home/user/capibara6/web
python3 -m http.server 8000

# Ahora localhost:8000 servirá archivos actualizados
```

### Opción 4: Usar un HTML Actualizado sin Archivos Obsoletos

Si no puedes actualizar el servidor remoto inmediatamente, puedes crear un HTML de prueba local:

```bash
# 1. Copiar chat.html a un archivo de prueba
cd /home/user/capibara6/web
cp chat.html chat-local.html

# 2. Servir localmente
python3 -m http.server 9000

# 3. Abrir en navegador
# http://localhost:9000/chat-local.html
```

Este archivo cargará los scripts correctos desde el repositorio actualizado:
- ✅ `chat-app.js` (EXISTE)
- ✅ `smart-mcp-integration.js` (EXISTE)
- ✅ `config.js` (EXISTE)
- ❌ NO `chat-page.js` (obsoleto)
- ❌ NO `n8n-manager.js` (obsoleto)

## 🔧 Verificación Después de Actualizar

### 1. Limpiar Caché del Navegador

```
Chrome/Edge: Ctrl + Shift + R
Firefox: Ctrl + F5
Safari: Cmd + Option + E, luego Cmd + R
```

### 2. Verificar en DevTools (F12)

**Pestaña "Network" (Red)**:

Archivos que DEBEN cargarse (desde `web/chat.html`):
```
✅ translations.js
✅ entropy-monitor.js
✅ smart-mcp-integration.js
✅ template-profiles.js
✅ rating-system.js
✅ tts-integration.js
✅ consensus-ui.js
✅ chat-app.js
✅ entropy-auto-inject.js
```

Archivos que NO deben aparecer:
```
❌ chat-page.js
❌ n8n-manager.js
```

**Pestaña "Console" (Consola)**:

Verificar puertos correctos:
```javascript
console.log('Backend URL:', CHATBOT_CONFIG.BACKEND_URL);
// Esperado: http://localhost:5001 o http://34.12.166.76:5001

console.log('MCP URL:', CHATBOT_CONFIG.SERVICES.MCP.url);
// Esperado: http://localhost:5003 o http://34.175.136.104:5003

console.log('Smart MCP:', SMART_MCP_CONFIG.serverUrl);
// Esperado: http://localhost:5003/api/mcp/augment
```

**NO debe haber referencias al puerto 8001** en las peticiones del frontend.

### 3. Verificar Errores en Consola

**✅ Correcto** (sin errores 404):
```
✅ Backend conectado
✅ Servicio TTS inicializado
✅ Servicio MCP inicializado
✅ Servicio RAG inicializado
```

**❌ Incorrecto** (si ves esto, el problema persiste):
```
❌ GET http://localhost:8001/api/mcp/status 404 (NOT FOUND)
❌ GET http://localhost:8001/api/n8n/templates/recommended 404 (NOT FOUND)
❌ POST http://localhost:8001/api/ai/generate 404 (NOT FOUND)
❌ Error en chat-page.js
❌ Error en n8n-manager.js
```

## 📊 Diagnóstico Avanzado

### Identificar el Servidor en Puerto 8000

```bash
# 1. Verificar si es un túnel SSH
ps aux | grep ssh | grep 8000

# Resultado esperado si hay túnel:
# ssh -L 8000:localhost:8000 user@vm-rag3

# 2. Verificar si es un servidor local
lsof -i :8000 -i :8001

# 3. Hacer request y ver headers
curl -I http://localhost:8000/

# Headers importantes:
# Server: nginx/1.18.0  -> Es nginx en VM
# Server: SimpleHTTP/0.6 Python/3.9.2 -> Es python http.server
# Server: uvicorn -> Es un servidor Python moderno
```

### Encontrar Archivos Obsoletos en el Servidor

Si tienes acceso al servidor:

```bash
# Buscar chat-page.js
find /home /var/www /opt -name "chat-page.js" 2>/dev/null

# Buscar n8n-manager.js
find /home /var/www /opt -name "n8n-manager.js" 2>/dev/null

# Buscar directorios que contengan ambos
find /home /var/www /opt -type f \( -name "chat-page.js" -o -name "n8n-manager.js" \) \
    -exec dirname {} \; | sort -u
```

### Ver Qué Archivos Carga el HTML Remoto

```bash
# Descargar el HTML que se está sirviendo
curl http://localhost:8000/chat.html > /tmp/chat-remote.html

# Ver los scripts que carga
grep -o '<script src="[^"]*"' /tmp/chat-remote.html

# Comparar con el HTML actual del repo
grep -o '<script src="[^"]*"' /home/user/capibara6/web/chat.html

# Buscar diferencias
diff <(grep -o '<script src="[^"]*"' /tmp/chat-remote.html | sort) \
     <(grep -o '<script src="[^"]*"' /home/user/capibara6/web/chat.html | sort)
```

## 🎯 Solución Rápida (TL;DR)

Si solo quieres que funcione rápido:

```bash
# 1. Cerrar cualquier servidor en puerto 8000
kill $(lsof -ti:8000) 2>/dev/null

# 2. Servir archivos actualizados desde este repo
cd /home/user/capibara6/web
python3 -m http.server 8000

# 3. Limpiar caché del navegador
# Ctrl + Shift + R

# 4. Recargar
# http://localhost:8000/chat.html
```

Si esto funciona, verás:
- ✅ NO errores 404 en puerto 8001
- ✅ Archivos correctos cargados (chat-app.js, smart-mcp-integration.js)
- ✅ Peticiones al backend en puerto 5001 (NO 8001)

## 📝 Resumen

| Problema | Causa | Solución |
|----------|-------|----------|
| Errores 404 en puerto 8001 | Archivos JS antiguos haciendo peticiones incorrectas | Actualizar servidor web con archivos del repo |
| chat-page.js not found | Archivo obsoleto que ya no existe | Servir desde `/home/user/capibara6/web` |
| n8n-manager.js not found | Archivo obsoleto que ya no existe | Servir desde `/home/user/capibara6/web` |
| Backend target incorrecto | Servidor antiguo con configuración obsoleta | Usar `web/config.js` actualizado |

**Archivos actuales correctos** (en `web/chat.html`):
- ✅ `chat-app.js` - Aplicación principal
- ✅ `smart-mcp-integration.js` - MCP v2.0
- ✅ `config.js` - Configuración actualizada

**Puertos correctos**:
- ✅ 5001 - Backend principal (GPT-OSS-20B)
- ✅ 5003 - MCP server (Context & RAG)
- ✅ 5002 - TTS server
- ✅ 8000 - RAG Bridge (VM rag3)
- ❌ 8001 - NO se usa en frontend actual

---

**Última actualización**: 2025-11-14
**Estado**: Problema identificado - Servidor remoto con archivos antiguos

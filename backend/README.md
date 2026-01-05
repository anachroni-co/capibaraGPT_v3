# Capibara6 Backend - Servidor de Chat con IA

Backend Flask para el chat de IA Capibara6 con soporte para GPT-OSS-20B y subida de archivos.

## 🚀 Inicio Rápido

### Prerequisitos

```bash
# Python 3.8 o superior
python3 --version

# Instalar dependencias
pip install flask flask-cors requests python-dotenv
```

### Configuración

1. **Crear archivo `.env` (opcional)**:
```bash
# Configuración del modelo GPT-OSS-20B
GPT_OSS_URL=http://34.175.215.109:8080
GPT_OSS_TIMEOUT=60

# Puerto del servidor (por defecto 5001)
PORT=5001

# Configuración SMTP (solo para server.py)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=info@anachroni.co
SMTP_PASSWORD=tu_password
FROM_EMAIL=info@anachroni.co
```

### Ejecutar el Servidor

**IMPORTANTE**: Para que el chat funcione, debes ejecutar `server_gptoss.py`:

```bash
cd backend
python3 server_gptoss.py
```

El servidor iniciará en `http://localhost:5001`

Verás este mensaje:
```
🦫 capibara6 Backend iniciado
🤖 Modelo: GPT-OSS-20B
🌐 URL del modelo: http://34.175.215.109:8080
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5001
```

### Acceder al Chat

1. Abre tu navegador
2. Navega a: `file:///ruta/a/capibara6/web/chat.html`
3. O usa un servidor web local:
   ```bash
   cd web
   python3 -m http.server 8000
   # Luego abre: http://localhost:8000/chat.html
   ```

## 📁 Servidores Disponibles

### 1. `server_gptoss.py` ⭐ (RECOMENDADO PARA CHAT)

**Servidor principal para el chat con IA**

**Características**:
- ✅ Endpoint `/api/chat` para chat con GPT-OSS-20B
- ✅ Soporte para subida de archivos (multipart/form-data)
- ✅ Almacenamiento de archivos en `user_data/uploads/`
- ✅ Conversaciones guardadas en JSON
- ✅ Streaming de respuestas
- ✅ Health check del modelo

**Endpoints**:
- `POST /api/chat` - Chat principal (acepta JSON o FormData con archivos)
- `POST /api/chat/stream` - Chat con streaming
- `GET /api/health` - Estado del servidor y modelo
- `GET /api/models` - Información del modelo
- `POST /api/save-conversation` - Guardar conversación

**Puerto**: 5001

**Cómo ejecutar**:
```bash
cd backend
python3 server_gptoss.py
```

### 2. `server.py`

**Servidor para gestión de emails y guardado de conversaciones**

**Características**:
- Envío de emails de confirmación
- Guardado de conversaciones
- NO tiene endpoint de chat

**Endpoints**:
- `POST /api/save-conversation` - Guardar y enviar emails
- `GET /api/health` - Health check

**Puerto**: 5000 (por defecto)

**Nota**: ⚠️ Este servidor NO es suficiente para el chat. El frontend requiere `/api/chat`.

**Cómo ejecutar**:
```bash
cd backend
python3 server.py
```

### 3. `capibara6_integrated_server.py`

**Servidor integrado con múltiples funcionalidades**

Si existe, incluye chat + TTS + MCP + E2B.

### 4. `main.py`

**Servidor FastAPI**

Si existe, puede tener endpoints diferentes (verifica el código).

## 🔧 Solución de Problemas

### ❌ El botón de enviar no funciona

**Causa**: El servidor no está ejecutándose o es el incorrecto.

**Solución**:
```bash
# 1. Verificar si hay un servidor corriendo
ps aux | grep python | grep server

# 2. Si no hay ninguno, iniciar server_gptoss.py
cd backend
python3 server_gptoss.py

# 3. Si hay uno incorrecto, detenerlo y ejecutar el correcto
killall python3  # o Ctrl+C en la terminal del servidor
python3 server_gptoss.py
```

### ❌ Error: "No se pudo conectar con el modelo"

**Causa**: El servidor GPT-OSS-20B no está disponible.

**Soluciones**:
1. Verifica que la IP en `.env` sea correcta
2. Verifica conectividad: `curl http://34.175.215.109:8080/health`
3. Cambia `GPT_OSS_URL` en `.env` si el servidor está en otra ubicación

### ❌ Error de CORS

**Causa**: Frontend y backend en dominios diferentes.

**Solución**: El servidor ya tiene CORS habilitado. Si persiste:
1. Usa un servidor web local para servir el frontend
2. O abre `chrome` con: `--disable-web-security --user-data-dir=/tmp/chrome`

### ❌ Los archivos no se suben

**Causa**: Permisos o configuración incorrecta.

**Solución**:
```bash
# Crear directorio de uploads
mkdir -p backend/user_data/uploads
chmod 755 backend/user_data/uploads

# Verificar que el servidor tiene permisos de escritura
ls -la backend/user_data/
```

### ❌ Error: "Address already in use"

**Causa**: El puerto 5001 ya está en uso.

**Solución**:
```bash
# Ver qué proceso está usando el puerto
lsof -i :5001

# Matar el proceso
kill -9 <PID>

# O cambiar el puerto en .env
PORT=5002
```

## 📊 Estructura de Datos

### Conversaciones guardadas

```json
{
  "timestamp": "2025-01-01T12:00:00",
  "user_message": "Hola, ¿cómo estás?",
  "ai_response": "¡Hola! Estoy bien, gracias...",
  "user_email": "usuario@example.com",
  "ip": "127.0.0.1",
  "user_agent": "Mozilla/5.0..."
}
```

Ubicación: `backend/user_data/conversations.json`

### Archivos subidos

Los archivos se guardan en: `backend/user_data/uploads/`

Formato del nombre: `YYYYMMDD_HHMMSS_nombre_original.ext`

Ejemplo: `20250110_143022_documento.pdf`

## 🔐 Seguridad

### Archivos permitidos

Por defecto, solo se permiten estos tipos:
- Imágenes: `png, jpg, jpeg, gif`
- Documentos: `pdf, doc, docx, txt`
- Datos: `csv, xlsx, xls`
- Presentaciones: `pptx, ppt`
- Comprimidos: `zip, rar`

### Tamaño máximo

10MB por archivo (configurable en `MAX_FILE_SIZE`)

### Validación

- Nombres de archivos sanitizados (secure_filename)
- Extensiones verificadas
- Tamaño validado

## 📧 Configuración SMTP (solo server.py)

Si usas `server.py` para envío de emails:

### Gmail
1. Ve a https://myaccount.google.com/apppasswords
2. Genera una "Contraseña de aplicación"
3. Usa esa contraseña en `SMTP_PASSWORD`

### Otros proveedores

**Outlook/Hotmail:**
```env
SMTP_SERVER=smtp.office365.com
SMTP_PORT=587
```

**Yahoo:**
```env
SMTP_SERVER=smtp.mail.yahoo.com
SMTP_PORT=587
```

## 🌐 Producción

Para producción, considera usar un servidor WSGI como Gunicorn:

```bash
# Instalar gunicorn
pip install gunicorn

# Ejecutar en producción
gunicorn -w 4 -b 0.0.0.0:5001 server_gptoss:app
```

O configurar Railway/Vercel según tus necesidades.

## 📝 Logs

Los logs se imprimen en la consola. Para guardarlos:

```bash
python3 server_gptoss.py 2>&1 | tee server.log
```

## 🆘 Soporte

Si tienes problemas:

1. Revisa los logs del servidor
2. Verifica el network tab del navegador (DevTools > Network)
3. Asegúrate de que el puerto 5001 esté libre: `lsof -i :5001`
4. Verifica la consola del navegador (F12) para errores JavaScript
5. Contacta con el equipo de Anachroni

---

**Desarrollado por**: Anachroni s.coop
**Modelo**: GPT-OSS-20B
**Versión**: 1.0

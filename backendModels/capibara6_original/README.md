# Backend capibara6

Servidor Flask para gestión de emails y conversaciones del chatbot.

## 🚀 Instalación

```bash
cd backend

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## ⚙️ Configuración

1. Copia el archivo de ejemplo:
```bash
cp .env.example .env
```

2. Edita `.env` con tus credenciales SMTP:
```env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=info@anachroni.co
SMTP_PASSWORD=tu_contraseña_de_aplicacion
FROM_EMAIL=info@anachroni.co
```

### 📧 Configurar Gmail

Si usas Gmail (`info@anachroni.co` en Gmail):

1. Ve a https://myaccount.google.com/apppasswords
2. Genera una "Contraseña de aplicación"
3. Usa esa contraseña en `SMTP_PASSWORD`

### 📧 Otros proveedores

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

**Servidor SMTP personalizado:**
Configura tu propio servidor SMTP.

## ▶️ Ejecutar

```bash
python server.py
```

El servidor estará disponible en: `http://localhost:5000`

## 🔌 Endpoints

### `POST /api/save-conversation`
Guarda la conversación y envía emails.

**Body:**
```json
{
  "email": "usuario@example.com",
  "conversations": [
    {
      "message": "Hola, me interesa capibara6",
      "timestamp": "2025-10-02T10:30:00.000Z"
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "email_sent": true,
  "admin_notified": true,
  "message": "Datos guardados correctamente"
}
```

### `GET /api/health`
Health check del servidor.

## 📁 Archivos generados

Los datos se guardan en `backend/user_data/`:
- `conversations.json` - Todas las conversaciones en JSON
- `user_YYYYMMDD_HHMMSS.txt` - Archivo de texto por cada usuario

## 🔒 Seguridad

- Nunca subas el archivo `.env` al repositorio
- Ya está incluido en `.gitignore`
- Usa contraseñas de aplicación, no tu contraseña personal

## 🐛 Troubleshooting

**Error de autenticación SMTP:**
- Verifica que la contraseña sea correcta
- Usa "Contraseña de aplicación" en Gmail
- Verifica que el servidor y puerto sean correctos

**Error de conexión:**
- Verifica que el firewall permita conexiones SMTP
- Algunos ISPs bloquean el puerto 587

**CORS errors:**
- El servidor ya tiene CORS habilitado
- Verifica que la URL del frontend sea correcta


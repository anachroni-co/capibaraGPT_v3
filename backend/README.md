# CapibaraGPT Backend - AI Chat Server

Flask backend for CapibaraGPT AI chat with support for multi-model inference and file uploads.

## Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python3 --version

# Install dependencies
pip install flask flask-cors requests python-dotenv
```

### Configuration

1. **Create `.env` file (optional)**:
```bash
# Model configuration
GPT_OSS_URL=http://34.175.215.109:8080
GPT_OSS_TIMEOUT=60

# Server port (default 5001)
PORT=5001

# SMTP configuration (only for server.py)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=info@anachroni.co
SMTP_PASSWORD=your_password
FROM_EMAIL=info@anachroni.co
```

### Running the Server

**IMPORTANT**: To enable chat functionality, run `server_gptoss.py`:

```bash
cd backend
python3 server_gptoss.py
```

Server will start at `http://localhost:5001`

You'll see:
```
Backend started
Model: GPT-OSS-20B
Model URL: http://34.175.215.109:8080
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5001
```

### Accessing the Chat

1. Open your browser
2. Navigate to: `file:///path/to/capibaraGPT_v3/web/chat.html`
3. Or use a local web server:
   ```bash
   cd web
   python3 -m http.server 8000
   # Then open: http://localhost:8000/chat.html
   ```

## Available Servers

### 1. `server_gptoss.py` (RECOMMENDED FOR CHAT)

**Main server for AI chat**

**Features**:
- `/api/chat` endpoint for chat with AI models
- File upload support (multipart/form-data)
- File storage in `user_data/uploads/`
- Conversations saved in JSON
- Response streaming
- Model health check

**Endpoints**:
- `POST /api/chat` - Main chat (accepts JSON or FormData with files)
- `POST /api/chat/stream` - Chat with streaming
- `GET /api/health` - Server and model status
- `GET /api/models` - Model information
- `POST /api/save-conversation` - Save conversation

**Port**: 5001

**How to run**:
```bash
cd backend
python3 server_gptoss.py
```

### 2. `server.py`

**Server for email management and conversation saving**

**Features**:
- Confirmation email sending
- Conversation saving
- NO chat endpoint

**Endpoints**:
- `POST /api/save-conversation` - Save and send emails
- `GET /api/health` - Health check

**Port**: 5000 (default)

**Note**: This server is NOT sufficient for chat. Frontend requires `/api/chat`.

### 3. `capibara6_integrated_server.py`

**Integrated server with multiple functionalities**

Includes chat + TTS + MCP + E2B.

### 4. `main.py`

**FastAPI server**

May have different endpoints (check the code).

## Troubleshooting

### Send button doesn't work

**Cause**: Server not running or wrong server.

**Solution**:
```bash
# 1. Check if a server is running
ps aux | grep python | grep server

# 2. If none, start server_gptoss.py
cd backend
python3 server_gptoss.py

# 3. If wrong server, stop and run correct one
killall python3  # or Ctrl+C in server terminal
python3 server_gptoss.py
```

### Error: "Could not connect to model"

**Cause**: AI model server unavailable.

**Solutions**:
1. Verify IP in `.env` is correct
2. Check connectivity: `curl http://34.175.215.109:8080/health`
3. Change `GPT_OSS_URL` in `.env` if server is elsewhere

### CORS Error

**Cause**: Frontend and backend on different domains.

**Solution**: Server has CORS enabled. If issue persists:
1. Use local web server to serve frontend
2. Or open Chrome with: `--disable-web-security --user-data-dir=/tmp/chrome`

### Files not uploading

**Cause**: Permissions or incorrect configuration.

**Solution**:
```bash
# Create uploads directory
mkdir -p backend/user_data/uploads
chmod 755 backend/user_data/uploads

# Verify server has write permissions
ls -la backend/user_data/
```

### Error: "Address already in use"

**Cause**: Port 5001 already in use.

**Solution**:
```bash
# See which process is using the port
lsof -i :5001

# Kill the process
kill -9 <PID>

# Or change port in .env
PORT=5002
```

## Data Structure

### Saved Conversations

```json
{
  "timestamp": "2025-01-01T12:00:00",
  "user_message": "Hello, how are you?",
  "ai_response": "Hello! I'm fine, thanks...",
  "user_email": "user@example.com",
  "ip": "127.0.0.1",
  "user_agent": "Mozilla/5.0..."
}
```

Location: `backend/user_data/conversations.json`

### Uploaded Files

Files are saved in: `backend/user_data/uploads/`

Filename format: `YYYYMMDD_HHMMSS_original_name.ext`

Example: `20250110_143022_document.pdf`

## Security

### Allowed Files

By default, only these types are allowed:
- Images: `png, jpg, jpeg, gif`
- Documents: `pdf, doc, docx, txt`
- Data: `csv, xlsx, xls`
- Presentations: `pptx, ppt`
- Archives: `zip, rar`

### Maximum Size

10MB per file (configurable in `MAX_FILE_SIZE`)

### Validation

- Sanitized filenames (secure_filename)
- Extension verification
- Size validation

## SMTP Configuration (server.py only)

If using `server.py` for email sending:

### Gmail
1. Go to https://myaccount.google.com/apppasswords
2. Generate an "App Password"
3. Use that password in `SMTP_PASSWORD`

### Other Providers

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

## Production

For production, consider using a WSGI server like Gunicorn:

```bash
# Install gunicorn
pip install gunicorn

# Run in production
gunicorn -w 4 -b 0.0.0.0:5001 server_gptoss:app
```

Or configure Railway/Vercel as needed.

## Logs

Logs are printed to console. To save them:

```bash
python3 server_gptoss.py 2>&1 | tee server.log
```

## Support

If you have problems:

1. Check server logs
2. Verify network tab in browser (DevTools > Network)
3. Ensure port 5001 is free: `lsof -i :5001`
4. Check browser console (F12) for JavaScript errors

---

**Developed by**: Anachroni s.coop
**Version**: 3.0

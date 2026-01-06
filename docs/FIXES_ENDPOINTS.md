# Endpoint and Configuration Fixes

## Date: 2025-11-13

## Reported Problem

404 errors in browser console:
```
GET http://localhost:8001/api/mcp/status 404 (NOT FOUND)
POST http://localhost:8001/api/ai/generate 404 (NOT FOUND)
GET http://34.175.136.104:5678/healthz net::ERR_CONNECTION_TIMED_OUT
```

## Analysis

The errors indicate:
1. **Wrong port** was being used for backend (8001 instead of 5001 or 5003)
2. **Endpoints don't exist** in current backend
3. **N8N service** is not accessible without VPN/tunnel

## Fixes Applied

### 1. File: `web/smart-mcp-integration.js`

**Changes:**
- Fixed port from **5001 → 5003** (actual MCP server port)
- Fixed endpoint from **`/api/mcp/analyze` → `/api/mcp/augment`** (correct endpoint)
- Added `healthUrl: 'http://localhost:5003/api/mcp/health'`
- Changed `enabled: true → false` (disabled by default)
- Updated health check to use correct port (5003 instead of 5010)

### 2. File: `web/config.js`

**Changes:**
- Added `N8N_ENABLED: false` to disable health check

### 3. File: `web/consensus-ui.js`

**Changes:**
- Commented out N8N health check that was causing timeout
- Added conditional checks based on configuration
- Fixed MCP endpoint from `/health` → `/api/mcp/health`

## Correct Backend Endpoints

### Main Server: `backend/server_gptoss.py` (Port 5001)
- `POST /api/chat` - Send message to chatbot
- `POST /api/chat/stream` - Chat with streaming
- `GET /api/health` - Server health check
- `GET /api/models` - List available models
- `POST /api/save-conversation` - Save conversation

### MCP Server: `backend/mcp_server.py` (Port 5003)
- `GET /api/mcp/contexts` - List available contexts
- `GET /api/mcp/context/<id>` - Get specific context
- `POST /api/mcp/augment` - Augment prompt with context (RAG)
- `GET /api/mcp/tools` - List available tools
- `POST /api/mcp/calculate` - Calculator
- `POST /api/mcp/verify` - Verify facts
- `GET /api/mcp/health` - MCP health check

### FastAPI Server: `backend/main.py` (Port 8000)
- `GET /health` - Health check
- `POST /api/v1/query` - Query model
- `GET /api/v1/models` - List models
- `POST /api/v1/e2b/execute` - Execute code in E2B

## Service Status

| Service | Port | Status | Notes |
|---------|------|--------|-------|
| Main Backend (Flask) | 5001 | Active | GPT-OSS server |
| MCP Server (Flask) | 5003 | Optional | Disabled by default |
| FastAPI Server | 8000 | Alternative | Not used by frontend |
| TTS Server | 5002 | Active | On VM 34.175.136.104 |
| N8N | 5678 | Requires VPN | Not publicly accessible |

## How to Verify Changes

1. **Clear browser cache:**
   ```
   Chrome/Edge: Ctrl+Shift+Delete
   Firefox: Ctrl+Shift+Delete
   Safari: Cmd+Option+E
   ```

2. **Verify servers are running:**
   ```bash
   # Main backend (port 5001)
   python3 backend/server_gptoss.py

   # MCP server (port 5003) - optional
   python3 backend/mcp_server.py

   # Web server
   python3 -m http.server 8000 --directory web
   ```

3. **Open browser console** (F12) and verify:
   - No 404 errors for MCP if disabled
   - No N8N timeout
   - Requests go to `localhost:5001` for main backend
   - MCP requests go to `localhost:5003` (if enabled)

## Recommendations

1. **To enable MCP:**
   - Start server: `python3 backend/mcp_server.py`
   - Change in `web/smart-mcp-integration.js`: `enabled: true`
   - Reload page with clean cache

2. **To access N8N:**
   - Set up SSH tunnel or VPN to VM 34.175.136.104
   - Or use N8N Cloud instead of self-hosted instance

3. **Monitoring:**
   - Regularly check backend logs with `tail -f backend/logs/*.log`
   - Use `/health` endpoint for automated health checks

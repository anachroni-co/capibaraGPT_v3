# Solución al Problema CORS MCP

## Descripción del Problema

El sistema Capibara6 presentaba un problema de seguridad de Cross-Origin Resource Sharing (CORS) cuando el frontend intentaba acceder al endpoint `http://localhost:8001/api/mcp/status`, que no existía en el servidor MCP original.

El error era:

```
Access to fetch at 'http://localhost:8001/api/mcp/status' from origin 'http://localhost:8000' has been blocked by CORS policy: Response to preflight request doesn't pass access control check: It does not have HTTP ok status.
```

## Análisis Técnico

1. El frontend estaba configurado para llamar al endpoint `/api/mcp/status`
2. El proxy CORS en `http://localhost:8001` intentaba redirigir esta solicitud a `http://34.175.136.104:5003/api/mcp/status`
3. El servidor MCP en el puerto 5003 solo tenía el endpoint `/api/mcp/health`, no `/api/mcp/status`
4. Como resultado, la solicitud devolvía un error 404, causando fallas de CORS

## Solución Implementada

Se modificó el archivo `proxy-cors.py` para manejar redirecciones inteligentes de los endpoints MCP:

```python
# Manejar casos especiales - redirigir diferentes variantes de status a health
actual_path = path
if path.startswith('/api/mcp/') and '/status' in path:
    # Reemplazar cualquier variante de /status con /health para el servidor MCP
    actual_path = path.replace('/status', '/health')
elif path.startswith('/api/v1/mcp/') and '/status' in path:
    # También manejar la variante con /v1
    actual_path = path.replace('/status', '/health')
elif path == '/api/mcp/status' or path == '/api/v1/mcp/status':
    # Casos específicos exactos
    actual_path = path.replace('/status', '/health')

url = target + actual_path
```

## Beneficios de la Solución

1. **Elimina el error CORS**: Ahora las solicitudes a `/api/mcp/status` se redirigen correctamente a `/api/mcp/health`
2. **Mantiene compatibilidad**: Las aplicaciones existentes que usan el endpoint incorrecto seguirán funcionando
3. **Centraliza la lógica**: No hay que modificar todas las referencias al endpoint en el código
4. **Es extensible**: Se puede aplicar la misma lógica para manejar otros endpoints desfasados

## Endpoints Afectados

- `/api/mcp/status` → redirigido a `/api/mcp/health`
- `/api/v1/mcp/status` → redirigido a `/api/v1/mcp/health`
- `/api/mcp/tool/status` → redirigido a `/api/mcp/tool/health`

## Validación

Se implementó un script de verificación (`scripts/cors_fix_verification.py`) que prueba que las redirecciones funcionen correctamente y que el problema CORS esté resuelto.

## Archivos Modificados

- `proxy-cors.py` - Añadida la lógica de redirección inteligente
- `docs/CORS_FIX.md` - Documentación de la solución (este archivo)
- `scripts/cors_fix_verification.py` - Script de verificación de la solución
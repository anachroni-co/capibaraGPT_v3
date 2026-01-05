from flask import Flask, request, Response
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

TARGETS = {
    '/api/ai': 'http://34.12.166.76:5001',
    '/api/chat': 'http://34.12.166.76:5001',
    '/api/mcp': 'http://34.175.136.104:5003',
    '/api/n8n': 'http://34.175.136.104:5678',
    '/api/tts': 'http://34.175.136.104:5002',
    '/health': 'http://34.12.166.76:5001'
}

@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy(path):
    path = '/' + path
    target = None

    for prefix, url in TARGETS.items():
        if path.startswith(prefix):
            target = url
            break

    if not target:
        return 'Not found', 404

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

    print(f'🔀 Proxy: {path} -> {url}')

    try:
        resp = requests.request(
            method=request.method,
            url=url,
            headers={k: v for k, v in request.headers if k != 'Host'},
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False
        )

        return Response(resp.content, resp.status_code, resp.headers.items())
    except Exception as e:
        print(f'❌ Error: {e}')
        return str(e), 500

if __name__ == '__main__':
    print('�� CORS Proxy listening on port 8001')
    app.run(host='0.0.0.0', port=8001, debug=False)

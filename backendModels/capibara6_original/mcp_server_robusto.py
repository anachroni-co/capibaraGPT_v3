"""
Smart MCP Server - Ultra Robusto
Versión con logging detallado para debugging
Puerto: 5010
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import sys

app = Flask(__name__)

# CORS muy permisivo
CORS(app, 
     origins='*',
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type'])

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    print("✅ Health check recibido")
    return jsonify({
        'status': 'healthy',
        'service': 'smart-mcp',
        'port': 5010,
        'version': 'robusto'
    })

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Analiza query"""
    
    # Manejar OPTIONS
    if request.method == 'OPTIONS':
        print("✓ OPTIONS request")
        return '', 204
    
    print("\n" + "="*50)
    print("📥 Request POST recibido en /analyze")
    print("="*50)
    
    try:
        # Log headers
        print(f"Headers: {dict(request.headers)}")
        
        # Obtener datos
        print(f"Content-Type: {request.content_type}")
        print(f"Data raw: {request.data}")
        
        # Intentar parsear JSON
        try:
            data = request.get_json(force=True)
            print(f"✓ JSON parseado: {data}")
        except Exception as json_error:
            print(f"❌ Error parseando JSON: {json_error}")
            return jsonify({
                'error': f'Invalid JSON: {str(json_error)}',
                'received_data': str(request.data)
            }), 400
        
        # Obtener query
        query = data.get('query', '') if data else ''
        
        if not query:
            print("⚠️ Query vacío")
            return jsonify({'error': 'Query is required'}), 400
        
        print(f"✅ Query: {query[:100]}")
        
        # Respuesta simple
        result = {
            'needs_context': False,
            'original_query': query,
            'augmented_prompt': query,
            'contexts_added': 0,
            'lightweight': True
        }
        
        print(f"✅ Enviando respuesta OK")
        print("="*50 + "\n")
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO:")
        print(f"Tipo: {type(e).__name__}")
        print(f"Mensaje: {str(e)}")
        traceback.print_exc()
        print("="*50 + "\n")
        
        return jsonify({
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Smart MCP Server - Versión Robusta con Logging")
    print("=" * 60)
    print("📌 Puerto: 5010")
    print("📊 Endpoints:")
    print("   GET  /health  - Health check")
    print("   POST /analyze - Analizar query")
    print("=" * 60)
    print("")
    
    app.run(
        host='0.0.0.0',
        port=5010,
        debug=True,
        use_reloader=False  # Desactivar reloader para evitar problemas
    )


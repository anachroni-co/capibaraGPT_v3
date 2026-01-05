"""
Smart MCP Server - Versión Simple Sin Errores
Solo devuelve query original (sin contexto por ahora)
Puerto: 5010
"""
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins='*')

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'smart-mcp-simple',
        'version': '1.0',
        'port': 5010
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analiza query - versión simple"""
    try:
        # Obtener datos
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data'}), 400
        
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        print(f"📝 Query recibida: {query[:100]}")
        
        # Devolver sin contexto adicional (funcional básico)
        result = {
            'needs_context': False,
            'original_query': query,
            'augmented_prompt': query,
            'contexts_added': 0,
            'lightweight': True
        }
        
        print(f"✅ Respuesta enviada")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Smart MCP Server - Versión Simple")
    print("=" * 60)
    print("📌 Puerto: 5010")
    print("📊 Modo: Passthrough (sin contexto adicional)")
    print("✅ Sin dependencias complejas")
    print("=" * 60)
    print("")
    
    app.run(
        host='0.0.0.0',
        port=5010,
        debug=False
    )


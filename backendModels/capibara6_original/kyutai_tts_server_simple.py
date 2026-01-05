"""
Servidor TTS Simplificado para VM
Devuelve fallback hasta que se implemente Kyutai correctamente
El frontend usará Web Speech API mientras tanto
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app, origins='*')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'service': 'tts-fallback-server',
        'status': 'healthy',
        'mode': 'fallback',
        'message': 'Kyutai TTS API en investigación - usando Web Speech API en frontend'
    })

@app.route('/tts', methods=['POST'])
def tts():
    """Endpoint que devuelve fallback para usar Web Speech API"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', 'es')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        print(f"📝 Request TTS: {len(text)} chars, lang={language}")
        print(f"⚠️  Devolviendo fallback - frontend usará Web Speech API")
        
        # Devolver fallback para que el frontend use Web Speech API
        result = {
            'fallback': True,
            'provider': 'Web Speech API (fallback)',
            'message': 'Kyutai TTS API en desarrollo',
            'characters': len(text),
            'language': language
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            'error': str(e),
            'fallback': True
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🎙️  TTS FALLBACK SERVER - Capibara6")
    print("=" * 60)
    print("⚠️  NOTA: Kyutai TTS API aún en investigación")
    print("💡 El frontend usará Web Speech API mientras tanto")
    print("🔗 Ver: KYUTAI_TTS_PENDIENTE.md para más información")
    print("=" * 60)
    print("")
    print("🌐 Iniciando servidor Flask en puerto 5002...")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5002,
        debug=False,
        threaded=True
    )


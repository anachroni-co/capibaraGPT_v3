# coqui_tts_server.py
# Servidor TTS basado en Coqui para el proyecto Capibara6

from flask import Flask, request, jsonify, send_file
import io
import logging
import os

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/api/tts/speak', methods=['POST'])
def tts_speak():
    try:
        data = request.get_json()
        text = data.get('text', '')
        voice = data.get('voice', 'default')
        
        # Simulación de generación de audio con Coqui TTS
        # En una implementación real, aquí se llamaría a la API de Coqui
        logger.info(f"Generando TTS para texto: {text[:50]}...")
        
        # Simular la generación de un archivo de audio
        # (en la implementación real, esto sería el archivo de audio real generado por Coqui)
        result = {
            'status': 'success',
            'message': 'Audio generado exitosamente',
            'text_processed': text,
            'voice_used': voice,
            'audio_url': '/api/tts/audio/mock'  # URL simulada
        }
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error en TTS: {e}")
        return jsonify({'error': 'Error en el servicio TTS'}), 500

@app.route('/api/tts/audio/mock', methods=['GET'])
def mock_audio():
    # En una implementación real, esto serviría el archivo de audio generado
    return jsonify({'status': 'mock audio endpoint'})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'coqui_tts_server',
        'tts_engine': 'coqui'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5004))  # Puerto 5004 para evitar conflicto con consensus
    app.run(host='0.0.0.0', port=port, debug=False)
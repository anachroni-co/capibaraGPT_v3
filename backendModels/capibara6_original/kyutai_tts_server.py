"""
Servidor Kyutai TTS para VM
Puerto: 5001
Optimizado para producción con GPU support
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import torch
import sys
import os

app = Flask(__name__)
CORS(app, origins=[
    'http://localhost:5500',
    'http://127.0.0.1:5500',
    'http://localhost:8000',
    'http://127.0.0.1:8000',
    'https://capibara6.vercel.app',
    'https://capibara6-kpdtkkw9k-anachroni.vercel.app',
    'https://*.vercel.app',
    '*'
])

# Configuración del modelo
KYUTAI_CONFIG = {
    'model_repo': 'kyutai/tts-1b-en_es',  # Modelo 1B multilingüe
    'sample_rate': 24000,
    'temperature': 0.7,
    'top_p': 0.9,
    'max_chars': 3000,
}

# Cache del modelo
_model_cache = None
_model_loading = False

def load_model():
    """Carga el modelo Kyutai TTS con manejo de errores robusto"""
    global _model_cache, _model_loading
    
    if _model_cache is not None:
        return _model_cache
    
    if _model_loading:
        raise Exception("Modelo ya está cargándose en otro thread")
    
    _model_loading = True
    
    try:
        print(f"📦 Cargando modelo Kyutai: {KYUTAI_CONFIG['model_repo']}")
        print(f"🔍 CUDA disponible: {torch.cuda.is_available()}")
        
        # Verificar memoria GPU si disponible
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Importar moshi
        try:
            from moshi import models
        except ImportError:
            print("❌ Error: biblioteca 'moshi' no encontrada")
            print("💡 Ejecutar: pip install moshi>=0.2.6")
            raise Exception("Moshi library not installed. Run: pip install moshi>=0.2.6")
        
        # Seleccionar device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Usando device: {device}")
        
        # Cargar modelo
        model = models.load_tts_model(
            hf_repo=KYUTAI_CONFIG['model_repo'],
            device=device
        )
        
        _model_cache = model
        _model_loading = False
        
        print(f"✅ Modelo Kyutai cargado exitosamente en {device}")
        print(f"📊 Sample rate: {KYUTAI_CONFIG['sample_rate']} Hz")
        return model
        
    except Exception as e:
        _model_loading = False
        print(f"❌ Error cargando modelo: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def synthesize_audio(text, language='es'):
    """Sintetiza texto a audio con Kyutai"""
    try:
        # Limitar caracteres
        if len(text) > KYUTAI_CONFIG['max_chars']:
            print(f"⚠️ Texto truncado de {len(text)} a {KYUTAI_CONFIG['max_chars']} caracteres")
            text = text[:KYUTAI_CONFIG['max_chars']]
        
        print(f"🎙️ Sintetizando: {len(text)} caracteres, idioma={language}")
        
        # Cargar modelo
        model = load_model()
        
        # Sintetizar
        audio_output = model.synthesize(
            text=text,
            language=language,
            temperature=KYUTAI_CONFIG['temperature'],
            top_p=KYUTAI_CONFIG['top_p']
        )
        
        # Convertir a WAV en memoria
        import io
        import soundfile as sf
        
        audio_buffer = io.BytesIO()
        sf.write(
            audio_buffer,
            audio_output,
            KYUTAI_CONFIG['sample_rate'],
            format='WAV'
        )
        audio_buffer.seek(0)
        audio_data = audio_buffer.read()
        
        print(f"✅ Audio generado: {len(audio_data)} bytes")
        return audio_data
        
    except Exception as e:
        print(f"❌ Error en síntesis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        model_loaded = _model_cache is not None
        
        return jsonify({
            'service': 'kyutai-tts',
            'status': 'healthy',
            'model': KYUTAI_CONFIG['model_repo'],
            'device': device,
            'gpu': gpu_name,
            'model_loaded': model_loaded,
            'sample_rate': KYUTAI_CONFIG['sample_rate'],
            'max_chars': KYUTAI_CONFIG['max_chars']
        })
    except Exception as e:
        return jsonify({
            'service': 'kyutai-tts',
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/tts', methods=['POST'])
def tts():
    """Endpoint principal de TTS"""
    try:
        # Obtener datos del request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text = data.get('text', '')
        language = data.get('language', 'es')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        print(f"📝 Request TTS: {len(text)} chars, lang={language}")
        
        # Sintetizar
        audio_data = synthesize_audio(text, language)
        
        # Convertir a base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        result = {
            'audioContent': audio_base64,
            'provider': 'Kyutai DSM TTS',
            'model': KYUTAI_CONFIG['model_repo'],
            'language': language,
            'characters': len(text),
            'sample_rate': KYUTAI_CONFIG['sample_rate'],
            'format': 'wav',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        print(f"✅ TTS exitoso")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error TTS: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': str(e),
            'fallback': True,
            'provider': 'Kyutai DSM TTS (error)'
        }), 500

@app.route('/preload', methods=['POST'])
def preload():
    """Pre-cargar modelo (útil para warmup)"""
    try:
        model = load_model()
        return jsonify({
            'status': 'success',
            'message': 'Modelo cargado exitosamente',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🎙️  KYUTAI TTS SERVER - Capibara6")
    print("=" * 60)
    print(f"📦 Modelo: {KYUTAI_CONFIG['model_repo']}")
    print(f"🔊 Sample rate: {KYUTAI_CONFIG['sample_rate']} Hz")
    print(f"📝 Max caracteres: {KYUTAI_CONFIG['max_chars']}")
    print(f"🔧 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    
    # Pre-cargar modelo al iniciar (opcional, comentar si quieres lazy loading)
    print("\n🚀 Pre-cargando modelo...")
    try:
        load_model()
        print("✅ Modelo pre-cargado exitosamente\n")
    except Exception as e:
        print(f"⚠️ No se pudo pre-cargar el modelo: {str(e)}")
        print("💡 Se cargará en el primer request\n")
    
    print("🌐 Iniciando servidor Flask en puerto 5001...")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=False,
        threaded=True
    )


"""
Servidor TTS con Coqui XTTS v2 - Con Clonación de Voz
Características:
- 3 voces predefinidas (2 mujeres, 1 hombre)
- Clonación de voz desde audio
- Multilingüe (16+ idiomas)
Puerto: 5002
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
import io
import sys
import os
import tempfile
import json
from pathlib import Path

app = Flask(__name__)
CORS(app, origins='*')

# Configuración del modelo
COQUI_CONFIG = {
    'model_name': 'tts_models/multilingual/multi-dataset/xtts_v2',
    'sample_rate': 24000,
    'max_chars': 3000,
    'speed': 1.0,
    'language': 'es',  # Español por defecto
}

# Directorio para voces de referencia
VOICES_DIR = Path(__file__).parent / 'voices_reference'
VOICES_DIR.mkdir(exist_ok=True)

# Voces predefinidas (speaker embeddings de XTTS v2)
PREDEFINED_VOICES = {
    'sofia': {
        'name': 'Sofía',
        'gender': 'female',
        'description': 'Voz femenina cálida y profesional',
        'language': 'es',
        'speaker_embedding': 'Claribel Dervla',  # Speaker de XTTS v2
    },
    'ana': {
        'name': 'Ana',
        'gender': 'female', 
        'description': 'Voz femenina joven y amigable',
        'language': 'es',
        'speaker_embedding': 'Daisy Studious',  # Speaker de XTTS v2
    },
    'carlos': {
        'name': 'Carlos',
        'gender': 'male',
        'description': 'Voz masculina clara y firme',
        'language': 'es',
        'speaker_embedding': 'Gilberto Mathias',  # Speaker de XTTS v2
    }
}

# Cache del modelo
_tts_model = None
_model_loading = False
_custom_voices = {}  # Voces clonadas por usuarios

def load_coqui_model():
    """Carga el modelo Coqui TTS"""
    global _tts_model, _model_loading
    
    if _tts_model is not None:
        return _tts_model
    
    if _model_loading:
        raise Exception("Modelo ya está cargándose")
    
    _model_loading = True
    
    try:
        print(f"📦 Cargando modelo XTTS v2...")
        
        from TTS.api import TTS
        
        # Cargar modelo con GPU si está disponible
        tts = TTS(model_name=COQUI_CONFIG['model_name'], progress_bar=True)
        
        _tts_model = tts
        _model_loading = False
        
        print("✅ Modelo XTTS v2 cargado exitosamente")
        print(f"   Idiomas soportados: {', '.join(tts.languages)}")
        print(f"   Speakers disponibles: {len(tts.speakers) if hasattr(tts, 'speakers') else 'N/A'}")
        
        return tts
        
    except Exception as e:
        _model_loading = False
        print(f"❌ Error al cargar modelo: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        tts = load_coqui_model()
        return jsonify({
            'status': 'healthy',
            'model': 'xtts_v2',
            'service': 'coqui-tts',
            'features': ['voice_cloning', 'multilingual', 'custom_voices'],
            'predefined_voices': len(PREDEFINED_VOICES),
            'custom_voices': len(_custom_voices),
            'languages': tts.languages if hasattr(tts, 'languages') else []
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 503

@app.route('/voices', methods=['GET'])
def list_voices():
    """Lista todas las voces disponibles (predefinidas + clonadas)"""
    try:
        voices = {
            'predefined': PREDEFINED_VOICES,
            'custom': {
                voice_id: {
                    'name': info['name'],
                    'created_at': info.get('created_at', 'unknown')
                }
                for voice_id, info in _custom_voices.items()
            }
        }
        
        return jsonify({
            'status': 'success',
            'voices': voices,
            'total_predefined': len(PREDEFINED_VOICES),
            'total_custom': len(_custom_voices)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/clone', methods=['POST'])
def clone_voice():
    """
    Clona una voz desde un archivo de audio
    Espera: multipart/form-data con 'audio' y 'name'
    """
    try:
        if 'audio' not in request.files:
            return jsonify({
                'status': 'error',
                'error': 'No se encontró archivo de audio'
            }), 400
        
        audio_file = request.files['audio']
        voice_name = request.form.get('name', 'Custom Voice')
        
        # Guardar audio temporal
        temp_path = VOICES_DIR / f"custom_{len(_custom_voices)}_{audio_file.filename}"
        audio_file.save(temp_path)
        
        # Generar ID único
        voice_id = f"custom_{len(_custom_voices) + 1}"
        
        # Guardar info de la voz
        _custom_voices[voice_id] = {
            'name': voice_name,
            'audio_path': str(temp_path),
            'created_at': __import__('datetime').datetime.now().isoformat()
        }
        
        print(f"✅ Voz clonada: {voice_name} (ID: {voice_id})")
        
        return jsonify({
            'status': 'success',
            'voice_id': voice_id,
            'name': voice_name,
            'message': 'Voz clonada exitosamente'
        })
        
    except Exception as e:
        print(f"❌ Error al clonar voz: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/tts', methods=['POST', 'OPTIONS'])
def synthesize():
    """
    Genera audio TTS
    Body: {
        "text": "texto a sintetizar",
        "language": "es" (opcional),
        "voice_id": "sofia" (opcional, predefinida o clonada)
    }
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Parsear request
        data = request.get_json(force=True)
        text = data.get('text', '').strip()
        language = data.get('language', COQUI_CONFIG['language'])
        voice_id = data.get('voice_id', 'sofia')  # Por defecto: Sofia
        
        if not text:
            return jsonify({
                'status': 'error',
                'error': 'No se proporcionó texto'
            }), 400
        
        # Límite de caracteres
        if len(text) > COQUI_CONFIG['max_chars']:
            text = text[:COQUI_CONFIG['max_chars']]
        
        print(f"\n🎙️ TTS Request:")
        print(f"   Texto: {text[:50]}...")
        print(f"   Idioma: {language}")
        print(f"   Voz: {voice_id}")
        
        # Cargar modelo
        tts = load_coqui_model()
        
        # Determinar speaker/audio de referencia
        speaker_wav = None
        speaker_name = None
        
        if voice_id in PREDEFINED_VOICES:
            # Usar voz predefinida
            voice_info = PREDEFINED_VOICES[voice_id]
            speaker_name = voice_info['speaker_embedding']
            print(f"   Usando voz predefinida: {voice_info['name']}")
            
        elif voice_id in _custom_voices:
            # Usar voz clonada
            voice_info = _custom_voices[voice_id]
            speaker_wav = voice_info['audio_path']
            print(f"   Usando voz clonada: {voice_info['name']}")
            
        else:
            # Voz por defecto
            speaker_name = PREDEFINED_VOICES['sofia']['speaker_embedding']
            print(f"   Voz no encontrada, usando Sofia por defecto")
        
        # Generar audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # XTTS v2: tts_to_file con speaker_wav o speaker
            if speaker_wav:
                tts.tts_to_file(
                    text=text,
                    file_path=tmp_path,
                    speaker_wav=speaker_wav,
                    language=language,
                    speed=COQUI_CONFIG['speed']
                )
            else:
                tts.tts_to_file(
                    text=text,
                    file_path=tmp_path,
                    speaker=speaker_name,
                    language=language,
                    speed=COQUI_CONFIG['speed']
                )
            
            # Leer audio generado
            with open(tmp_path, 'rb') as f:
                audio_data = f.read()
            
            # Limpiar archivo temporal
            os.unlink(tmp_path)
            
            # Convertir a base64
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            print(f"✅ Audio generado: {len(audio_data)} bytes")
            
            return jsonify({
                'status': 'success',
                'audio': audio_b64,
                'format': 'wav',
                'sample_rate': COQUI_CONFIG['sample_rate'],
                'model': 'xtts_v2',
                'voice': voice_id,
                'language': language,
                'text_length': len(text)
            })
            
        except Exception as e:
            # Limpiar archivo temporal si existe
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        
    except Exception as e:
        print(f"❌ Error en TTS: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/test-voice', methods=['POST'])
def test_voice():
    """
    Prueba una voz con una frase de ejemplo
    Body: {"voice_id": "sofia"}
    """
    try:
        data = request.get_json(force=True)
        voice_id = data.get('voice_id', 'sofia')
        
        # Texto de prueba
        test_texts = {
            'es': '¡Hola! Esta es una demostración de mi voz.',
            'en': 'Hello! This is a demonstration of my voice.',
            'fr': 'Bonjour! Ceci est une démonstration de ma voix.',
        }
        
        language = data.get('language', 'es')
        text = test_texts.get(language, test_texts['es'])
        
        # Reutilizar endpoint de TTS
        return synthesize()
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🎙️  COQUI TTS SERVER - Capibara6")
    print("=" * 60)
    print(f"📦 Modelo: XTTS v2 (Máxima Calidad + Clonación)")
    print(f"🌍 Multilingüe: 16+ idiomas disponibles")
    print(f"🔊 Sample rate: {COQUI_CONFIG['sample_rate']} Hz")
    print(f"📝 Max caracteres: {COQUI_CONFIG['max_chars']}")
    print(f"🌐 Idioma por defecto: {COQUI_CONFIG['language']}")
    print(f"✨ Características: Clonación de voz + 3 voces predefinidas")
    print(f"👥 Voces predefinidas: {', '.join(PREDEFINED_VOICES.keys())}")
    print("=" * 60)
    
    # Pre-cargar modelo al iniciar (recomendado)
    print("\n🚀 Pre-cargando modelo Coqui TTS...")
    try:
        load_coqui_model()
        print("✅ Modelo pre-cargado. Servidor listo.\n")
    except Exception as e:
        print(f"⚠️  No se pudo pre-cargar el modelo: {e}")
        print("💡 Se cargará en el primer request\n")
    
    print("🌐 Iniciando servidor Flask en puerto 5002...")
    app.run(
        host='0.0.0.0',
        port=5002,
        debug=False,
        threaded=True
    )


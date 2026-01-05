#!/usr/bin/env python3
"""
Servidor de Consenso - Capibara6
Maneja múltiples modelos y consenso entre respuestas
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from models_config import (
    MODELS_CONFIG, CONSENSUS_CONFIG, PROMPT_TEMPLATES,
    get_active_models, get_model_config, format_prompt,
    get_system_info
)

app = Flask(__name__)
CORS(app, origins=['http://localhost:8000', 'http://127.0.0.1:8000', 'http://localhost:5500', 'http://127.0.0.1:5500'])

# ============================================
# CLASE DE CONSENSO
# ============================================

class ModelConsensus:
    def __init__(self):
        self.active_models = get_active_models()
        self.consensus_config = CONSENSUS_CONFIG
    
    async def query_model(self, session: aiohttp.ClientSession, model_id: str, 
                         prompt: str, template_id: str = 'general') -> Dict[str, Any]:
        """Consulta un modelo específico"""
        model_config = get_model_config(model_id)
        if not model_config:
            return {'error': f'Modelo {model_id} no encontrado'}
        
        try:
            # Formatear prompt
            formatted_prompt = format_prompt(model_id, template_id, prompt)
            
            # Preparar parámetros
            params = {
                'prompt': formatted_prompt,
                **model_config['parameters']
            }
            
            # Realizar consulta
            start_time = time.time()
            async with session.post(
                model_config['server_url'],
                json=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    duration = time.time() - start_time
                    
                    return {
                        'model_id': model_id,
                        'response': result.get('content', ''),
                        'duration': duration,
                        'tokens_generated': result.get('tokens_generated', 0),
                        'tokens_evaluated': result.get('tokens_evaluated', 0),
                        'success': True
                    }
                else:
                    return {
                        'model_id': model_id,
                        'error': f'Error HTTP {response.status}',
                        'success': False
                    }
                    
        except asyncio.TimeoutError:
            return {
                'model_id': model_id,
                'error': 'Timeout',
                'success': False
            }
        except Exception as e:
            return {
                'model_id': model_id,
                'error': str(e),
                'success': False
            }
    
    async def get_consensus(self, prompt: str, template_id: str = 'general', 
                           models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Obtiene consenso entre múltiples modelos"""
        if models is None:
            models = self.active_models
        
        if len(models) < self.consensus_config['min_models']:
            # Si no hay suficientes modelos, usar solo uno
            models = [self.consensus_config['fallback_model']]
        
        # Consultar todos los modelos en paralelo
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.query_model(session, model_id, prompt, template_id)
                for model_id in models
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar resultados
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append({'error': str(result)})
            elif result.get('success', False):
                successful_results.append(result)
            else:
                failed_results.append(result)
        
        if not successful_results:
            return {
                'error': 'Todos los modelos fallaron',
                'failed_models': failed_results,
                'consensus': False
            }
        
        # Aplicar método de consenso
        if len(successful_results) == 1:
            # Solo un modelo exitoso
            return {
                'response': successful_results[0]['response'],
                'model_used': successful_results[0]['model_id'],
                'duration': successful_results[0]['duration'],
                'tokens_generated': successful_results[0]['tokens_generated'],
                'tokens_evaluated': successful_results[0]['tokens_evaluated'],
                'consensus': False,
                'models_queried': len(models),
                'successful_models': len(successful_results)
            }
        
        # Múltiples modelos exitosos - aplicar consenso
        if self.consensus_config['voting_method'] == 'weighted':
            return self._weighted_consensus(successful_results)
        else:
            return self._simple_consensus(successful_results)
    
    def _weighted_consensus(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consenso ponderado basado en pesos de modelos"""
        weights = self.consensus_config['model_weights']
        
        # Calcular peso total
        total_weight = sum(weights.get(r['model_id'], 0.5) for r in results)
        
        # Seleccionar respuesta con mayor peso
        best_result = max(results, key=lambda r: weights.get(r['model_id'], 0.5))
        
        return {
            'response': best_result['response'],
            'model_used': best_result['model_id'],
            'duration': best_result['duration'],
            'tokens_generated': best_result['tokens_generated'],
            'tokens_evaluated': best_result['tokens_evaluated'],
            'consensus': True,
            'consensus_method': 'weighted',
            'models_queried': len(results),
            'successful_models': len(results),
            'total_weight': total_weight,
            'selected_weight': weights.get(best_result['model_id'], 0.5)
        }
    
    def _simple_consensus(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consenso simple - selecciona la primera respuesta exitosa"""
        selected = results[0]  # Por ahora, seleccionar la primera
        
        return {
            'response': selected['response'],
            'model_used': selected['model_id'],
            'duration': selected['duration'],
            'tokens_generated': selected['tokens_generated'],
            'tokens_evaluated': selected['tokens_evaluated'],
            'consensus': True,
            'consensus_method': 'simple',
            'models_queried': len(results),
            'successful_models': len(results)
        }

# ============================================
# INSTANCIA GLOBAL
# ============================================

consensus = ModelConsensus()

# ============================================
# RUTAS API
# ============================================

@app.route('/api/consensus/query', methods=['POST'])
async def query_consensus():
    """Endpoint principal para consultas con consenso"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        template_id = data.get('template', 'general')
        models = data.get('models')  # Lista opcional de modelos específicos
        
        if not prompt:
            return jsonify({'error': 'Prompt requerido'}), 400
        
        # Obtener consenso
        result = await consensus.get_consensus(prompt, template_id, models)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/consensus/models', methods=['GET'])
def get_models():
    """Obtiene información de los modelos disponibles"""
    try:
        info = get_system_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/consensus/templates', methods=['GET'])
def get_templates():
    """Obtiene las plantillas de prompts disponibles"""
    try:
        return jsonify(PROMPT_TEMPLATES)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/consensus/config', methods=['GET'])
def get_config():
    """Obtiene la configuración del consenso"""
    try:
        return jsonify(CONSENSUS_CONFIG)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/consensus/health', methods=['GET'])
async def health_check():
    """Health check del sistema de consenso"""
    try:
        # Verificar conectividad con modelos
        health_status = {}
        
        async with aiohttp.ClientSession() as session:
            for model_id in get_active_models():
                model_config = get_model_config(model_id)
                try:
                    async with session.get(
                        model_config['server_url'].replace('/completion', '/health'),
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        health_status[model_id] = {
                            'status': 'healthy' if response.status == 200 else 'unhealthy',
                            'response_time': response.headers.get('X-Response-Time', 'N/A')
                        }
                except:
                    health_status[model_id] = {'status': 'unreachable'}
        
        return jsonify({
            'status': 'healthy',
            'consensus_enabled': CONSENSUS_CONFIG['enabled'],
            'active_models': len(get_active_models()),
            'models_health': health_status
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# FUNCIÓN PARA STREAMING
# ============================================

async def stream_consensus(prompt: str, template_id: str = 'general', 
                          models: Optional[List[str]] = None):
    """Streaming de consenso (para implementar más tarde)"""
    # Por ahora, devolver resultado normal
    result = await consensus.get_consensus(prompt, template_id, models)
    yield f"data: {json.dumps(result)}\n\n"

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("🤖 Iniciando Servidor de Consenso Capibara6...")
    print("=" * 50)
    
    info = get_system_info()
    print(f"Modelos activos: {info['active_models']}")
    print(f"Consenso habilitado: {info['consensus_enabled']}")
    
    for model_id in info['models_list']:
        config = get_model_config(model_id)
        print(f"  • {config['name']} ({config['hardware']})")
    
    print(f"\n🌐 Servidor ejecutándose en: http://localhost:5002")
    print("📋 Endpoints disponibles:")
    print("  • POST /api/consensus/query - Consulta con consenso")
    print("  • GET  /api/consensus/models - Información de modelos")
    print("  • GET  /api/consensus/templates - Plantillas de prompts")
    print("  • GET  /api/consensus/config - Configuración del consenso")
    print("  • GET  /api/consensus/health - Health check")
    
    app.run(host='0.0.0.0', port=5002, debug=True)

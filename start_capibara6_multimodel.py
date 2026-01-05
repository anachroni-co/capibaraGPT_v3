#!/usr/bin/env python3
"""
Script para iniciar el servidor vLLM multi-modelo Capibara6 con sistema de consenso
Excluyendo el modelo problemático (GPT-OSS-20B) que causaba errores
"""
import os
import sys
import json
import subprocess
import time
import signal
import atexit
from pathlib import Path

# Asegurar que vllm-source-modified esté en el path
sys.path.insert(0, '/home/elect/capibara6/vllm-source-modified')

# Variables de entorno para ARM-Axion
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_ENABLE_V1_ENGINE"] = "0"
os.environ["TORCHINDUCTOR_DISABLED"] = "1"
os.environ["TORCH_COMPILE_BACKEND"] = "eager"
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"

def load_config():
    """Cargar la configuración del servidor"""
    config_path = Path("/home/elect/capibara6/capibara6_config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Configuración por defecto
        return {
            "models": {
                "fast_response": {
                    "name": "phi-4-mini",
                    "local_path": "/home/elect/models/phi-4-mini",
                    "dtype": "float16",
                    "quantization": "awq"
                },
                "balanced": {
                    "name": "qwen2.5-coder-1.5b",
                    "local_path": "/home/elect/models/qwen2.5-coder-1.5b",
                    "dtype": "float16",
                    "quantization": "awq"
                },
                "general": {
                    "name": "mistral-7b-instruct-v0.2",
                    "local_path": "/home/elect/models/mistral-7b-instruct-v0.2",
                    "dtype": "float16",
                    "quantization": "awq"
                },
                "multimodal": {
                    "name": "gemma-3-27b-it-awq",
                    "local_path": "/home/elect/models/gemma-3-27b-it-awq",
                    "dtype": "bfloat16",
                    "quantization": None
                }
            },
            "consensus_enabled": True,
            "api_settings": {
                "default_model": "phi-4-mini",
                "port": 8080
            }
        }

def check_model_exists(model_config):
    """Verificar si un modelo existe en el sistema"""
    local_path = model_config.get('local_path', '/home/elect/models/unknown')
    path = Path(local_path)
    return path.exists() and any(path.iterdir())

def start_vllm_server(model_name, model_path, port, dtype="float16", quantization="awq"):
    """Iniciar un servidor vLLM para un modelo específico"""
    print(f"🚀 Iniciando servidor vLLM para {model_name} en puerto {port}...")

    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", dtype,
        "--api-key", "EMPTY"
    ]

    # Agregar opción de cuantización si está configurada
    if quantization:
        cmd.extend(["--quantization", quantization])

    # Para ARM-Axion en CPU, usar settings apropiados
    cmd.extend([
        "--tensor-parallel-size", "1",  # Solo 1 GPU/CPU
        "--max-model-len", "4096"  # Limitar longitud del modelo para evitar errores de memoria
    ])

    print(f"Comando: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(cmd)
        print(f"✅ Servidor {model_name} iniciado con PID {process.pid}")
        return process
    except Exception as e:
        print(f"❌ Error iniciando servidor {model_name}: {e}")
        return None

def start_consensus_server():
    """Iniciar el servidor de consenso"""
    print("🚀 Iniciando servidor de consenso...")
    
    # Buscar el servidor de consenso en el sistema
    consensus_server_paths = [
        "/home/elect/capibara6/backendModels/capibara6_original/consensus_server.py",
        "/home/elect/capibara6/vm-bounty2/servers/consensus_server.py",
        "/home/elect/capibara6/archived/legacy_backend/consensus_server.py",
        "/home/elect/capibara6/backend/consensus_server.py"
    ]
    
    server_path = None
    for path in consensus_server_paths:
        if Path(path).exists():
            server_path = path
            break
    
    if not server_path:
        print("❌ No se encontró el servidor de consenso")
        # Crear un servidor de consenso básico
        create_basic_consensus_server()
        server_path = "/home/elect/capibara6/basic_consensus_server.py"
    
    try:
        process = subprocess.Popen([
            "python3", server_path,
            "--port", "5005"
        ])
        print(f"✅ Servidor de consenso iniciado con PID {process.pid}")
        return process
    except Exception as e:
        print(f"❌ Error iniciando servidor de consenso: {e}")
        return None

def create_basic_consensus_server():
    """Crear un servidor de consenso básico si no se encuentra"""
    print("🔧 Creando servidor de consenso básico...")
    
    basic_server_content = '''
import json
import time
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
import requests

app = Flask(__name__)

# Configuración de modelos
models = {
    "phi-4-mini": {"url": "http://localhost:8081/v1", "weight": 0.7},
    "qwen2.5-coder-1.5b": {"url": "http://localhost:8082/v1", "weight": 0.8}, 
    "mistral-7b-instruct-v0.2": {"url": "http://localhost:8083/v1", "weight": 0.6},
    "gemma-3-27b-it-awq": {"url": "http://localhost:8084/v1", "weight": 0.9}
}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "consensus"})

@app.route('/api/consensus/query', methods=['POST'])
def consensus_query():
    data = request.json
    query = data.get("prompt", "")
    models_to_use = data.get("models", list(models.keys()))
    
    # Obtener respuestas de múltiples modelos
    responses = []
    
    def query_model(model_name):
        try:
            model_info = models[model_name]
            url = f"{model_info['url']}/chat/completions"
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": query}],
                "max_tokens": 512,
                "temperature": 0.7
            }
            
            response = requests.post(url, json=payload, headers={"Authorization": "Bearer EMPTY"})
            if response.status_code == 200:
                result = response.json()
                return {
                    "model": model_name,
                    "response": result["choices"][0]["message"]["content"],
                    "weight": model_info["weight"]
                }
        except Exception as e:
            print(f"Error querying {model_name}: {e}")
        return None

    with ThreadPoolExecutor(max_workers=len(models_to_use)) as executor:
        futures = [executor.submit(query_model, model) for model in models_to_use]
        for future in futures:
            result = future.result()
            if result:
                responses.append(result)
    
    if not responses:
        return jsonify({"error": "No models responded"}), 500
    
    # Aplicar consenso ponderado
    weighted_response = apply_weighted_consensus(responses)
    
    return jsonify({
        "consensus_response": weighted_response,
        "individual_responses": responses,
        "consensus_applied": True
    })

def apply_weighted_consensus(responses):
    """Aplicar consenso ponderado basado en pesos de modelos"""
    # Para simplificar, devolver la respuesta del modelo con mayor peso
    highest_weight_resp = max(responses, key=lambda x: x["weight"])
    return highest_weight_resp["response"]

@app.route('/api/consensus/models', methods=['GET'])
def get_models():
    return jsonify({"models": list(models.keys())})

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5005
    app.run(host="0.0.0.0", port=port, debug=False)
'''
    
    with open("/home/elect/capibara6/basic_consensus_server.py", "w") as f:
        f.write(basic_server_content)
    print("✅ Servidor de consenso básico creado")

def main():
    print("🚀 Iniciando sistema multi-modelo Capibara6 con consenso...")
    
    # Cargar configuración
    config = load_config()
    print(f"📋 Configuración cargada: {len(config['models'])} modelos")
    
    # Verificar modelos
    available_models = {}
    for model_key, model_config in config['models'].items():
        if check_model_exists(model_config):
            available_models[model_key] = model_config
            print(f"✅ Modelo {model_key} disponible: {model_config['name']}")
        else:
            print(f"❌ Modelo {model_key} no encontrado: {model_config['name']}")
    
    if not available_models:
        print("❌ No hay modelos disponibles para iniciar")
        return
    
    # Asignar puertos dinámicos para cada modelo
    processes = []
    port = 8081  # Comenzar desde 8081 para no interferir con el puerto principal
    
    # Iniciar servidores vLLM para cada modelo
    for model_key, model_config in available_models.items():
        process = start_vllm_server(
            model_name=model_config['name'],
            model_path=model_config['local_path'],
            port=port,
            dtype=model_config.get('dtype', 'float16'),
            quantization=model_config.get('quantization', 'awq')
        )
        if process:
            processes.append(process)
            port += 1  # Asignar siguiente puerto
    
    # Iniciar servidor de consenso
    consensus_process = start_consensus_server()
    if consensus_process:
        processes.append(consensus_process)
    
    print(f"✅ {len(processes)} servicios iniciados correctamente")
    print("📋 Servicios disponibles:")
    print("   - vLLM servers: puerto 8081-8084 (o siguientes)")
    print("   - Consenso: puerto 5005")
    
    # Registrar función de limpieza
    def cleanup():
        print("\\n🛑 Deteniendo servicios...")
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
        print("✅ Servicios detenidos")
    
    atexit.register(cleanup)
    
    try:
        # Mantener el proceso principal vivo
        while True:
            time.sleep(1)
            # Verificar si los procesos están activos
            active_processes = [p for p in processes if p.poll() is None]
            if len(active_processes) != len(processes):
                print("⚠️ Algunos procesos se han detenido inesperadamente")
                break
    except KeyboardInterrupt:
        print("\\n Interrupción recibida, deteniendo servicios...")
        cleanup()

if __name__ == "__main__":
    main()
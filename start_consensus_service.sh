#!/bin/bash
# Script para iniciar el servidor de consenso para Capibara6

echo "🚀 INICIANDO SERVIDOR DE CONSENSO CAPIBARA6"
echo "   Conectado a modelos en http://localhost:8081"
echo ""

# Directorios
CAPIBARA_ROOT="/home/elect/capibara6"
BACKEND_DIR="$CAPIBARA_ROOT/vm-bounty2"  # Common location for backend services

# Variables de entorno
export PYTHONPATH="$CAPIBARA_ROOT/vllm-source-modified:$CAPIBARA_ROOT/arm-axion-optimizations:$PYTHONPATH"

# Puerto para el servidor de consenso
PORT=5005

echo "📍 Directorios:"
echo "   - Backend: $BACKEND_DIR"
echo ""

# Buscar y ejecutar el servidor de consenso
echo "🔍 Buscando servidor de consenso..."
if [ -f "$CAPIBARA_ROOT/backendModels/capibara6_original/consensus_server.py" ]; then
    CONSENSUS_SERVER="$CAPIBARA_ROOT/backendModels/capibara6_original/consensus_server.py"
    echo "✅ Encontrado en backendModels"
elif [ -f "$CAPIBARA_ROOT/vm-bounty2/servers/consensus_server.py" ]; then
    CONSENSUS_SERVER="$CAPIBARA_ROOT/vm-bounty2/servers/consensus_server.py"
    echo "✅ Encontrado en vm-bounty2"
elif [ -f "$CAPIBARA_ROOT/backend/consensus_server.py" ]; then
    CONSENSUS_SERVER="$CAPIBARA_ROOT/backend/consensus_server.py"
    echo "✅ Encontrado en backend"
elif [ -f "$CAPIBARA_ROOT/archived/legacy_backend/consensus_server.py" ]; then
    CONSENSUS_SERVER="$CAPIBARA_ROOT/archived/legacy_backend/consensus_server.py"
    echo "✅ Encontrado en archived (legacy)"
elif [ -f "$CAPIBARA_ROOT/basic_consensus_server.py" ]; then
    CONSENSUS_SERVER="$CAPIBARA_ROOT/basic_consensus_server.py"
    echo "✅ Usando servidor de consenso básico"
else
    echo "❌ No se encontró el servidor de consenso, creando uno básico..."
    cat > "$CAPIBARA_ROOT/consensus_server_simple.py" << 'EOF'
import json
import time
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

app = Flask(__name__)

# Configuración de modelos en el servidor multi-modelo
MULTI_MODEL_URL = "http://localhost:8081/v1"

models = {
    "phi4_fast": {"weight": 0.7},
    "mistral_balanced": {"weight": 0.6},
    "qwen_coder": {"weight": 0.8},
    "gemma3_multimodal": {"weight": 0.9}
}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy", 
        "service": "consensus",
        "connected_to": "http://localhost:8081"
    })

@app.route('/api/consensus/query', methods=['POST'])
def consensus_query():
    data = request.json
    query = data.get("prompt", data.get("query", ""))
    models_to_use = data.get("models", list(models.keys()))
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 512)
    
    if not query:
        return jsonify({"error": "Prompt/query is required"}), 400
    
    # Obtener respuestas de múltiples modelos
    responses = []
    
    def query_model(model_name):
        try:
            url = f"{MULTI_MODEL_URL}/chat/completions"
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": query}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            response = requests.post(
                url, 
                json=payload, 
                headers={
                    "Authorization": "Bearer EMPTY",
                    "Content-Type": "application/json"
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return {
                    "model": model_name,
                    "response": content,
                    "weight": models[model_name]["weight"],
                    "tokens": len(content.split())  # Estimación simple
                }
        except Exception as e:
            print(f"Error querying {model_name}: {e}")
        return None

    # Consultar modelos en paralelo
    with ThreadPoolExecutor(max_workers=len(models_to_use)) as executor:
        future_to_model = {executor.submit(query_model, model): model for model in models_to_use}
        
        for future in as_completed(future_to_model):
            result = future.result()
            if result:
                responses.append(result)
    
    if not responses:
        return jsonify({"error": "No models responded successfully"}), 500
    
    # Aplicar consenso ponderado
    consensus_response = apply_weighted_consensus(responses)
    
    return jsonify({
        "consensus_response": consensus_response,
        "individual_responses": responses,
        "consensus_applied": True,
        "total_models_queried": len(models_to_use),
        "successful_responses": len(responses)
    })

def apply_weighted_consensus(responses):
    """Aplicar consenso ponderado basado en pesos de modelos"""
    # Para simplificar, devolver la respuesta del modelo con mayor peso
    highest_weight_resp = max(responses, key=lambda x: x["weight"])
    return highest_weight_resp["response"]

@app.route('/api/consensus/models', methods=['GET'])
def get_models():
    return jsonify({
        "models": list(models.keys()),
        "model_weights": {k: v["weight"] for k, v in models.items()}
    })

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5005
    app.run(host="0.0.0.0", port=port, debug=False)
EOF
    CONSENSUS_SERVER="$CAPIBARA_ROOT/consensus_server_simple.py"
    echo "✅ Servidor de consenso básico creado y listo para usar"
fi

echo ""
echo "🌐 Iniciando servidor de consenso en puerto $PORT..."
echo "   Conectando con: http://localhost:8081"
echo "   Accede a: http://localhost:$PORT"
echo "   Para detener: Ctrl+C"
echo ""

# Iniciar el servidor de consenso
python3 "$CONSENSUS_SERVER" "$PORT"
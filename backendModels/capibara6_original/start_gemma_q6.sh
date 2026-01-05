#!/bin/bash

# Script para iniciar Gemma 2-12B con Q6_K (Balance Calidad/Velocidad)
# Uso: ./start_gemma_q6.sh

MODEL_PATH="/mnt/data/models/gemma-2-12b-it-Q6_K.gguf"
PORT=8080
CTX_SIZE=8192
THREADS=16

echo "============================================"
echo "  🚀 Gemma 2-12B Q6_K Server"
echo "============================================"
echo "📦 Modelo: Q6_K (Balance perfecto)"
echo "🔢 Context: $CTX_SIZE tokens"
echo "🧵 Threads: $THREADS"
echo "🌐 Puerto: $PORT"
echo "============================================"
echo ""

# Verificar que el modelo existe
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ ERROR: Modelo no encontrado en $MODEL_PATH"
    echo ""
    echo "📥 Descarga el modelo con:"
    echo "huggingface-cli download bartowski/gemma-2-12b-it-GGUF gemma-2-12b-it-Q6_K.gguf --local-dir /mnt/data/models/ --local-dir-use-symlinks False"
    exit 1
fi

# Verificar que llama-server existe
if [ ! -f ~/llama.cpp/build/bin/llama-server ]; then
    echo "❌ ERROR: llama-server no encontrado"
    echo "Verifica que llama.cpp esté compilado en ~/llama.cpp"
    exit 1
fi

echo "✅ Modelo encontrado"
echo "✅ llama-server encontrado"
echo ""
echo "🚀 Iniciando servidor..."
echo ""

cd ~/llama.cpp

./build/bin/llama-server \
  --host 0.0.0.0 \
  --port $PORT \
  --model "$MODEL_PATH" \
  --ctx-size $CTX_SIZE \
  --n-threads $THREADS \
  --n-gpu-layers 0 \
  --flash-attn \
  --cont-batching \
  --metrics \
  --log-disable


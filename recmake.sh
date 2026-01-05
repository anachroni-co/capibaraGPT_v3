cat > ~/setup_gguf.sh << 'EOF'
#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Optimización Gemma 3 para ARM (GGUF/llama.cpp) - FIXED           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# 1. Instalar dependencias
echo "🚀 [1/5] Instalando dependencias..."
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-pip

# 2. Clonar llama.cpp (si no existe)
echo "🚀 [2/5] Verificando llama.cpp..."
cd ~/capibara6
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
fi
cd llama.cpp

# 3. Compilar con CMake (NUEVO MÉTODO)
echo "🚀 [3/5] Compilando llama.cpp con CMake..."
mkdir -p build
cd build
cmake .. -DGGML_NATIVE=ON  # Optimización nativa para ARM
cmake --build . --config Release -j$(nproc)
cd ..

# 4. Instalar dependencias Python
echo "🚀 [4/5] Instalando dependencias Python..."
pip3 install -r requirements.txt

# 5. Convertir y Quantizar
echo "🚀 [5/5] Convirtiendo Gemma 3 a GGUF Q4_K_M..."
echo "   Origen: /home/elect/models/gemma-3-27b-it (52GB)"
echo "   Destino: /home/elect/models/gemma-3-27b-it.Q4_K_M.gguf (~16GB)"

# Verificar script de conversión
if [ -f "convert_hf_to_gguf.py" ]; then
    CONVERT_SCRIPT="convert_hf_to_gguf.py"
else
    # Fallback para versiones muy nuevas/viejas
    CONVERT_SCRIPT="convert.py"
fi

echo "   Usando script: $CONVERT_SCRIPT"

# Convertir a FP16 primero
python3 $CONVERT_SCRIPT /home/elect/models/gemma-3-27b-it --outfile /home/elect/models/gemma-3-27b-it.fp16.gguf

# Localizar binario de quantización
if [ -f "./build/bin/llama-quantize" ]; then
    QUANTIZE_BIN="./build/bin/llama-quantize"
else
    QUANTIZE_BIN="./build/llama-quantize"
fi

# Quantizar a Q4_K_M
$QUANTIZE_BIN /home/elect/models/gemma-3-27b-it.fp16.gguf /home/elect/models/gemma-3-27b-it.Q4_K_M.gguf Q4_K_M

# Limpiar intermedio
rm /home/elect/models/gemma-3-27b-it.fp16.gguf

echo ""
echo "✅ ¡Conversión completada!"
echo "Modelo listo en: /home/elect/models/gemma-3-27b-it.Q4_K_M.gguf"
EOF

# Ejecutar script corregido
chmod +x ~/setup_gguf.sh
bash ~/setup_gguf.sh

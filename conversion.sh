cat > ~/setup_gguf.sh << 'EOF'
#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Optimización Gemma 3 para ARM (GGUF/llama.cpp)                   ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# 1. Instalar dependencias
echo "🚀 [1/5] Instalando dependencias..."
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-pip

# 2. Clonar llama.cpp
echo "🚀 [2/5] Clonando llama.cpp..."
cd ~/capibara6
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
fi
cd llama.cpp

# 3. Compilar (Optimizado para ARM NEON)
echo "🚀 [3/5] Compilando llama.cpp (esto tomará unos minutos)..."
make -j$(nproc)

# 4. Instalar dependencias Python
echo "🚀 [4/5] Instalando dependencias Python..."
pip3 install -r requirements.txt

# 5. Convertir y Quantizar
echo "🚀 [5/5] Convirtiendo Gemma 3 a GGUF Q4_K_M..."
echo "   Origen: /home/elect/models/gemma-3-27b-it (52GB)"
echo "   Destino: /home/elect/models/gemma-3-27b-it.Q4_K_M.gguf (~16GB)"
echo "   Esto puede tomar 10-20 minutos..."

# Convertir a FP16 primero
python3 convert_hf_to_gguf.py /home/elect/models/gemma-3-27b-it --outfile /home/elect/models/gemma-3-27b-it.fp16.gguf

# Quantizar a Q4_K_M (Balance perfecto velocidad/calidad)
./llama-quantize /home/elect/models/gemma-3-27b-it.fp16.gguf /home/elect/models/gemma-3-27b-it.Q4_K_M.gguf Q4_K_M

# Limpiar intermedio
rm /home/elect/models/gemma-3-27b-it.fp16.gguf

echo ""
echo "✅ ¡Conversión completada!"
echo "Modelo listo en: /home/elect/models/gemma-3-27b-it.Q4_K_M.gguf"
EOF

# Dar permisos y ejecutar
chmod +x ~/setup_gguf.sh
bash ~/setup_gguf.sh

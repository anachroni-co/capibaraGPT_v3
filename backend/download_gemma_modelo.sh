#!/bin/bash

# Script para descargar modelos optimizados de Gemma 2-12B
# Uso: ./download_gemma_modelo.sh [q8|q6|q5]

echo "============================================"
echo "  📥 Descargador de Modelos Gemma 2-12B"
echo "============================================"
echo ""

# Verificar argumento
if [ -z "$1" ]; then
    echo "❌ ERROR: Debes especificar qué modelo descargar"
    echo ""
    echo "Uso: ./download_gemma_modelo.sh [q8|q6|q5]"
    echo ""
    echo "Opciones:"
    echo "  q8  → Q8_0 (~13 GB) - Máxima calidad ⭐⭐⭐⭐⭐"
    echo "  q6  → Q6_K (~10 GB) - Balance perfecto ⭐⭐⭐⭐"
    echo "  q5  → Q5_K_M (~8.5 GB) - Ligero mejorado ⭐⭐⭐⭐"
    echo ""
    exit 1
fi

# Crear directorio si no existe
mkdir -p /mnt/data/models

# Verificar que huggingface-cli está instalado
if ! command -v huggingface-cli &> /dev/null; then
    echo "📦 Instalando huggingface-hub..."
    pip install huggingface-hub
fi

case "$1" in
    q8|Q8)
        MODEL_FILE="gemma-2-12b-it-Q8_0.gguf"
        SIZE="~13 GB"
        QUALITY="⭐⭐⭐⭐⭐ Máxima calidad"
        ;;
    q6|Q6)
        MODEL_FILE="gemma-2-12b-it-Q6_K.gguf"
        SIZE="~10 GB"
        QUALITY="⭐⭐⭐⭐ Balance perfecto"
        ;;
    q5|Q5)
        MODEL_FILE="gemma-2-12b-it-Q5_K_M.gguf"
        SIZE="~8.5 GB"
        QUALITY="⭐⭐⭐⭐ Ligero mejorado"
        ;;
    *)
        echo "❌ ERROR: Opción no válida: $1"
        echo "Usa: q8, q6 o q5"
        exit 1
        ;;
esac

echo "📦 Modelo seleccionado: $MODEL_FILE"
echo "💾 Tamaño aproximado: $SIZE"
echo "⭐ Calidad: $QUALITY"
echo "📁 Destino: /mnt/data/models/"
echo ""

# Verificar si ya existe
if [ -f "/mnt/data/models/$MODEL_FILE" ]; then
    echo "⚠️  El modelo ya existe:"
    ls -lh "/mnt/data/models/$MODEL_FILE"
    echo ""
    read -p "¿Descargar de nuevo? (s/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        echo "❌ Descarga cancelada"
        exit 0
    fi
fi

echo "📥 Iniciando descarga..."
echo "⏱️  Esto puede tardar 10-20 minutos..."
echo ""

huggingface-cli download \
  bartowski/gemma-2-12b-it-GGUF \
  "$MODEL_FILE" \
  --local-dir /mnt/data/models/ \
  --local-dir-use-symlinks False

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "  ✅ Descarga completada exitosamente"
    echo "============================================"
    echo ""
    echo "📁 Archivo: /mnt/data/models/$MODEL_FILE"
    ls -lh "/mnt/data/models/$MODEL_FILE"
    echo ""
    echo "🚀 Siguiente paso: Iniciar el servidor"
    
    case "$1" in
        q8|Q8)
            echo "   ./start_gemma_q8.sh"
            ;;
        q6|Q6)
            echo "   ./start_gemma_q6.sh"
            ;;
        q5|Q5)
            echo "   ./start_gemma_q5.sh"
            ;;
    esac
    echo ""
else
    echo ""
    echo "❌ ERROR: La descarga falló"
    echo ""
    echo "💡 Intenta de nuevo o usa wget:"
    echo "cd /mnt/data/models/"
    echo "wget https://huggingface.co/bartowski/gemma-2-12b-it-GGUF/resolve/main/$MODEL_FILE"
    exit 1
fi


#!/bin/bash
# Script para descargar el modelo Aya Expanse 8B con manejo de errores y autenticación

echo "🔐 Verificando autenticación de Hugging Face..."

# Verificar si está configurado el token de Hugging Face
if [ ! -f ~/.cache/huggingface/token ]; then
    echo "⚠️  No se encontró token de Hugging Face. Por favor, ejecuta:"
    echo "huggingface-cli login"
    echo "Y obtén acceso a https://huggingface.co/CohereLabs/aya-expanse-8b"
    exit 1
fi

echo "✅ Token de Hugging Face encontrado"

# Directorio de destino
MODEL_DIR="/home/elect/models/aya-expanse-8b"
mkdir -p "$MODEL_DIR"

echo "🔄 Iniciando descarga del modelo CohereLabs/aya-expanse-8b..."
echo "   Destino: $MODEL_DIR"

# Intentar descargar el modelo con autenticación
python3 -c "
import os
from huggingface_hub import snapshot_download, login
from huggingface_hub.utils import LocalTokenNotFoundError

print('Descargando modelo: CohereLabs/aya-expanse-8b')
try:
    result = snapshot_download(
        repo_id='CohereLabs/aya-expanse-8b',
        local_dir='$MODEL_DIR',
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=2,
        tqdm_class=None  # Desactivar barra de progreso para limpiar salida
    )
    print('✅ Descarga completada exitosamente')
    print(f'Archivo guardados en: {result}')
except Exception as e:
    print(f'❌ Error durante la descarga: {e}')
    print('💡 Posibles causas:')
    print('   - No tienes acceso al modelo (restringido)')
    print('   - Token de Hugging Face no tiene permisos')
    print('   - Problemas de conexión o red')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 ¡Modelo Aya Expanse 8B descargado exitosamente!"
    echo "   Ubicación: $MODEL_DIR"
    echo ""
    echo "📝 Contenido del modelo:"
    ls -la "$MODEL_DIR"
    
    # Verificar que los archivos del modelo estén presentes
    if [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/tokenizer.json" ]; then
        echo "✅ Archivos esenciales del modelo presentes"
        echo "✅ Modelo listo para integración"
    else
        echo "⚠️  Algunos archivos del modelo pueden faltar"
        echo "   Verifique el contenido antes de integrar al sistema"
    fi
else
    echo ""
    echo "❌ Falló la descarga del modelo"
    echo "💡 Por favor, asegúrate de tener acceso al repositorio:"
    echo "   https://huggingface.co/CohereLabs/aya-expanse-8b"
    echo ""
    echo "Para solicitar acceso:"
    echo "1. Visita: https://huggingface.co/CohereLabs/aya-expanse-8b"
    echo "2. Haz clic en 'Files and versions'"
    echo "3. Solicita acceso en la sección correspondiente"
    echo "4. Una vez otorgado, vuelve a ejecutar este script"
fi
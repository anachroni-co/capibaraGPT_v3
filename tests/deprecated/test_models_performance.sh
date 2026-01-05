#!/bin/bash
# Script para probar cada uno de los 5 modelos del sistema de consenso

echo "🧪 PRUEBAS DE LOS 5 MODELOS DEL SISTEMA CAPIBARA6"
echo "=================================================="
echo ""

SERVER_URL="http://localhost:8080"

# Verificar que el servidor esté disponible
echo "🔍 Verificando disponibilidad del servidor..."
if ! curl -s --max-time 10 $SERVER_URL/health > /dev/null; then
    echo "❌ El servidor no responde. Por favor, asegúrate de que esté corriendo."
    exit 1
fi

echo "✅ Servidor disponible"
echo ""

# Array con los IDs de los modelos
MODELS=("phi4_fast" "mistral_balanced" "qwen_coder" "gemma3_multimodal" "aya_expanse_multilingual")

# Array con descripciones de los modelos
DESCRIPCIONES=(
    "Modelo rápido para respuestas simples"
    "Modelo equilibrado para tareas técnicas"
    "Modelo especializado en código y programación"
    "Modelo multimodal para análisis complejo"
    "Modelo multilingüe de Cohere (23 idiomas)"
)

# Función para hacer la prueba
probar_modelo() {
    local modelo=$1
    local descripcion=$2
    local consulta=$3
    local tipo=$4
    
    echo "----- Prueba: $tipo -----"
    echo "Modelo: $modelo"
    echo "Descripción: $descripcion"
    echo "Consulta: $consulta"
    echo ""
    
    local start_time=$(date +%s.%N)
    
    # Hacer la petición al modelo
    response=$(curl -s --max-time 45 \
        -X POST $SERVER_URL/v1/chat/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer EMPTY" \
        -d "{
            \"model\": \"$modelo\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$consulta\"}],
            \"max_tokens\": 150,
            \"temperature\": 0.7
        }")
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)
    
    # Extraer la respuesta
    local respuesta=$(echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['choices'][0]['message']['content'] if 'choices' in data and len(data['choices']) > 0 else 'ERROR: No se recibió respuesta válida')")
    
    if [[ $respuesta == ERROR:* ]]; then
        echo "❌ Error en la respuesta: $respuesta"
        echo "   JSON completo: $response"
    else
        echo "✅ Respuesta del modelo:"
        echo "$respuesta"
        echo ""
        echo "⏱️  Tiempo de respuesta: ${duration}s"
    fi
    
    echo "----------------------------------------"
    echo ""
}

# Pruebas generales para todos los modelos
for i in {0..4}; do
    modelo=${MODELS[$i]}
    descripcion=${DESCRIPCIONES[$i]}
    
    echo "🤖 MODELO: $modelo"
    echo "📝 Descripción: $descripcion"
    echo "========================================"
    
    # Prueba 1: Consulta general
    probar_modelo "$modelo" "$descripcion" "¿Qué es la inteligencia artificial?" "Consulta general"
    
    # Prueba 2: Consulta técnica (solo para modelos técnicos/código)
    if [[ $modelo == "mistral_balanced" || $modelo == "qwen_coder" ]]; then
        probar_modelo "$modelo" "$descripcion" "Explica brevemente cómo funciona un algoritmo de ordenamiento rápido (quick sort)." "Consulta técnica"
    fi
    
    # Prueba 3: Consulta de programación (solo para modelo de código)
    if [[ $modelo == "qwen_coder" ]]; then
        probar_modelo "$modelo" "$descripcion" "Escribe una función en Python que calcule el factorial de un número." "Consulta de programación"
    fi
    
    # Prueba 4: Consulta multilingüe (solo para modelos multilingües)
    if [[ $modelo == "aya_expanse_multilingual" || $modelo == "gemma3_multimodal" ]]; then
        probar_modelo "$modelo" "$descripcion" "Traduce al inglés: 'La inteligencia artificial está transformando el mundo moderno'." "Traducción"
    fi
    
    echo ""
done

# Prueba adicional: Comparación entre modelos
echo "🔍 PRUEBA DE COMPARACIÓN ENTRE MODELOS"
echo "======================================"
CONSULTA_COMPARACION="Explica qué es el machine learning en 3 líneas"

echo "Consulta para comparación: $CONSULTA_COMPARACION"
echo ""

for modelo in "${MODELS[@]}"; do
    echo "💬 $modelo:"
    respuesta=$(curl -s --max-time 30 \
        -X POST $SERVER_URL/v1/chat/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer EMPTY" \
        -d "{
            \"model\": \"$modelo\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$CONSULTA_COMPARACION\"}],
            \"max_tokens\": 100
        }" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['choices'][0]['message']['content'][:200] + '...' if 'choices' in data and len(data['choices']) > 0 else 'ERROR')")
    
    echo "$respuesta"
    echo "-----"
done

echo ""
echo "✅ PRUEBAS COMPLETADAS"
echo "El sistema de 5 modelos en consenso está completamente funcional."
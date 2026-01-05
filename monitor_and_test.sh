#!/bin/bash
# Script para monitorear el servidor ARM-Axion y probarlo cuando esté listo

LOG_FILE="/tmp/vllm_arm_axion.log"
SERVER_URL="http://localhost:8080"

echo "════════════════════════════════════════════════════════════"
echo "  MONITOR Y TEST - SERVIDOR ARM-AXION MULTI-MODELO"
echo "════════════════════════════════════════════════════════════"
echo ""

# Función para verificar si el servidor responde
check_server() {
    curl -s "$SERVER_URL/" > /dev/null 2>&1
    return $?
}

# Monitorear el log
echo "📊 Monitoreando carga de modelos..."
echo "   (Presiona Ctrl+C cuando veas 'Application startup complete')"
echo ""

# Mostrar las últimas líneas relevantes
tail -f "$LOG_FILE" | grep --line-buffered -E "Loading|✅|❌|Warming|Application|INFO:     Uvicorn" &
TAIL_PID=$!

# Esperar señal del usuario
echo ""
read -p "Presiona ENTER cuando el servidor esté listo..."

# Detener el tail
kill $TAIL_PID 2>/dev/null

# Verificar que el servidor responda
echo ""
echo "🔍 Verificando servidor..."
if check_server; then
    echo "✅ Servidor disponible en $SERVER_URL"
else
    echo "❌ Servidor no responde aún. Espera un poco más."
    exit 1
fi

# Mostrar modelos disponibles
echo ""
echo "📚 Modelos disponibles:"
curl -s "$SERVER_URL/models" | python3 -m json.tool 2>/dev/null || echo "Error obteniendo modelos"

# Menú de pruebas
while true; do
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "¿Qué quieres hacer?"
    echo "  1. Probar modelo qwen25-coder (código - más pequeño)"
    echo "  2. Probar modelo phi4-fast (respuestas rápidas)"
    echo "  3. Probar modelo mistral7b-balanced (equilibrado)"
    echo "  4. Ver estado del servidor"
    echo "  5. Ejecutar CLI interactiva"
    echo "  6. Salir"
    echo "════════════════════════════════════════════════════════════"
    read -p "Opción (1-6): " choice

    case $choice in
        1)
            echo ""
            echo "🔄 Probando qwen25-coder..."
            curl -X POST "$SERVER_URL/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d '{
                    "model": "qwen_coder",
                    "messages": [{"role": "user", "content": "Escribe una función Python para sumar dos números"}],
                    "max_tokens": 100
                }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Respuesta:'); print(r.get('choices',[{}])[0].get('message',{}).get('content','Error'))" 2>/dev/null || echo "Error en la consulta"
            ;;
        2)
            echo ""
            echo "🔄 Probando phi4-fast..."
            curl -X POST "$SERVER_URL/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d '{
                    "model": "phi4_fast",
                    "messages": [{"role": "user", "content": "Hola, ¿cómo estás?"}],
                    "max_tokens": 50
                }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Respuesta:'); print(r.get('choices',[{}])[0].get('message',{}).get('content','Error'))" 2>/dev/null || echo "Error en la consulta"
            ;;
        3)
            echo ""
            echo "🔄 Probando mistral7b-balanced..."
            curl -X POST "$SERVER_URL/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d '{
                    "model": "mistral_balanced",
                    "messages": [{"role": "user", "content": "Explica qué es vLLM en una frase"}],
                    "max_tokens": 50
                }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Respuesta:'); print(r.get('choices',[{}])[0].get('message',{}).get('content','Error'))" 2>/dev/null || echo "Error en la consulta"
            ;;
        4)
            echo ""
            echo "🖥️  Estado del servidor:"
            curl -s "$SERVER_URL/" | python3 -m json.tool 2>/dev/null
            echo ""
            curl -s "$SERVER_URL/health" | python3 -m json.tool 2>/dev/null
            ;;
        5)
            echo ""
            echo "🚀 Ejecutando CLI interactiva..."
            python3 /home/elect/capibara6/test_multi_models_cli.py
            ;;
        6)
            echo ""
            echo "👋 ¡Hasta luego!"
            exit 0
            ;;
        *)
            echo "❌ Opción inválida"
            ;;
    esac
done

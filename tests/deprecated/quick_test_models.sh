#!/bin/bash
# Script rápido para probar modelos desde terminal

BASE_URL="http://localhost:8080"

# Colores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  PRUEBA RÁPIDA DE MODELOS - ARM-Axion Multi-Model System${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

# Función para probar un modelo
test_model() {
    local model_id=$1
    local model_name=$2
    local query=$3

    echo -e "\n${YELLOW}🔄 Probando: ${model_name}${NC}"
    echo -e "   Consulta: ${query}"

    start_time=$(date +%s.%N)

    response=$(curl -s -X POST "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${model_id}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"${query}\"}],
            \"max_tokens\": 100,
            \"temperature\": 0.7
        }")

    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)

    # Extraer respuesta usando jq si está disponible, sino usar grep
    if command -v jq &> /dev/null; then
        text=$(echo "$response" | jq -r '.choices[0].message.content // "Error"')
        if [ "$text" != "Error" ]; then
            echo -e "${GREEN}✅ Respuesta (${elapsed}s):${NC}"
            echo "$text" | fold -s -w 70 | sed 's/^/   /'
        else
            echo -e "${RED}❌ Error en respuesta${NC}"
            echo "$response" | jq '.'
        fi
    else
        # Fallback sin jq
        if echo "$response" | grep -q "content"; then
            echo -e "${GREEN}✅ Respuesta (${elapsed}s):${NC}"
            echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "$response"
        else
            echo -e "${RED}❌ Error en respuesta${NC}"
            echo "$response"
        fi
    fi
}

# Verificar que el servidor está corriendo
echo -e "\n${BLUE}📡 Verificando servidor...${NC}"
server_info=$(curl -s "${BASE_URL}/")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Servidor disponible${NC}"
    if command -v jq &> /dev/null; then
        echo "$server_info" | jq '.'
    else
        echo "$server_info"
    fi
else
    echo -e "${RED}❌ Servidor no disponible en ${BASE_URL}${NC}"
    echo "   Inicia el servidor con: ./start_vllm_arm_axion.sh"
    exit 1
fi

# Menú de opciones
echo -e "\n${BLUE}Selecciona una opción:${NC}"
echo "  1. Probar Phi4-Fast (el más rápido)"
echo "  2. Probar Qwen2.5-Coder (experto en código)"
echo "  3. Probar Mistral7B (equilibrado)"
echo "  4. Probar Gemma3-27B (análisis complejo)"
echo "  5. Probar GPT-OSS-20B (razonamiento avanzado)"
echo "  6. Probar TODOS los modelos con una consulta"
echo "  7. Prueba personalizada (especificar modelo y consulta)"
echo ""

read -p "Opción (1-7): " option

case $option in
    1)
        query=${1:-"¿Qué es ARM Axion?"}
        test_model "phi4-fast" "Phi4-Fast (14B)" "$query"
        ;;
    2)
        query=${1:-"Escribe una función Python para calcular factorial"}
        test_model "qwen25-coder" "Qwen2.5-Coder (1.5B)" "$query"
        ;;
    3)
        query=${1:-"Explica qué es vLLM y sus ventajas"}
        test_model "mistral7b-balanced" "Mistral7B (7B)" "$query"
        ;;
    4)
        query=${1:-"Analiza las diferencias entre arquitecturas ARM y x86"}
        test_model "gemma3-27b" "Gemma3-27B (27B)" "$query"
        ;;
    5)
        query=${1:-"Explica la teoría de la relatividad en términos simples"}
        test_model "gptoss-20b" "GPT-OSS-20B (20B)" "$query"
        ;;
    6)
        read -p "Consulta para todos los modelos: " query
        echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}  PROBANDO ${query:0:40}... EN 5 MODELOS${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

        test_model "phi4-fast" "Phi4-Fast" "$query"
        echo ""
        test_model "qwen25-coder" "Qwen2.5-Coder" "$query"
        echo ""
        test_model "mistral7b-balanced" "Mistral7B" "$query"
        echo ""
        test_model "gemma3-27b" "Gemma3-27B" "$query"
        echo ""
        test_model "gptoss-20b" "GPT-OSS-20B" "$query"
        ;;
    7)
        echo "Modelos disponibles:"
        echo "  - phi4-fast"
        echo "  - qwen25-coder"
        echo "  - mistral7b-balanced"
        echo "  - gemma3-27b"
        echo "  - gptoss-20b"
        read -p "Modelo: " model_id
        read -p "Consulta: " query
        test_model "$model_id" "$model_id" "$query"
        ;;
    *)
        echo -e "${RED}❌ Opción inválida${NC}"
        exit 1
        ;;
esac

echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Prueba completada${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

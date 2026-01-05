#!/bin/bash
# Script de Inicio Rápido - Capibara6 con 5 Modelos ARM-Axion
# Este script inicia todos los servicios necesarios para usar el sistema completo

echo "🚀 INICIANDO SERVICIOS CAPIBARA6 - 5 MODELOS ARM-Axion"
echo "================================================================"

# Cambiar al directorio de trabajo
cd /home/elect/capibara6

echo "📁 Directorio de trabajo: $(pwd)"
echo ""

# Verificar disponibilidad de modelos
echo "📦 Verificando modelos disponibles..."
if [ -d "/home/elect/models/phi-4-mini" ]; then
    echo "✅ phi4:mini: Disponible"
else
    echo "❌ phi4:mini: No encontrado"
fi

if [ -d "/home/elect/models/qwen2.5-coder-1.5b" ]; then
    echo "✅ qwen2.5-coder-1.5b: Disponible"
else
    echo "❌ qwen2.5-coder-1.5b: No encontrado"
fi

if [ -d "/home/elect/models/gemma-3-27b-it-awq" ]; then
    echo "✅ gemma-3-27b-it-awq: Disponible"
else
    echo "❌ gemma-3-27b-it-awq: No encontrado"
fi

if [ -d "/home/elect/models/mistral-7b-instruct-v0.2" ]; then
    echo "✅ mistral-7b-instruct-v0.2: Disponible"
else
    echo "❌ mistral-7b-instruct-v0.2: No encontrado"
fi

if [ -d "/home/elect/models/gpt-oss-20b" ]; then
    echo "✅ gpt-oss-20b: Disponible"
else
    echo "❌ gpt-oss-20b: No encontrado"
fi

echo ""

# Verificar si el servidor ya está corriendo
echo "🔍 Verificando estado de servicios..."
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Advertencia: El puerto 8000 ya está en uso"
    echo "   Revisa si ya hay un servidor corriendo o libera el puerto"
else
    echo "✅ Puerto 8000 disponible"
fi

echo ""

# Mostrar comandos para iniciar servicios
echo "🔧 Para iniciar el servidor con los 5 modelos ARM-Axion optimizados:"
echo ""
echo "   cd /home/elect/capibara6/arm-axion-optimizations/vllm-integration"
echo "   python3 multi_model_server.py --config config.five_models.optimized.json --host 0.0.0.0 --port 8000"
echo ""
echo "# Una vez iniciado, puedes probar:"
echo "curl http://localhost:8000/v1/models"
echo ""

# Mostrar comandos para pruebas
echo "🧪 Para probar los modelos una vez iniciado el servidor:"
echo ""
echo "   # Probar cliente real:"
echo "   cd /home/elect/capibara6"
echo "   python3 real_model_tester.py"
echo ""
echo "   # Probar interfaz completa:"
echo "   python3 interactive_test_interface_optimized.py"
echo ""

echo "💡 NOTA: La configuración completa de los 5 modelos está en:"
echo "   /home/elect/capibara6/five_model_config.json"
echo ""

echo "🎉 ¡Sistema Capibara6 con 5 modelos ARM-Axion optimizados está listo!"
echo "   - phi4:mini (rápido)"
echo "   - qwen2.5-coder-1.5b (técnico)"  
echo "   - gemma-3-27b-it-awq (multimodal)"
echo "   - mistral-7b-instruct-v0.2 (general)"
echo "   - gpt-oss-20b (complejo)"
echo ""
echo "   Todos con optimizaciones ARM-Axion (NEON + ACL + cuantización)"
echo "   Router semántico, sistema de consenso y pruebas integradas disponibles"

echo "================================================================"
echo "✅ LISTO PARA INICIAR LOS SERVICIOS!"
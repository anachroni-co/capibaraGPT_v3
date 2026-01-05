#!/bin/bash
# Script para habilitar/deshabilitar el modo consenso en la configuración

CONFIG_FILE="/home/elect/capibara6/arm-axion-optimizations/vllm_integration/config.five_models.optimized.json"

echo "════════════════════════════════════════════════════════════"
echo "  CONFIGURACIÓN DE MODO CONSENSO"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "El modo consenso permite consultar múltiples modelos y"
echo "combinar sus respuestas para obtener mejores resultados."
echo ""
echo "Opciones:"
echo "  1. Habilitar modo consenso"
echo "  2. Deshabilitar modo consenso"
echo "  3. Ver configuración actual"
echo "  4. Salir"
echo ""

read -p "Selecciona opción (1-4): " option

case $option in
    1)
        echo ""
        echo "🔧 Habilitando modo consenso..."

        # Crear backup
        cp "$CONFIG_FILE" "${CONFIG_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
        echo "✅ Backup creado"

        # Habilitar consenso (usando python para editar JSON de forma segura)
        python3 << EOF
import json

with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)

config['enable_consensus'] = True
config['consensus_model'] = 'gemma3-27b'  # Usar el modelo más potente como arbitro

with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Modo consenso habilitado")
print(f"   Modelo arbitro: {config['consensus_model']}")
EOF

        echo ""
        echo "⚠️  IMPORTANTE: Debes reiniciar el servidor para aplicar cambios:"
        echo "   1. Detener servidor actual (Ctrl+C)"
        echo "   2. Ejecutar: ./start_vllm_arm_axion.sh"
        ;;

    2)
        echo ""
        echo "🔧 Deshabilitando modo consenso..."

        # Crear backup
        cp "$CONFIG_FILE" "${CONFIG_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
        echo "✅ Backup creado"

        # Deshabilitar consenso
        python3 << EOF
import json

with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)

config['enable_consensus'] = False
config['consensus_model'] = None

with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Modo consenso deshabilitado")
EOF

        echo ""
        echo "⚠️  IMPORTANTE: Debes reiniciar el servidor para aplicar cambios"
        ;;

    3)
        echo ""
        echo "📋 Configuración actual:"
        echo ""
        python3 << EOF
import json

with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)

enabled = config.get('enable_consensus', False)
model = config.get('consensus_model', None)

status = "✅ HABILITADO" if enabled else "❌ DESHABILITADO"
print(f"   Modo consenso: {status}")
if enabled and model:
    print(f"   Modelo arbitro: {model}")

# Mostrar configuración de expertos
if 'experts' in config:
    print(f"\n   Modelos disponibles: {len(config['experts'])}")
    for expert in config['experts']:
        print(f"      - {expert['model_id']} (peso: {expert.get('routing_weight', 1.0)})")
EOF
        ;;

    4)
        echo "👋 Saliendo..."
        exit 0
        ;;

    *)
        echo "❌ Opción inválida"
        exit 1
        ;;
esac

echo ""
echo "════════════════════════════════════════════════════════════"

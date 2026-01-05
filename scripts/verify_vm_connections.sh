#!/bin/bash
# Script para verificar conexiones entre VMs de GCloud
# Verifica IPs, servicios y conectividad entre Bounty2, rag3 y gpt-oss-20b

set -e

PROJECT="mamba-001"
BOUNTY2_ZONE="europe-west4-a"
RAG3_ZONE="europe-west2-c"
GPTOSS_ZONE="europe-southwest1-b"

echo "🔍 Obteniendo IPs de las VMs..."
echo "=================================="

# Obtener IPs externas
BOUNTY2_IP=$(gcloud compute instances describe bounty2 --zone=$BOUNTY2_ZONE --project=$PROJECT --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || echo "")
RAG3_IP=$(gcloud compute instances describe rag3 --zone=$RAG3_ZONE --project=$PROJECT --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || echo "")
GPTOSS_IP=$(gcloud compute instances describe gpt-oss-20b --zone=$GPTOSS_ZONE --project=$PROJECT --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || echo "")

# Obtener IPs internas
BOUNTY2_INTERNAL=$(gcloud compute instances describe bounty2 --zone=$BOUNTY2_ZONE --project=$PROJECT --format="value(networkInterfaces[0].networkIP)" 2>/dev/null || echo "")
RAG3_INTERNAL=$(gcloud compute instances describe rag3 --zone=$RAG3_ZONE --project=$PROJECT --format="value(networkInterfaces[0].networkIP)" 2>/dev/null || echo "")
GPTOSS_INTERNAL=$(gcloud compute instances describe gpt-oss-20b --zone=$GPTOSS_ZONE --project=$PROJECT --format="value(networkInterfaces[0].networkIP)" 2>/dev/null || echo "")

# Obtener nombres de red VPC
BOUNTY2_NETWORK=$(gcloud compute instances describe bounty2 --zone=$BOUNTY2_ZONE --project=$PROJECT --format="value(networkInterfaces[0].network)" 2>/dev/null | awk -F'/' '{print $NF}' || echo "")
RAG3_NETWORK=$(gcloud compute instances describe rag3 --zone=$RAG3_ZONE --project=$PROJECT --format="value(networkInterfaces[0].network)" 2>/dev/null | awk -F'/' '{print $NF}' || echo "")
GPTOSS_NETWORK=$(gcloud compute instances describe gpt-oss-20b --zone=$GPTOSS_ZONE --project=$PROJECT --format="value(networkInterfaces[0].network)" 2>/dev/null | awk -F'/' '{print $NF}' || echo "")

echo ""
echo "📊 Información de VMs:"
echo "======================"
echo "Bounty2 (Ollama):"
echo "  - IP Externa: ${BOUNTY2_IP:-NO DISPONIBLE}"
echo "  - IP Interna: ${BOUNTY2_INTERNAL:-NO DISPONIBLE}"
echo "  - Zona: $BOUNTY2_ZONE"
echo "  - Red VPC: ${BOUNTY2_NETWORK:-NO DISPONIBLE}"
echo ""
echo "rag3 (Base de Datos RAG):"
echo "  - IP Externa: ${RAG3_IP:-NO DISPONIBLE}"
echo "  - IP Interna: ${RAG3_INTERNAL:-NO DISPONIBLE}"
echo "  - Zona: $RAG3_ZONE"
echo "  - Red VPC: ${RAG3_NETWORK:-NO DISPONIBLE}"
echo ""
echo "gpt-oss-20b (TTS, MCP, N8n, Bridge):"
echo "  - IP Externa: ${GPTOSS_IP:-NO DISPONIBLE}"
echo "  - IP Interna: ${GPTOSS_INTERNAL:-NO DISPONIBLE}"
echo "  - Zona: $GPTOSS_ZONE"
echo "  - Red VPC: ${GPTOSS_NETWORK:-NO DISPONIBLE}"
echo ""

# Verificar si están en la misma red VPC
if [ "$BOUNTY2_NETWORK" = "$RAG3_NETWORK" ] && [ "$RAG3_NETWORK" = "$GPTOSS_NETWORK" ] && [ -n "$BOUNTY2_NETWORK" ]; then
    echo "✅ Todas las VMs están en la misma red VPC: $BOUNTY2_NETWORK"
    echo "   Esto permite comunicación de alta velocidad entre VMs"
else
    echo "⚠️  Las VMs NO están en la misma red VPC:"
    echo "   - Bounty2: ${BOUNTY2_NETWORK:-N/A}"
    echo "   - rag3: ${RAG3_NETWORK:-N/A}"
    echo "   - gpt-oss-20b: ${GPTOSS_NETWORK:-N/A}"
    echo "   Se recomienda moverlas a la misma red para mejor rendimiento"
fi

echo ""
echo "🔌 Verificando servicios en cada VM..."
echo "======================================"

# Función para verificar servicios en una VM
check_services() {
    local VM_NAME=$1
    local ZONE=$2
    local IP=$3
    
    if [ -z "$IP" ]; then
        echo "⚠️  No se pudo obtener IP de $VM_NAME"
        return
    fi
    
    echo ""
    echo "📡 Verificando $VM_NAME ($IP)..."
    
    # Conectar y verificar puertos comunes
    echo "  Verificando puertos comunes..."
    
    # Ollama (11434)
    if [ "$VM_NAME" = "bounty2" ]; then
        echo "    - Puerto 11434 (Ollama): Verificando..."
        timeout 3 bash -c "echo > /dev/tcp/$IP/11434" 2>/dev/null && echo "      ✅ Puerto 11434 abierto" || echo "      ❌ Puerto 11434 cerrado o inaccesible"
    fi
    
    # Backend Flask (5000, 5001)
    for port in 5000 5001; do
        timeout 3 bash -c "echo > /dev/tcp/$IP/$port" 2>/dev/null && echo "      ✅ Puerto $port abierto" || echo "      ⚠️  Puerto $port cerrado o inaccesible"
    done
    
    # MCP (5003, 5010)
    if [ "$VM_NAME" = "gpt-oss-20b" ]; then
        for port in 5003 5010; do
            timeout 3 bash -c "echo > /dev/tcp/$IP/$port" 2>/dev/null && echo "      ✅ Puerto $port (MCP) abierto" || echo "      ⚠️  Puerto $port cerrado o inaccesible"
        done
    fi
    
    # TTS (5002)
    if [ "$VM_NAME" = "gpt-oss-20b" ]; then
        timeout 3 bash -c "echo > /dev/tcp/$IP/5002" 2>/dev/null && echo "      ✅ Puerto 5002 (TTS) abierto" || echo "      ⚠️  Puerto 5002 cerrado o inaccesible"
    fi
    
    # RAG API (8000)
    if [ "$VM_NAME" = "rag3" ]; then
        timeout 3 bash -c "echo > /dev/tcp/$IP/8000" 2>/dev/null && echo "      ✅ Puerto 8000 (RAG API) abierto" || echo "      ⚠️  Puerto 8000 cerrado o inaccesible"
    fi
    
    # N8n (5678)
    if [ "$VM_NAME" = "gpt-oss-20b" ]; then
        timeout 3 bash -c "echo > /dev/tcp/$IP/5678" 2>/dev/null && echo "      ✅ Puerto 5678 (N8n) abierto" || echo "      ⚠️  Puerto 5678 cerrado o inaccesible"
    fi
}

# Verificar servicios (solo si tenemos IPs)
if [ -n "$BOUNTY2_IP" ]; then
    check_services "bounty2" "$BOUNTY2_ZONE" "$BOUNTY2_IP"
fi

if [ -n "$RAG3_IP" ]; then
    check_services "rag3" "$RAG3_ZONE" "$RAG3_IP"
fi

if [ -n "$GPTOSS_IP" ]; then
    check_services "gpt-oss-20b" "$GPTOSS_ZONE" "$GPTOSS_IP"
fi

echo ""
echo "📝 Generando archivo de configuración..."
echo "========================================"

# Generar archivo de configuración
cat > vm_config.json << EOF
{
  "vms": {
    "bounty2": {
      "name": "bounty2",
      "zone": "$BOUNTY2_ZONE",
      "ip_external": "$BOUNTY2_IP",
      "ip_internal": "$BOUNTY2_INTERNAL",
      "network": "$BOUNTY2_NETWORK",
      "services": {
        "ollama": {
          "port": 11434,
          "endpoint": "http://${BOUNTY2_INTERNAL:-$BOUNTY2_IP}:11434",
          "models": ["gpt-oss-20B", "mixtral", "phi-mini3"]
        },
        "backend": {
          "port": 5001,
          "endpoint": "http://${BOUNTY2_INTERNAL:-$BOUNTY2_IP}:5001"
        }
      }
    },
    "rag3": {
      "name": "rag3",
      "zone": "$RAG3_ZONE",
      "ip_external": "$RAG3_IP",
      "ip_internal": "$RAG3_INTERNAL",
      "network": "$RAG3_NETWORK",
      "services": {
        "rag_api": {
          "port": 8000,
          "endpoint": "http://${RAG3_INTERNAL:-$RAG3_IP}:8000"
        }
      }
    },
    "gpt-oss-20b": {
      "name": "gpt-oss-20b",
      "zone": "$GPTOSS_ZONE",
      "ip_external": "$GPTOSS_IP",
      "ip_internal": "$GPTOSS_INTERNAL",
      "network": "$GPTOSS_NETWORK",
      "services": {
        "tts": {
          "port": 5002,
          "endpoint": "http://${GPTOSS_INTERNAL:-$GPTOSS_IP}:5002"
        },
        "mcp": {
          "port": 5003,
          "endpoint": "http://${GPTOSS_INTERNAL:-$GPTOSS_IP}:5003"
        },
        "mcp_alt": {
          "port": 5010,
          "endpoint": "http://${GPTOSS_INTERNAL:-$GPTOSS_IP}:5010"
        },
        "n8n": {
          "port": 5678,
          "endpoint": "http://${GPTOSS_INTERNAL:-$GPTOSS_IP}:5678"
        },
        "bridge": {
          "port": 5000,
          "endpoint": "http://${GPTOSS_INTERNAL:-$GPTOSS_IP}:5000"
        }
      }
    }
  },
  "network": {
    "same_vpc": $([ "$BOUNTY2_NETWORK" = "$RAG3_NETWORK" ] && [ "$RAG3_NETWORK" = "$GPTOSS_NETWORK" ] && echo "true" || echo "false"),
    "vpc_name": "$BOUNTY2_NETWORK"
  }
}
EOF

echo "✅ Configuración guardada en: vm_config.json"
echo ""
echo "🎯 Próximos pasos:"
echo "=================="
echo "1. Revisar vm_config.json para verificar las IPs"
echo "2. Verificar que los servicios estén corriendo en cada VM"
echo "3. Configurar firewall rules si es necesario"
echo "4. Actualizar archivos de configuración del backend y frontend"
echo ""


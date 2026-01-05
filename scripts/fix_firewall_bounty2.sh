#!/bin/bash
# Script para configurar firewall en Bounty2 y permitir puertos 5000 y 5001

echo "🔧 Configurando firewall para Bounty2..."

# Conectar a la VM y deshabilitar/limpiar firewall interno
gcloud compute ssh --zone=europe-west4-a bounty2 --project=mamba-001 << 'EOF'
    echo "📋 Verificando firewall interno..."
    
    # Deshabilitar UFW si está activo
    if command -v ufw &> /dev/null; then
        echo "🛑 Deshabilitando UFW..."
        sudo ufw --force disable || true
        sudo ufw --force reset || true
    fi
    
    # Limpiar reglas iptables que bloqueen 5000/5001
    echo "🧹 Limpiando reglas iptables..."
    sudo iptables -D INPUT -p tcp --dport 5000 -j DROP 2>/dev/null || true
    sudo iptables -D INPUT -p tcp --dport 5001 -j DROP 2>/dev/null || true
    
    # Permitir puertos 5000 y 5001
    sudo iptables -I INPUT -p tcp --dport 5000 -j ACCEPT 2>/dev/null || true
    sudo iptables -I INPUT -p tcp --dport 5001 -j ACCEPT 2>/dev/null || true
    
    # Guardar reglas iptables si es posible
    if command -v iptables-save &> /dev/null; then
        sudo iptables-save | sudo tee /etc/iptables/rules.v4 > /dev/null 2>&1 || true
    fi
    
    echo "✅ Firewall interno configurado"
    echo "📊 Estado de puertos:"
    sudo netstat -tlnp | grep -E ':5000|:5001' || sudo ss -tlnp | grep -E ':5000|:5001' || echo "No hay servicios escuchando en 5000/5001"
EOF

echo "✅ Script completado"


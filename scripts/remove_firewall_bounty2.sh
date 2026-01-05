#!/bin/bash
# Script para eliminar reglas de firewall que bloqueen puertos en Bounty2

echo "🔧 Eliminando reglas de firewall en Bounty2..."

# Conectar a la VM y limpiar firewall
gcloud compute ssh --zone=europe-west4-a bounty2 --project=mamba-001 << 'EOF'
    echo "📋 Estado actual del firewall..."
    
    # Verificar UFW
    if command -v ufw &> /dev/null; then
        echo "🛑 Deshabilitando UFW completamente..."
        sudo ufw --force disable
        sudo uptables --flush || true
    fi
    
    # Limpiar todas las reglas iptables que bloqueen nuestros puertos
    echo "🧹 Eliminando reglas iptables que bloqueen puertos 5000 y 5001..."
    
    # Listar reglas actuales
    echo "📊 Reglas actuales:"
    sudo iptables -L INPUT -n --line-numbers | head -20
    
    # Eliminar reglas DROP/REJECT para puertos 5000 y 5001
    sudo iptables -D INPUT -p tcp --dport 5000 -j DROP 2>/dev/null || true
    sudo iptables -D INPUT -p tcp --dport 5001 -j DROP 2>/dev/null || true
    sudo iptables -D INPUT -p tcp --dport 5000 -j REJECT 2>/dev/null || true
    sudo iptables -D INPUT -p tcp --dport 5001 -j REJECT 2>/dev/null || true
    
    # Asegurar que los puertos están permitidos
    echo "✅ Añadiendo reglas para permitir puertos..."
    sudo iptables -I INPUT 1 -p tcp --dport 5000 -j ACCEPT 2>/dev/null || true
    sudo iptables -I INPUT 1 -p tcp --dport 5001 -j ACCEPT 2>/dev/null || true
    
    # Verificar servicios escuchando
    echo "📊 Servicios escuchando en puertos 5000/5001:"
    sudo netstat -tlnp 2>/dev/null | grep -E ':5000|:5001' || sudo ss -tlnp 2>/dev/null | grep -E ':5000|:5001' || echo "No hay servicios escuchando"
    
    echo "✅ Firewall limpiado"
EOF

echo "✅ Script completado"


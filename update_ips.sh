#!/bin/bash
# Script de actualización automática de IPs para el sistema Capibara6

CURRENT_EXTERNAL_IP=$(curl -s -H "Metadata-Flavor: Google" http://169.254.169.254/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip)
CURRENT_INTERNAL_IP=$(hostname -I | awk '{print $1}')  # Primera IP interna

echo "📍 IP Externa actual: $CURRENT_EXTERNAL_IP"
echo "📍 IP Interna actual: $CURRENT_INTERNAL_IP"

# Variables de entorno antiguas (para encontrar y reemplazar)
OLD_EXTERNAL_IP_REGEX="34\.175\.(48|255)\.[0-9]{1,3}"

# Actualizar todos los archivos de API con la IP nueva
find /home/elect/capibara6/api/ -name "*.js" -type f -exec sed -i "s/$OLD_EXTERNAL_IP_REGEX/$CURRENT_EXTERNAL_IP/g" {} \;

# Actualizar archivos de configuración si existen
if [ -f "/home/elect/capibara6/backend/models_config.py" ]; then
    sed -i "s/$OLD_EXTERNAL_IP_REGEX/$CURRENT_EXTERNAL_IP/g" /home/elect/capibara6/backend/models_config.py
fi

if [ -f "/home/elect/capibara6/backend/config/infrastructure_config.py" ]; then
    sed -i "s/$OLD_EXTERNAL_IP_REGEX/$CURRENT_EXTERNAL_IP/g" /home/elect/capibara6/backend/config/infrastructure_config.py
fi

if [ -f "/home/elect/capibara6/backend/gateway_server.py" ]; then
    sed -i "s/10\.204\.0\.[0-9]{1,3}/$CURRENT_INTERNAL_IP/g" /home/elect/capibara6/backend/gateway_server.py
fi

echo "✅ Actualización de IPs completada"
echo "📝 Archivos actualizados con:"
echo "   - IP Externa: $CURRENT_EXTERNAL_IP"
echo "   - IP Interna: $CURRENT_INTERNAL_IP"
#!/bin/bash
# Script para iniciar el servidor proxy local
# Este servidor maneja los problemas CORS entre el frontend y el backend real

# Instalar dependencias si no están instaladas
pip3 install flask requests --break-system-packages

# Iniciar el servidor proxy local
python3 /mnt/c/Users/elect/Capibara6.com/capibara6/backend/local_proxy_server.py
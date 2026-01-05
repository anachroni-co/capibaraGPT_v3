@echo off
REM Script para arreglar el error 500 ejecutando el script de reinicio en la VM

echo 🔧 Arreglando Error 500 - Reiniciando Backend
echo =============================================

set VM_NAME=gpt-oss-20b
set ZONE=europe-southwest1-b
set PROJECT=mamba-001

echo.
echo 📥 Subiendo script de reinicio a la VM...
echo.

REM Crear script temporal en la VM
gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT% --command="
cat > /tmp/restart_backend.sh << 'SCRIPT_EOF'
#!/bin/bash
echo '🔄 Reiniciando Servidor Integrado Capibara6'
echo '=========================================='

# Encontrar directorio del backend
if [ -d ~/capibara6/backend ]; then
    BACKEND_DIR=~/capibara6/backend
elif [ -d backend ]; then
    BACKEND_DIR=backend
else
    BACKEND_DIR=\$(find ~ -name 'capibara6_integrated_server.py' 2>/dev/null | head -1 | xargs dirname)
fi

echo '📁 Directorio: '\$BACKEND_DIR
cd \$BACKEND_DIR

echo ''
echo '1️⃣ Deteniendo servidor existente...'
pkill -f 'capibara6_integrated_server' || pkill -f 'python.*5001' || echo 'No hay procesos para detener'
sleep 2

echo ''
echo '2️⃣ Verificando puerto 5001...'
if lsof -i :5001 > /dev/null 2>&1; then
    echo '⚠️ Puerto 5001 aún en uso, forzando...'
    fuser -k 5001/tcp 2>/dev/null || sudo kill -9 \$(lsof -t -i:5001) 2>/dev/null || echo 'No se pudo liberar'
    sleep 2
fi

echo ''
echo '3️⃣ Verificando modelo GPT-OSS-20B...'
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo '✅ Modelo GPT-OSS-20B está respondiendo'
else
    echo '❌ Modelo GPT-OSS-20B no responde'
fi

echo ''
echo '4️⃣ Verificando dependencias...'
python3 -c 'import flask; import requests; print(\"✅ Dependencias OK\")' || pip3 install flask flask-cors requests

echo ''
echo '5️⃣ Iniciando servidor integrado...'
nohup python3 capibara6_integrated_server.py > ../logs/server_\$(date +%%Y%%m%%d_%%H%%M%%S).log 2>&1 &
sleep 3

echo ''
echo '6️⃣ Verificando servidor...'
if curl -s http://localhost:5001/health > /dev/null 2>&1; then
    echo '✅ Servidor responde correctamente'
    curl -s http://localhost:5001/health | head -5
else
    echo '❌ Servidor no responde'
    tail -20 ../logs/server_*.log 2>/dev/null || echo 'No hay logs'
fi

echo ''
echo '✅ Proceso completado'
SCRIPT_EOF
chmod +x /tmp/restart_backend.sh
/tmp/restart_backend.sh
"

echo.
echo ✅ Proceso completado
echo.
echo 🧪 Verificando respuesta del servidor...
gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT% --command="sleep 2 && curl -s http://localhost:5001/health | head -10"

echo.
echo 💡 Si el error persiste, revisa los logs:
echo    gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT%
echo    tail -f ~/capibara6/logs/server_*.log
echo.
pause

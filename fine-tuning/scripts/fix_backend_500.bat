@echo off
REM Script para diagnosticar y arreglar el error 500 en el backend

echo 🔧 Diagnosticando y Arreglando Error 500
echo ========================================

set VM_NAME=gpt-oss-20b
set ZONE=europe-southwest1-b
set PROJECT=mamba-001

echo.
echo 1️⃣ Verificando si el servidor está corriendo...
gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT% --command="ps aux | grep -E 'capibara6_integrated_server|python.*5001' | grep -v grep"

echo.
echo 2️⃣ Verificando puertos...
gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT% --command="sudo netstat -tuln | grep -E ':(5001|8080)'"

echo.
echo 3️⃣ Verificando respuesta del modelo GPT-OSS-20B:
gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT% --command="curl -s http://localhost:8080/health || echo 'Modelo no responde'"

echo.
echo 4️⃣ Verificando logs de error recientes:
gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT% --command="tail -30 backend/logs/errors.log 2>/dev/null || tail -30 ~/capibara6/backend/logs/errors.log 2>/dev/null || echo 'No se encontraron logs de error'"

echo.
echo 5️⃣ Reiniciando servidor integrado...
echo    Deteniendo procesos existentes:
gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT% --command="pkill -f 'capibara6_integrated_server' || pkill -f 'python.*5001' || echo 'No hay procesos para detener'"

echo    Esperando 3 segundos...
timeout /t 3 /nobreak > nul

echo    Iniciando servidor:
gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT% --command="cd ~/capibara6/backend && nohup python3 capibara6_integrated_server.py > ../logs/server.log 2>&1 &"

echo.
echo 6️⃣ Verificando que el servidor está corriendo:
timeout /t 3 /nobreak > nul
gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT% --command="ps aux | grep capibara6_integrated_server | grep -v grep"

echo.
echo 7️⃣ Verificando respuesta del servidor:
gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT% --command="sleep 2 && curl -s http://localhost:5001/health || echo 'Servidor aún no responde'"

echo.
echo ✅ Proceso completado
echo.
echo 💡 Si el error persiste, revisa los logs:
echo    gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT%
echo    tail -f ~/capibara6/logs/server.log
echo.
pause

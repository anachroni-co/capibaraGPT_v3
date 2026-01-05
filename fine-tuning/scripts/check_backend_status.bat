@echo off
REM Script para verificar el estado del backend de Capibara6 desde Windows
REM Ejecuta comandos en la VM para verificar servicios

echo 🔍 Verificando Estado del Backend Capibara6
echo ==========================================

set VM_NAME=gpt-oss-20b
set ZONE=europe-southwest1-b
set PROJECT=mamba-001

echo.
echo 📊 Conectando a la VM y verificando servicios...
echo.

echo 1️⃣ Verificando procesos Python activos:
gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT% --command="ps aux | grep -E '(python|flask)' | grep -v grep || echo 'No hay procesos Python activos'"

echo.
echo 2️⃣ Verificando puertos en uso:
echo    Puerto 5001 (Servidor integrado):
gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT% --command="netstat -tuln | grep :5001 || echo 'Puerto 5001 no está en uso'"

echo    Puerto 5002 (TTS Server):
gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT% --command="netstat -tuln | grep :5002 || echo 'Puerto 5002 no está en uso'"

echo    Puerto 5003 (MCP Server):
gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT% --command="netstat -tuln | grep :5003 || echo 'Puerto 5003 no está en uso'"

echo    Puerto 8080 (GPT-OSS-20B):
gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT% --command="netstat -tuln | grep :8080 || echo 'Puerto 8080 no está en uso'"

echo.
echo 3️⃣ Verificando respuestas de los servicios:
echo    Modelo GPT-OSS-20B:
gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT% --command="curl -s http://localhost:8080/health 2>&1 | head -3 || echo 'Modelo no responde'"

echo    Servidor Integrado:
gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT% --command="curl -s http://localhost:5001/health 2>&1 | head -3 || echo 'Servidor integrado no responde'"

echo.
echo 4️⃣ Verificando espacio en disco:
gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT% --command="df -h | grep -E '(Filesystem|/dev/)'"

echo.
echo 5️⃣ Verificando servicios systemd:
gcloud compute ssh --zone %ZONE% %VM_NAME% --project %PROJECT% --command="systemctl list-units | grep -E '(llama|capibara|gpt)' || echo 'No hay servicios systemd relacionados'"

echo.
echo ✅ Verificación completada
echo.
pause

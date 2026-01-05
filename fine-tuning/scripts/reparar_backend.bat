@echo off
REM Script simple para reparar el error 500 del backend

echo 🔧 Reparando Backend - Error 500
echo ==================================

echo.
echo Verificando autenticación en Google Cloud...
gcloud auth list >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ❌ No estás autenticado en Google Cloud
    echo.
    echo Ejecuta primero:
    echo    gcloud auth login
    echo.
    pause
    exit /b 1
)

echo ✅ Autenticación OK
echo.

echo 🔄 Reiniciando servidor integrado en la VM...
echo.

gcloud compute ssh --zone "europe-southwest1-b" "gpt-oss-20b" --project "mamba-001" --command="cd ~/capibara6/backend && pkill -f capibara6_integrated_server 2>/dev/null; sleep 2; nohup python3 capibara6_integrated_server.py > ../logs/server_$(date +%%Y%%m%%d_%%H%%M%%S).log 2>&1 &; sleep 3; echo 'Verificando servidor...'; curl -s http://localhost:5001/health | head -5 || echo 'Servidor aún iniciando...'"

echo.
echo ✅ Proceso completado
echo.
echo 🧪 Prueba ahora en el navegador: https://www.capibara6.com
echo.
echo 💡 Si el error persiste, revisa los logs con:
echo    gcloud compute ssh --zone "europe-southwest1-b" "gpt-oss-20b" --project "mamba-001"
echo    tail -f ~/capibara6/logs/server_*.log
echo.
pause

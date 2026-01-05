@echo off
REM Script para monitorear el entrenamiento GPT-OSS-20B desde Windows

echo 🔍 Monitoreando Fine-tuning GPT-OSS-20B
echo ======================================

set TPU_NAME=tx-5-oss-20b
set PROJECT_ID=mamba-001
set ZONE=us-central1-a

:monitor_loop
echo.
echo 📊 Estado del entrenamiento - %date% %time%
echo ==========================================

echo 🔗 Conectando a la VM TPU...
gcloud compute tpus tpu-vm ssh --zone %ZONE% --project %PROJECT_ID% %TPU_NAME% --command="
echo '📋 Estado del sistema:'
df -h /mnt/1pb-storage | head -2
echo ''
echo '📊 Procesos de entrenamiento:'
ps aux | grep -E '(t5x|python)' | grep -v grep
echo ''
echo '📁 Últimos logs:'
if [ -d /mnt/1pb-storage/logs/training ]; then
    ls -la /mnt/1pb-storage/logs/training/ | tail -3
    echo ''
    echo '📝 Últimas líneas del log:'
    if [ -f /mnt/1pb-storage/logs/training/training_*.log ]; then
        tail -10 /mnt/1pb-storage/logs/training/training_*.log | tail -5
    fi
fi
echo ''
echo '💾 Checkpoints disponibles:'
if [ -d /mnt/1pb-storage/checkpoints ]; then
    ls -la /mnt/1pb-storage/checkpoints/ | tail -5
fi
"

echo.
echo ⏰ Esperando 60 segundos antes del siguiente check...
echo Presiona Ctrl+C para salir
timeout /t 60 /nobreak > nul
goto monitor_loop

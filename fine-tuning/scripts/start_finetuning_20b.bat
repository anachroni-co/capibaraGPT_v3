@echo off
REM Script para iniciar fine-tuning GPT-OSS-20B desde Windows
REM Ejecuta la configuración completa en la VM TPU

echo 🚀 Iniciando Fine-tuning GPT-OSS-20B
echo ===================================

REM Verificar autenticación en Google Cloud
echo 🔍 Verificando autenticación en Google Cloud...
gcloud auth list
if %errorlevel% neq 0 (
    echo ❌ Error: No estás autenticado en Google Cloud
    echo Ejecuta: gcloud auth login
    pause
    exit /b 1
)

REM Variables
set TPU_NAME=tx-5-oss-20b
set PROJECT_ID=mamba-001
set ZONE=us-central1-a

echo 📋 Verificando estado de la TPU...
gcloud compute tpus describe %TPU_NAME% --zone=%ZONE% --project=%PROJECT_ID%
if %errorlevel% neq 0 (
    echo ❌ Error: TPU %TPU_NAME% no encontrada
    pause
    exit /b 1
)

echo 🔗 Conectando a la VM TPU y configurando entorno...
echo ⚠️ Esto puede tomar 30-60 minutos para instalar todas las dependencias

gcloud compute tpus tpu-vm ssh --zone %ZONE% --project %PROJECT_ID% %TPU_NAME% --command="
echo '📥 Descargando script de configuración...'
mkdir -p ~/capibara6-finetuning
cd ~/capibara6-finetuning

echo '🚀 Ejecutando configuración completa...'
curl -sSL https://raw.githubusercontent.com/tu-usuario/capibara6/main/fine-tuning/scripts/setup_complete_environment.sh -o setup_complete_environment.sh
chmod +x setup_complete_environment.sh
./setup_complete_environment.sh

echo '✅ Configuración completada. Iniciando entrenamiento...'
source ~/.bashrc
/mnt/1pb-storage/launch_training.sh
"

echo.
echo ✅ Fine-tuning iniciado en la VM TPU
echo 📊 Para monitorear el progreso:
echo    gcloud compute tpus tpu-vm ssh --zone %ZONE% --project %PROJECT_ID% %TPU_NAME%
echo    tensorboard --logdir=/mnt/1pb-storage/logs/tensorboard
echo.
echo ⚠️ El entrenamiento puede tomar varias horas. Revisa los logs regularmente.
pause

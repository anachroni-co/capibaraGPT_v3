@echo off
REM Script para iniciar fine-tuning GPT-OSS-20B basado en configuración existente
REM Usa la configuración de Capibara6 que ya tienes funcionando

echo 🚀 Iniciando Fine-tuning GPT-OSS-20B (Capibara6)
echo ===============================================

REM Variables
set TPU_NAME=tx-5-oss-20b
set PROJECT_ID=mamba-001
set ZONE=us-central1-a
set VM_IP=34.175.215.109

echo 📋 Configuración detectada:
echo VM IP: %VM_IP%
echo Proyecto: %PROJECT_ID%
echo TPU: %TPU_NAME%
echo Zona: %ZONE%

echo.
echo 🔍 Verificando autenticación en Google Cloud...
gcloud auth list
if %errorlevel% neq 0 (
    echo ❌ Error: No estás autenticado en Google Cloud
    echo Ejecuta: gcloud auth login
    pause
    exit /b 1
)

echo.
echo 📋 Verificando estado de la TPU...
gcloud compute tpus describe %TPU_NAME% --zone=%ZONE% --project=%PROJECT_ID%
if %errorlevel% neq 0 (
    echo ❌ Error: TPU %TPU_NAME% no encontrada
    pause
    exit /b 1
)

echo.
echo 🔗 Conectando a la VM TPU y configurando entorno...
echo ⚠️ Esto puede tomar 30-60 minutos para instalar todas las dependencias

gcloud compute tpus tpu-vm ssh --zone %ZONE% --project %PROJECT_ID% %TPU_NAME% --command="
echo '📥 Descargando script de configuración basado en Capibara6...'
mkdir -p ~/capibara6-finetuning
cd ~/capibara6-finetuning

echo '🚀 Ejecutando configuración completa basada en tu setup existente...'
curl -sSL https://raw.githubusercontent.com/tu-usuario/capibara6/main/fine-tuning/scripts/setup_finetuning_from_existing.sh -o setup_finetuning_from_existing.sh
chmod +x setup_finetuning_from_existing.sh
./setup_finetuning_from_existing.sh

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
echo 🤖 El modelo se entrenará basándose en tu configuración existente de Capibara6
echo ⚠️ El entrenamiento puede tomar varias horas. Revisa los logs regularmente.
pause

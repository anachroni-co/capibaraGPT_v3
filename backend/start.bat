@echo off
REM Script de inicio rápido para el backend de capibara6 (Windows)

echo 🦫 Iniciando backend de capibara6...

REM Verificar si existe el entorno virtual
if not exist "venv\" (
    echo 📦 Creando entorno virtual...
    python -m venv venv
)

REM Activar entorno virtual
echo 🔌 Activando entorno virtual...
call venv\Scripts\activate.bat

REM Instalar/actualizar dependencias
echo 📥 Instalando dependencias...
pip install -q -r requirements.txt

REM Verificar si existe .env
if not exist ".env" (
    echo ⚠️  Archivo .env no encontrado!
    echo 📝 Copia env.example a .env y configura tus credenciales SMTP:
    echo    copy env.example .env
    echo    notepad .env
    exit /b 1
)

REM Crear directorio de datos
if not exist "user_data\" mkdir user_data

REM Iniciar servidor
echo 🚀 Iniciando servidor en http://localhost:5000
python server.py


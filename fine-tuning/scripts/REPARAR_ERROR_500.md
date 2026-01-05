# 🔧 Guía para Reparar el Error 500 en el Backend

## Paso 1: Autenticarse en Google Cloud

```cmd
gcloud auth login
```

## Paso 2: Ejecutar el Script de Reparación

```cmd
fine-tuning\scripts\fix_500_error.bat
```

## Paso 3: Si el Script no Funciona, Ejecutar Manualmente

### Conectar a la VM:
```cmd
gcloud compute ssh --zone "europe-southwest1-b" "gpt-oss-20b" --project "mamba-001"
```

### En la VM, ejecutar estos comandos:

```bash
# 1. Verificar qué procesos están corriendo
ps aux | grep python | grep -v grep

# 2. Verificar puertos en uso
sudo netstat -tuln | grep -E ":(5001|8080)"

# 3. Verificar si el modelo GPT-OSS-20B responde
curl http://localhost:8080/health

# 4. Detener servidor integrado si está corriendo
pkill -f capibara6_integrated_server

# 5. Ir al directorio del backend
cd ~/capibara6/backend

# 6. Verificar que existe el archivo
ls -la capibara6_integrated_server.py

# 7. Iniciar el servidor
nohup python3 capibara6_integrated_server.py > ../logs/server.log 2>&1 &

# 8. Esperar 3 segundos y verificar
sleep 3
curl http://localhost:5001/health

# 9. Ver logs si hay errores
tail -30 ../logs/server.log
```

## Paso 4: Verificar desde el Navegador

Después de reiniciar el servidor, prueba en el navegador:
- Abre https://www.capibara6.com
- Envía un mensaje de prueba
- Si el error persiste, revisa los logs

## Comandos Útiles para Diagnóstico

```bash
# Ver logs en tiempo real
tail -f ~/capibara6/logs/server.log

# Ver errores recientes
tail -50 ~/capibara6/logs/errors.log

# Verificar procesos Python
ps aux | grep python

# Verificar conexión del servidor al modelo
curl -v http://localhost:8080/completion -X POST -H "Content-Type: application/json" -d '{"prompt":"test","n_predict":10}'

# Reiniciar todo el sistema
pkill -f python
sleep 2
cd ~/capibara6/backend
nohup python3 capibara6_integrated_server.py > ../logs/server.log 2>&1 &
```

## Problemas Comunes y Soluciones

### 1. Error: "Address already in use"
```bash
# Liberar puerto 5001
sudo fuser -k 5001/tcp
# O
sudo kill -9 $(lsof -t -i:5001)
```

### 2. Error: "Connection refused" en puerto 8080
```bash
# Verificar si el modelo está corriendo
ps aux | grep llama-server
# Si no está, iniciarlo según tu configuración
```

### 3. Error: "Module not found"
```bash
# Instalar dependencias
cd ~/capibara6/backend
pip3 install -r requirements.txt
```

### 4. Error: "Permission denied"
```bash
# Verificar permisos
chmod +x capibara6_integrated_server.py
```

## Verificación Final

Después de reiniciar, verifica que todo esté funcionando:

```bash
# 1. Servidor integrado responde
curl http://localhost:5001/health

# 2. Modelo GPT-OSS-20B responde  
curl http://localhost:8080/health

# 3. Endpoint /api/chat funciona
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"hola","conversation":[]}'
```

Si todos estos comandos funcionan, el backend debería estar operativo.

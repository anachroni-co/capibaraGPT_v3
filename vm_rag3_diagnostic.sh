#!/bin/bash
# VM rag3 - Script de diagnóstico completo
# Recopila información sobre servicios, bases de datos y puertos activos

set -e

OUTPUT_FILE="vm_rag3_diagnostic_$(date +%Y%m%d_%H%M%S).txt"

echo "🔍 VM rag3 - Diagnóstico Completo de Infraestructura" | tee "$OUTPUT_FILE"
echo "Fecha: $(date)" | tee -a "$OUTPUT_FILE"
echo "════════════════════════════════════════════════════════════════" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Información del sistema
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "1. INFORMACIÓN DEL SISTEMA" | tee -a "$OUTPUT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Hostname:" | tee -a "$OUTPUT_FILE"
hostname | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Sistema Operativo:" | tee -a "$OUTPUT_FILE"
uname -a | tee -a "$OUTPUT_FILE"
cat /etc/os-release | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "IP Address:" | tee -a "$OUTPUT_FILE"
ip addr show | grep "inet " | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Puertos en escucha
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "2. PUERTOS EN ESCUCHA" | tee -a "$OUTPUT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Todos los puertos TCP en escucha:" | tee -a "$OUTPUT_FILE"
sudo netstat -tlnp 2>/dev/null || sudo ss -tlnp 2>/dev/null | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Buscar puertos específicos de bases de datos
echo "Puertos específicos de bases de datos:" | tee -a "$OUTPUT_FILE"
echo "  - Milvus (19530):" | tee -a "$OUTPUT_FILE"
sudo netstat -tlnp 2>/dev/null | grep ":19530" || echo "    No encontrado" | tee -a "$OUTPUT_FILE"

echo "  - Nebula Graph Meta (9559):" | tee -a "$OUTPUT_FILE"
sudo netstat -tlnp 2>/dev/null | grep ":9559" || echo "    No encontrado" | tee -a "$OUTPUT_FILE"

echo "  - Nebula Graph Storage (9779):" | tee -a "$OUTPUT_FILE"
sudo netstat -tlnp 2>/dev/null | grep ":9779" || echo "    No encontrado" | tee -a "$OUTPUT_FILE"

echo "  - Nebula Graph Query (9669):" | tee -a "$OUTPUT_FILE"
sudo netstat -tlnp 2>/dev/null | grep ":9669" || echo "    No encontrado" | tee -a "$OUTPUT_FILE"

echo "  - ChromaDB (8000):" | tee -a "$OUTPUT_FILE"
sudo netstat -tlnp 2>/dev/null | grep ":8000" || echo "    No encontrado" | tee -a "$OUTPUT_FILE"

echo "  - Neo4j (7687):" | tee -a "$OUTPUT_FILE"
sudo netstat -tlnp 2>/dev/null | grep ":7687" || echo "    No encontrado" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Procesos en ejecución
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "3. PROCESOS RELEVANTES" | tee -a "$OUTPUT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Procesos Python:" | tee -a "$OUTPUT_FILE"
ps aux | grep python | grep -v grep | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Procesos con 'milvus':" | tee -a "$OUTPUT_FILE"
ps aux | grep -i milvus | grep -v grep | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Procesos con 'nebula':" | tee -a "$OUTPUT_FILE"
ps aux | grep -i nebula | grep -v grep | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Procesos con 'chroma':" | tee -a "$OUTPUT_FILE"
ps aux | grep -i chroma | grep -v grep | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Procesos con 'bridge':" | tee -a "$OUTPUT_FILE"
ps aux | grep -i bridge | grep -v grep | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Procesos con 'rag':" | tee -a "$OUTPUT_FILE"
ps aux | grep -i rag | grep -v grep | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Contenedores Docker
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "4. CONTENEDORES DOCKER" | tee -a "$OUTPUT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

if command -v docker &> /dev/null; then
    echo "Contenedores en ejecución:" | tee -a "$OUTPUT_FILE"
    sudo docker ps | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"

    echo "Todos los contenedores (incluyendo detenidos):" | tee -a "$OUTPUT_FILE"
    sudo docker ps -a | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"

    echo "Imágenes Docker instaladas:" | tee -a "$OUTPUT_FILE"
    sudo docker images | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"

    echo "Docker Compose:" | tee -a "$OUTPUT_FILE"
    find /home -name "docker-compose.yml" -o -name "docker-compose.yaml" 2>/dev/null | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
else
    echo "Docker no está instalado" | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
fi

# Servicios systemd
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "5. SERVICIOS SYSTEMD" | tee -a "$OUTPUT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Servicios activos relacionados con bases de datos:" | tee -a "$OUTPUT_FILE"
systemctl list-units --type=service --state=running | grep -E "(milvus|nebula|chroma|postgres|redis|mongo)" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Todos los servicios de bases de datos:" | tee -a "$OUTPUT_FILE"
systemctl list-unit-files | grep -E "(milvus|nebula|chroma|postgres|redis|mongo)" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Buscar instalaciones de software
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "6. INSTALACIONES DE SOFTWARE" | tee -a "$OUTPUT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Milvus:" | tee -a "$OUTPUT_FILE"
which milvus 2>/dev/null || echo "  No encontrado en PATH" | tee -a "$OUTPUT_FILE"
find /opt -name "*milvus*" -type d 2>/dev/null | tee -a "$OUTPUT_FILE"
find /usr/local -name "*milvus*" -type d 2>/dev/null | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Nebula Graph:" | tee -a "$OUTPUT_FILE"
which nebula-graphd 2>/dev/null || echo "  No encontrado en PATH" | tee -a "$OUTPUT_FILE"
find /opt -name "*nebula*" -type d 2>/dev/null | tee -a "$OUTPUT_FILE"
find /usr/local -name "*nebula*" -type d 2>/dev/null | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "ChromaDB:" | tee -a "$OUTPUT_FILE"
which chroma 2>/dev/null || echo "  No encontrado en PATH" | tee -a "$OUTPUT_FILE"
pip list 2>/dev/null | grep -i chroma || pip3 list 2>/dev/null | grep -i chroma || echo "  No encontrado en pip" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Archivos de configuración
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "7. ARCHIVOS DE CONFIGURACIÓN" | tee -a "$OUTPUT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Archivos .env:" | tee -a "$OUTPUT_FILE"
find /home -name ".env" -type f 2>/dev/null | head -20 | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Archivos de configuración YAML/JSON:" | tee -a "$OUTPUT_FILE"
find /home -name "*.yaml" -o -name "*.yml" -o -name "*config.json" 2>/dev/null | grep -E "(milvus|nebula|chroma|bridge|rag)" | head -20 | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Directorios del proyecto
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "8. ESTRUCTURA DE DIRECTORIOS DEL PROYECTO" | tee -a "$OUTPUT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Buscar proyectos en /home:" | tee -a "$OUTPUT_FILE"
find /home -maxdepth 3 -type d -name "*capibara*" -o -name "*rag*" -o -name "*bridge*" 2>/dev/null | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Archivos Python en directorio home:" | tee -a "$OUTPUT_FILE"
find /home -maxdepth 4 -name "*.py" -type f 2>/dev/null | grep -E "(bridge|rag|milvus|nebula|chroma)" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Logs recientes
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "9. LOGS RECIENTES" | tee -a "$OUTPUT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Logs de systemd para servicios relevantes:" | tee -a "$OUTPUT_FILE"
for service in milvus nebula-graphd nebula-metad chromadb; do
    if systemctl list-units --all | grep -q "$service"; then
        echo "  Logs de $service:" | tee -a "$OUTPUT_FILE"
        sudo journalctl -u "$service" -n 20 --no-pager 2>/dev/null | tee -a "$OUTPUT_FILE"
    fi
done
echo "" | tee -a "$OUTPUT_FILE"

# Verificar conectividad a puertos
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "10. VERIFICACIÓN DE ENDPOINTS" | tee -a "$OUTPUT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Probando endpoints HTTP comunes:" | tee -a "$OUTPUT_FILE"
for port in 8000 8001 5000 5001 19530 9669; do
    echo "  Puerto $port:" | tee -a "$OUTPUT_FILE"
    curl -s -o /dev/null -w "    HTTP Status: %{http_code}\n" http://localhost:$port 2>/dev/null | tee -a "$OUTPUT_FILE" || echo "    No accesible" | tee -a "$OUTPUT_FILE"
done
echo "" | tee -a "$OUTPUT_FILE"

# Resumen de capacidades instaladas
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "11. RESUMEN DE CAPACIDADES" | tee -a "$OUTPUT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Python:" | tee -a "$OUTPUT_FILE"
python3 --version 2>&1 | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Paquetes Python relevantes:" | tee -a "$OUTPUT_FILE"
pip3 list 2>/dev/null | grep -E "(milvus|pymilvus|nebula|chroma|chromadb|langchain|openai|faiss|pinecone)" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Node.js:" | tee -a "$OUTPUT_FILE"
node --version 2>&1 | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Docker:" | tee -a "$OUTPUT_FILE"
docker --version 2>&1 | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "Docker Compose:" | tee -a "$OUTPUT_FILE"
docker-compose --version 2>&1 | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Finalizar
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "DIAGNÓSTICO COMPLETO" | tee -a "$OUTPUT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"
echo "✅ Reporte guardado en: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"
echo "Para compartir este reporte:" | tee -a "$OUTPUT_FILE"
echo "  cat $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"
echo "O descargarlo:" | tee -a "$OUTPUT_FILE"
echo "  gcloud compute scp --zone \"europe-west2-c\" rag3:~/$OUTPUT_FILE . --project \"mamba-001\"" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

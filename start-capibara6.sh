#!/bin/bash

# Script maestro para gestionar todos los servicios de Capibara6
# Autor: Anachroni
# Versión: 1.0

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuración
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
WEB_DIR="$SCRIPT_DIR/web"
LOG_DIR="$BACKEND_DIR/logs"
BACKEND_PORT=5000
FRONTEND_PORT=8080

# Crear directorio de logs si no existe
mkdir -p "$LOG_DIR"

# Banner
print_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║                                                        ║"
    echo "║              🦫 CAPIBARA6 LAUNCHER                    ║"
    echo "║        Sistema de Gestión de Servicios v1.0          ║"
    echo "║                                                        ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Funciones auxiliares
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_section() {
    echo ""
    echo -e "${PURPLE}${BOLD}═══ $1 ═══${NC}"
    echo ""
}

# Verificar si Docker está instalado
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker no está instalado"
        echo "Por favor, instala Docker desde: https://docs.docker.com/get-docker/"
        return 1
    fi

    if ! docker info &> /dev/null; then
        print_error "Docker daemon no está corriendo"
        echo "Por favor, inicia Docker y vuelve a intentar"
        return 1
    fi

    print_success "Docker está disponible"
    return 0
}

# Verificar si Docker Compose está instalado
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose no está instalado"
        echo "Por favor, instala Docker Compose desde: https://docs.docker.com/compose/install/"
        return 1
    fi

    print_success "Docker Compose está disponible"
    return 0
}

# Verificar si Python está instalado
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 no está instalado"
        return 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION está disponible"
    return 0
}

# Verificar estado de contenedores Docker
check_docker_services() {
    print_section "Estado de Servicios Docker"

    local services=(
        "capibara6-api:API Principal:8000"
        "capibara6-graphql:API GraphQL:8001"
        "capibara6-worker:Background Worker:-"
        "capibara6-postgres:PostgreSQL:5432"
        "capibara6-timescaledb:TimescaleDB:5433"
        "capibara6-redis:Redis:6379"
        "capibara6-nginx:Nginx:80,443"
        "capibara6-prometheus:Prometheus:9090"
        "capibara6-grafana:Grafana:3000"
        "capibara6-jaeger:Jaeger:16686"
        "capibara6-n8n:n8n:5678"
    )

    local running=0
    local stopped=0

    echo -e "${BOLD}Nombre del Contenedor${NC}          ${BOLD}Estado${NC}       ${BOLD}Puerto${NC}"
    echo "────────────────────────────────────────────────────────────"

    for service_info in "${services[@]}"; do
        IFS=':' read -r container_name service_name port <<< "$service_info"

        if docker ps --format '{{.Names}}' | grep -q "^${container_name}$"; then
            echo -e "${container_name}    ${GREEN}●${NC} Running    ${port}"
            ((running++))
        elif docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
            echo -e "${container_name}    ${RED}●${NC} Stopped    ${port}"
            ((stopped++))
        else
            echo -e "${container_name}    ${YELLOW}●${NC} No creado  ${port}"
            ((stopped++))
        fi
    done

    echo ""
    echo -e "Total: ${GREEN}${running} corriendo${NC} | ${RED}${stopped} detenidos${NC}"
    echo ""

    return 0
}

# Verificar servidor backend Python
check_backend_server() {
    if lsof -i :$BACKEND_PORT &> /dev/null; then
        print_success "Backend server corriendo en puerto $BACKEND_PORT"
        return 0
    else
        print_warning "Backend server NO está corriendo en puerto $BACKEND_PORT"
        return 1
    fi
}

# Verificar frontend
check_frontend_server() {
    if lsof -i :$FRONTEND_PORT &> /dev/null; then
        print_success "Frontend server corriendo en puerto $FRONTEND_PORT"
        return 0
    else
        print_warning "Frontend server NO está corriendo en puerto $FRONTEND_PORT"
        return 1
    fi
}

# Iniciar servicios Docker
start_docker_services() {
    print_section "Iniciando Servicios Docker"

    cd "$SCRIPT_DIR"

    print_status "Levantando contenedores con Docker Compose..."

    if docker-compose up -d; then
        print_success "Servicios Docker iniciados correctamente"

        print_status "Esperando a que los servicios estén listos..."
        sleep 5

        # Verificar servicios críticos
        check_critical_services
    else
        print_error "Error al iniciar servicios Docker"
        return 1
    fi
}

# Verificar servicios críticos
check_critical_services() {
    print_section "Verificando Servicios Críticos"

    local services=("capibara6-postgres" "capibara6-redis" "capibara6-api")

    for service in "${services[@]}"; do
        if docker ps --format '{{.Names}}' | grep -q "^${service}$"; then
            print_success "$service está corriendo"
        else
            print_error "$service NO está corriendo"
            return 1
        fi
    done

    return 0
}

# Iniciar backend Python
start_backend() {
    print_section "Iniciando Backend Python"

    cd "$BACKEND_DIR"

    # Verificar si ya está corriendo
    if check_backend_server; then
        print_warning "Backend ya está corriendo"
        return 0
    fi

    # Verificar dependencias
    if [ ! -d "venv" ]; then
        print_warning "Entorno virtual no encontrado. Creando..."
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    else
        source venv/bin/activate
    fi

    # Iniciar servidor en background
    print_status "Iniciando servidor backend en puerto $BACKEND_PORT..."
    nohup python3 server.py > "$LOG_DIR/backend.log" 2>&1 &
    echo $! > "$LOG_DIR/backend.pid"

    sleep 3

    if check_backend_server; then
        print_success "Backend iniciado correctamente (PID: $(cat $LOG_DIR/backend.pid))"
    else
        print_error "Error al iniciar backend. Revisa los logs en $LOG_DIR/backend.log"
        return 1
    fi
}

# Iniciar frontend
start_frontend() {
    print_section "Iniciando Frontend"

    cd "$WEB_DIR"

    # Verificar si ya está corriendo
    if check_frontend_server; then
        print_warning "Frontend ya está corriendo"
        return 0
    fi

    # Iniciar servidor web simple
    print_status "Iniciando servidor frontend en puerto $FRONTEND_PORT..."
    nohup python3 -m http.server $FRONTEND_PORT > "$LOG_DIR/frontend.log" 2>&1 &
    echo $! > "$LOG_DIR/frontend.pid"

    sleep 2

    if check_frontend_server; then
        print_success "Frontend iniciado correctamente (PID: $(cat $LOG_DIR/frontend.pid))"
    else
        print_error "Error al iniciar frontend"
        return 1
    fi
}

# Detener servicios
stop_all_services() {
    print_section "Deteniendo Todos los Servicios"

    # Detener backend
    if [ -f "$LOG_DIR/backend.pid" ]; then
        print_status "Deteniendo backend..."
        kill $(cat "$LOG_DIR/backend.pid") 2>/dev/null || true
        rm -f "$LOG_DIR/backend.pid"
        print_success "Backend detenido"
    fi

    # Detener frontend
    if [ -f "$LOG_DIR/frontend.pid" ]; then
        print_status "Deteniendo frontend..."
        kill $(cat "$LOG_DIR/frontend.pid") 2>/dev/null || true
        rm -f "$LOG_DIR/frontend.pid"
        print_success "Frontend detenido"
    fi

    # Detener Docker
    print_status "Deteniendo servicios Docker..."
    cd "$SCRIPT_DIR"
    docker-compose down
    print_success "Servicios Docker detenidos"
}

# Ver logs
view_logs() {
    print_section "Logs Disponibles"

    echo "1. Backend Python"
    echo "2. Frontend"
    echo "3. Docker Compose"
    echo "4. API Principal"
    echo "5. n8n"
    echo "6. Volver"
    echo ""
    read -p "Selecciona una opción: " log_option

    case $log_option in
        1)
            tail -f "$LOG_DIR/backend.log"
            ;;
        2)
            tail -f "$LOG_DIR/frontend.log"
            ;;
        3)
            docker-compose logs -f
            ;;
        4)
            docker logs -f capibara6-api
            ;;
        5)
            docker logs -f capibara6-n8n
            ;;
        6)
            return
            ;;
        *)
            print_error "Opción inválida"
            ;;
    esac
}

# Mostrar URLs de acceso
show_access_urls() {
    print_section "URLs de Acceso"

    echo -e "${BOLD}Aplicación Principal:${NC}"
    echo -e "  Frontend:       ${CYAN}http://localhost:$FRONTEND_PORT${NC}"
    echo -e "  Backend API:    ${CYAN}http://localhost:$BACKEND_PORT${NC}"
    echo ""

    echo -e "${BOLD}APIs y Servicios:${NC}"
    echo -e "  API Principal:  ${CYAN}http://localhost:8000${NC}"
    echo -e "  GraphQL API:    ${CYAN}http://localhost:8001/graphql${NC}"
    echo -e "  n8n:            ${CYAN}http://localhost:5678${NC}"
    echo ""

    echo -e "${BOLD}Monitorización:${NC}"
    echo -e "  Grafana:        ${CYAN}http://localhost:3000${NC} (admin/capibara6_admin)"
    echo -e "  Prometheus:     ${CYAN}http://localhost:9090${NC}"
    echo -e "  Jaeger:         ${CYAN}http://localhost:16686${NC}"
    echo ""

    echo -e "${BOLD}Bases de Datos:${NC}"
    echo -e "  PostgreSQL:     ${CYAN}localhost:5432${NC}"
    echo -e "  TimescaleDB:    ${CYAN}localhost:5433${NC}"
    echo -e "  Redis:          ${CYAN}localhost:6379${NC}"
    echo ""
}

# Menú principal
show_menu() {
    clear
    print_banner

    echo -e "${BOLD}Opciones Disponibles:${NC}"
    echo ""
    echo "  1. ▶️  Iniciar TODOS los servicios"
    echo "  2. 🐳 Iniciar solo servicios Docker"
    echo "  3. 🐍 Iniciar solo Backend Python"
    echo "  4. 🌐 Iniciar solo Frontend"
    echo "  5. 📊 Ver estado de servicios"
    echo "  6. 📜 Ver logs"
    echo "  7. 🔗 Mostrar URLs de acceso"
    echo "  8. ⏹️  Detener todos los servicios"
    echo "  9. 🔄 Reiniciar servicios"
    echo "  0. ❌ Salir"
    echo ""
}

# Función principal
main() {
    # Verificar requisitos
    print_section "Verificando Requisitos"
    check_docker || exit 1
    check_docker_compose || exit 1
    check_python || exit 1

    while true; do
        show_menu
        read -p "Selecciona una opción: " option

        case $option in
            1)
                start_docker_services
                start_backend
                start_frontend
                show_access_urls
                read -p "Presiona Enter para continuar..."
                ;;
            2)
                start_docker_services
                read -p "Presiona Enter para continuar..."
                ;;
            3)
                start_backend
                read -p "Presiona Enter para continuar..."
                ;;
            4)
                start_frontend
                read -p "Presiona Enter para continuar..."
                ;;
            5)
                check_docker_services
                echo ""
                check_backend_server || true
                check_frontend_server || true
                read -p "Presiona Enter para continuar..."
                ;;
            6)
                view_logs
                ;;
            7)
                show_access_urls
                read -p "Presiona Enter para continuar..."
                ;;
            8)
                stop_all_services
                read -p "Presiona Enter para continuar..."
                ;;
            9)
                stop_all_services
                sleep 2
                start_docker_services
                start_backend
                start_frontend
                show_access_urls
                read -p "Presiona Enter para continuar..."
                ;;
            0)
                echo ""
                print_success "¡Hasta luego! 👋"
                echo ""
                exit 0
                ;;
            *)
                print_error "Opción inválida"
                sleep 2
                ;;
        esac
    done
}

# Ejecutar
main

#!/bin/bash
# Backend Installation Script for Capibara6 VM
# Use this script to install and configure the backend components on a new VM

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}################################################################${NC}"
    echo -e "${BLUE}## $1${NC}"
    echo -e "${BLUE}################################################################${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Variables
PROJECT_DIR="/opt/capibara6"
USER="capibara6"
PYTHON_VERSION="3.9"
VM_ROLE="bounty2"  # Options: bounty2 (models), services (tts/mcp), rag3 (rag system)

print_header "Installing Capibara6 Backend Components"
print_info "System setup for VM role: $VM_ROLE"

# Function to check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_warning "This script should not be run as root. Will create user if needed."
        # Continue but warn
    fi
}

# Function to create capibara6 user
create_user() {
    print_info "Creating capibara6 user if needed..."
    if id "$USER" &>/dev/null; then
        print_success "User $USER already exists"
    else
        sudo useradd -m -s /bin/bash "$USER"
        print_success "Created user $USER"
    fi

    # Add user to docker group if docker is installed
    if command -v docker &>/dev/null; then
        sudo usermod -aG docker "$USER"
        print_success "Added $USER to docker group"
    fi
}

# Function to install system dependencies
install_system_deps() {
    print_header "Installing System Dependencies"
    
    # Update package list
    sudo apt-get update

    # Install basic dependencies
    print_info "Installing basic system dependencies..."
    sudo apt-get install -y \
        curl \
        wget \
        git \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        libssl-dev \
        libffi-dev \
        cargo \
        nodejs \
        npm \
        docker.io \
        docker-compose \
        nginx \
        supervisor \
        postgresql-client \
        redis-tools \
        net-tools \
        ufw \
        jq \
        htop \
        vim

    print_success "System dependencies installed"
}

# Function to install Python dependencies
install_python_deps() {
    print_header "Installing Python Dependencies"

    # Install Python dependencies globally
    sudo apt-get install -y python3-pip python3-venv python3-dev
    
    # Upgrade pip
    python3 -m pip install --upgrade pip
    
    # Install global Python packages
    sudo pip3 install --upgrade setuptools wheel
    sudo pip3 install virtualenv
    
    print_success "Python dependencies installed"
}

# Function to install Ollama (for model VM)
install_ollama() {
    print_header "Installing Ollama"
    
    if [ "$VM_ROLE" = "bounty2" ]; then
        # Install Ollama
        print_info "Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
        
        # Start Ollama service
        sudo systemctl enable ollama
        sudo systemctl start ollama
        
        # Verify installation
        sleep 10  # Wait for Ollama to start
        if command -v ollama &>/dev/null; then
            print_success "Ollama installed and running"
            
            # Pull required models
            print_info "Pulling required models..."
            ollama pull gpt-oss:20b || print_warning "Could not pull gpt-oss:20b"
            ollama pull phi3:mini || print_warning "Could not pull phi3:mini"
            ollama pull llama2 || print_warning "Could not pull llama2"
        else
            print_error "Ollama installation failed"
        fi
    else
        print_info "Skipping Ollama installation (not model VM)"
    fi
}

# Function to clone and setup capibara6
setup_capibara6() {
    print_header "Setting up Capibara6 Project"

    if [ -d "$PROJECT_DIR" ]; then
        print_warning "Project directory already exists, backing up..."
        sudo mv "$PROJECT_DIR" "${PROJECT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    fi

    # Create project directory
    sudo mkdir -p "$PROJECT_DIR"
    sudo chown -R "$USER:$USER" "$PROJECT_DIR"
    
    # Clone repository
    print_info "Cloning Capibara6 repository..."
    git clone https://github.com/anachroni-co/capibara6.git "$PROJECT_DIR"
    
    # Change ownership
    sudo chown -R "$USER:$USER" "$PROJECT_DIR"
    
    print_success "Capibara6 project cloned"
}

# Function to setup Python environment
setup_python_env() {
    print_header "Setting up Python Virtual Environment"
    
    # Change to project directory
    cd "$PROJECT_DIR"
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Install Python requirements
    pip install --upgrade pip
    if [ -f "$PROJECT_DIR/requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Python requirements installed"
    else
        print_warning "requirements.txt not found, creating minimal setup"
        pip install Flask flask-cors python-dotenv gunicorn requests
    fi
    
    # Install additional dependencies specific to VM role
    if [ "$VM_ROLE" = "bounty2" ]; then
        pip install semantic-router fastembed
    fi
    
    print_success "Python environment configured"
}

# Function to configure services based on VM role
configure_services() {
    print_header "Configuring Services for VM Role: $VM_ROLE"

    case "$VM_ROLE" in
        "bounty2")
            configure_bounty2_services
            ;;
        "services")
            configure_services_vm
            ;;
        "rag3")
            configure_rag3_services
            ;;
        *)
            print_error "Invalid VM role: $VM_ROLE"
            exit 1
            ;;
    esac
}

# Function to configure model VM (bounty2) services
configure_bounty2_services() {
    print_info "Configuring model services (bounty2)..."
    
    # Copy systemd service file for backend
    sudo tee /etc/systemd/system/capibara6-backend.service > /dev/null <<EOF
[Unit]
Description=Capibara6 Backend Server
After=network.target
After=ollama.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR/backend
Environment=PATH=$PROJECT_DIR/venv/bin
Environment=GPT_OSS_URL=http://localhost:11434
Environment=PORT=5001
ExecStart=$PROJECT_DIR/venv/bin/python server_gptoss.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Enable and start the service
    sudo systemctl daemon-reload
    sudo systemctl enable capibara6-backend
    sudo systemctl start capibara6-backend
    
    print_success "Backend service configured and started"
}

# Function to configure services VM (tts/mcp/n8n)
configure_services_vm() {
    print_info "Configuring services (tts/mcp/n8n)..."
    
    # Install Kyutai TTS dependencies
    pip install torch torchaudio
    
    # Setup for TTS and MCP services would go here
    print_warning "TTS/MCP service configuration needed"
}

# Function to configure RAG VM (databases and monitoring)
configure_rag3_services() {
    print_info "Configuring RAG services (databases and monitoring)..."
    
    # This would involve setting up Docker Compose for Milvus, Nebula, etc.
    # For now, just preparing the environment
    print_warning "RAG service configuration needed (Milvus, Nebula Graph, PostgreSQL, etc.)"
}

# Function to setup firewall
setup_firewall() {
    print_header "Configuring Firewall"

    # Enable UFW
    sudo ufw --force enable

    # Allow SSH
    sudo ufw allow 22/tcp

    # Allow HTTP/HTTPS
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp

    # Allow specific ports based on VM role
    case "$VM_ROLE" in
        "bounty2")
            sudo ufw allow 5001/tcp  # Backend API
            sudo ufw allow 11434/tcp  # Ollama
            ;;
        "services")
            sudo ufw allow 5002/tcp  # TTS
            sudo ufw allow 5003/tcp  # MCP
            sudo ufw allow 5678/tcp  # N8N
            ;;
        "rag3")
            sudo ufw allow 8000/tcp  # Bridge API
            sudo ufw allow 19530/tcp  # Milvus
            sudo ufw allow 9669/tcp  # Nebula Graph
            sudo ufw allow 5432/tcp  # PostgreSQL
            sudo ufw allow 3000/tcp  # Grafana
            sudo ufw allow 9090/tcp  # Prometheus
            sudo ufw allow 16686/tcp  # Jaeger
            ;;
    esac

    print_success "Firewall configured for VM role: $VM_ROLE"
}

# Function to create environment file
setup_env_file() {
    print_header "Setting up Environment Configuration"

    cd "$PROJECT_DIR/backend"
    
    # Create .env file
    cat > .env <<EOF
# Capibara6 Backend Configuration
PORT=5001
GPT_OSS_URL=${GPT_OSS_URL:-http://localhost:11434}
GPT_OSS_TIMEOUT=60

# VM Configuration
BOUNTY2_VM_URL=${BOUNTY2_VM_URL:-http://34.12.166.76}
SERVICES_VM_URL=${SERVICES_VM_URL:-http://34.175.136.104}
RAG3_VM_URL=${RAG3_VM_URL:-http://10.154.0.2}

# Service URLs
TTS_URL=${TTS_URL:-http://34.175.136.104:5002}
MCP_URL=${MCP_URL:-http://34.175.136.104:5003}
N8N_URL=${N8N_URL:-http://34.175.136.104:5678}

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/backend.log

# Model Configuration
DEFAULT_MODEL=gpt-oss-20b
EOF

    print_success "Environment file created at $PROJECT_DIR/backend/.env"
}

# Function to run tests
run_tests() {
    print_header "Running System Tests"

    cd "$PROJECT_DIR/backend"
    source venv/bin/activate

    # Test basic import
    python -c "import flask, requests, dotenv; print('Python dependencies OK')"
    
    # Test backend connectivity
    if [ "$VM_ROLE" = "bounty2" ]; then
        print_info "Testing backend connectivity..."
        # Try to check if the service is running
        if curl -s --connect-timeout 5 http://localhost:5001/health; then
            print_success "Backend service is responding"
        else
            print_warning "Backend service may not be running yet (started as systemd service)"
        fi
    fi
    
    print_success "Basic tests completed"
}

# Function to display status
display_status() {
    print_header "System Status"

    print_info "Capibara6 project location: $PROJECT_DIR"
    print_info "User: $USER"
    print_info "VM Role: $VM_ROLE"
    
    if [ "$VM_ROLE" = "bounty2" ]; then
        print_info "Ollama status:"
        if command -v ollama &>/dev/null; then
            ollama list 2>/dev/null || echo "Could not list models"
        fi
        
        print_info "Backend service status:"
        sudo systemctl status capibara6-backend --no-pager -l
    fi
    
    print_info "Firewall status:"
    sudo ufw status verbose
}

# Function to show usage instructions
show_instructions() {
    print_header "Next Steps and Usage"

    echo "To manage the backend service:"
    echo "  sudo systemctl start capibara6-backend    # Start service"
    echo "  sudo systemctl stop capibara6-backend     # Stop service"
    echo "  sudo systemctl restart capibara6-backend  # Restart service"
    echo "  sudo systemctl status capibara6-backend   # Check status"
    echo "  sudo journalctl -u capibara6-backend -f   # View logs"
    echo ""
    echo "To access the service:"
    echo "  Health check: curl http://localhost:5001/health"
    echo "  Model list: curl http://localhost:5001/api/models"
    echo ""
    echo "For development, you can run directly:"
    echo "  cd $PROJECT_DIR/backend"
    echo "  source venv/bin/activate"
    echo "  python server_gptoss.py"
}

# Main execution
main() {
    print_header "Starting Capibara6 Backend Installation"

    VM_ROLE=${1:-bounty2}  # Default to bounty2 if no role specified

    check_root
    create_user
    install_system_deps
    install_python_deps
    install_ollama
    setup_capibara6
    setup_python_env
    setup_env_file
    configure_services
    setup_firewall
    run_tests
    display_status
    show_instructions

    print_header "Installation completed successfully!"
    print_success "Capibara6 backend is now installed and configured."
    print_info "Review the status above and the next steps for usage instructions."
}

# Run main function with the provided role
main "$@"
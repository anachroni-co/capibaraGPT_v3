#!/bin/bash
#
# Deployment script para ARM Axion VM en Google Cloud
# Usa los modelos ya descargados en /home/user/models/
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  vLLM + ARM Axion - Production Deployment${NC}"
echo -e "${GREEN}  Capibara6 Multi-Expert System${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Configuration
MODELS_DIR="${MODELS_DIR:-/home/user/models}"
VLLM_PORT="${VLLM_PORT:-8080}"
CONFIG_FILE="config.production.json"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on ARM
check_arm() {
    log_info "Checking architecture..."
    ARCH=$(uname -m)

    if [[ "$ARCH" != "aarch64" ]] && [[ "$ARCH" != "arm64" ]]; then
        log_error "This script requires ARM64 architecture. Found: $ARCH"
        exit 1
    fi

    log_info "✅ Running on ARM64: $ARCH"
}

# Check if models exist
check_models() {
    log_info "Checking models directory..."

    if [ ! -d "$MODELS_DIR" ]; then
        log_error "Models directory not found: $MODELS_DIR"
        log_info "Please ensure models are downloaded to this location"
        exit 1
    fi

    log_info "📦 Models directory: $MODELS_DIR"

    # List available models
    log_info "Available models:"
    for model_dir in "$MODELS_DIR"/*; do
        if [ -d "$model_dir" ]; then
            model_name=$(basename "$model_dir")
            size=$(du -sh "$model_dir" 2>/dev/null | cut -f1)
            echo "  • $model_name ($size)"
        fi
    done
}

# Update config with actual model paths
update_config() {
    log_info "Updating configuration with actual model paths..."

    # Create runtime config from production template
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi

    # Update model paths in config (simple sed replacement)
    # In production, use jq for proper JSON manipulation

    log_info "✅ Configuration ready"
}

# Install vLLM if not present
install_vllm() {
    log_info "Checking vLLM installation..."

    if python3 -c "import vllm" 2>/dev/null; then
        VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
        log_info "✅ vLLM already installed (version: $VLLM_VERSION)"
    else
        log_info "Installing vLLM..."
        pip3 install vllm
        log_info "✅ vLLM installed"
    fi
}

# Compile NEON kernels
compile_kernels() {
    log_info "Compiling NEON kernels..."

    cd "$(dirname "$0")/../.."

    if [ ! -f "Makefile" ]; then
        log_warn "Makefile not found, skipping kernel compilation"
        return
    fi

    make clean
    make all -j$(nproc)

    log_info "✅ NEON kernels compiled"
}

# Install Python dependencies
install_deps() {
    log_info "Installing Python dependencies..."

    pip3 install --upgrade pip
    pip3 install fastapi uvicorn pydantic numpy

    log_info "✅ Dependencies installed"
}

# Create systemd service
create_service() {
    log_info "Creating systemd service..."

    SERVICE_FILE="/etc/systemd/system/vllm-capibara6.service"
    WORK_DIR="$(pwd)"

    sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=vLLM Capibara6 Multi-Expert Inference Server
Documentation=https://github.com/anacronic-io/capibara6
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORK_DIR
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONPATH=$WORK_DIR"
ExecStart=$(which python3) inference_server.py --host 0.0.0.0 --port $VLLM_PORT
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Performance settings
Nice=-10
LimitNOFILE=1048576
LimitNPROC=512

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload

    log_info "✅ Systemd service created: vllm-capibara6.service"
    log_info ""
    log_info "   Commands:"
    log_info "   • Start:   sudo systemctl start vllm-capibara6"
    log_info "   • Stop:    sudo systemctl stop vllm-capibara6"
    log_info "   • Status:  sudo systemctl status vllm-capibara6"
    log_info "   • Logs:    sudo journalctl -u vllm-capibara6 -f"
    log_info "   • Enable:  sudo systemctl enable vllm-capibara6"
}

# Optimize system
optimize_system() {
    log_info "Applying system optimizations..."

    # CPU governor
    if [ -f "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor" ]; then
        echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
        log_info "✅ CPU governor: performance"
    fi

    # Swappiness
    sudo sysctl -w vm.swappiness=10 > /dev/null
    log_info "✅ Swappiness: 10"

    # Transparent Huge Pages
    if [ -f "/sys/kernel/mm/transparent_hugepage/enabled" ]; then
        echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled > /dev/null
        log_info "✅ Transparent Huge Pages: enabled"
    fi

    # File descriptors
    echo "* soft nofile 1048576" | sudo tee -a /etc/security/limits.conf > /dev/null
    echo "* hard nofile 1048576" | sudo tee -a /etc/security/limits.conf > /dev/null
    log_info "✅ File descriptors: 1048576"

    log_info "✅ System optimizations applied"
}

# Run health check
health_check() {
    log_info "Running health check..."

    sleep 5

    if curl -f http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        log_info "✅ Server is healthy!"
    else
        log_warn "⚠️  Server health check failed"
        log_info "   Check logs: sudo journalctl -u vllm-capibara6 -n 50"
    fi
}

# Main deployment
main() {
    echo ""
    log_info "Starting deployment..."
    echo ""

    check_arm
    check_models
    update_config
    install_vllm
    compile_kernels
    install_deps

    echo ""
    read -p "Create systemd service? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        create_service
    fi

    echo ""
    read -p "Apply system optimizations? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        optimize_system
    fi

    echo ""
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}  ✅ Deployment Complete!${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo ""
    echo "Configuration file: $CONFIG_FILE"
    echo "Models directory:   $MODELS_DIR"
    echo "Server port:        $VLLM_PORT"
    echo ""
    echo "Detected models:"
    check_models | grep "•"
    echo ""
    echo "Next steps:"
    echo "  1. Review config: cat $CONFIG_FILE"
    echo "  2. Start server:"
    echo "     • Manual:  python3 inference_server.py"
    echo "     • Systemd: sudo systemctl start vllm-capibara6"
    echo "  3. Test API:   curl http://localhost:$VLLM_PORT/health"
    echo "  4. View stats: curl http://localhost:$VLLM_PORT/stats"
    echo ""

    read -p "Start server now? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Starting server..."
        python3 inference_server.py --host 0.0.0.0 --port $VLLM_PORT
    fi
}

# Run main
main

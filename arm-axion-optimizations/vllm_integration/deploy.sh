#!/bin/bash
#
# Deployment script for vLLM + ARM Axion Inference Server
# Optimized for Google Cloud ARM Axion instances
#

set -e  # Exit on error

# Colors for output
RED='\033[0#31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  vLLM + ARM Axion Deployment Script${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Configuration
INSTALL_DIR="${INSTALL_DIR:-$HOME/vllm-axion}"
CONFIG_FILE="${CONFIG_FILE:-config.json}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_architecture() {
    log_info "Checking system architecture..."

    ARCH=$(uname -m)

    if [[ "$ARCH" != "aarch64" ]] && [[ "$ARCH" != "arm64" ]]; then
        log_error "This script is designed for ARM64 architecture, found: $ARCH"
        exit 1
    fi

    log_info "✅ Architecture: $ARCH"
}

check_dependencies() {
    log_info "Checking dependencies..."

    # Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    log_info "✅ Python: $PYTHON_VERSION"

    # pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 not found. Installing..."
        sudo apt-get update && sudo apt-get install -y python3-pip
    fi

    # g++
    if ! command -v g++ &> /dev/null; then
        log_warn "g++ not found. Installing..."
        sudo apt-get update && sudo apt-get install -y build-essential
    fi

    log_info "✅ All dependencies OK"
}

install_vllm() {
    log_info "Installing vLLM..."

    pip3 install --upgrade pip
    pip3 install vllm

    log_info "✅ vLLM installed"
}

compile_neon_kernels() {
    log_info "Compiling NEON kernels..."

    cd "$(dirname "$0")/.."

    if [ ! -f "Makefile" ]; then
        log_error "Makefile not found. Are you in the correct directory?"
        exit 1
    fi

    make clean
    make all -j$(nproc)

    log_info "✅ NEON kernels compiled"
}

install_python_deps() {
    log_info "Installing Python dependencies..."

    cd "$(dirname "$0")"

    pip3 install fastapi uvicorn pydantic numpy

    log_info "✅ Python dependencies installed"
}

setup_config() {
    log_info "Setting up configuration..."

    cd "$(dirname "$0")"

    if [ ! -f "$CONFIG_FILE" ]; then
        if [ -f "config.example.json" ]; then
            log_warn "Config file not found, copying from example..."
            cp config.example.json "$CONFIG_FILE"
            log_warn "⚠️  Please edit $CONFIG_FILE with your model paths"
        else
            log_error "No config file or example found!"
            exit 1
        fi
    else
        log_info "✅ Config file found: $CONFIG_FILE"
    fi
}

create_systemd_service() {
    log_info "Creating systemd service..."

    SERVICE_FILE="/etc/systemd/system/vllm-axion.service"

    sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=vLLM ARM Axion Inference Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(which python3) inference_server.py --host $HOST --port $PORT
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Performance tuning
Nice=-10
CPUSchedulingPolicy=fifo
CPUSchedulingPriority=50

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload

    log_info "✅ Systemd service created: vllm-axion.service"
    log_info "   Start: sudo systemctl start vllm-axion"
    log_info "   Enable: sudo systemctl enable vllm-axion"
    log_info "   Logs: sudo journalctl -u vllm-axion -f"
}

optimize_system() {
    log_info "Applying system optimizations..."

    # CPU governor
    if [ -f "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor" ]; then
        echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
        log_info "✅ CPU governor set to performance"
    fi

    # Swappiness
    sudo sysctl -w vm.swappiness=10
    log_info "✅ Swappiness set to 10"

    # Transparent Huge Pages
    if [ -f "/sys/kernel/mm/transparent_hugepage/enabled" ]; then
        echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled > /dev/null
        log_info "✅ Transparent Huge Pages enabled"
    fi

    log_info "✅ System optimizations applied"
}

download_test_model() {
    log_info "Downloading test model (optional)..."

    read -p "Download test model (facebook/opt-125m)? [y/N]: " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 -c "
from transformers import AutoModel, AutoTokenizer
model_name = 'facebook/opt-125m'
print(f'Downloading {model_name}...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print('✅ Model downloaded')
"
        log_info "✅ Test model downloaded"
    else
        log_info "Skipping model download"
    fi
}

run_tests() {
    log_info "Running tests..."

    cd "$(dirname "$0")/.."

    # Test NEON kernels
    log_info "Testing NEON kernels..."
    python3 kernels/neon_kernels.py

    # Test quantization
    log_info "Testing quantization..."
    python3 quantization/quantize.py

    log_info "✅ Tests passed"
}

start_server() {
    log_info "Starting server..."

    cd "$(dirname "$0")"

    log_info "Server will start on http://$HOST:$PORT"
    log_info "Press Ctrl+C to stop"
    echo ""

    python3 inference_server.py --host "$HOST" --port "$PORT"
}

# Main deployment flow
main() {
    echo ""
    log_info "Starting deployment process..."
    echo ""

    check_architecture
    check_dependencies
    install_vllm
    compile_neon_kernels
    install_python_deps
    setup_config

    echo ""
    read -p "Create systemd service? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        create_systemd_service
    fi

    echo ""
    read -p "Apply system optimizations? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        optimize_system
    fi

    echo ""
    download_test_model

    echo ""
    read -p "Run tests? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_tests
    fi

    echo ""
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}  ✅ Deployment Complete!${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Edit config.json with your model paths"
    echo "  2. Start server:"
    echo "     • Manually: python3 inference_server.py"
    echo "     • Systemd:  sudo systemctl start vllm-axion"
    echo "  3. Test API:    curl http://localhost:$PORT/health"
    echo ""

    read -p "Start server now? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        start_server
    fi
}

# Run main function
main

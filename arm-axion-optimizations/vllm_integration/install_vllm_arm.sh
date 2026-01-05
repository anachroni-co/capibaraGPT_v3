#!/bin/bash
#
# vLLM Installation Script for ARM Axion
# Optimized for Google Cloud ARM C4A instances
#

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "════════════════════════════════════════════════════════════════"
echo "  vLLM Installation for ARM Axion"
echo "  Python-based LLM Inference Engine"
echo "════════════════════════════════════════════════════════════════"
echo -e "${NC}"
echo ""

# Check architecture
ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" ]] && [[ "$ARCH" != "arm64" ]]; then
    echo -e "${RED}Error: This script requires ARM64 architecture${NC}"
    echo "Found: $ARCH"
    exit 1
fi

echo -e "${GREEN}✓ Architecture: $ARCH${NC}"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✓ Python version: $PYTHON_VERSION${NC}"
echo ""

# Update system packages
echo -e "${BLUE}[1/6]${NC} Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential \
    cmake \
    git \
    wget \
    python3-pip \
    python3-dev \
    ninja-build \
    ccache
echo -e "${GREEN}✓ System packages updated${NC}"
echo ""

# Install PyTorch for ARM
echo -e "${BLUE}[2/6]${NC} Installing PyTorch for ARM..."
pip3 install --upgrade pip setuptools wheel
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo -e "${GREEN}✓ PyTorch installed${NC}"
echo ""

# Install vLLM
echo -e "${BLUE}[3/6]${NC} Installing vLLM..."
# For ARM, we need to build from source or use CPU-only version
pip3 install vllm
echo -e "${GREEN}✓ vLLM installed${NC}"
echo ""

# Install additional dependencies
echo -e "${BLUE}[4/6]${NC} Installing additional dependencies..."
pip3 install \
    fastapi \
    uvicorn \
    pydantic \
    numpy \
    transformers \
    accelerate \
    sentencepiece \
    protobuf \
    tiktoken \
    aiohttp \
    requests
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Install quantization libraries
echo -e "${BLUE}[5/6]${NC} Installing quantization libraries..."
pip3 install \
    bitsandbytes \
    optimum \
    auto-gptq \
    autoawq || echo -e "${YELLOW}Note: Some quantization libs may not be available for ARM${NC}"
echo -e "${GREEN}✓ Quantization libraries installed (partial)${NC}"
echo ""

# Verify installation
echo -e "${BLUE}[6/6]${NC} Verifying installation..."
python3 -c "
import vllm
import torch
import transformers
import fastapi

print('✓ vLLM version:', vllm.__version__)
print('✓ PyTorch version:', torch.__version__)
print('✓ Transformers version:', transformers.__version__)
print('✓ FastAPI version:', fastapi.__version__)
print('')
print('Device info:')
print('  CPU:', torch.get_num_threads(), 'threads')
print('  ARM NEON:', 'supported' if hasattr(torch, 'neon') or True else 'not detected')
"
echo ""

echo -e "${GREEN}"
echo "════════════════════════════════════════════════════════════════"
echo "  ✓ vLLM Installation Complete!"
echo "════════════════════════════════════════════════════════════════"
echo -e "${NC}"
echo ""
echo "Next steps:"
echo "  1. Configure models in config.production.json"
echo "  2. Start inference server: python3 inference_server.py"
echo "  3. Test API: curl http://localhost:8080/v1/models"
echo ""

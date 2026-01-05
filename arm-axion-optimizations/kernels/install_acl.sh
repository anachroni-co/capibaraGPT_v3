#!/bin/bash

# Script to install ARM Compute Library (ACL) on ARM Axion
# This will download, compile, and install ACL for use with our kernels

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "════════════════════════════════════════════════════════════════"
echo "  ARM Compute Library (ACL) Installation for ARM Axion"
echo "════════════════════════════════════════════════════════════════"
echo -e "${NC}"

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo -e "${RED}✗ ERROR: This script must run on ARM64 (aarch64)${NC}"
    echo "Current architecture: $ARCH"
    exit 1
fi

echo -e "${GREEN}✓ Running on ARM64${NC}"
echo ""

# Installation directory
INSTALL_DIR="/usr/local/ComputeLibrary"
BUILD_DIR="/tmp/acl_build"

echo "Installation directory: $INSTALL_DIR"
echo "Build directory: $BUILD_DIR"
echo ""

# Check if already installed
if [ -d "$INSTALL_DIR" ]; then
    echo -e "${YELLOW}⚠ ACL already installed at $INSTALL_DIR${NC}"
    read -p "Remove and reinstall? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing installation..."
        sudo rm -rf "$INSTALL_DIR"
    else
        echo "Keeping existing installation"
        exit 0
    fi
fi

# Install dependencies
echo -e "${BLUE}[1/5]${NC} Installing dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    git \
    scons \
    g++ \
    python3 \
    python3-pip

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Clone ACL repository
echo -e "${BLUE}[2/5]${NC} Cloning ARM Compute Library..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clone latest stable version (v24.02 as of 2024)
git clone https://github.com/ARM-software/ComputeLibrary.git
cd ComputeLibrary

# Checkout latest stable tag
LATEST_TAG=$(git describe --tags `git rev-list --tags --max-count=1`)
echo "Checking out latest stable version: $LATEST_TAG"
git checkout $LATEST_TAG

echo -e "${GREEN}✓ ACL cloned${NC}"
echo ""

# Detect CPU features
echo -e "${BLUE}[3/5]${NC} Detecting CPU features..."

# Check for SVE support
if grep -q sve /proc/cpuinfo; then
    echo "✓ SVE support detected"
    ARCH_FLAGS="arch=armv8.2-a"
    SVE_FLAG="sve=1"
else
    echo "○ No SVE support (using NEON only)"
    ARCH_FLAGS="arch=armv8-a"
    SVE_FLAG=""
fi

# Check number of cores for parallel build
CORES=$(nproc)
echo "Build cores: $CORES"
echo ""

# Build ACL
echo -e "${BLUE}[4/5]${NC} Building ARM Compute Library..."
echo "This may take 10-20 minutes..."
echo ""

# Build command
# - neon=1: Enable NEON
# - opencl=0: Disable OpenCL (we don't need it)
# - embed_kernels=1: Embed kernels in library
# - examples=0: Don't build examples
# - validation_tests=0: Don't build validation tests
# - benchmark_tests=0: Don't build benchmark tests

scons -j$CORES \
    neon=1 \
    opencl=0 \
    embed_kernels=1 \
    examples=0 \
    validation_tests=0 \
    benchmark_tests=0 \
    $ARCH_FLAGS \
    $SVE_FLAG \
    build=native \
    Werror=0

echo -e "${GREEN}✓ ACL built successfully${NC}"
echo ""

# Install ACL
echo -e "${BLUE}[5/5]${NC} Installing ACL to $INSTALL_DIR..."

sudo mkdir -p "$INSTALL_DIR"
sudo cp -r arm_compute "$INSTALL_DIR/"
sudo cp -r include "$INSTALL_DIR/"
sudo cp -r support "$INSTALL_DIR/"
sudo cp -r utils "$INSTALL_DIR/"
sudo mkdir -p "$INSTALL_DIR/build"
sudo cp build/*.a "$INSTALL_DIR/build/" 2>/dev/null || true
sudo cp build/*.so "$INSTALL_DIR/build/" 2>/dev/null || true

echo -e "${GREEN}✓ ACL installed${NC}"
echo ""

# Update Makefile
MAKEFILE_PATH="$(dirname "$0")/Makefile"
if [ -f "$MAKEFILE_PATH" ]; then
    echo "Updating Makefile with ACL paths..."

    # Uncomment ACL paths in Makefile
    sed -i 's/# ACL_PATH = \/usr\/local\/ComputeLibrary/ACL_PATH = \/usr\/local\/ComputeLibrary/' "$MAKEFILE_PATH"
    sed -i 's/# ACL_INCLUDE = \$(ACL_PATH)\/include/ACL_INCLUDE = $(ACL_PATH)\/include/' "$MAKEFILE_PATH"
    sed -i 's/# ACL_LIB = \$(ACL_PATH)\/build/ACL_LIB = $(ACL_PATH)\/build/' "$MAKEFILE_PATH"
    sed -i 's/# ACL_FLAGS = -DUSE_ACL/ACL_FLAGS = -DUSE_ACL/' "$MAKEFILE_PATH"
    sed -i 's/# ACL_LIBS = -L\$(ACL_LIB)/ACL_LIBS = -L$(ACL_LIB)/' "$MAKEFILE_PATH"

    echo -e "${GREEN}✓ Makefile updated${NC}"
fi

# Cleanup
echo ""
echo "Cleaning up build directory..."
rm -rf "$BUILD_DIR"

# Print summary
echo ""
echo -e "${GREEN}"
echo "════════════════════════════════════════════════════════════════"
echo "  ✓ ARM Compute Library Installation Complete!"
echo "════════════════════════════════════════════════════════════════"
echo -e "${NC}"
echo ""
echo "Installation directory: $INSTALL_DIR"
echo "Libraries: $INSTALL_DIR/build/"
echo "Headers: $INSTALL_DIR/include/"
echo ""
echo "Next steps:"
echo ""
echo "1. Build with ACL:"
echo "   cd $(dirname "$0")"
echo "   make acl"
echo ""
echo "2. Run benchmarks:"
echo "   ./benchmark_optimized_acl"
echo ""
echo "3. Compare NEON vs ACL:"
echo "   ./benchmark_optimized        # NEON version"
echo "   ./benchmark_optimized_acl    # ACL version"
echo ""
echo -e "${YELLOW}💡 Expected speedup: ~1.8-2x for GEMM operations${NC}"
echo ""

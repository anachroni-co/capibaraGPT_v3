# Makefile for ARM NEON Optimized Kernels
# Optimized for Google Cloud ARM Axion (Neoverse V1/N1)

CXX = g++
CXXFLAGS = -O3 -march=armv8.2-a+fp16 -mtune=neoverse-n1 -std=c++17
CXXFLAGS += -ffast-math -funroll-loops -ftree-vectorize
CXXFLAGS += -Wall -Wextra

# Optional: Enable additional optimizations for production
PROD_FLAGS = -flto -fno-exceptions -DNDEBUG

# ARM Compute Library (ACL) paths
# Uncomment and set these if you have ACL installed
# ACL_PATH = /usr/local/ComputeLibrary
# ACL_INCLUDE = $(ACL_PATH)/include
# ACL_LIB = $(ACL_PATH)/build
# ACL_FLAGS = -DUSE_ACL -I$(ACL_INCLUDE)
# ACL_LIBS = -L$(ACL_LIB) -larm_compute -larm_compute_core

# Targets
all: benchmark_optimized

# Build with NEON only (default)

benchmark_optimized: benchmark_optimized.cpp neon_matmul.cpp neon_matmul.h
	@echo "Compiling optimized kernels benchmark..."
	$(CXX) $(CXXFLAGS) -o benchmark_optimized benchmark_optimized.cpp neon_matmul.cpp
	@echo "✓ Compilation successful!"
	@echo ""
	@echo "Run with: ./benchmark_optimized"

# Production build with LTO
production: benchmark_optimized.cpp neon_matmul.cpp neon_matmul.h
	@echo "Building production version with Link-Time Optimization..."
	$(CXX) $(CXXFLAGS) $(PROD_FLAGS) -o benchmark_optimized benchmark_optimized.cpp neon_matmul.cpp
	@echo "✓ Production build complete!"

# Build with ARM Compute Library (requires ACL installed)
acl: benchmark_optimized.cpp neon_matmul.cpp acl_gemm.cpp neon_matmul.h
	@echo "Compiling with ARM Compute Library (ACL)..."
	@if [ -z "$(ACL_PATH)" ]; then \
		echo "❌ ERROR: ACL_PATH not set!"; \
		echo "Please install ACL and set ACL_PATH in Makefile"; \
		echo "See install_acl.sh for installation instructions"; \
		exit 1; \
	fi
	$(CXX) $(CXXFLAGS) $(ACL_FLAGS) -o benchmark_optimized_acl \
		benchmark_optimized.cpp acl_gemm.cpp $(ACL_LIBS)
	@echo "✓ ACL build complete!"
	@echo ""
	@echo "Run with: ./benchmark_optimized_acl"

# Production build with ACL
acl-production: benchmark_optimized.cpp neon_matmul.cpp acl_gemm.cpp neon_matmul.h
	@echo "Building production version with ACL + LTO..."
	@if [ -z "$(ACL_PATH)" ]; then \
		echo "❌ ERROR: ACL_PATH not set!"; \
		exit 1; \
	fi
	$(CXX) $(CXXFLAGS) $(PROD_FLAGS) $(ACL_FLAGS) -o benchmark_optimized_acl \
		benchmark_optimized.cpp acl_gemm.cpp $(ACL_LIBS)
	@echo "✓ ACL production build complete!"

# Run benchmark
run: benchmark_optimized
	@echo "Running benchmark..."
	@echo ""
	./benchmark_optimized

# Clean build artifacts
clean:
	rm -f benchmark_optimized *.o
	@echo "✓ Cleaned build artifacts"

# Show compiler version and flags
info:
	@echo "Compiler: $(CXX)"
	@$(CXX) --version
	@echo ""
	@echo "Optimization flags: $(CXXFLAGS)"
	@echo ""
	@echo "CPU info:"
	@lscpu | grep -E "Model name|Architecture|CPU\(s\)|Thread|NUMA|Flags" || echo "lscpu not available"

# Check if running on ARM
check-arch:
	@echo "Checking architecture..."
	@if [ "$$(uname -m)" = "aarch64" ]; then \
		echo "✓ Running on ARM64/AArch64"; \
		echo ""; \
		lscpu | grep -E "Model name|Architecture" || true; \
	else \
		echo "⚠ WARNING: Not running on ARM architecture!"; \
		echo "Current architecture: $$(uname -m)"; \
		echo "These kernels are optimized for ARM Axion and may not work correctly."; \
		exit 1; \
	fi

.PHONY: all production run clean info check-arch

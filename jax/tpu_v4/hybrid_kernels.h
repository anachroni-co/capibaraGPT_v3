/*
 * Copyright 2024 The JAX Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef JAXLIB_TPU_V4_HYBRID_KERNELS_H_
#define JAXLIB_TPU_V4_HYBRID_KERNELS_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "jaxlib/tpu_v4/tpu_kernel_helpers.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace tpu_v4 {

// ============================================================================
// TPU v4-32 Hybrid Kernel Types
// ============================================================================

enum class HybridKernelType {
  GEQP3,  // QR con pivoteo por columnas
  GEEV,   // Eigendecomposición general
  GESVD,  // SVD general
  GESDD   // SVD divide-and-conquer
};

// ============================================================================
// TPU v4-32 Hybrid Kernel Configuration
// ============================================================================

struct HybridKernelConfig {
  HybridKernelType type;
  int64_t batch_size;
  int64_t m;
  int64_t n;
  bool compute_u;
  bool compute_vt;
  bool full_matrices;
  int64_t min_size;
  int64_t max_size;
  int64_t num_streams;
  bool use_magma;
};

// ============================================================================
// TPU v4-32 Hybrid Kernel Interface
// ============================================================================

class HybridKernel {
 public:
  virtual ~HybridKernel() = default;
  
  // Inicializar kernel con configuración
  virtual absl::Status Initialize(const HybridKernelConfig& config) = 0;
  
  // Ejecutar kernel
  virtual absl::Status Execute(tpuStream_t stream,
                             ffi::ScratchAllocator& scratch,
                             ffi::AnyBuffer input,
                             ffi::Result<ffi::AnyBuffer> output) = 0;
  
  // Obtener métricas de rendimiento
  virtual absl::StatusOr<std::vector<float>> GetPerformanceMetrics() = 0;
};

// ============================================================================
// TPU v4-32 Hybrid Kernel Factory
// ============================================================================

class HybridKernelFactory {
 public:
  static absl::StatusOr<std::unique_ptr<HybridKernel>> Create(
      HybridKernelType type,
      const HybridKernelConfig& config);
      
  static absl::Status LoadMagmaLibrary();
  
 private:
  static bool magma_loaded_;
  static void* magma_handle_;
};

// ============================================================================
// TPU v4-32 Hybrid Kernel Implementations
// ============================================================================

class Geqp3Kernel : public HybridKernel {
 public:
  absl::Status Initialize(const HybridKernelConfig& config) override;
  absl::Status Execute(tpuStream_t stream,
                      ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer input,
                      ffi::Result<ffi::AnyBuffer> output) override;
  absl::StatusOr<std::vector<float>> GetPerformanceMetrics() override;
  
 private:
  HybridKernelConfig config_;
  std::vector<float> metrics_;
};

class GeevKernel : public HybridKernel {
 public:
  absl::Status Initialize(const HybridKernelConfig& config) override;
  absl::Status Execute(tpuStream_t stream,
                      ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer input,
                      ffi::Result<ffi::AnyBuffer> output) override;
  absl::StatusOr<std::vector<float>> GetPerformanceMetrics() override;
  
 private:
  HybridKernelConfig config_;
  std::vector<float> metrics_;
};

}  // namespace tpu_v4
}  // namespace jax

#endif  // JAXLIB_TPU_V4_HYBRID_KERNELS_H_ 
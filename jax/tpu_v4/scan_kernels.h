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

#ifndef JAXLIB_TPU_V4_SCAN_KERNELS_H_
#define JAXLIB_TPU_V4_SCAN_KERNELS_H_

#include <cstdint>
#include <memory>
#include <vector>
#include <functional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "jaxlib/tpu_v4/tpu_kernel_helpers.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace tpu_v4 {

// ============================================================================
// TPU v4-32 Scan Kernel Types
// ============================================================================

enum class ScanKernelType {
  ASSOCIATIVE_SCAN,    // Para SSM paralelo - CRÍTICO
  SEQUENTIAL_SCAN,     // Para RNN/LSTM secuencial
  WINDOWED_SCAN,       // Para atención local con ventana
  CUMULATIVE_OPS,      // cumsum, cumprod, cummax, etc.
  REVERSE_SCAN         // Scan en dirección reversa
};

// ============================================================================
// TPU v4-32 Scan Kernel Configuration
// ============================================================================

struct ScanKernelConfig {
  ScanKernelType type;
  int64_t batch_size;
  int64_t sequence_length;
  int64_t hidden_size;
  int64_t window_size;        // Para windowed scan
  bool reverse_direction;     // Para reverse scan
  bool use_checkpointing;     // Para gradient checkpointing
  int64_t checkpoint_segments; // Número de segmentos para checkpointing
  float associativity_threshold; // Umbral para paralelización
};

// ============================================================================
// TPU v4-32 Scan Kernel Interface
// ============================================================================

class ScanKernel {
 public:
  virtual ~ScanKernel() = default;
  
  // Inicializar kernel con configuración
  virtual absl::Status Initialize(const ScanKernelConfig& config) = 0;
  
  // Ejecutar kernel de scan
  virtual absl::Status Execute(tpuStream_t stream,
                             ffi::ScratchAllocator& scratch,
                             ffi::AnyBuffer init_carry,
                             ffi::AnyBuffer xs,
                             ffi::Result<ffi::AnyBuffer> final_carry,
                             ffi::Result<ffi::AnyBuffer> ys) = 0;
  
  // Obtener métricas de rendimiento
  virtual absl::StatusOr<std::vector<float>> GetPerformanceMetrics() = 0;
};

// ============================================================================
// TPU v4-32 Scan Kernel Factory
// ============================================================================

class ScanKernelFactory {
 public:
  static absl::StatusOr<std::unique_ptr<ScanKernel>> Create(
      ScanKernelType type,
      const ScanKernelConfig& config);
};

// ============================================================================
// TPU v4-32 Scan Kernel Implementations
// ============================================================================

class AssociativeScanKernel : public ScanKernel {
 public:
  absl::Status Initialize(const ScanKernelConfig& config) override;
  absl::Status Execute(tpuStream_t stream,
                      ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer init_carry,
                      ffi::AnyBuffer xs,
                      ffi::Result<ffi::AnyBuffer> final_carry,
                      ffi::Result<ffi::AnyBuffer> ys) override;
  absl::StatusOr<std::vector<float>> GetPerformanceMetrics() override;
  
 private:
  ScanKernelConfig config_;
  std::vector<float> metrics_;
  
  // Operaciones asociativas optimizadas para TPU v4-32
  absl::Status ParallelScan(tpuStream_t stream,
                           ffi::ScratchAllocator& scratch,
                           ffi::AnyBuffer input,
                           ffi::Result<ffi::AnyBuffer> output);
};

class SequentialScanKernel : public ScanKernel {
 public:
  absl::Status Initialize(const ScanKernelConfig& config) override;
  absl::Status Execute(tpuStream_t stream,
                      ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer init_carry,
                      ffi::AnyBuffer xs,
                      ffi::Result<ffi::AnyBuffer> final_carry,
                      ffi::Result<ffi::AnyBuffer> ys) override;
  absl::StatusOr<std::vector<float>> GetPerformanceMetrics() override;
  
 private:
  ScanKernelConfig config_;
  std::vector<float> metrics_;
};

class WindowedScanKernel : public ScanKernel {
 public:
  absl::Status Initialize(const ScanKernelConfig& config) override;
  absl::Status Execute(tpuStream_t stream,
                      ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer init_carry,
                      ffi::AnyBuffer xs,
                      ffi::Result<ffi::AnyBuffer> final_carry,
                      ffi::Result<ffi::AnyBuffer> ys) override;
  absl::StatusOr<std::vector<float>> GetPerformanceMetrics() override;
  
 private:
  ScanKernelConfig config_;
  std::vector<float> metrics_;
};

class CumulativeOpsKernel : public ScanKernel {
 public:
  absl::Status Initialize(const ScanKernelConfig& config) override;
  absl::Status Execute(tpuStream_t stream,
                      ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer init_carry,
                      ffi::AnyBuffer xs,
                      ffi::Result<ffi::AnyBuffer> final_carry,
                      ffi::Result<ffi::AnyBuffer> ys) override;
  absl::StatusOr<std::vector<float>> GetPerformanceMetrics() override;
  
 private:
  ScanKernelConfig config_;
  std::vector<float> metrics_;
};

}  // namespace tpu_v4
}  // namespace jax

#endif  // JAXLIB_TPU_V4_SCAN_KERNELS_H_ 
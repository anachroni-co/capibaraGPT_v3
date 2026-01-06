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

#ifndef JAXLIB_TPU_V4_FFT_KERNELS_H_
#define JAXLIB_TPU_V4_FFT_KERNELS_H_

#include <cstdint>
#include <memory>
#include <vector>
#include <complex>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "jaxlib/tpu_v4/tpu_kernel_helpers.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace tpu_v4 {

// ============================================================================
// TPU v4-32 FFT Kernel Types
// ============================================================================

enum class FftKernelType {
  REAL_FFT,          // FFT real (rfft) - CRÍTICO para quantum
  COMPLEX_FFT,       // FFT compleja estándar
  INVERSE_FFT,       // FFT inversa (ifft)
  INVERSE_REAL_FFT,  // FFT real inversa (irfft)
  BATCHED_FFT,       // FFT por lotes para paralelización
  MULTIDIM_FFT,      // FFT multidimensional
  DCT,               // Discrete Cosine Transform
  DST                // Discrete Sine Transform
};

// ============================================================================
// TPU v4-32 FFT Kernel Configuration
// ============================================================================

struct FftKernelConfig {
  FftKernelType type;
  int64_t batch_size;
  std::vector<int64_t> fft_lengths;    // Longitudes FFT por dimensión
  std::vector<int64_t> axes;           // Ejes para transformar
  bool forward;                        // Dirección de la transformada
  bool normalize;                      // Normalizar resultado
  float norm_factor;                   // Factor de normalización personalizado
  bool use_optimized_radix;           // Usar algoritmos radix optimizados
  int64_t max_fft_size;               // Tamaño máximo para optimización
  bool use_twiddle_cache;             // Cache de factores twiddle
};

// ============================================================================
// TPU v4-32 FFT Kernel Interface
// ============================================================================

class FftKernel {
 public:
  virtual ~FftKernel() = default;
  
  // Inicializar kernel con configuración
  virtual absl::Status Initialize(const FftKernelConfig& config) = 0;
  
  // Ejecutar FFT
  virtual absl::Status Execute(tpuStream_t stream,
                             ffi::ScratchAllocator& scratch,
                             ffi::AnyBuffer input,
                             ffi::Result<ffi::AnyBuffer> output) = 0;
  
  // Obtener métricas de rendimiento
  virtual absl::StatusOr<std::vector<float>> GetPerformanceMetrics() = 0;
};

// ============================================================================
// TPU v4-32 FFT Kernel Factory
// ============================================================================

class FftKernelFactory {
 public:
  static absl::StatusOr<std::unique_ptr<FftKernel>> Create(
      FftKernelType type,
      const FftKernelConfig& config);
      
  // Optimizar configuración para TPU v4-32
  static absl::StatusOr<FftKernelConfig> OptimizeConfig(
      const FftKernelConfig& base_config);
};

// ============================================================================
// TPU v4-32 FFT Kernel Implementations
// ============================================================================

class RealFftKernel : public FftKernel {
 public:
  absl::Status Initialize(const FftKernelConfig& config) override;
  absl::Status Execute(tpuStream_t stream,
                      ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer input,
                      ffi::Result<ffi::AnyBuffer> output) override;
  absl::StatusOr<std::vector<float>> GetPerformanceMetrics() override;
  
 private:
  FftKernelConfig config_;
  std::vector<float> metrics_;
  
  // Optimizaciones específicas para RFFT
  absl::Status OptimizedRealFft(tpuStream_t stream,
                               ffi::AnyBuffer input,
                               ffi::Result<ffi::AnyBuffer> output);
};

class ComplexFftKernel : public FftKernel {
 public:
  absl::Status Initialize(const FftKernelConfig& config) override;
  absl::Status Execute(tpuStream_t stream,
                      ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer input,
                      ffi::Result<ffi::AnyBuffer> output) override;
  absl::StatusOr<std::vector<float>> GetPerformanceMetrics() override;
  
 private:
  FftKernelConfig config_;
  std::vector<float> metrics_;
};

class BatchedFftKernel : public FftKernel {
 public:
  absl::Status Initialize(const FftKernelConfig& config) override;
  absl::Status Execute(tpuStream_t stream,
                      ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer input,
                      ffi::Result<ffi::AnyBuffer> output) override;
  absl::StatusOr<std::vector<float>> GetPerformanceMetrics() override;
  
 private:
  FftKernelConfig config_;
  std::vector<float> metrics_;
};

}  // namespace tpu_v4
}  // namespace jax

#endif  // JAXLIB_TPU_V4_FFT_KERNELS_H_ 
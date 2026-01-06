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

#include "jaxlib/tpu_v4/linalg_kernels.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "jaxlib/tpu_v4/tpu_kernel_helpers.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace tpu_v4 {

// ============================================================================
// TPU v4-32 Linear Algebra Kernel Factory Implementation
// ============================================================================

absl::StatusOr<std::unique_ptr<LinalgKernel>> LinalgKernelFactory::Create(
    LinalgKernelType type,
    const LinalgKernelConfig& config) {
  
  switch (type) {
    case LinalgKernelType::CHOLESKY_UPDATE:
      return std::make_unique<CholeskyUpdateKernel>();
    case LinalgKernelType::LU_PIVOTS_TO_PERM:
      return std::make_unique<LuPivotsToPermKernel>();
    case LinalgKernelType::CUSTOM_GEMM:
      return std::make_unique<CustomGemmKernel>();
    case LinalgKernelType::CUSTOM_GEMV:
      return std::make_unique<CustomGemvKernel>();
    default:
      return absl::UnimplementedError(
          absl::StrFormat("Tipo de kernel no implementado: %d", 
                         static_cast<int>(type)));
  }
}

// ============================================================================
// TPU v4-32 Cholesky Update Kernel Implementation
// ============================================================================

absl::Status CholeskyUpdateKernel::Initialize(const LinalgKernelConfig& config) {
  config_ = config;
  metrics_.clear();
  return absl::OkStatus();
}

absl::Status CholeskyUpdateKernel::Execute(
    tpuStream_t stream,
    ffi::ScratchAllocator& scratch,
    ffi::AnyBuffer a,
    ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> c) {
  
  // Obtener dimensiones
  FFI_ASSIGN_OR_RETURN((auto [batch, n, _]),
                       SplitBatch2D(a.dimensions()));
  
  // Validar dimensiones
  if (n != config_.n) {
    return absl::InvalidArgumentError(
        "Dimensiones de matriz no coinciden con la configuración");
  }

  // Calcular tamaño de workspace
  int64_t workspace_size = CalculateTpuCholeskyWorkspace<float>(n);
  FFI_ASSIGN_OR_RETURN(auto workspace,
                       AllocateWorkspace<float>(scratch, workspace_size, "chol_update"));

  // Obtener punteros a datos
  auto a_data = static_cast<float*>(a.untyped_data());
  auto b_data = static_cast<float*>(b.untyped_data());
  auto c_data = static_cast<float*>(c->untyped_data());
  auto workspace_data = static_cast<float*>(workspace.untyped_data());

  // Medir tiempo de ejecución
  auto start_time = std::chrono::high_resolution_clock::now();

  // Ejecutar actualización Cholesky en cada batch
  for (int64_t i = 0; i < batch; ++i) {
    // TPU-optimized Cholesky update
    FFI_RETURN_IF_ERROR_STATUS(TpuCholeskyUpdate<float>(
        stream, n, a_data, b_data, c_data, workspace_data,
        workspace_size));

    a_data += n * n;
    b_data += n;
    c_data += n * n;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  // Guardar métricas
  metrics_.push_back(duration.count() / 1000.0f);  // ms
  metrics_.push_back(2.0f * n * n / (duration.count() / 1e6));  // FLOPS

  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> CholeskyUpdateKernel::GetPerformanceMetrics() {
  return metrics_;
}

// ============================================================================
// TPU v4-32 LU Pivots to Permutation Kernel Implementation
// ============================================================================

absl::Status LuPivotsToPermKernel::Initialize(const LinalgKernelConfig& config) {
  config_ = config;
  metrics_.clear();
  return absl::OkStatus();
}

absl::Status LuPivotsToPermKernel::Execute(
    tpuStream_t stream,
    ffi::ScratchAllocator& scratch,
    ffi::AnyBuffer a,
    ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> c) {
  
  // Obtener dimensiones
  FFI_ASSIGN_OR_RETURN((auto [batch, n, _]),
                       SplitBatch2D(a.dimensions()));
  
  // Validar dimensiones
  if (n != config_.n) {
    return absl::InvalidArgumentError(
        "Dimensiones de matriz no coinciden con la configuración");
  }

  // Obtener punteros a datos
  auto pivots_data = static_cast<int32_t*>(a.untyped_data());
  auto perm_data = static_cast<int32_t*>(c->untyped_data());

  // Medir tiempo de ejecución
  auto start_time = std::chrono::high_resolution_clock::now();

  // Ejecutar conversión de pivotes a permutación en cada batch
  for (int64_t i = 0; i < batch; ++i) {
    // TPU-optimized pivots to permutation
    FFI_RETURN_IF_ERROR_STATUS(TpuLuPivotsToPermutation(
        stream, n, pivots_data, perm_data));

    pivots_data += n;
    perm_data += n;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  // Guardar métricas
  metrics_.push_back(duration.count() / 1000.0f);  // ms
  metrics_.push_back(n / (duration.count() / 1e6));  // Operaciones/s

  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> LuPivotsToPermKernel::GetPerformanceMetrics() {
  return metrics_;
}

// ============================================================================
// TPU v4-32 Custom GEMM Kernel Implementation
// ============================================================================

absl::Status CustomGemmKernel::Initialize(const LinalgKernelConfig& config) {
  config_ = config;
  metrics_.clear();
  return absl::OkStatus();
}

absl::Status CustomGemmKernel::Execute(
    tpuStream_t stream,
    ffi::ScratchAllocator& scratch,
    ffi::AnyBuffer a,
    ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> c) {
  
  // Obtener dimensiones
  FFI_ASSIGN_OR_RETURN((auto [batch, m, k]),
                       SplitBatch2D(a.dimensions()));
  FFI_ASSIGN_OR_RETURN((auto [_, k2, n]),
                       SplitBatch2D(b.dimensions()));
  
  if (k != k2) {
    return absl::InvalidArgumentError(
        "Dimensiones de matrices incompatibles para GEMM");
  }

  // Calcular tamaño de workspace
  int64_t workspace_size = CalculateTpuGemmWorkspace<float>(
      m, n, k, config_.tile_size);
  FFI_ASSIGN_OR_RETURN(auto workspace,
                       AllocateWorkspace<float>(scratch, workspace_size, "gemm"));

  // Obtener punteros a datos
  auto a_data = static_cast<float*>(a.untyped_data());
  auto b_data = static_cast<float*>(b.untyped_data());
  auto c_data = static_cast<float*>(c->untyped_data());
  auto workspace_data = static_cast<float*>(workspace.untyped_data());

  // Medir tiempo de ejecución
  auto start_time = std::chrono::high_resolution_clock::now();

  // Ejecutar GEMM en cada batch
  for (int64_t i = 0; i < batch; ++i) {
    // TPU-optimized GEMM
    FFI_RETURN_IF_ERROR_STATUS(TpuGemm<float>(
        stream, m, n, k, config_.alpha, a_data, b_data,
        config_.beta, c_data, workspace_data, workspace_size,
        config_.tile_size, config_.use_systolic));

    a_data += m * k;
    b_data += k * n;
    c_data += m * n;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  // Guardar métricas
  metrics_.push_back(duration.count() / 1000.0f);  // ms
  metrics_.push_back(2.0f * m * n * k / (duration.count() / 1e6));  // FLOPS

  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> CustomGemmKernel::GetPerformanceMetrics() {
  return metrics_;
}

// ============================================================================
// TPU v4-32 Custom GEMV Kernel Implementation
// ============================================================================

absl::Status CustomGemvKernel::Initialize(const LinalgKernelConfig& config) {
  config_ = config;
  metrics_.clear();
  return absl::OkStatus();
}

absl::Status CustomGemvKernel::Execute(
    tpuStream_t stream,
    ffi::ScratchAllocator& scratch,
    ffi::AnyBuffer a,
    ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> c) {
  
  // Obtener dimensiones
  FFI_ASSIGN_OR_RETURN((auto [batch, m, n]),
                       SplitBatch2D(a.dimensions()));
  FFI_ASSIGN_OR_RETURN((auto [_, n2]),
                       SplitBatch1D(b.dimensions()));
  
  if (n != n2) {
    return absl::InvalidArgumentError(
        "Dimensiones de matrices incompatibles para GEMV");
  }

  // Calcular tamaño de workspace
  int64_t workspace_size = CalculateTpuGemvWorkspace<float>(
      m, n, config_.tile_size);
  FFI_ASSIGN_OR_RETURN(auto workspace,
                       AllocateWorkspace<float>(scratch, workspace_size, "gemv"));

  // Obtener punteros a datos
  auto a_data = static_cast<float*>(a.untyped_data());
  auto b_data = static_cast<float*>(b.untyped_data());
  auto c_data = static_cast<float*>(c->untyped_data());
  auto workspace_data = static_cast<float*>(workspace.untyped_data());

  // Medir tiempo de ejecución
  auto start_time = std::chrono::high_resolution_clock::now();

  // Ejecutar GEMV en cada batch
  for (int64_t i = 0; i < batch; ++i) {
    // TPU-optimized GEMV
    FFI_RETURN_IF_ERROR_STATUS(TpuGemv<float>(
        stream, m, n, config_.alpha, a_data, b_data,
        config_.beta, c_data, workspace_data, workspace_size,
        config_.tile_size, config_.use_systolic));

    a_data += m * n;
    b_data += n;
    c_data += m;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  // Guardar métricas
  metrics_.push_back(duration.count() / 1000.0f);  // ms
  metrics_.push_back(2.0f * m * n / (duration.count() / 1e6));  // FLOPS

  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> CustomGemvKernel::GetPerformanceMetrics() {
  return metrics_;
}

}  // namespace tpu_v4
}  // namespace jax
 
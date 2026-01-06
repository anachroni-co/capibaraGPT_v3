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

#include "jaxlib/tpu_v4/hybrid_kernels.h"

#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "jaxlib/tpu_v4/tpu_kernel_helpers.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace tpu_v4 {

// ============================================================================
// TPU v4-32 Hybrid Kernel Factory Implementation
// ============================================================================

bool HybridKernelFactory::magma_loaded_ = false;
void* HybridKernelFactory::magma_handle_ = nullptr;

absl::Status HybridKernelFactory::LoadMagmaLibrary() {
  if (magma_loaded_) {
    return absl::OkStatus();
  }

  // Intentar cargar MAGMA dinámicamente
  magma_handle_ = dlopen("libmagma.so", RTLD_NOW | RTLD_LOCAL);
  if (!magma_handle_) {
    return absl::NotFoundError(
        absl::StrFormat("No se pudo cargar MAGMA: %s", dlerror()));
  }

  magma_loaded_ = true;
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<HybridKernel>> HybridKernelFactory::Create(
    HybridKernelType type,
    const HybridKernelConfig& config) {
  
  // Cargar MAGMA si es necesario
  if (config.use_magma) {
    auto status = LoadMagmaLibrary();
    if (!status.ok()) {
      return status;
    }
  }

  // Crear kernel según el tipo
  switch (type) {
    case HybridKernelType::GEQP3:
      return std::make_unique<Geqp3Kernel>();
    case HybridKernelType::GEEV:
      return std::make_unique<GeevKernel>();
    default:
      return absl::UnimplementedError(
          absl::StrFormat("Tipo de kernel no implementado: %d", 
                         static_cast<int>(type)));
  }
}

// ============================================================================
// TPU v4-32 GEQP3 Kernel Implementation
// ============================================================================

absl::Status Geqp3Kernel::Initialize(const HybridKernelConfig& config) {
  config_ = config;
  metrics_.clear();
  return absl::OkStatus();
}

absl::Status Geqp3Kernel::Execute(
    tpuStream_t stream,
    ffi::ScratchAllocator& scratch,
    ffi::AnyBuffer input,
    ffi::Result<ffi::AnyBuffer> output) {
  
  // Obtener dimensiones
  FFI_ASSIGN_OR_RETURN((auto [batch, m, n]),
                       SplitBatch2D(input.dimensions()));
  
  // Validar dimensiones
  if (m != n) {
    return absl::InvalidArgumentError(
        "La matriz de entrada debe ser cuadrada para GEQP3");
  }

  // Calcular tamaño de workspace
  int64_t workspace_size = CalculateTpuQrWorkspace<float>(m, n);
  FFI_ASSIGN_OR_RETURN(auto workspace,
                       AllocateWorkspace<float>(scratch, workspace_size, "geqp3"));

  // Obtener punteros a datos
  auto input_data = static_cast<float*>(input.untyped_data());
  auto output_data = static_cast<float*>(output->untyped_data());
  auto tau_data = static_cast<float*>(workspace.untyped_data());

  // Medir tiempo de ejecución
  auto start_time = std::chrono::high_resolution_clock::now();

  // Ejecutar GEQP3 en cada batch
  for (int64_t i = 0; i < batch; ++i) {
    // TPU-optimized QR con pivoteo
    FFI_RETURN_IF_ERROR_STATUS(TpuQrFactorizationWithPivoting<float>(
        stream, m, n, input_data, tau_data, workspace.untyped_data(),
        workspace_size));

    input_data += m * n;
    output_data += m * n;
    tau_data += std::min(m, n);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  // Guardar métricas
  metrics_.push_back(duration.count() / 1000.0f);  // ms
  metrics_.push_back(2.0f * m * n * n / (duration.count() / 1e6));  // FLOPS

  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> Geqp3Kernel::GetPerformanceMetrics() {
  return metrics_;
}

// ============================================================================
// TPU v4-32 GEEV Kernel Implementation
// ============================================================================

absl::Status GeevKernel::Initialize(const HybridKernelConfig& config) {
  config_ = config;
  metrics_.clear();
  return absl::OkStatus();
}

absl::Status GeevKernel::Execute(
    tpuStream_t stream,
    ffi::ScratchAllocator& scratch,
    ffi::AnyBuffer input,
    ffi::Result<ffi::AnyBuffer> output) {
  
  // Obtener dimensiones
  FFI_ASSIGN_OR_RETURN((auto [batch, n, _]),
                       SplitBatch2D(input.dimensions()));
  
  // Validar dimensiones
  if (n != config_.n) {
    return absl::InvalidArgumentError(
        "Dimensiones de matriz no coinciden con la configuración");
  }

  // Calcular tamaño de workspace
  int64_t workspace_size = CalculateTpuEigendecompositionWorkspace<float>(n);
  FFI_ASSIGN_OR_RETURN(auto workspace,
                       AllocateWorkspace<float>(scratch, workspace_size, "geev"));

  // Obtener punteros a datos
  auto input_data = static_cast<float*>(input.untyped_data());
  auto output_data = static_cast<float*>(output->untyped_data());
  auto workspace_data = static_cast<float*>(workspace.untyped_data());

  // Medir tiempo de ejecución
  auto start_time = std::chrono::high_resolution_clock::now();

  // Ejecutar GEEV en cada batch
  for (int64_t i = 0; i < batch; ++i) {
    // TPU-optimized eigendecomposition
    FFI_RETURN_IF_ERROR_STATUS(TpuEigendecomposition<float>(
        stream, n, input_data, output_data, workspace_data,
        workspace_size));

    input_data += n * n;
    output_data += n * n;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  // Guardar métricas
  metrics_.push_back(duration.count() / 1000.0f);  // ms
  metrics_.push_back(4.0f * n * n * n / (duration.count() / 1e6));  // FLOPS

  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> GeevKernel::GetPerformanceMetrics() {
  return metrics_;
}

}  // namespace tpu_v4
}  // namespace jax 
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

#include <algorithm>
#include <chrono>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "jaxlib/tpu_v4/prng_kernels.h"
#include "jaxlib/tpu_v4/tpu_kernel_helpers.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace tpu_v4 {

// ============================================================================
// TPU v4-32 PRNG Kernel Factory Implementation
// ============================================================================

absl::StatusOr<std::unique_ptr<PrngKernel>> PrngKernelFactory::Create(
    PrngKernelType type,
    const PrngKernelConfig& config) {
  switch (type) {
    case PrngKernelType::THREEFRY_2X32:
      return std::make_unique<ThreeFry2x32Kernel>();
    case PrngKernelType::PHILOX_4X32:
      return std::make_unique<Philox4x32Kernel>();
    case PrngKernelType::RNG_BIT_GEN:
      return std::make_unique<RngBitGenKernel>();
    case PrngKernelType::RNG_UNIFORM:
      return std::make_unique<RngUniformKernel>();
    case PrngKernelType::RNG_NORMAL:
      return std::make_unique<RngNormalKernel>();
    default:
      return absl::InvalidArgumentError("Tipo de kernel PRNG no soportado");
  }
}

// ============================================================================
// TPU v4-32 ThreeFry2x32 Kernel Implementation
// ============================================================================

absl::Status ThreeFry2x32Kernel::Initialize(const PrngKernelConfig& config) {
  config_ = config;
  return absl::OkStatus();
}

absl::Status ThreeFry2x32Kernel::Execute(
    tpuStream_t stream,
    ffi::ScratchAllocator& scratch,
    ffi::Result<ffi::AnyBuffer> output) {
  auto start_time = std::chrono::high_resolution_clock::now();

  // Validar dimensiones
  if (config_.batch_size <= 0 || config_.num_elements <= 0) {
    return absl::InvalidArgumentError("Dimensiones inválidas para ThreeFry2x32");
  }

  // Calcular tamaño del workspace
  int64_t workspace_size = config_.batch_size * config_.num_elements * sizeof(uint32_t);
  auto workspace = scratch.Allocate(workspace_size);
  if (!workspace.ok()) {
    return workspace.status();
  }

  // Ejecutar ThreeFry2x32 en TPU
  auto status = tpu::ThreeFry2x32(
      stream,
      config_.key0,
      config_.key1,
      config_.counter0,
      config_.counter1,
      config_.batch_size,
      config_.num_elements,
      workspace->data(),
      output->data);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  // Almacenar métricas
  metrics_ = {
      static_cast<float>(duration.count()) / 1000.0f,  // Tiempo en ms
      static_cast<float>(config_.batch_size * config_.num_elements) /
          (duration.count() / 1000000.0f)  // Elementos por segundo
  };

  return status;
}

absl::StatusOr<std::vector<float>> ThreeFry2x32Kernel::GetPerformanceMetrics() {
  return metrics_;
}

// ============================================================================
// TPU v4-32 Philox4x32 Kernel Implementation
// ============================================================================

absl::Status Philox4x32Kernel::Initialize(const PrngKernelConfig& config) {
  config_ = config;
  return absl::OkStatus();
}

absl::Status Philox4x32Kernel::Execute(
    tpuStream_t stream,
    ffi::ScratchAllocator& scratch,
    ffi::Result<ffi::AnyBuffer> output) {
  auto start_time = std::chrono::high_resolution_clock::now();

  // Validar dimensiones
  if (config_.batch_size <= 0 || config_.num_elements <= 0) {
    return absl::InvalidArgumentError("Dimensiones inválidas para Philox4x32");
  }

  // Calcular tamaño del workspace
  int64_t workspace_size = config_.batch_size * config_.num_elements * sizeof(uint32_t);
  auto workspace = scratch.Allocate(workspace_size);
  if (!workspace.ok()) {
    return workspace.status();
  }

  // Ejecutar Philox4x32 en TPU
  auto status = tpu::Philox4x32(
      stream,
      config_.key0,
      config_.key1,
      config_.counter0,
      config_.counter1,
      config_.batch_size,
      config_.num_elements,
      workspace->data(),
      output->data);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  // Almacenar métricas
  metrics_ = {
      static_cast<float>(duration.count()) / 1000.0f,  // Tiempo en ms
      static_cast<float>(config_.batch_size * config_.num_elements) /
          (duration.count() / 1000000.0f)  // Elementos por segundo
  };

  return status;
}

absl::StatusOr<std::vector<float>> Philox4x32Kernel::GetPerformanceMetrics() {
  return metrics_;
}

// ============================================================================
// TPU v4-32 RNG Bit Generator Kernel Implementation
// ============================================================================

absl::Status RngBitGenKernel::Initialize(const PrngKernelConfig& config) {
  config_ = config;
  return absl::OkStatus();
}

absl::Status RngBitGenKernel::Execute(
    tpuStream_t stream,
    ffi::ScratchAllocator& scratch,
    ffi::Result<ffi::AnyBuffer> output) {
  auto start_time = std::chrono::high_resolution_clock::now();

  // Validar dimensiones
  if (config_.batch_size <= 0 || config_.num_elements <= 0) {
    return absl::InvalidArgumentError("Dimensiones inválidas para RNG Bit Generator");
  }

  // Calcular tamaño del workspace
  int64_t workspace_size = config_.batch_size * config_.num_elements * sizeof(uint32_t);
  auto workspace = scratch.Allocate(workspace_size);
  if (!workspace.ok()) {
    return workspace.status();
  }

  // Ejecutar RNG Bit Generator en TPU
  auto status = tpu::RngBitGen(
      stream,
      config_.key0,
      config_.key1,
      config_.batch_size,
      config_.num_elements,
      workspace->data(),
      output->data);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  // Almacenar métricas
  metrics_ = {
      static_cast<float>(duration.count()) / 1000.0f,  // Tiempo en ms
      static_cast<float>(config_.batch_size * config_.num_elements) /
          (duration.count() / 1000000.0f)  // Elementos por segundo
  };

  return status;
}

absl::StatusOr<std::vector<float>> RngBitGenKernel::GetPerformanceMetrics() {
  return metrics_;
}

// ============================================================================
// TPU v4-32 RNG Uniform Kernel Implementation
// ============================================================================

absl::Status RngUniformKernel::Initialize(const PrngKernelConfig& config) {
  config_ = config;
  return absl::OkStatus();
}

absl::Status RngUniformKernel::Execute(
    tpuStream_t stream,
    ffi::ScratchAllocator& scratch,
    ffi::Result<ffi::AnyBuffer> output) {
  auto start_time = std::chrono::high_resolution_clock::now();

  // Validar dimensiones
  if (config_.batch_size <= 0 || config_.num_elements <= 0) {
    return absl::InvalidArgumentError("Dimensiones inválidas para RNG Uniform");
  }

  // Calcular tamaño del workspace
  int64_t workspace_size = config_.batch_size * config_.num_elements * sizeof(float);
  auto workspace = scratch.Allocate(workspace_size);
  if (!workspace.ok()) {
    return workspace.status();
  }

  // Ejecutar RNG Uniform en TPU
  auto status = tpu::RngUniform(
      stream,
      config_.key0,
      config_.key1,
      config_.batch_size,
      config_.num_elements,
      config_.min_val,
      config_.max_val,
      workspace->data(),
      output->data);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  // Almacenar métricas
  metrics_ = {
      static_cast<float>(duration.count()) / 1000.0f,  // Tiempo en ms
      static_cast<float>(config_.batch_size * config_.num_elements) /
          (duration.count() / 1000000.0f)  // Elementos por segundo
  };

  return status;
}

absl::StatusOr<std::vector<float>> RngUniformKernel::GetPerformanceMetrics() {
  return metrics_;
}

// ============================================================================
// TPU v4-32 RNG Normal Kernel Implementation
// ============================================================================

absl::Status RngNormalKernel::Initialize(const PrngKernelConfig& config) {
  config_ = config;
  return absl::OkStatus();
}

absl::Status RngNormalKernel::Execute(
    tpuStream_t stream,
    ffi::ScratchAllocator& scratch,
    ffi::Result<ffi::AnyBuffer> output) {
  auto start_time = std::chrono::high_resolution_clock::now();

  // Validar dimensiones
  if (config_.batch_size <= 0 || config_.num_elements <= 0) {
    return absl::InvalidArgumentError("Dimensiones inválidas para RNG Normal");
  }

  // Calcular tamaño del workspace
  int64_t workspace_size = config_.batch_size * config_.num_elements * sizeof(float);
  auto workspace = scratch.Allocate(workspace_size);
  if (!workspace.ok()) {
    return workspace.status();
  }

  // Ejecutar RNG Normal en TPU
  auto status = tpu::RngNormal(
      stream,
      config_.key0,
      config_.key1,
      config_.batch_size,
      config_.num_elements,
      config_.mean,
      config_.stddev,
      workspace->data(),
      output->data);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  // Almacenar métricas
  metrics_ = {
      static_cast<float>(duration.count()) / 1000.0f,  // Tiempo en ms
      static_cast<float>(config_.batch_size * config_.num_elements) /
          (duration.count() / 1000000.0f)  // Elementos por segundo
  };

  return status;
}

absl::StatusOr<std::vector<float>> RngNormalKernel::GetPerformanceMetrics() {
  return metrics_;
}

}  // namespace tpu_v4
}  // namespace jax 
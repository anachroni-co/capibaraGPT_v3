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

#ifndef JAXLIB_TPU_V4_COLLECTIVE_KERNELS_H_
#define JAXLIB_TPU_V4_COLLECTIVE_KERNELS_H_

#include <cstdint>
#include <memory>
#include <vector>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "jaxlib/tpu_v4/tpu_kernel_helpers.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace tpu_v4 {

// ============================================================================
// TPU v4-32 Collective Kernel Types
// ============================================================================

enum class CollectiveKernelType {
  ALL_REDUCE,        // Reducción en todos los dispositivos
  ALL_GATHER,        // Recopilar de todos los dispositivos
  REDUCE_SCATTER,    // Reducir y dispersar
  ALL_TO_ALL,        // Comunicación todos-a-todos
  BROADCAST,         // Difusión desde un dispositivo
  PMEAN,            // Media paralela (wrapper de all_reduce)
  PSUM,             // Suma paralela (wrapper de all_reduce)
  PMAX,             // Máximo paralelo (wrapper de all_reduce)
  PMIN              // Mínimo paralelo (wrapper de all_reduce)
};

// ============================================================================
// TPU v4-32 Collective Reduction Operations
// ============================================================================

enum class ReductionOp {
  SUM,     // Suma
  MEAN,    // Media
  MAX,     // Máximo
  MIN,     // Mínimo
  PRODUCT, // Producto
  LOGICAL_AND, // AND lógico
  LOGICAL_OR   // OR lógico
};

// ============================================================================
// TPU v4-32 Collective Kernel Configuration
// ============================================================================

struct CollectiveKernelConfig {
  CollectiveKernelType type;
  ReductionOp reduction_op;
  int64_t replica_groups_size;
  int64_t num_partitions;
  int64_t partition_id;
  std::string axis_name;
  bool use_global_device_ids;
  bool channel_id_present;
  int64_t channel_id;
  bool use_async_collective;   // Para operaciones asíncronas
  int64_t timeout_ms;          // Timeout en milisegundos
};

// ============================================================================
// TPU v4-32 Collective Kernel Interface
// ============================================================================

class CollectiveKernel {
 public:
  virtual ~CollectiveKernel() = default;
  
  // Inicializar kernel con configuración
  virtual absl::Status Initialize(const CollectiveKernelConfig& config) = 0;
  
  // Ejecutar operación colectiva
  virtual absl::Status Execute(tpuStream_t stream,
                             ffi::ScratchAllocator& scratch,
                             ffi::AnyBuffer input,
                             ffi::Result<ffi::AnyBuffer> output) = 0;
  
  // Obtener métricas de rendimiento
  virtual absl::StatusOr<std::vector<float>> GetPerformanceMetrics() = 0;
  
  // Sincronizar operación (para operaciones asíncronas)
  virtual absl::Status Synchronize(tpuStream_t stream) = 0;
};

// ============================================================================
// TPU v4-32 Collective Kernel Factory
// ============================================================================

class CollectiveKernelFactory {
 public:
  static absl::StatusOr<std::unique_ptr<CollectiveKernel>> Create(
      CollectiveKernelType type,
      const CollectiveKernelConfig& config);
      
  // Verificar disponibilidad de comunicación colectiva
  static bool IsCollectiveAvailable();
  
  // Obtener información de topología TPU
  static absl::StatusOr<std::vector<int32_t>> GetTpuTopology();
};

// ============================================================================
// TPU v4-32 Collective Kernel Implementations
// ============================================================================

class AllReduceKernel : public CollectiveKernel {
 public:
  absl::Status Initialize(const CollectiveKernelConfig& config) override;
  absl::Status Execute(tpuStream_t stream,
                      ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer input,
                      ffi::Result<ffi::AnyBuffer> output) override;
  absl::StatusOr<std::vector<float>> GetPerformanceMetrics() override;
  absl::Status Synchronize(tpuStream_t stream) override;
  
 private:
  CollectiveKernelConfig config_;
  std::vector<float> metrics_;
  
  // Optimizaciones específicas para TPU v4-32
  absl::Status OptimizedAllReduce(tpuStream_t stream,
                                 ffi::AnyBuffer input,
                                 ffi::Result<ffi::AnyBuffer> output);
};

class AllGatherKernel : public CollectiveKernel {
 public:
  absl::Status Initialize(const CollectiveKernelConfig& config) override;
  absl::Status Execute(tpuStream_t stream,
                      ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer input,
                      ffi::Result<ffi::AnyBuffer> output) override;
  absl::StatusOr<std::vector<float>> GetPerformanceMetrics() override;
  absl::Status Synchronize(tpuStream_t stream) override;
  
 private:
  CollectiveKernelConfig config_;
  std::vector<float> metrics_;
};

class ReduceScatterKernel : public CollectiveKernel {
 public:
  absl::Status Initialize(const CollectiveKernelConfig& config) override;
  absl::Status Execute(tpuStream_t stream,
                      ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer input,
                      ffi::Result<ffi::AnyBuffer> output) override;
  absl::StatusOr<std::vector<float>> GetPerformanceMetrics() override;
  absl::Status Synchronize(tpuStream_t stream) override;
  
 private:
  CollectiveKernelConfig config_;
  std::vector<float> metrics_;
};

class AllToAllKernel : public CollectiveKernel {
 public:
  absl::Status Initialize(const CollectiveKernelConfig& config) override;
  absl::Status Execute(tpuStream_t stream,
                      ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer input,
                      ffi::Result<ffi::AnyBuffer> output) override;
  absl::StatusOr<std::vector<float>> GetPerformanceMetrics() override;
  absl::Status Synchronize(tpuStream_t stream) override;
  
 private:
  CollectiveKernelConfig config_;
  std::vector<float> metrics_;
};

class PmeanKernel : public CollectiveKernel {
 public:
  absl::Status Initialize(const CollectiveKernelConfig& config) override;
  absl::Status Execute(tpuStream_t stream,
                      ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer input,
                      ffi::Result<ffi::AnyBuffer> output) override;
  absl::StatusOr<std::vector<float>> GetPerformanceMetrics() override;
  absl::Status Synchronize(tpuStream_t stream) override;
  
 private:
  CollectiveKernelConfig config_;
  std::vector<float> metrics_;
};

}  // namespace tpu_v4
}  // namespace jax

#endif  // JAXLIB_TPU_V4_COLLECTIVE_KERNELS_H_ 
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

#ifndef JAXLIB_TPU_V4_ATTENTION_KERNELS_H_
#define JAXLIB_TPU_V4_ATTENTION_KERNELS_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "jaxlib/tpu_v4/tpu_kernel_helpers.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace tpu_v4 {

// ============================================================================
// TPU v4-32 Attention Kernel Types
// ============================================================================

enum class AttentionKernelType {
  MULTI_HEAD_ATTENTION,    // Atención multi-cabeza estándar
  FLASH_ATTENTION,         // Atención eficiente en memoria
  SPARSE_ATTENTION,        // Atención sparse para contextos largos
  CROSS_ATTENTION,         // Atención cruzada multimodal
  GROUPED_QUERY_ATTENTION, // Grouped Query Attention (GQA)
  SLIDING_WINDOW_ATTENTION // Atención con ventana deslizante
};

// ============================================================================
// TPU v4-32 Attention Kernel Configuration
// ============================================================================

struct AttentionKernelConfig {
  AttentionKernelType type;
  int64_t batch_size;
  int64_t sequence_length;
  int64_t num_heads;
  int64_t head_dim;
  int64_t kv_heads;           // Para GQA
  int64_t window_size;        // Para sliding window
  float scale;                // Factor de escala (1/sqrt(head_dim))
  bool use_causal_mask;       // Máscara causal para autoregresión
  bool use_flash_attention;   // Usar optimización Flash Attention
  int64_t block_size_q;       // Tamaño de bloque para queries
  int64_t block_size_kv;      // Tamaño de bloque para keys/values
  float sparsity_threshold;   // Umbral para atención sparse
};

// ============================================================================
// TPU v4-32 Attention Kernel Interface
// ============================================================================

class AttentionKernel {
 public:
  virtual ~AttentionKernel() = default;
  
  // Inicializar kernel con configuración
  virtual absl::Status Initialize(const AttentionKernelConfig& config) = 0;
  
  // Ejecutar kernel de atención
  virtual absl::Status Execute(tpuStream_t stream,
                             ffi::ScratchAllocator& scratch,
                             ffi::AnyBuffer query,
                             ffi::AnyBuffer key,
                             ffi::AnyBuffer value,
                             ffi::AnyBuffer mask,
                             ffi::Result<ffi::AnyBuffer> output,
                             ffi::Result<ffi::AnyBuffer> attention_weights) = 0;
  
  // Obtener métricas de rendimiento
  virtual absl::StatusOr<std::vector<float>> GetPerformanceMetrics() = 0;
};

// ============================================================================
// TPU v4-32 Attention Kernel Factory
// ============================================================================

class AttentionKernelFactory {
 public:
  static absl::StatusOr<std::unique_ptr<AttentionKernel>> Create(
      AttentionKernelType type,
      const AttentionKernelConfig& config);
};

// ============================================================================
// TPU v4-32 Attention Kernel Implementations
// ============================================================================

class MultiHeadAttentionKernel : public AttentionKernel {
 public:
  absl::Status Initialize(const AttentionKernelConfig& config) override;
  absl::Status Execute(tpuStream_t stream,
                      ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer query,
                      ffi::AnyBuffer key,
                      ffi::AnyBuffer value,
                      ffi::AnyBuffer mask,
                      ffi::Result<ffi::AnyBuffer> output,
                      ffi::Result<ffi::AnyBuffer> attention_weights) override;
  absl::StatusOr<std::vector<float>> GetPerformanceMetrics() override;
  
 private:
  AttentionKernelConfig config_;
  std::vector<float> metrics_;
  
  // Optimizaciones específicas para TPU v4-32
  absl::Status OptimizedMatMul(tpuStream_t stream,
                              ffi::AnyBuffer a,
                              ffi::AnyBuffer b,
                              ffi::Result<ffi::AnyBuffer> c);
};

class FlashAttentionKernel : public AttentionKernel {
 public:
  absl::Status Initialize(const AttentionKernelConfig& config) override;
  absl::Status Execute(tpuStream_t stream,
                      ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer query,
                      ffi::AnyBuffer key,
                      ffi::AnyBuffer value,
                      ffi::AnyBuffer mask,
                      ffi::Result<ffi::AnyBuffer> output,
                      ffi::Result<ffi::AnyBuffer> attention_weights) override;
  absl::StatusOr<std::vector<float>> GetPerformanceMetrics() override;
  
 private:
  AttentionKernelConfig config_;
  std::vector<float> metrics_;
  
  // Implementación Flash Attention optimizada para TPU
  absl::Status FlashAttentionTiled(tpuStream_t stream,
                                  ffi::ScratchAllocator& scratch,
                                  ffi::AnyBuffer query,
                                  ffi::AnyBuffer key,
                                  ffi::AnyBuffer value,
                                  ffi::Result<ffi::AnyBuffer> output);
};

class SparseAttentionKernel : public AttentionKernel {
 public:
  absl::Status Initialize(const AttentionKernelConfig& config) override;
  absl::Status Execute(tpuStream_t stream,
                      ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer query,
                      ffi::AnyBuffer key,
                      ffi::AnyBuffer value,
                      ffi::AnyBuffer mask,
                      ffi::Result<ffi::AnyBuffer> output,
                      ffi::Result<ffi::AnyBuffer> attention_weights) override;
  absl::StatusOr<std::vector<float>> GetPerformanceMetrics() override;
  
 private:
  AttentionKernelConfig config_;
  std::vector<float> metrics_;
};

class CrossAttentionKernel : public AttentionKernel {
 public:
  absl::Status Initialize(const AttentionKernelConfig& config) override;
  absl::Status Execute(tpuStream_t stream,
                      ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer query,
                      ffi::AnyBuffer key,
                      ffi::AnyBuffer value,
                      ffi::AnyBuffer mask,
                      ffi::Result<ffi::AnyBuffer> output,
                      ffi::Result<ffi::AnyBuffer> attention_weights) override;
  absl::StatusOr<std::vector<float>> GetPerformanceMetrics() override;
  
 private:
  AttentionKernelConfig config_;
  std::vector<float> metrics_;
};

class GroupedQueryAttentionKernel : public AttentionKernel {
 public:
  absl::Status Initialize(const AttentionKernelConfig& config) override;
  absl::Status Execute(tpuStream_t stream,
                      ffi::ScratchAllocator& scratch,
                      ffi::AnyBuffer query,
                      ffi::AnyBuffer key,
                      ffi::AnyBuffer value,
                      ffi::AnyBuffer mask,
                      ffi::Result<ffi::AnyBuffer> output,
                      ffi::Result<ffi::AnyBuffer> attention_weights) override;
  absl::StatusOr<std::vector<float>> GetPerformanceMetrics() override;
  
 private:
  AttentionKernelConfig config_;
  std::vector<float> metrics_;
};

}  // namespace tpu_v4
}  // namespace jax

#endif  // JAXLIB_TPU_V4_ATTENTION_KERNELS_H_ 
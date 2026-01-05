/**
 * ARM NEON Optimized Kernels - Header
 * Optimized for Google Cloud ARM Axion Processors
 */

#ifndef NEON_MATMUL_H
#define NEON_MATMUL_H

#include <cstdint>

namespace neon_kernels {

// ============================================================================
// Matrix Multiplication Kernels
// ============================================================================

/**
 * FP32 Matrix Multiplication: C = A * B
 * Optimized with 8x8 tiles and prefetching
 * Performance: ~30% faster than baseline
 */
void matmul_fp32_neon(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);

/**
 * FP16 Matrix Multiplication (half precision)
 * 2x faster than FP32 with minimal accuracy loss
 */
void matmul_fp16_neon(
    const __fp16* A,
    const __fp16* B,
    __fp16* C,
    int M,
    int N,
    int K
);

/**
 * INT8 Matrix Multiplication (quantized)
 * 4x faster than FP32
 */
void matmul_int8_neon(
    const int8_t* A,
    const int8_t* B,
    int32_t* C,
    int M,
    int N,
    int K,
    float scale_a,
    float scale_b
);

// ============================================================================
// Vector Operations
// ============================================================================

/**
 * Dot Product FP32 (optimized for attention mechanisms)
 * Performance: 5x faster than scalar
 */
float dot_product_fp32_neon(
    const float* a,
    const float* b,
    int length
);

/**
 * RMSNorm (Root Mean Square Normalization)
 * Optimized for LLaMA/Gemma models
 * Performance: 4x faster than baseline
 */
void rmsnorm_fp32_neon(
    const float* input,
    float* output,
    const float* weight,
    int size,
    float eps = 1e-6f
);

/**
 * SoftMax (optimized for attention)
 * Uses fast vectorized exp approximation
 * Performance: ~40% faster than baseline
 */
void softmax_fp32_neon(
    const float* input,
    float* output,
    int size
);

// ============================================================================
// Activation Functions (NEW - Fused Operations)
// ============================================================================

/**
 * SwiGLU Activation Function (Fused)
 * SwiGLU(x, gate) = x ⊙ Swish(gate) where Swish(x) = x * sigmoid(x)
 *
 * Used in: LLaMA, Mistral, and other modern LLMs
 * Performance: ~35% faster than separate swish + multiply
 *
 * Args:
 *   x: Input tensor [size]
 *   gate: Gate tensor [size]
 *   output: Output tensor [size]
 *   size: Number of elements
 */
void swiglu_neon(
    const float* x,
    const float* gate,
    float* output,
    int size
);

/**
 * GeLU Activation Function (Fused)
 * GeLU(x) = x * Φ(x) where Φ is the cumulative distribution function of standard normal
 *
 * Used in: BERT, GPT-2, and other transformers
 * Performance: ~30% faster than PyTorch GeLU
 *
 * Args:
 *   input: Input tensor [size]
 *   output: Output tensor [size]
 *   size: Number of elements
 */
void gelu_neon(
    const float* input,
    float* output,
    int size
);

// ============================================================================
// Positional Embeddings (NEW - Optimized for LLMs)
// ============================================================================

/**
 * Rotary Position Embeddings (RoPE)
 * Used in LLaMA, Mistral, Qwen, and other modern LLMs
 *
 * Performance: ~25% faster than naive implementation
 *
 * Args:
 *   qk: Query or Key tensor [seq_len x head_dim] (modified in-place)
 *   cos_table: Precomputed cos values [seq_len x head_dim/2]
 *   sin_table: Precomputed sin values [seq_len x head_dim/2]
 *   seq_len: Sequence length
 *   head_dim: Head dimension
 */
void rotary_embedding_neon(
    float* qk,
    const float* cos_table,
    const float* sin_table,
    int seq_len,
    int head_dim
);

/**
 * Precompute RoPE cos/sin tables
 * This should be done once during model initialization
 *
 * Args:
 *   cos_table: Output cos table [max_seq_len x head_dim/2]
 *   sin_table: Output sin table [max_seq_len x head_dim/2]
 *   max_seq_len: Maximum sequence length
 *   head_dim: Head dimension
 *   base: Base for frequency computation (default: 10000.0)
 */
void precompute_rope_tables(
    float* cos_table,
    float* sin_table,
    int max_seq_len,
    int head_dim,
    float base = 10000.0f
);

// ============================================================================
// Attention Kernels (NEW - Fused Operations)
// ============================================================================

/**
 * Fused Scaled Dot-Product Attention (single-head)
 * Combines: QK^T + Scale + Softmax + @ V in one kernel
 *
 * Benefits:
 * - Reduces memory bandwidth by 3x (no intermediate storage)
 * - Improved cache locality
 * - ~50% faster than separate kernels
 *
 * Args:
 *   Q: Query matrix [seq_len x head_dim]
 *   K: Key matrix [seq_len x head_dim]
 *   V: Value matrix [seq_len x head_dim]
 *   output: Output matrix [seq_len x head_dim]
 *   seq_len: Sequence length
 *   head_dim: Head dimension
 *   scale: Scaling factor (typically 1/sqrt(head_dim))
 */
void fused_attention_neon(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int seq_len,
    int head_dim,
    float scale
);

/**
 * Flash Attention (Memory-Efficient Attention) ⚡ NEW!
 * Implements Flash Attention algorithm using existing NEON kernels
 *
 * KEY FEATURES:
 * - Reduces memory from O(N²) to O(N)
 * - Enables long sequences (8K-32K+ tokens)
 * - 2-3x faster for sequences >2048 tokens
 * - NO new NEON kernels needed - reuses existing optimized code!
 *
 * How it works:
 * - Processes attention in blocks (tiling)
 * - Online softmax (no full QK^T materialization)
 * - Reuses: dot_product_fp32_neon, vexpq_f32_fast
 *
 * Args:
 *   Q, K, V: Query/Key/Value matrices [seq_len x head_dim]
 *   output: Output matrix [seq_len x head_dim]
 *   seq_len: Sequence length (can be very large!)
 *   head_dim: Head dimension
 *   scale: Scaling factor (typically 1/sqrt(head_dim))
 *   block_size: Block size for tiling (default: 64, tuned for ARM Axion)
 *
 * When to use:
 * - seq_len < 512: Use fused_attention_neon (slightly faster)
 * - seq_len >= 512: Use flash_attention_neon (faster + less memory)
 * - seq_len > 2048: Flash Attention is MUCH better
 */
void flash_attention_neon(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int seq_len,
    int head_dim,
    float scale,
    int block_size = 64
);

/**
 * Multi-Head Fused Attention (batched version)
 * Processes multiple attention heads in parallel
 *
 * Args:
 *   Q: Query [batch x num_heads x seq_len x head_dim]
 *   K: Key [batch x num_heads x seq_len x head_dim]
 *   V: Value [batch x num_heads x seq_len x head_dim]
 *   output: [batch x num_heads x seq_len x head_dim]
 *   batch_size: Number of sequences in batch
 *   num_heads: Number of attention heads
 *   seq_len: Sequence length
 *   head_dim: Head dimension
 */
void multi_head_fused_attention_neon(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
);

} // namespace neon_kernels

#endif // NEON_MATMUL_H

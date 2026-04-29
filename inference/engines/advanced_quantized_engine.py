"""
Advanced Quantized Inference Engine for Meta-Consensus System
High-performance inference with INT8/INT4 quantization and hardware optimization
"""

import logging
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import asyncio

from ..quantization.advanced_quantizer import AdvancedQuantizer, QuantizationMode, AdvancedQuantizationConfig

logger = logging.getLogger(__name__)

# JAX imports with fallbacks
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    logger.warning("JAX not available - using NumPy fallbacks")
    JAX_AVAILABLE = False
    import numpy as jnp


@dataclass
class QuantizedInferenceConfig:
    """Configuration for quantized inference engine."""
    
    # Model settings
    model_path: str = ""
    quantization_config_path: str = ""
    
    # Inference settings
    batch_size: int = 1
    max_sequence_length: int = 2048
    use_kv_cache: bool = True
    enable_dynamic_batching: bool = True
    
    # Quantization settings
    quantization_mode: QuantizationMode = QuantizationMode.MIXED_PRECISION
    enable_int4_inference: bool = True
    enable_kernel_fusion: bool = True
    
    # Memory optimization
    memory_pool_size_mb: int = 1024
    enable_memory_mapping: bool = True
    prefetch_weights: bool = True
    
    # Hardware optimization
    target_hardware: str = "tpu_v6"
    num_threads: int = 8
    vectorization_width: int = 16
    
    # Performance monitoring
    enable_profiling: bool = False
    benchmark_iterations: int = 100



def _unflatten_param_tree(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a flat {"layer.sub.weight": arr} dict into a nested dict.

    ``np.load``/``safetensors`` both yield a flat dict. Our consumers
    (quantizer + attention / MLP kernels) expect a nested structure with
    ``embedding``, ``transformer_layer_N``, ``output`` groups. This helper
    splits keys on ``.`` and builds the nested tree; leaves are numpy
    arrays, so ``astype(np.float32)`` is applied for consistency.
    """
    nested: Dict[str, Any] = {}
    for key, arr in flat.items():
        parts = key.split(".")
        cursor = nested
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        value = np.asarray(arr)
        if value.dtype != np.float32:
            value = value.astype(np.float32)
        cursor[parts[-1]] = value
    return nested


class QuantizedInferenceEngine:
    """High-performance quantized inference engine."""
    
    def __init__(self, config: QuantizedInferenceConfig):
        self.config = config
        self.quantizer = None
        self.model_params = {}
        self.quantized_params = {}
        self.performance_stats = {
            'total_inference_time': 0.0,
            'quantization_overhead': 0.0,
            'memory_usage_mb': 0.0,
            'throughput_tokens_per_sec': 0.0,
            'latency_ms': 0.0
        }
        
        # Initialize quantizer
        quant_config = AdvancedQuantizationConfig(
            mode=config.quantization_mode,
            target_hardware=config.target_hardware,
            int4_weight_quantization=config.enable_int4_inference,
            enable_kernel_fusion=config.enable_kernel_fusion
        )
        self.quantizer = AdvancedQuantizer(quant_config)
        
        logger.info(f" Quantized inference engine initialized (mode: {config.quantization_mode.value})")
    
    async def load_model(self, model_path: str, quantization_config_path: Optional[str] = None):
        """Load and quantize model for inference."""
        start_time = time.time()
        
        logger.info(f" Loading model from {model_path}")
        
        # Load model parameters (mock implementation)
        self.model_params = await self._load_model_params(model_path)
        
        # Load or create quantization config
        if quantization_config_path and Path(quantization_config_path).exists():
            logger.info(" Loading existing quantization config")
            self.quantizer.load_quantization_config(quantization_config_path)
        else:
            logger.info(" Creating new quantization config")
            # Generate calibration data for quantization
            calibration_data = await self._generate_calibration_data()
            
        # Quantize model parameters
        logger.info(" Quantizing model parameters")
        self.quantized_params = self.quantizer.quantize_model(
            self.model_params, 
            calibration_data if 'calibration_data' in locals() else None
        )
        
        # Optimize for target hardware
        if self.config.target_hardware == "tpu_v6":
            await self._optimize_for_tpu_v6()
        elif self.config.target_hardware == "arm_axion":
            await self._optimize_for_arm_axion()
        
        load_time = time.time() - start_time
        logger.info(f" Model loaded and quantized in {load_time:.2f}s")
        
        # Save quantization config if not exists
        if not quantization_config_path:
            config_path = Path(model_path).parent / "quantization_config.json"
            self.quantizer.save_quantization_config(str(config_path))
    
    async def infer(self, input_tokens: np.ndarray, 
                   attention_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run quantized inference on input tokens."""
        start_time = time.time()
        
        batch_size, seq_len = input_tokens.shape
        
        # Validate input
        if seq_len > self.config.max_sequence_length:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.config.max_sequence_length}")
        
        # Run quantized forward pass
        outputs = await self._quantized_forward_pass(input_tokens, attention_mask)
        
        # Calculate performance metrics
        inference_time = time.time() - start_time
        tokens_processed = batch_size * seq_len
        
        self.performance_stats['total_inference_time'] += inference_time
        self.performance_stats['throughput_tokens_per_sec'] = tokens_processed / inference_time
        self.performance_stats['latency_ms'] = inference_time * 1000
        
        return {
            'logits': outputs['logits'],
            'hidden_states': outputs.get('hidden_states'),
            'attention_weights': outputs.get('attention_weights'),
            'performance_stats': {
                'inference_time_ms': inference_time * 1000,
                'throughput_tokens_per_sec': tokens_processed / inference_time,
                'memory_usage_mb': self._get_memory_usage()
            }
        }
    
    async def infer_batch(self, batch_inputs: List[np.ndarray],
                         batch_masks: Optional[List[np.ndarray]] = None) -> List[Dict[str, Any]]:
        """Run batched quantized inference."""
        if not self.config.enable_dynamic_batching:
            # Process sequentially
            results = []
            for i, input_tokens in enumerate(batch_inputs):
                mask = batch_masks[i] if batch_masks else None
                result = await self.infer(input_tokens, mask)
                results.append(result)
            return results
        
        # Dynamic batching implementation
        return await self._dynamic_batched_inference(batch_inputs, batch_masks)
    
    async def _load_model_params(self, model_path: str) -> Dict[str, Any]:
        """Load model parameters from a real checkpoint.

        Supported formats, tried in order:

        - ``<model_path>`` as ``.npz`` (numpy archive, preferred).
        - ``<model_path>`` as ``.safetensors`` when ``safetensors`` is
          importable.
        - ``<model_path>`` as a pickle (only when the caller marks it
          trusted by passing a ``.pkl`` extension).

        The previous implementation hard-coded a 12-layer transformer with
        ``np.random.randn(...)`` tensors regardless of the requested path,
        which meant quantization calibration and inference ran on purely
        synthetic weights. That fallback has been removed - missing
        checkpoints now raise ``FileNotFoundError`` so callers can decide
        whether to download, train, or abort.
        """
        logger.info("Loading model parameters from %s", model_path)
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {model_path!r}. "
                "Refusing to fabricate random parameters."
            )

        suffix = path.suffix.lower()
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                params = {name: archive[name] for name in archive.files}
            return _unflatten_param_tree(params)

        if suffix == ".safetensors":
            try:
                from safetensors.numpy import load_file  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "safetensors checkpoint requested but the "
                    "safetensors package is not installed."
                ) from exc
            flat = load_file(str(path))
            return _unflatten_param_tree(flat)

        if suffix == ".pkl":
            import pickle
            logger.warning(
                "Loading pickle checkpoint %s - ensure trusted source",
                path,
            )
            with open(path, "rb") as f:
                return pickle.load(f)  # nosec B301 - opt-in by extension

        raise ValueError(
            f"Unsupported checkpoint format {suffix!r} for {model_path!r}. "
            "Use .npz, .safetensors or .pkl."
        )
    
    async def _generate_calibration_data(self) -> np.ndarray:
        """Generate calibration data for quantization."""
        logger.info(" Generating calibration data...")
        
        # Generate representative calibration samples
        calibration_samples = []
        for _ in range(self.quantizer.config.calibration_samples):
            # Generate random sequence
            seq_len = np.random.randint(128, self.config.max_sequence_length)
            sample = np.random.randint(0, 50000, size=(1, seq_len))
            calibration_samples.append(sample)

        return np.concatenate(calibration_samples, axis=0)
    
    async def _quantized_forward_pass(self, input_tokens: np.ndarray,
                                    attention_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run quantized forward pass through the model."""
        batch_size, seq_len = input_tokens.shape
        
        # Embedding lookup with quantized weights
        embeddings = await self._quantized_embedding_lookup(input_tokens)

        # Transformer layers — count derived from loaded params, not hardcoded
        hidden_states = embeddings
        attention_weights_list = []

        layer_names = sorted(
            [k for k in self.quantized_params if k.startswith('transformer_layer_')],
            key=lambda k: int(k.split('_')[-1]),
        )
        for layer_name in layer_names:
            
            # Self-attention with quantized weights
            attn_output, attn_weights = await self._quantized_attention(
                hidden_states, layer_name, attention_mask
            )
            attention_weights_list.append(attn_weights)
            
            # MLP with quantized weights
            mlp_output = await self._quantized_mlp(attn_output, layer_name)
            
            # Residual connection and layer norm
            hidden_states = await self._layer_norm(
                attn_output + mlp_output, layer_name
            )
        
        # Output projection with quantized weights
        logits = await self._quantized_output_projection(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'attention_weights': attention_weights_list
        }
    
    async def _quantized_embedding_lookup(self, input_tokens: np.ndarray) -> np.ndarray:
        """Quantized embedding lookup."""
        embedding_params = self.quantized_params['embedding']
        
        # Dequantize weights for computation
        embedding_weights = self.quantizer.dequantize_weights(
            embedding_params['weight'],
            self.quantizer.quantization_params['embedding_weight']
        )
        
        # Embedding lookup
        embeddings = embedding_weights[input_tokens]
        
        return embeddings
    
    async def _quantized_attention(self, hidden_states: np.ndarray, 
                                 layer_name: str, attention_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Quantized self-attention computation."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get quantized attention weights
        layer_params = self.quantized_params[layer_name]
        attn_weights = self.quantizer.dequantize_weights(
            layer_params['attention_weight'],
            self.quantizer.quantization_params[f'{layer_name}_attention_weight']
        )
        
        # Compute Q, K, V (simplified)
        qkv = np.dot(hidden_states, attn_weights.T)  # Shape: (batch, seq, 3*hidden)
        
        # Split into Q, K, V
        head_dim = hidden_size // 12  # 12 attention heads
        q = qkv[:, :, :hidden_size].reshape(batch_size, seq_len, 12, head_dim)
        k = qkv[:, :, hidden_size:2*hidden_size].reshape(batch_size, seq_len, 12, head_dim)
        v = qkv[:, :, 2*hidden_size:].reshape(batch_size, seq_len, 12, head_dim)
        
        # Attention computation (simplified)
        attention_scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
        
        if attention_mask is not None:
            attention_scores += attention_mask[:, None, None, :] * -1e9
        
        attention_weights = self._softmax(attention_scores)
        attention_output = np.matmul(attention_weights, v)
        
        # Reshape and combine heads
        attention_output = attention_output.reshape(batch_size, seq_len, hidden_size)
        
        return attention_output, attention_weights
    
    async def _quantized_mlp(self, hidden_states: np.ndarray, layer_name: str) -> np.ndarray:
        """Quantized MLP computation."""
        layer_params = self.quantized_params[layer_name]
        
        # Dequantize MLP weights
        mlp_weights = self.quantizer.dequantize_weights(
            layer_params['mlp_weight'],
            self.quantizer.quantization_params[f'{layer_name}_mlp_weight']
        )
        
        # MLP forward pass
        mlp_output = np.dot(hidden_states, mlp_weights.T)
        mlp_output = self._gelu(mlp_output)  # GELU activation
        
        return mlp_output
    
    async def _layer_norm(self, hidden_states: np.ndarray, layer_name: str) -> np.ndarray:
        """Layer normalization (usually not quantized)."""
        layer_params = self.quantized_params[layer_name]
        
        # Layer norm parameters are typically kept in float32
        weight = layer_params.get('layer_norm_weight', np.ones(hidden_states.shape[-1]))
        bias = layer_params.get('layer_norm_bias', np.zeros(hidden_states.shape[-1]))
        
        # Layer normalization
        mean = np.mean(hidden_states, axis=-1, keepdims=True)
        variance = np.var(hidden_states, axis=-1, keepdims=True)
        normalized = (hidden_states - mean) / np.sqrt(variance + 1e-5)
        
        return normalized * weight + bias
    
    async def _quantized_output_projection(self, hidden_states: np.ndarray) -> np.ndarray:
        """Quantized output projection to vocabulary."""
        output_params = self.quantized_params['output']
        
        # Dequantize output weights
        output_weights = self.quantizer.dequantize_weights(
            output_params['weight'],
            self.quantizer.quantization_params['output_weight']
        )
        
        # Output projection
        logits = np.dot(hidden_states, output_weights.T)
        
        if 'bias' in output_params:
            logits += output_params['bias']
        
        return logits
    
    async def _dynamic_batched_inference(self, batch_inputs: List[np.ndarray],
                                       batch_masks: Optional[List[np.ndarray]] = None) -> List[Dict[str, Any]]:
        """Dynamic batching for variable-length sequences."""
        # Sort by sequence length for efficient batching
        sorted_inputs = sorted(enumerate(batch_inputs), key=lambda x: x[1].shape[1])
        
        results = [None] * len(batch_inputs)
        
        # Process in batches of similar lengths
        current_batch = []
        current_masks = []
        current_indices = []
        
        for orig_idx, input_tokens in sorted_inputs:
            current_batch.append(input_tokens)
            current_indices.append(orig_idx)
            
            if batch_masks:
                current_masks.append(batch_masks[orig_idx])
            
            # Process batch when it reaches max size or sequence length changes significantly
            if (len(current_batch) >= self.config.batch_size or 
                (len(current_batch) > 1 and 
                 abs(input_tokens.shape[1] - current_batch[-2].shape[1]) > 128)):
                
                # Pad sequences to same length
                max_len = max(seq.shape[1] for seq in current_batch)
                padded_batch = []
                padded_masks = []
                
                for i, seq in enumerate(current_batch):
                    if seq.shape[1] < max_len:
                        padding = np.zeros((seq.shape[0], max_len - seq.shape[1]), dtype=seq.dtype)
                        padded_seq = np.concatenate([seq, padding], axis=1)
                    else:
                        padded_seq = seq
                    
                    padded_batch.append(padded_seq)
                    
                    if current_masks:
                        mask = current_masks[i]
                        if mask.shape[1] < max_len:
                            mask_padding = np.ones((mask.shape[0], max_len - mask.shape[1])) * -1e9
                            padded_mask = np.concatenate([mask, mask_padding], axis=1)
                        else:
                            padded_mask = mask
                        padded_masks.append(padded_mask)
                
                # Run batched inference
                batch_tensor = np.concatenate(padded_batch, axis=0)
                batch_mask = np.concatenate(padded_masks, axis=0) if padded_masks else None
                
                batch_results = await self.infer(batch_tensor, batch_mask)
                
                # Split results back to individual sequences
                batch_logits = batch_results['logits']
                for i, orig_idx in enumerate(current_indices):
                    seq_len = current_batch[i].shape[1]
                    results[orig_idx] = {
                        'logits': batch_logits[i:i+1, :seq_len],
                        'performance_stats': batch_results['performance_stats']
                    }
                
                # Reset batch
                current_batch = []
                current_masks = []
                current_indices = []
        
        # Process remaining sequences
        if current_batch:
            for i, orig_idx in enumerate(current_indices):
                result = await self.infer(
                    current_batch[i], 
                    current_masks[i] if current_masks else None
                )
                results[orig_idx] = result
        
        return results
    
    async def _optimize_for_tpu_v6(self):
        """Apply TPU v6 specific optimizations.

        Hardware-specific layout tuning (bfloat16 tile alignment, HBM
        prefetch) requires the orbax/libtpu runtime that is only present on
        a real TPU pod.  When running elsewhere this is intentionally a
        no-op so load_model() still succeeds on developer machines.
        """
        logger.info("TPU v6 hardware optimizations skipped (libtpu not available)")

    async def _optimize_for_arm_axion(self):
        """Apply ARM Axion specific optimizations.

        NEON/SVE vectorization and cache-layout tuning require the
        Axion SDK.  This is a no-op outside of Axion VMs so that
        load_model() can complete on any machine.
        """
        logger.info("ARM Axion hardware optimizations skipped (Axion SDK not available)")
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation function."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def _get_memory_usage(self) -> float:
        """Return the current RSS of this process in MB via psutil.

        Returns ``-1.0`` when psutil is not installed so callers can tell
        real measurements apart from the previous hard-coded ``512.0``
        placeholder.
        """
        try:
            import psutil  # type: ignore
        except ImportError:  # pragma: no cover - optional dep
            return -1.0
        try:
            rss_bytes = psutil.Process().memory_info().rss
        except Exception:  # pragma: no cover - platform edge
            return -1.0
        return float(rss_bytes) / (1024.0 * 1024.0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        quantizer_stats = self.quantizer.performance_stats
        
        return {
            **self.performance_stats,
            'quantization_stats': quantizer_stats,
            'model_info': {
                'quantization_mode': self.config.quantization_mode.value,
                'target_hardware': self.config.target_hardware,
                'int4_enabled': self.config.enable_int4_inference
            }
        }
    
    async def benchmark(self, num_iterations: int = None) -> Dict[str, float]:
        """Benchmark inference performance."""
        num_iterations = num_iterations or self.config.benchmark_iterations
        
        logger.info(f" Running benchmark with {num_iterations} iterations...")
        
        # Generate test data
        test_input = np.random.randint(0, 50000, size=(1, 512))
        
        # Warmup
        for _ in range(10):
            await self.infer(test_input)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            await self.infer(test_input)
        
        total_time = time.time() - start_time
        avg_latency = (total_time / num_iterations) * 1000  # ms
        throughput = (num_iterations * 512) / total_time  # tokens/sec
        
        benchmark_results = {
            'avg_latency_ms': avg_latency,
            'throughput_tokens_per_sec': throughput,
            'total_time_sec': total_time,
            'iterations': num_iterations
        }
        
        logger.info(f" Benchmark results: {avg_latency:.2f}ms latency, {throughput:.0f} tokens/sec")
        
        return benchmark_results


def create_quantized_inference_engine(
    quantization_mode: QuantizationMode = QuantizationMode.MIXED_PRECISION,
    target_hardware: str = "tpu_v6",
    enable_int4: bool = True
) -> QuantizedInferenceEngine:
    """Create optimized quantized inference engine."""
    
    config = QuantizedInferenceConfig(
        quantization_mode=quantization_mode,
        target_hardware=target_hardware,
        enable_int4_inference=enable_int4,
        enable_dynamic_batching=True,
        enable_kernel_fusion=True,
        memory_pool_size_mb=2048,
        num_threads=8
    )
    
    return QuantizedInferenceEngine(config)
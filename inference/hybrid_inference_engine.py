#!/usr/bin/env python3
"""
Hybrid Inference Engine for CapibaraGPT v3

Hybrid inference system that supports multiple backends:
- TPU v6e: For large models trained on TPU
- GPU: For local development and testing
- CPU: Universal fallback

Features:
- Automatic detection of available hardware
- Backend-specific optimizations
- Load balancing between multiple instances
- Intelligent model caching
- Real-time token streaming
"""

import os
import sys
import asyncio
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Union, AsyncGenerator, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle

# Core libraries
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# JAX/TPU support
try:
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from flax.training import train_state
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# PyTorch support
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class InferenceBackend(Enum):
    """Available inference backend types."""
    TPU_V6E = "tpu_v6e"
    GPU_CUDA = "gpu_cuda"
    CPU = "cpu"
    AUTO = "auto"

@dataclass
class InferenceConfig:
    """Inference configuration."""
    backend: InferenceBackend = InferenceBackend.AUTO
    model_path: str = ""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    
    # Backend-specific settings
    tpu_mesh_shape: Tuple[int, int] = (1, 1)  # For TPU v6e
    gpu_device_id: int = 0    # For GPU
    
    # Performance
    batch_size: int = 1
    enable_streaming: bool = True
    enable_caching: bool = True
    cache_size_mb: int = 1024

@dataclass
class InferenceResult:
    """Inference result."""
    text: str
    tokens_generated: int
    time_taken: float
    tokens_per_second: float
    backend_used: str
    model_name: str
    prompt_tokens: int
    cached: bool = False

class TPUv6eInferenceEngine:
    """Inference engine optimized for TPU v6e."""
    
    def __init__(self, model_path: str, config: InferenceConfig):
        self.model_path = model_path
        self.config = config
        self.tokenizer = None
        self.model = None
        self.mesh = None
        self.compiled_generate = None
        self.loaded = False
        
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available for TPU inference")
    
    def load_model(self):
        """Load model on TPU v6e."""
        logger.info(f" Loading model on TPU v6e: {self.model_path}")

        try:
            # Configure JAX for TPU
            jax.config.update('jax_platforms', 'tpu')

            # Create mesh for TPU v6e
            devices = jax.devices()
            if len(devices) == 0:
                raise RuntimeError("No TPU devices found")

            # Configure mesh based on available devices
            num_devices = len(devices)
            if num_devices >= 64:  # TPU v6e 8x8
                mesh_shape = (8, 8)
            elif num_devices >= 32:  # TPU v4-32
                mesh_shape = (4, 8)
            elif num_devices >= 8:
                mesh_shape = (2, 4)
            else:
                mesh_shape = (1, num_devices)
            
            devices_array = np.array(devices[:mesh_shape[0] * mesh_shape[1]]).reshape(mesh_shape)
            self.mesh = jax.sharding.Mesh(devices_array, ('batch', 'model'))

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model parameters
            self._load_model_params()
            
            # Compile generation function
            self._compile_generation_function()
            
            self.loaded = True
            logger.info(f" TPU v6e model loaded successfully with mesh {mesh_shape}")
            
        except Exception as e:
            logger.error(f" Failed to load model on TPU v6e: {e}")
            raise
    
    def _load_model_params(self):
        """Load model parameters from a real JAX/Flax checkpoint.

        The previous behaviour silently fell back to
        ``{"dummy": jnp.array([1.0])}`` when no checkpoint file was
        present, which meant downstream generation ran on fabricated
        parameters without any signal to the caller. We now:

        - Try the documented pickle path (``checkpoint.pkl``).
        - Try flax.serialization msgpack (``checkpoint.msgpack`` or
          ``params.msgpack``) when flax is importable.
        - Try orbax (``checkpoint/`` directory) when orbax is importable.
        - Raise ``FileNotFoundError`` if none of those are usable. No
          synthetic fallback is ever returned.
        """
        base = Path(self.model_path)
        pickle_path = base / "checkpoint.pkl"
        if pickle_path.exists():
            logger.warning(
                "Loading pickle checkpoint from %s - ensure trusted source",
                pickle_path,
            )
            with open(pickle_path, "rb") as f:
                self.model_params = pickle.load(f)  # nosec B301 - trusted checkpoint
            return

        # Flax msgpack (primary portable format)
        try:
            from flax import serialization  # type: ignore
        except Exception:  # pragma: no cover - optional dep path
            serialization = None  # type: ignore

        if serialization is not None:
            for name in ("checkpoint.msgpack", "params.msgpack"):
                msgpack_path = base / name
                if msgpack_path.exists():
                    logger.info("Loading flax msgpack checkpoint from %s", msgpack_path)
                    with open(msgpack_path, "rb") as f:
                        self.model_params = serialization.msgpack_restore(f.read())
                    return

        # Orbax directory checkpoint (modern JAX format)
        orbax_dir = base / "checkpoint"
        if orbax_dir.is_dir():
            try:
                import orbax.checkpoint as ocp  # type: ignore
                logger.info("Loading orbax checkpoint from %s", orbax_dir)
                ckptr = ocp.StandardCheckpointer()
                self.model_params = ckptr.restore(orbax_dir.resolve())
                return
            except Exception as exc:  # pragma: no cover - optional dep path
                logger.warning("orbax restore failed for %s: %s", orbax_dir, exc)

        # No real checkpoint available. Refuse to fabricate parameters.
        raise FileNotFoundError(
            f"No model checkpoint found under {self.model_path!r}. "
            "Expected one of: checkpoint.pkl, checkpoint.msgpack, "
              "params.msgpack, or an orbax checkpoint/ directory. "
            "Aborting instead of returning mock parameters."
        )
    
    def _compile_generation_function(self):
        """Compile a real JAX generation step when a Flax module is wired.

        The engine exposes an optional ``self.model_module`` attribute
        (a Flax ``nn.Module`` callable as ``module.apply(params, input_ids,
        position)`` and returning logits). If it is present, we JIT-compile
        the real forward pass. Otherwise we leave
        ``self.compiled_generate = None`` so the ``generate`` path can
        fail fast with a clear message instead of producing logits made of
        ``jnp.ones(...)``.
        """
        module = getattr(self, "model_module", None)
        if module is None or self.model_params is None:
            self.compiled_generate = None
            logger.warning(
                "No Flax model_module attached to %s; generate() will fail "
                "fast until a real module is provided.",
                type(self).__name__,
            )
            return

        @jax.jit
        def generate_step(params, input_ids, position):
            # Real forward pass: the attached module returns logits.
            return module.apply(params, input_ids, position)

        self.compiled_generate = generate_step
    
    async def generate(self, prompt: str, **kwargs) -> InferenceResult:
        """Generate text using TPU v6e."""
        if not self.loaded:
            self.load_model()

        if self.compiled_generate is None:
            raise RuntimeError(
                "TPU inference engine has no compiled generation "
                "function. Attach a Flax ``model_module`` before "
                "calling generate(); the previous mock fallback has "
                "been removed (BACKLOG-004)."
            )

        start_time = time.time()

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="np", padding=True)
        input_ids = jnp.array(inputs["input_ids"])
        
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)
        
        generated_tokens = []
        current_ids = input_ids
        rng = jax.random.PRNGKey(0)

        # Autoregressive generation
        with self.mesh:
            for step in range(max_tokens):
                # Forward pass
                logits = self.compiled_generate(self.model_params, current_ids, step)

                # Sampling: temperature > 0 → multinomial; temperature == 0 → greedy
                if temperature > 0:
                    rng, sample_rng = jax.random.split(rng)
                    log_probs = jax.nn.log_softmax(logits[:, -1, :] / temperature)
                    next_token = jax.random.categorical(sample_rng, log_probs)[..., None]
                else:
                    next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
                
                generated_tokens.append(int(next_token[0]))
                current_ids = jnp.concatenate([current_ids, next_token], axis=1)
                
                # Check for EOS
                if int(next_token[0]) == self.tokenizer.eos_token_id:
                    break

        # Decode result
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        end_time = time.time()
        time_taken = end_time - start_time
        tokens_per_second = len(generated_tokens) / time_taken if time_taken > 0 else 0
        
        return InferenceResult(
            text=generated_text,
            tokens_generated=len(generated_tokens),
            time_taken=time_taken,
            tokens_per_second=tokens_per_second,
            backend_used="tpu_v6e",
            model_name=self.model_path,
            prompt_tokens=len(inputs["input_ids"][0])
        )
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text streaming one decoded token at a time."""
        if not self.loaded:
            self.load_model()

        if self.compiled_generate is None:
            raise RuntimeError(
                "TPU inference engine has no compiled generation function. "
                "Attach a Flax model_module before calling generate_stream()."
            )

        inputs = self.tokenizer(prompt, return_tensors="np", padding=True)
        current_ids = jnp.array(inputs["input_ids"])
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)
        rng = jax.random.PRNGKey(0)

        with self.mesh:
            for step in range(max_tokens):
                logits = self.compiled_generate(self.model_params, current_ids, step)

                if temperature > 0:
                    rng, sample_rng = jax.random.split(rng)
                    log_probs = jax.nn.log_softmax(logits[:, -1, :] / temperature)
                    next_token = jax.random.categorical(sample_rng, log_probs)[..., None]
                else:
                    next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)

                token_id = int(next_token[0])
                current_ids = jnp.concatenate([current_ids, next_token], axis=1)

                if token_id == self.tokenizer.eos_token_id:
                    break

                token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                if token_text:
                    yield token_text

class GPUInferenceEngine:
    """Inference engine for GPU/CPU using PyTorch."""
    
    def __init__(self, model_path: str, config: InferenceConfig):
        self.model_path = model_path
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = None
        self.loaded = False
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for GPU inference")
    
    def load_model(self):
        """Load model on GPU/CPU."""
        logger.info(f" Loading model on GPU/CPU: {self.model_path}")

        try:
            # Detect device
            if torch.cuda.is_available() and self.config.backend == InferenceBackend.GPU_CUDA:
                self.device = f"cuda:{self.config.gpu_device_id}"
            else:
                self.device = "cpu"

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                device_map="auto" if "cuda" in self.device else None,
                trust_remote_code=True
            )
            
            if "cuda" not in self.device:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.loaded = True
            
            logger.info(f" Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f" Failed to load model on GPU/CPU: {e}")
            raise
    
    async def generate(self, prompt: str, **kwargs) -> InferenceResult:
        """Generate text using GPU/CPU."""
        if not self.loaded:
            self.load_model()

        start_time = time.time()

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Configure generation parameters
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)
        top_p = kwargs.get('top_p', self.config.top_p)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=self.config.do_sample,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_ids = outputs[0][len(inputs["input_ids"][0]):]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        end_time = time.time()
        time_taken = end_time - start_time
        tokens_per_second = len(generated_ids) / time_taken if time_taken > 0 else 0
        
        return InferenceResult(
            text=generated_text,
            tokens_generated=len(generated_ids),
            time_taken=time_taken,
            tokens_per_second=tokens_per_second,
            backend_used=self.device,
            model_name=self.model_path,
            prompt_tokens=len(inputs["input_ids"][0])
        )

class HybridInferenceEngine:
    """Hybrid inference engine with support for multiple backends."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.engines = {}
        self.active_engine = None
        self.model_cache = {}
        self.performance_stats = {}

        # Detect available hardware
        self.available_backends = self._detect_available_backends()

        # Select optimal backend
        if config.backend == InferenceBackend.AUTO:
            self.selected_backend = self._select_optimal_backend()
        else:
            self.selected_backend = config.backend
        
        logger.info(f" Hybrid engine initialized with backend: {self.selected_backend}")
        logger.info(f" Available backends: {[b.value for b in self.available_backends]}")
    
    def _detect_available_backends(self) -> List[InferenceBackend]:
        """Detect available backends."""
        backends = []

        # TPU v6e
        if JAX_AVAILABLE:
            try:
                devices = jax.devices()
                if devices and devices[0].platform == 'tpu':
                    backends.append(InferenceBackend.TPU_V6E)
            except Exception:
                pass

        # GPU CUDA
        if TORCH_AVAILABLE and torch.cuda.is_available():
            backends.append(InferenceBackend.GPU_CUDA)

        # CPU (always available)
        backends.append(InferenceBackend.CPU)

        return backends
    
    def _select_optimal_backend(self) -> InferenceBackend:
        """Select optimal backend based on available hardware."""
        # Priority: TPU v6e > GPU > CPU
        if InferenceBackend.TPU_V6E in self.available_backends:
            return InferenceBackend.TPU_V6E
        elif InferenceBackend.GPU_CUDA in self.available_backends:
            return InferenceBackend.GPU_CUDA
        else:
            return InferenceBackend.CPU
    
    def load_model(self, model_path: str):
        """Load model on the selected backend."""
        self.config.model_path = model_path

        if model_path in self.model_cache:
            logger.info(f" Using cached model: {model_path}")
            self.active_engine = self.model_cache[model_path]
            return

        # Create engine based on backend
        if self.selected_backend == InferenceBackend.TPU_V6E:
            engine = TPUv6eInferenceEngine(model_path, self.config)
        else:  # GPU or CPU
            engine = GPUInferenceEngine(model_path, self.config)

        # Load model
        engine.load_model()

        # Cache and activate
        if self.config.enable_caching:
            self.model_cache[model_path] = engine
        
        self.active_engine = engine
        logger.info(f" Model loaded successfully on {self.selected_backend.value}")
    
    async def generate(self, prompt: str, **kwargs) -> InferenceResult:
        """Generate text using the active engine."""
        if not self.active_engine:
            raise RuntimeError("No model loaded")

        start_time = time.time()

        try:
            # Use specific engine
            if hasattr(self.active_engine, 'generate') and asyncio.iscoroutinefunction(self.active_engine.generate):
                result = await self.active_engine.generate(prompt, **kwargs)
            else:
                result = await self.active_engine.generate(prompt, **kwargs)

            # Update statistics
            self._update_performance_stats(result)

            return result

        except Exception as e:
            logger.error(f" Generation failed: {e}")
            # Fallback to CPU if there's an error
            if self.selected_backend != InferenceBackend.CPU:
                logger.warning(" Falling back to CPU")
                await self._fallback_to_cpu(prompt, **kwargs)
            raise
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text with streaming."""
        if not self.active_engine:
            raise RuntimeError("No model loaded")

        if hasattr(self.active_engine, 'generate_stream'):
            async for token in self.active_engine.generate_stream(prompt, **kwargs):
                yield token
        else:
            # Fallback for engines that don't support streaming: yield full result
            result = await self.generate(prompt, **kwargs)
            yield result.text
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get current backend information."""
        return {
            'selected_backend': self.selected_backend.value,
            'available_backends': [b.value for b in self.available_backends],
            'model_loaded': self.active_engine is not None,
            'model_path': self.config.model_path,
            'cache_size': len(self.model_cache),
            'performance_stats': self.performance_stats
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def switch_backend(self, backend: InferenceBackend):
        """Switch backend dynamically."""
        if backend not in self.available_backends:
            raise ValueError(f"Backend {backend.value} not available")
        
        logger.info(f" Switching from {self.selected_backend.value} to {backend.value}")
        
        self.selected_backend = backend
        self.active_engine = None
        
        # Reload model if configured
        if self.config.model_path:
            self.load_model(self.config.model_path)
    
    def _update_performance_stats(self, result: InferenceResult):
        """Update performance statistics."""
        backend = result.backend_used
        if backend not in self.performance_stats:
            self.performance_stats[backend] = {
                'total_requests': 0,
                'total_tokens': 0,
                'total_time': 0,
                'avg_tokens_per_second': 0,
                'last_request_time': 0
            }
        
        stats = self.performance_stats[backend]
        stats['total_requests'] += 1
        stats['total_tokens'] += result.tokens_generated
        stats['total_time'] += result.time_taken
        stats['avg_tokens_per_second'] = stats['total_tokens'] / stats['total_time']
        stats['last_request_time'] = time.time()
    
    async def _fallback_to_cpu(self, prompt: str, **kwargs):
        """Fallback to CPU in case of error."""
        original_backend = self.selected_backend
        try:
            self.switch_backend(InferenceBackend.CPU)
            return await self.generate(prompt, **kwargs)
        except Exception as e:
            # Restore original backend if fallback also fails
            self.selected_backend = original_backend
            raise e

# Factory function
def create_inference_engine(config: InferenceConfig = None) -> HybridInferenceEngine:
    """Create hybrid inference engine."""
    if config is None:
        config = InferenceConfig()

    return HybridInferenceEngine(config)

# Convenience functions
async def quick_generate(prompt: str, model_path: str = "", **kwargs) -> str:
    """Quick function to generate text."""
    config = InferenceConfig(model_path=model_path)
    engine = create_inference_engine(config)
    
    if model_path:
        engine.load_model(model_path)
    
    result = await engine.generate(prompt, **kwargs)
    return result.text

# Benchmark function
async def benchmark_backends(prompt: str, model_path: str, rounds: int = 3) -> Dict[str, Any]:
    """Performance benchmark across backends."""
    results = {}
    
    for backend in [InferenceBackend.TPU_V6E, InferenceBackend.GPU_CUDA, InferenceBackend.CPU]:
        try:
            config = InferenceConfig(backend=backend, model_path=model_path)
            engine = create_inference_engine(config)
            
            if backend not in engine.available_backends:
                continue
            
            engine.load_model(model_path)
            
            times = []
            tokens_per_second = []
            
            for _ in range(rounds):
                result = await engine.generate(prompt)
                times.append(result.time_taken)
                tokens_per_second.append(result.tokens_per_second)
            
            results[backend.value] = {
                'avg_time': sum(times) / len(times),
                'avg_tokens_per_second': sum(tokens_per_second) / len(tokens_per_second),
                'min_time': min(times),
                'max_time': max(times)
            }
            
        except Exception as e:
            logger.warning(f"Benchmark failed for {backend.value}: {e}")
            results[backend.value] = {'error': str(e)}
    
    return results
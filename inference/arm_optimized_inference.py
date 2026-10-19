#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CapibaraGPT v3 ARM Optimized Inference Engine
==============================================

Optimización específica para Google Cloud ARM Axion VMs:
- Configuración ARM64 nativa
- Optimizaciones de memoria
- Threading optimizado para Axion
- Compatible con modelos Cascade

Usage:
    from capibara.inference.arm_optimized_inference import ARMInferenceEngine
    
    engine = ARMInferenceEngine("path/to/model")
    result = engine.generate("Hello world", max_tokens=100)
"""

import os
import sys
import json
import logging
import threading
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import multiprocessing as mp

# Core ML libraries
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    TextGenerationPipeline,
    set_seed
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM

# ARM optimizations
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX Runtime not available. Install with: pip install onnxruntime")

# Performance monitoring
import psutil

logger = logging.getLogger(__name__)

class ARMInferenceEngine:
    """Motor de inferencia optimizado for ARM Axion"""
    
    def __init__(
        self, 
        model_path: str,
        device: str = "cpu",
        max_memory_gb: float = 16.0,
        num_threads: Optional[int] = None,
        use_onnx: bool = False
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.max_memory_gb = max_memory_gb
        self.use_onnx = use_onnx and ONNX_AVAILABLE
        
        # ARM Axion optimizations
        self.num_threads = num_threads or self._detect_optimal_threads()
        self._configure_arm_optimizations()
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.config = None
        self.pipeline = None
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "total_time_ms": 0,
            "avg_tokens_per_second": 0,
            "memory_peak_mb": 0
        }
        
        # Load model
        self._load_model()
        
        logger.info(f" ARM Inference Engine initialized")
        logger.info(f"    Model: {self.model_path}")
        logger.info(f"   ️ Device: {self.device}")
        logger.info(f"    Threads: {self.num_threads}")
        logger.info(f"    Max Memory: {self.max_memory_gb}GB")
        logger.info(f"    ONNX: {self.use_onnx}")
    
    def _detect_optimal_threads(self) -> int:
        """Detecta número óptimo de threads for ARM Axion"""
        cpu_count = mp.cpu_count()
        
        # ARM Axion specific optimizations
        if cpu_count >= 16:  # t2a-standard-16
            return min(cpu_count - 2, 14)  # Leave 2 cores for system
        elif cpu_count >= 8:  # t2a-standard-8
            return min(cpu_count - 1, 7)
        else:
            return max(cpu_count - 1, 1)
    
    def _configure_arm_optimizations(self):
        """Configura optimizaciones específicas for ARM"""
        
        # Threading configuration
        torch.set_num_threads(self.num_threads)
        os.environ["OMP_NUM_THREADS"] = str(self.num_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.num_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(self.num_threads)
        
        # Memory optimizations
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        # ARM specific optimizations
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        
        # cpu optimizations for ARM
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = True
        
        logger.info(f"️ ARM optimizations configured ({self.num_threads} threads)")
    
    def _load_model(self):
        """load model with optimizaciones ARM"""
        try:
            start_time = time.time()
            
            # Load tokenizer
            logger.info(" Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load config
            self.config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Memory-efficient loading for ARM
            logger.info(" Loading model with ARM optimizations...")
            
            model_kwargs = {
                "torch_dtype": torch.float32,  # ARM optimized precision
                "device_map": None,  # Load to cpu first
                "low_cpu_mem_usage": True,
                "trust_remote_code": True
            }
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # ARM optimization: Enable compilation if available
            if hasattr(torch, 'compile') and self.device == 'cpu':
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info(" Model compiled for ARM optimization")
                except Exception as e:
                    logger.warning(f"Failed to compile model: {e}")
            
            # Set to eval mode
            self.model.eval()
            
            # Create pipeline
            self.pipeline = TextGenerationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                torch_dtype=torch.float32
            )
            
            load_time = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            logger.info(f" Model loaded successfully")
            logger.info(f"   ️ Load time: {load_time:.2f}s")
            logger.info(f"    Memory usage: {memory_usage:.1f}MB")
            
            # Store peak memory
            self.stats["memory_peak_mb"] = memory_usage
            
        except Exception as e:
            logger.error(f" Error loading model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs
    ) -> str:
        """Genera texto with optimizaciones ARM"""
        
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Set random seed for reproducibility
            if not do_sample:
                set_seed(42)
            
            # Generation parameters optimized for ARM
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "num_return_sequences": num_return_sequences,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,  # ARM benefits from caching
                **kwargs
            }
            
            # Generate using pipeline for ARM optimization
            with torch.no_grad():
                results = self.pipeline(
                    prompt,
                    **generation_kwargs
                )
            
            # Extract generated text
            if isinstance(results, list):
                generated_text = results[0]["generated_text"]
            else:
                generated_text = results["generated_text"]
            
            # Remove original prompt from result
            response = generated_text[len(prompt):].strip()
            
            # Update statistics
            generation_time = (time.time() - start_time) * 1000  # ms
            tokens_generated = len(self.tokenizer.tokenize(response))
            
            self.stats["total_requests"] += 1
            self.stats["total_tokens_generated"] += tokens_generated
            self.stats["total_time_ms"] += generation_time
            self.stats["avg_tokens_per_second"] = (
                self.stats["total_tokens_generated"] / 
                (self.stats["total_time_ms"] / 1000)
            )
            
            logger.info(f" Generated {tokens_generated} tokens in {generation_time:.1f}ms")
            logger.info(f" Speed: {(tokens_generated / (generation_time / 1000)):.1f} tokens/second")
            
            return response
            
        except Exception as e:
            logger.error(f" Error during generation: {e}")
            raise
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        **kwargs
    ) -> List[str]:
        """Generación en lote optimizada for ARM"""
        
        results = []
        
        # ARM optimization: Process in smaller batches to avoid memory issues
        batch_size = min(len(prompts), 4)  # ARM memory limitation
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            batch_results = []
            for prompt in batch_prompts:
                result = self.generate(prompt, max_tokens=max_tokens, **kwargs)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """information del model cargado"""
        if not self.model or not self.config:
            return {"error": "Model not loaded"}
        
        return {
            "model_path": str(self.model_path),
            "architecture": self.config.architectures[0] if self.config.architectures else "unknown",
            "vocab_size": self.config.vocab_size,
            "hidden_size": getattr(self.config, 'hidden_size', 'unknown'),
            "num_layers": getattr(self.config, 'num_hidden_layers', 'unknown'),
            "num_attention_heads": getattr(self.config, 'num_attention_heads', 'unknown'),
            "max_position_embeddings": getattr(self.config, 'max_position_embeddings', 'unknown'),
            "torch_dtype": str(self.model.dtype),
            "device": str(self.model.device),
            "memory_footprint_mb": self.stats["memory_peak_mb"]
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Estadísticas de rendimiento"""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            **self.stats,
            "current_memory_mb": current_memory,
            "cpu_percent": cpu_percent,
            "threads_used": self.num_threads,
            "onnx_enabled": self.use_onnx
        }
    
    def benchmark(self, num_iterations: int = 10, prompt: str = "Tell me about ARM processors") -> Dict[str, Any]:
        """Ejecuta benchmark de rendimiento en ARM"""
        logger.info(f" Starting ARM benchmark ({num_iterations} iterations)")
        
        # Warm up
        self.generate(prompt, max_tokens=50)
        
        # Benchmark
        times = []
        tokens_generated = []
        
        for i in range(num_iterations):
            start_time = time.time()
            result = self.generate(prompt, max_tokens=50)
            end_time = time.time()
            
            generation_time = (end_time - start_time) * 1000  # ms
            num_tokens = len(self.tokenizer.tokenize(result))
            
            times.append(generation_time)
            tokens_generated.append(num_tokens)
            
            logger.info(f"  Iteration {i+1}/{num_iterations}: {generation_time:.1f}ms, {num_tokens} tokens")
        
        # Statistics
        avg_time = sum(times) / len(times)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        avg_tokens_per_second = avg_tokens / (avg_time / 1000)
        
        benchmark_results = {
            "iterations": num_iterations,
            "avg_generation_time_ms": avg_time,
            "min_generation_time_ms": min(times),
            "max_generation_time_ms": max(times),
            "avg_tokens_generated": avg_tokens,
            "avg_tokens_per_second": avg_tokens_per_second,
            "total_benchmark_time_ms": sum(times),
            "arm_optimized": True,
            "threads_used": self.num_threads
        }
        
        logger.info(f" Benchmark completed: {avg_tokens_per_second:.1f} tokens/second avg")
        
        return benchmark_results
    
    def cleanup(self):
        """Limpia recursos del model"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if self.pipeline:
            del self.pipeline
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info(" ARM Inference Engine cleaned up")

# Convenience function for direct usage
def run_inference(
    model_path: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """function de conveniencia for inferencia directa en ARM"""
    
    engine = ARMInferenceEngine(model_path)
    try:
        result = engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return result
    finally:
        engine.cleanup()

# CLI interface
def main():
    """Interfaz de line de comandos"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CapibaraGPT v3 ARM Optimized Inference")
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("--prompt", default="Hello! How are you?", help="Prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Initialize engine
        engine = ARMInferenceEngine(args.model_path)
        
        # Show model info
        if args.verbose:
            model_info = engine.get_model_info()
            logger.info("\n Model Information:")
            for key, value in model_info.items():
                logger.info(f"   {key}: {value}")
        
        if args.benchmark:
            # Run benchmark
            logger.info(f"\n Running ARM benchmark...")
            results = engine.benchmark(args.iterations, args.prompt)
            logger.info("\n Benchmark Results:")
            for key, value in results.items():
                logger.info(f"   {key}: {value}")
        else:
            # Single generation
            logger.info(f"\n Generating response...")
            logger.info(f"Prompt: {args.prompt}")
            
            start_time = time.time()
            response = engine.generate(
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            end_time = time.time()
            
            logger.info(f"\nResponse: {response}")
            logger.info(f"\n️ Generation time: {(end_time - start_time) * 1000:.1f}ms")
            
        # Show performance stats
        if args.verbose:
            stats = engine.get_performance_stats()
            logger.info("\n Performance Stats:")
            for key, value in stats.items():
                logger.info(f"   {key}: {value}")
        
    except Exception as e:
        logger.error(f" Error: {e}")
        sys.exit(1)
    finally:
        if 'engine' in locals():
            engine.cleanup()

if __name__ == "__main__":
    main()
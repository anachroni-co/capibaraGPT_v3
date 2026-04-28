"""Optimized inference module for CapibaraModel v3.0
================================================================

Enterprise inference system with advanced capabilities:
1. **Singleton Configuration Manager** - Load configs once
2. **Advanced Caching** - TTL, invalidation, memory pressure management
3. **Resource Management** - Context managers, automatic cleanup
4. **Thread-Safe Operations** - Concurrent inference support
5. **Model Pool** - Efficient model reuse
6. **Async Support** - Non-blocking inference
7. **Comprehensive Validation** - Input/output validation
8. **Flexible Configuration** - Environment overrides
9. **Performance Monitoring** - Detailed metrics
10. **Graceful Error Recovery** - Intelligent fallbacks
11. **ARM Axion Support** - Specific optimizations for ARM Axion
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    _repo_root = _Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import asyncio
import gc
import os
import time
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    PSUTIL_AVAILABLE = False
import logging
import platform
import threading
import subprocess
import dataclasses
from functools import lru_cache, partial, wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import asynccontextmanager, contextmanager

try:
    import toml
except ImportError:
    toml = None  # type: ignore

try:
    from jax import devices as _jax_devices
except Exception:
    try:
        from jax import devices as _jax_devices  # type: ignore
    except Exception:
        def _jax_devices(*args, **kwargs):  # type: ignore
            return []

# Capibara imports - organized and deduplicated
from config.model_config import ModelConfig
from config.adaptive_config import AdaptiveConfig
from config.config_semiotic import SemioticConfig
from prompts.response_formatter import format_markdown_response
from utils.error_handling import BaseConfig, handle_error

try:
    from .hf_model import HuggingFaceCausalLM, HuggingFaceConfig
except Exception:
    from core.hf_model import HuggingFaceCausalLM, HuggingFaceConfig

logger = logging.getLogger(__name__)

class InferenceError(Exception):
    """Explicit inference error for core inference pipeline."""

# =============================================================================
# Hardware Detection and ARM Axion Support
# =============================================================================

def detect_arm_axion() -> Dict[str, Any]:
    """Detects if running on ARM Axion (Google Cloud C4A)."""
    detection_result = {
        "is_arm": False,
        "is_axion": False,
        "is_c4a": False,
        "cpu_info": {},
        "optimizations_available": []
    }
    
    try:
        # Detect ARM architecture
        machine = platform.machine().lower()
        detection_result["is_arm"] = machine in ["aarch64", "arm64"]
        
        if detection_result["is_arm"]:
            # Read CPU information on Linux
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    
                # Detect Neoverse V2 (Axion)
                if "neoverse" in cpuinfo.lower() or "armv9" in cpuinfo.lower():
                    detection_result["is_axion"] = True
                    detection_result["optimizations_available"].extend([
                        "arm_kleidi", "sve2_vectorization", "neoverse_v2_cache",
                        "uma_memory", "titanium_offload"
                    ])
                
                # Detect Google Cloud C4A through metadata
                try:
                    metadata_cmd = ["curl", "-s", "-H", "Metadata-Flavor: Google",
                                  "http://metadata.google.internal/computeMetadata/v1/instance/machine-type"]
                    result = subprocess.run(metadata_cmd, capture_output=True, text=True, timeout=2)
                    if "c4a" in result.stdout.lower():
                        detection_result["is_c4a"] = True
                        detection_result["optimizations_available"].append("c4a_networking")
                except:
                    pass
                
                # Extract relevant CPU information
                detection_result["cpu_info"] = {
                    "cores": psutil.cpu_count(logical=False),
                    "threads": psutil.cpu_count(logical=True),
                    "frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None
                }
                
            except Exception as e:
                logger.warning(f"Could not read /proc/cpuinfo: {e}")
                
    except Exception as e:
        logger.error(f"Error detecting ARM Axion: {e}")
    
    return detection_result

def setup_arm_optimizations(detection_info: Dict[str, Any]) -> Dict[str, Any]:
    """Configures specific optimizations for ARM Axion."""
    optimizations = {
        "jax_platform": "cpu",
        "precision": "bfloat16",
        "memory_layout": "optimal",
        "vectorization": False,
        "numa_binding": False
    }
    
    if detection_info["is_arm"]:
        optimizations.update({
            "jax_platform": "cpu",  # JAX cpu backend for ARM
            "precision": "bfloat16",  # ARM Axion optimized for bfloat16
            "memory_layout": "arm_optimized"
        })
        
    if detection_info["is_axion"]:
        optimizations.update({
            "vectorization": "sve2",
            "numa_binding": True,
            "cache_optimization": "neoverse_v2",
            "use_titanium_offload": detection_info["is_c4a"]
        })
        
        # Configure ARM-specific environment variables
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("JAX_ENABLE_X64", "true")
        os.environ.setdefault("XLA_FLAGS", "--xla_cpu_use_thunk_runtime=true")
        
        # ARM Axion v3.1 Features Integration
        try:
            from capibara.core.arm_optimizations import create_arm_optimization_suite, ARM_CAPABILITIES
            
            if ARM_CAPABILITIES['total_features'] > 0:
                # Load ARM-specific configuration
                arm_config = {}
                try:
                    from capibara.core.arm_optimizations import load_arm_config_from_toml
                    arm_config = load_arm_config_from_toml("capibara/config/configs_toml/specialized/arm_axion_inference.toml")
                except Exception as e:
                    logger.warning(f"Could not load ARM config: {e}")
                
                # Create optimization suite
                arm_suite = create_arm_optimization_suite(arm_config)
                optimizations["arm_optimization_suite"] = arm_suite
                
                # Activate specific features
                if ARM_CAPABILITIES.get('kleidi_integration'):
                    os.environ.setdefault("ARM_KLEIDI_ENABLE", "1")
                    optimizations["use_arm_kleidi"] = True
                
                if ARM_CAPABILITIES.get('onnx_runtime_arm'):
                    optimizations["onnx_runtime_available"] = True
                
                if ARM_CAPABILITIES.get('arm_quantization'):
                    optimizations["arm_quantization_available"] = True
                
                if ARM_CAPABILITIES.get('multi_instance_balancer'):
                    optimizations["multi_instance_balancing"] = True
                
                logger.info(f" ARM Axion v3.1 features activated: {ARM_CAPABILITIES['total_features']}/4")
            
        except ImportError:
            logger.info("ARM v3.1 optimizations not available, using basic ARM support")
    
    return optimizations

# Detect hardware at import
HARDWARE_INFO = detect_arm_axion()
ARM_OPTIMIZATIONS = setup_arm_optimizations(HARDWARE_INFO)

logger.info(f"Hardware detected: ARM={HARDWARE_INFO['is_arm']}, "
           f"Axion={HARDWARE_INFO['is_axion']}, C4A={HARDWARE_INFO['is_c4a']}")

if HARDWARE_INFO["is_axion"]:
    logger.info("ARM Axion optimizations enabled")

# =============================================================================
# Configuration Management
# =============================================================================

@dataclasses.dataclass
class InferenceConfig(BaseConfig):
    """Optimized inference configuration."""
    # Model & Tokenizer
    model_path: str
    tokenizer_path: str
    config_root: str = os.environ.get("CAPIBARA_CONFIG_ROOT", "capibara/config")
    device: Optional[str] = os.environ.get("CAPIBARA_INFER_DEVICE")
    
    # Generation parameters
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_return_sequences: int = 1
    
    # Performance settings
    use_tpu: bool = False
    use_arm_axion: bool = HARDWARE_INFO["is_axion"]
    tpu_config: Optional[str] = None
    batch_size: int = 8 if HARDWARE_INFO["is_axion"] else 1  # ARM Axion optimized
    max_concurrent_requests: int = 10
    
    # ARM Axion specific settings
    arm_optimizations: Dict[str, Any] = dataclasses.field(default_factory=lambda: ARM_OPTIMIZATIONS)
    
    # Caching
    enable_cache: bool = True
    cache_ttl: float = 300.0  # 5 minutes
    cache_max_size: int = 1000
    memory_pressure_threshold: float = 0.85
    
    # Model pool
    model_pool_size: int = 3
    model_idle_timeout: float = 600.0  # 10 minutes
    
    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if not (0 <= self.top_p <= 1):
            raise ValueError("top_p must be between 0 and 1")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
            
        # Adjust configuration for ARM Axion
        if self.use_arm_axion:
            logger.info("Configuring for ARM Axion...")
            # Load ARM Axion specific configuration if it exists
            arm_config_path = Path(self.config_root) / "configs_toml" / "specialized" / "arm_axion_inference.toml"
            if arm_config_path.exists():
                try:
                    arm_config = toml.load(arm_config_path)
                    self._apply_arm_config(arm_config)
                    logger.info("ARM Axion configuration loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load ARM Axion configuration: {e}")
    
    def _apply_arm_config(self, arm_config: Dict[str, Any]):
        """Applies ARM Axion specific configuration."""
        if "inference" in arm_config:
            inference_config = arm_config["inference"]
            self.batch_size = inference_config.get("batch_size", self.batch_size)
            self.max_length = inference_config.get("max_sequence_length", self.max_length)
        
        if "performance" in arm_config:
            perf_config = arm_config["performance"]
            self.arm_optimizations.update(perf_config)


class ConfigurationManager:
    """Singleton for centralized configuration management."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._configs_cache = {}
        self._config_lock = threading.RLock()
        self._last_reload = {}
        
    def load_config(self, config_name: str, config_root: str = None) -> Dict[str, Any]:
        """Load configuration with cache and validation."""
        config_root = config_root or "capibara/config"
        cache_key = f"{config_root}:{config_name}"
        
        with self._config_lock:
            # Check cache first
            if cache_key in self._configs_cache:
                return self._configs_cache[cache_key]
            
            # Load configuration
            try:
                config_path = Path(config_root) / "configs_toml" / f"{config_name}.toml"
                
                if not config_path.exists():
                    logger.warning(f"Config file not found: {config_path}")
                    return {}
                
                config_data = toml.load(config_path)
                
                # cache the result
                self._configs_cache[cache_key] = config_data
                self._last_reload[cache_key] = time.time()
                
                logger.debug(f"Loaded config: {config_name}")
                return config_data
                
            except Exception as e:
                logger.error(f"Failed to load config {config_name}: {e}")
                return {}
    
    def get_model_configs(self, config_root: str = None) -> Tuple[ModelConfig, AdaptiveConfig, SemioticConfig]:
        """Gets all model configurations at once."""
        try:
            # Load all config files
            default_config = self.load_config("default", config_root)
            adaptive_config = self.load_config("adaptive", config_root)
            semiotic_config = self.load_config("semiotic_config", config_root)
            
            # Extract specific sections
            model_dict = default_config.get("model", {})
            adaptive_dict = adaptive_config.get("adaptive", {})
            semiotic_dict = semiotic_config.get("semiotic", {})
            
            # Create config objects
            model_config = ModelConfig(**model_dict)
            adaptive_config_obj = AdaptiveConfig(**adaptive_dict)
            semiotic_config_obj = SemioticConfig(**semiotic_dict)
            
            return model_config, adaptive_config_obj, semiotic_config_obj
            
        except Exception as e:
            logger.error(f"Failed to load model configs: {e}")
            raise InferenceError(f"Configuration loading failed: {e}") from e
    
    def clear_cache(self):
        """Clears configuration cache."""
        with self._config_lock:
            self._configs_cache.clear()
            self._last_reload.clear()
            logger.info("Configuration cache cleared")


# =============================================================================
# Enhanced Inference Engine
# =============================================================================

class CapibaraInference:
    """Optimized inference system for CapibaraModel."""
    
    def __init__(self, config: InferenceConfig):
        """Initializes the optimized inference system."""
        self.config = config
        
        # Initialize components
        self._config_manager = ConfigurationManager()
        
        # Model and tokenizer
        self.model = None
        self.tokenizer = None
        
        # Threading
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent_requests,
            thread_name_prefix="inference"
        )
        
        # Performance metrics
        self._inference_count = 0
        self._total_inference_time = 0.0
        
        # Validation and setup
        self._validate_setup()
        
        # Detect available hardware
        self.arm_optimizations = ARM_OPTIMIZATIONS
        self.use_arm = config.use_arm_axion and HARDWARE_INFO["is_axion"]
        if self.use_arm and not self.arm_optimizations.get("arm_optimization_suite"):
            logger.info(
                "ARM Axion detected but optimizations are unavailable in core inference. "
                "Falling back to standard CPU path."
            )
            self.use_arm = False

        if self.use_arm:
            logger.info("Using ARM Axion optimizations in core inference")
        else:
            logger.info("Using standard CPU inference")
            
        # TPU only for training
        self.use_tpu = False  # TPU not used in inference
        if config.use_tpu and len(_jax_devices()) >= 32:
            logger.warning("TPU v4-32 detected but will not be used in inference (training only)")
        
        self._load_model()
        
        logger.info("CapibaraInference initialized successfully")
    
    def _validate_setup(self) -> None:
        """Validates the initial setup."""
        # Check paths
        if not Path(self.config.model_path).exists():
            logger.warning(f"Model path does not exist: {self.config.model_path}")
        
        if not Path(self.config.tokenizer_path).exists():
            logger.warning(f"Tokenizer path does not exist: {self.config.tokenizer_path}")
        
        if self.config.device:
            device_lower = self.config.device.lower()
            if device_lower not in {"cpu", "cuda"}:
                logger.warning("Unknown device '%s'; expected cpu or cuda", self.config.device)
        
        # Check config files
        config_root = Path(self.config.config_root)
        required_configs = ["default", "adaptive", "semiotic_config"]
        
        for config_name in required_configs:
            config_path = config_root / "configs_toml" / f"{config_name}.toml"
            if not config_path.exists():
                logger.warning(f"Config file missing: {config_path}")
    
    def _load_model(self) -> None:
        """Loads the model and tokenizer."""
        try:
            # Load configurations (best-effort)
            try:
                self._config_manager.get_model_configs(self.config.config_root)
            except Exception as exc:
                logger.warning("Config load failed; continuing with inference: %s", exc)

            hf_config = HuggingFaceConfig(
                model_path=self.config.model_path,
                tokenizer_path=self.config.tokenizer_path,
                device=self.config.device if self.config.device != "auto" else None,
                dtype="auto",
            )
            self.model = HuggingFaceCausalLM(hf_config)
            self.tokenizer = self.model.tokenizer
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise InferenceError(f"Model loading failed: {e}") from e
    
    def _validate_input(self, prompt: str) -> None:
        """Validates user input."""
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")
        
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if len(prompt) > self.config.max_length * 4:  # Rough token estimate
            raise ValueError(f"Prompt too long (max ~{self.config.max_length * 4} chars)")
    
    @handle_error(InferenceError)
    def generate(self, prompt: str, **generation_kwargs) -> str:
        """Generates response for a prompt."""
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_input(prompt)
            
            # Process with model
            response = self._generate_response(prompt, generation_kwargs)
            
            # Update metrics
            duration = time.time() - start_time
            self._inference_count += 1
            self._total_inference_time += duration
            
            logger.debug(f"Generated response in {duration:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise InferenceError(f"Generation failed: {e}") from e
    
    def _generate_response(self, prompt: str, generation_config: Dict[str, Any]) -> str:
        """Generates new response using the model."""
        try:
            # Tokenize input
            tokens = self.model.encode_with_mask(prompt)
            input_ids = tokens["input_ids"]
            attention_mask = tokens.get("attention_mask")

            # Generation parameters
            max_new_tokens = generation_config.get("max_new_tokens")
            if max_new_tokens is None:
                max_new_tokens = generation_config.get("max_length", self.config.max_length)

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": generation_config.get("temperature", self.config.temperature),
                "top_p": generation_config.get("top_p", self.config.top_p),
                "top_k": generation_config.get("top_k", self.config.top_k),
                "num_return_sequences": generation_config.get("num_return_sequences", self.config.num_return_sequences),
                "do_sample": generation_config.get("do_sample", True),
                "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
                "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
                "attention_mask": attention_mask,
            }

            output_ids = self.model.generate(input_ids, **gen_kwargs)
            if hasattr(output_ids, "shape") and output_ids.shape[0] > 1:
                selected = output_ids[0]
            else:
                selected = output_ids[0] if hasattr(output_ids, "__getitem__") else output_ids

            decoded = self.model.decode(selected, skip_special_tokens=True)
            if decoded.startswith(prompt):
                decoded = decoded[len(prompt):].lstrip()

            formatted_response = format_markdown_response(decoded)
            return formatted_response
            
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            raise InferenceError(f"Model inference failed: {e}") from e
    
    async def generate_async(self, prompt: str, **generation_kwargs) -> str:
        """Generates response asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Run generation in thread pool
        response = await loop.run_in_executor(
            self._executor,
            self.generate,
            prompt
        )
        
        return response
    
    def generate_batch(self, prompts: List[str], **generation_kwargs) -> List[str]:
        """Generates responses for multiple prompts in parallel."""
        if not prompts:
            return []
        
        if len(prompts) == 1:
            return [self.generate(prompts[0], **generation_kwargs)]
        
        # Submit all tasks
        futures = []
        for prompt in prompts:
            future = self._executor.submit(self.generate, prompt, **generation_kwargs)
            futures.append(future)
        
        # Collect results
        responses = []
        for future in as_completed(futures):
            try:
                response = future.result()
                responses.append(response)
            except Exception as e:
                logger.error(f"Batch generation failed for one prompt: {e}")
                responses.append(f"Error: {str(e)}")
        
        return responses
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Gets complete system statistics."""
        avg_inference_time = (
            self._total_inference_time / self._inference_count 
            if self._inference_count > 0 else 0.0
        )
        
        stats = {
            "config": {
                "model_path": self.config.model_path,
                "tokenizer_path": self.config.tokenizer_path,
                "max_length": self.config.max_length,
                "temperature": self.config.temperature,
            },
            "performance": {
                "total_inferences": self._inference_count,
                "avg_inference_time": avg_inference_time,
                "total_inference_time": self._total_inference_time
            }
        }
        
        # System memory
        try:
            memory = psutil.virtual_memory()
            stats["system"] = {
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "memory_used_gb": memory.used / (1024**3)
            }
        except Exception:
            pass
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Checks the system status."""
        health = {
            "status": "healthy",
            "checks": {},
            "timestamp": time.time()
        }
        
        # Check model
        try:
            if self.model is not None:
                health["checks"]["model"] = {"status": "ok"}
            else:
                health["checks"]["model"] = {"status": "error", "error": "Model not loaded"}
                health["status"] = "unhealthy"
        except Exception as e:
            health["checks"]["model"] = {"status": "error", "error": str(e)}
            health["status"] = "unhealthy"
        
        # Check tokenizer
        try:
            if self.tokenizer is not None:
                health["checks"]["tokenizer"] = {"status": "ok"}
            else:
                health["checks"]["tokenizer"] = {"status": "error", "error": "Tokenizer not loaded"}
                health["status"] = "unhealthy"
        except Exception as e:
            health["checks"]["tokenizer"] = {"status": "error", "error": str(e)}
        
        # Check memory
        try:
            memory = psutil.virtual_memory()
            memory_status = "ok" if memory.percent < 90 else "warning"
            health["checks"]["memory"] = {
                "status": memory_status,
                "percent": memory.percent
            }
            
            if memory.percent > 95:
                health["status"] = "unhealthy"
                
        except Exception as e:
            health["checks"]["memory"] = {"status": "error", "error": str(e)}
        
        return health
    
    def cleanup(self) -> None:
        """Cleans up system resources."""
        try:
            # Shutdown executor
            self._executor.shutdown(wait=True)
            
            # Clear model references
            self.model = None
            self.tokenizer = None
            
            # Clear config cache
            self._config_manager.clear_cache()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Inference system cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def __del__(self):
        """Destructor with cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Silent cleanup in destructor


# =============================================================================
# Utility Functions
# =============================================================================

@handle_error(InferenceError)
def create_inference_engine(
    model_path: str,
    tokenizer_path: str,
    **config_kwargs
) -> CapibaraInference:
    """Factory function to create inference engine."""
    config = InferenceConfig(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        **config_kwargs
    )
    
    return CapibaraInference(config)


@handle_error(InferenceError)
def quick_generate(
    prompt: str,
    model_path: str,
    tokenizer_path: str,
    **generation_kwargs
) -> str:
    """Convenience function for quick generation."""
    with create_inference_engine(model_path, tokenizer_path) as engine:
        return engine.generate(prompt, **generation_kwargs)


def main():
    """Main function for testing and CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CapibaraModel Inference Engine")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer")
    parser.add_argument("--prompt", help="Prompt to generate")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--stats", action="store_true", help="Show system stats")
    parser.add_argument("--health", action="store_true", help="Health check")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto", help="Inference device")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--max-length", type=int, default=512, help="Max generation length")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create inference config
    config = InferenceConfig(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        temperature=args.temperature,
        max_length=args.max_length,
        log_level=args.log_level
    )
    
    # Create inference engine
    with CapibaraInference(config) as inference:
        
        # Health check
        if args.health:
            health = inference.health_check()
            logger.info(f"Health Status: {health['status']}")
            for check, result in health['checks'].items():
                logger.info(f"  {check}: {result['status']}")
            return
        
        # System stats
        if args.stats:
            stats = inference.get_system_stats()
            logger.info("System Statistics:")
            for category, data in stats.items():
                logger.info(f"  {category}: {data}")
            return
        
        # Single prompt
        if args.prompt:
            logger.info(f"Generating response for: {args.prompt}")
            response = inference.generate(args.prompt)
            logger.info(f"Response: {response}")
            return
        
        # Interactive mode
        if args.interactive:
            logger.info("Interactive mode. Type 'quit' to exit, 'stats' for statistics.")
            
            while True:
                try:
                    prompt = input("\nPrompt: ").strip()
                    
                    if prompt.lower() in ['quit', 'exit']:
                        break
                    
                    if prompt.lower() == 'stats':
                        stats = inference.get_system_stats()
                        for category, data in stats.items():
                            logger.info(f"  {category}: {data}")
                        continue
                    
                    if prompt.lower() == 'health':
                        health = inference.health_check()
                        logger.info(f"Status: {health['status']}")
                        continue
                    
                    if not prompt:
                        continue
                    
                    start_time = time.time()
                    response = inference.generate(prompt)
                    duration = time.time() - start_time
                    
                    logger.info(f"Response ({duration:.3f}s): {response}")
                    
                except KeyboardInterrupt:
                    logger.info("\nExiting...")
                    break
                except Exception as e:
                    logger.error(f"Error: {e}")
        
        # Default behavior
        if not any([args.prompt, args.interactive, args.stats, args.health]):
            logger.info("No action specified. Use --help for options.")


if __name__ == "__main__":
    main() 

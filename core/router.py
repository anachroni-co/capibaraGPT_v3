"""
Router for dynamic module selection with TPU v4-32 optimizations.
Enhanced with proper encoding handling, training integration, and verification.
"""

import logging
import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# JAX imports with proper error handling
try:
    from capibara.jax import nn
    from capibara.jax import numpy as jnp
    import jax  # noqa: F401
    JAX_AVAILABLE = True
except ImportError as e:
    logging.warning(f"JAX not available: {e}")
    JAX_AVAILABLE = False

# Verification imports
try:
    from capibara.core.verification import (
        ComprehensiveVerificationSystem,
        AlignmentConfig,
    )
except Exception:
    ComprehensiveVerificationSystem = None  # type: ignore
    AlignmentConfig = None  # type: ignore

# Configuration imports
try:
    import toml  # type: ignore
    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False
    logging.warning("TOML not available, using JSON fallback")

# Training integration imports
try:
    from capibara.training.expanded_expert_cores_strategy import (
        ExpandedExpertCoresStrategy, 
        ExpertCoreType,  # noqa: F401
    )
    from capibara.training.huggingface_consensus_strategy import (
        HuggingFaceConsensusStrategy,
    )
    TRAINING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Training modules not available: {e}")
    TRAINING_AVAILABLE = False

# CSA Expert imports
try:
    from capibara.sub_models.csa_expert import CSAExpert, CSAExpertConfig
    from capibara.interfaces.isub_models import ExpertContext
    CSA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"CSA Expert not available: {e}")
    CSA_AVAILABLE = False

# Import decorators for optimization
try:
    from capibara.interfaces.decorators import for_router_functions, smart_decorator
except ImportError:
    # Fallback if decorators not available
    def for_router_functions(func):
        return func
    def smart_decorator(func):
        return func

logger = logging.getLogger(__name__)

class RouterType(Enum):
    STANDARD = "standard"
    EXPERT_CORE = "expert_core"
    CONSENSUS = "consensus"
    HYBRID = "hybrid"
    TRAINING_OPTIMIZED = "training_optimized"

@dataclass
class RouterConfig:
    router_type: RouterType = RouterType.STANDARD
    hidden_size: int = 768
    num_heads: int = 8
    dropout_rate: float = 0.1
    dtype: str = "bfloat16"
    cache_size: int = 1000
    use_training_integration: bool = True
    expert_cores_enabled: bool = True
    consensus_enabled: bool = True
    csa_expert_enabled: bool = True
    encoding: str = "utf-8"
    config_path: Optional[str] = None

@dataclass
class RoutingResult:
    success: bool
    selected_module: str
    confidence: float
    processing_time: float
    cache_hit: bool = False
    training_integration_used: bool = False
    expert_cores_used: List[str] = None
    csa_expert_used: bool = False
    csa_results: Optional[List[Any]] = None
    error_message: Optional[str] = None
    # Verification fields
    verification: Optional[Dict[str, Any]] = None
    safety_level: Optional[str] = None
    safety_info: Optional[Dict[str, Any]] = None

class EnhancedRouter:
    """Enhanced router with proper encoding handling and training integration."""
    
    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()
        self._setup_logging()
        self._load_configuration()
        self._initialize_components()

        # Initialize verification system
        self.verification_system = None
        if AlignmentConfig is not None and ComprehensiveVerificationSystem is not None:
            try:
                self.verification_system = ComprehensiveVerificationSystem(AlignmentConfig())
            except Exception as e:
                logger.warning(f"No se pudo inicializar verificación en Router: {e}")
                self.verification_system = None

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            encoding=self.config.encoding,
        )

    def _load_configuration(self):
        config_path = self.config.config_path
        if not config_path:
            config_root = Path(os.environ.get("CAPIBARA_CONFIG_ROOT", "capibara/config"))
            config_path = config_root / "configs_toml/production/unified_router.toml"
        if isinstance(config_path, str):
            config_path = Path(config_path)

        try:
            if TOML_AVAILABLE and config_path.suffix == ".toml":
                with open(config_path, "r", encoding=self.config.encoding) as f:
                    config_data = toml.load(f)
            else:
                json_path = config_path.with_suffix('.json')
                if json_path.exists():
                    with open(json_path, "r", encoding=self.config.encoding) as f:
                        config_data = json.load(f)
                else:
                    config_data = self._get_default_config()
            self._apply_config(config_data)
        except Exception as e:
            logger.warning(f"Error loading configuration: {e}")
            self._apply_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "core": {
                "hidden_size": self.config.hidden_size,
                "num_experts": self.config.num_heads,
                "dropout_rate": self.config.dropout_rate,
            },
            "performance": {
                "dtype": self.config.dtype,
            },
            "training": {
                "integration_enabled": self.config.use_training_integration,
                "expert_cores_enabled": self.config.expert_cores_enabled,
                "consensus_enabled": self.config.consensus_enabled,
            },
        }

    def _apply_config(self, config_data: Dict[str, Any]):
        core_config = config_data.get("core", {})
        perf_config = config_data.get("performance", {})
        training_config = config_data.get("training", {})
        self.hidden_size = core_config.get("hidden_size", self.config.hidden_size)
        self.num_heads = core_config.get("num_experts", self.config.num_heads)
        self.dropout_rate = core_config.get("dropout_rate", self.config.dropout_rate)
        self.dtype = self._get_dtype(perf_config.get("dtype", self.config.dtype))
        self.use_training_integration = training_config.get("integration_enabled", self.config.use_training_integration)
        self.expert_cores_enabled = training_config.get("expert_cores_enabled", self.config.expert_cores_enabled)
        self.consensus_enabled = training_config.get("consensus_enabled", self.config.consensus_enabled)

    def _apply_default_config(self):
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_heads
        self.dropout_rate = self.config.dropout_rate
        self.dtype = self._get_dtype(self.config.dtype)
        self.use_training_integration = self.config.use_training_integration
        self.expert_cores_enabled = self.config.expert_cores_enabled
        self.consensus_enabled = self.config.consensus_enabled

    def _get_dtype(self, dtype_str: str):
        if not JAX_AVAILABLE:
            return None
        dtype_map = {
            "bfloat16": jnp.bfloat16,
            "float32": jnp.float32,
            "float16": jnp.float16,
        }
        return dtype_map.get(dtype_str, jnp.float32)

    def _initialize_components(self):
        self.routing_cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.expert_strategy = None
        self.consensus_strategy = None
        if JAX_AVAILABLE:
            self._initialize_jax_components()
        if TRAINING_AVAILABLE and self.use_training_integration:
            self._initialize_training_integration()

    def _initialize_jax_components(self):
        try:
            # FIXED: Use proper Flax attention module
            from flax import linen as fnn
            self.attention = fnn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.hidden_size,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
            )
        except Exception as e:
            logger.error(f"Error initializing JAX components: {e}")
            self.attention = None

    def _initialize_training_integration(self):
        try:
            if self.expert_cores_enabled and TRAINING_AVAILABLE:
                self.expert_strategy = ExpandedExpertCoresStrategy({})
                logger.info("Expert cores strategy initialized")
            else:
                self.expert_strategy = None
            if self.consensus_enabled and TRAINING_AVAILABLE:
                self.consensus_strategy = HuggingFaceConsensusStrategy({})
                logger.info("Consensus strategy initialized")
            else:
                self.consensus_strategy = None
        except Exception as e:
            logger.error(f"Error initializing training integration: {e}")
            self.expert_strategy = None
            self.consensus_strategy = None

    @smart_decorator
    async def route_request(
        self,
        input_data: Union[str, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        training_mode: bool = False,
    ) -> RoutingResult:
        start_time = time.time()
        try:
            # Pre-verificación
            if isinstance(input_data, str) and self.verification_system is not None:
                input_embedding = jnp.ones((768,)) if JAX_AVAILABLE else None
                input_verification = self.verification_system.verify_output(input_embedding, input_data)
                if input_verification.get("safety_level") == "blocked":
                    return RoutingResult(
                        success=False,
                        selected_module="blocked",
                        confidence=0.0,
                        processing_time=time.time() - start_time,
                        error_message="Input blocked due to safety concerns",
                        safety_info=input_verification,
                        safety_level="blocked",
                    )

            processed_input = self._process_text_input(input_data) if isinstance(input_data, str) else input_data

            cache_key = self._compute_cache_key(processed_input)
            if use_cache and not training_mode and cache_key in self.routing_cache:
                self.cache_hits += 1
                result = RoutingResult(
                    success=True,
                    selected_module="cached",
                    confidence=1.0,
                    processing_time=time.time() - start_time,
                    cache_hit=True,
                )
                if self.verification_system is not None:
                    output_embedding = jnp.ones((768,)) if JAX_AVAILABLE else None
                    output_verification = self.verification_system.verify_output(output_embedding)
                    result.verification = output_verification
                    result.safety_level = output_verification.get("safety_level")
                return result

            self.cache_misses += 1

            if self.config.router_type == RouterType.EXPERT_CORE and self.expert_strategy:
                result = await self._route_with_expert_cores(processed_input, context)
            elif self.config.router_type == RouterType.CONSENSUS and self.consensus_strategy:
                result = await self._route_with_consensus(processed_input, context)
            elif self.config.router_type == RouterType.HYBRID:
                result = await self._route_hybrid(processed_input, context)
            else:
                result = await self._route_standard(processed_input, context)

            if use_cache and not training_mode:
                self._update_cache(cache_key, result)

            if result.success and self.verification_system is not None:
                output_embedding = jnp.ones((768,)) if JAX_AVAILABLE else None
                output_verification = self.verification_system.verify_output(output_embedding)
                result.verification = output_verification
                result.safety_level = output_verification.get("safety_level")

            return result
        except Exception as e:
            logger.error(f"Error in routing: {e}")
            return RoutingResult(
                success=False,
                selected_module="error",
                confidence=0.0,
                processing_time=time.time() - start_time,
                error_message=str(e),
            )

    def _process_text_input(self, text: str) -> Dict[str, Any]:
        try:
            if not isinstance(text, str):
                text = str(text)
            return {"text": text, "length": len(text), "encoding": self.config.encoding, "processed": True}
        except Exception as e:
            logger.error(f"Error processing text input: {e}")
            return {"text": "", "error": str(e)}

    def _compute_cache_key(self, data: Dict[str, Any]) -> str:
        try:
            if data is None:
                return "none_data"
            key_data = {"text": data.get("text", ""), "length": data.get("length", 0), "type": data.get("type", "unknown")}
            return json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error computing cache key: {e}")
            return str(hash(str(data)))

    def _update_cache(self, key: str, result: RoutingResult):
        if len(self.routing_cache) >= self.config.cache_size:
            self._cleanup_cache()
        self.routing_cache[key] = result

    def _cleanup_cache(self):
        num_to_remove = len(self.routing_cache) - self.config.cache_size + 100
        for _ in range(num_to_remove):
            if self.routing_cache:
                self.routing_cache.pop(next(iter(self.routing_cache)))

    async def _route_standard(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> RoutingResult:
        if not JAX_AVAILABLE or not getattr(self, 'attention', None):
            try:
                if input_data is None:
                    return RoutingResult(success=True, selected_module="default_module", confidence=0.5, processing_time=0.0)
                text = input_data.get("text", "")
                if text:
                    if "spanish" in text.lower() or "español" in text.lower():
                        selected_module = "spanish_module"
                    elif "training" in text.lower():
                        selected_module = "training_module"
                    elif "test" in text.lower():
                        selected_module = "test_module"
                    else:
                        selected_module = "default_module"
                    return RoutingResult(success=True, selected_module=selected_module, confidence=0.8, processing_time=0.0)
                else:
                    return RoutingResult(success=True, selected_module="default_module", confidence=0.5, processing_time=0.0)
            except Exception as e:
                logger.error(f"Error in fallback routing: {e}")
                return RoutingResult(success=False, selected_module="standard", confidence=0.0, processing_time=0.0, error_message=str(e))

        try:
            text = input_data.get("text", "")
            length = len(text)
            input_array = jnp.asarray([[float(length)]]) if JAX_AVAILABLE else None
            scores = self.attention(input_array, input_array, input_array)
            module_scores = jnp.mean(scores, axis=-1)
            selected_module_idx = jnp.argmax(module_scores)
            modules = ["module_1", "module_2", "module_3", "module_4"]
            selected_module = modules[int(selected_module_idx)] if int(selected_module_idx) < len(modules) else "default"
            return RoutingResult(success=True, selected_module=selected_module, confidence=float(jnp.max(module_scores)), processing_time=0.0)
        except Exception as e:
            logger.error(f"Error in standard routing: {e}")
            return RoutingResult(success=False, selected_module="standard", confidence=0.0, processing_time=0.0, error_message=str(e))

# Legacy compatibility class
class CoreIntegratedTokenRouter(EnhancedRouter):
    def __init__(self, hidden_size: int = 768, num_heads: int = 8, dropout_rate: float = 0.1, 
                 cache_size: int = 1000, dtype=None, use_core_primitives: bool = True):
        config = RouterConfig(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            cache_size=cache_size,
            dtype=str(dtype) if dtype else "float32",
            use_training_integration=use_core_primitives,
        )

        super().__init__(config)

# -----------------------------------------------------------------------------
# Backwards-compatibility aliases and factory helpers expected by core/__init__
# -----------------------------------------------------------------------------

class Router(EnhancedRouter):
    """Back-compat alias: some modules import `Router` from core.router."""
    pass


def create_enhanced_router(config: Optional[RouterConfig] = None) -> EnhancedRouter:
    return EnhancedRouter(config)


def create_core_integrated_router(config: Optional[RouterConfig] = None) -> CoreIntegratedTokenRouter:
    if config is None:
        return CoreIntegratedTokenRouter()
    return CoreIntegratedTokenRouter(
        hidden_size=getattr(config, "hidden_size", 768),
        num_heads=getattr(config, "num_heads", 8),
        dropout_rate=getattr(config, "dropout_rate", 0.1),
        cache_size=getattr(config, "cache_size", 1000),
        dtype=getattr(config, "dtype", "float32"),
        use_core_primitives=getattr(config, "use_training_integration", True),
    )
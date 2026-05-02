"""
Modular Model - Core Implementation
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import os

try:
    import toml
    TOML_AVAILABLE = True
except ImportError:
    toml = None
    TOML_AVAILABLE = False

try:
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp
from capibara.core.memory_monitors import CoreIntegratedMemoryMonitor
from capibara.core.metrics import MetricsCollector
from capibara.core.module_registry import ModuleRegistry
from capibara.core.router import CoreIntegratedTokenRouter as Router
from capibara.interfaces.imodules import IModule

logger = logging.getLogger(__name__)


class ModularCapibaraError(Exception):
    """Base exception for modular model errors."""
    pass


class ModuleLoadingError(ModularCapibaraError):
    """Error when loading a module."""
    pass


class ModularConfig:
    """Configuration for ModularCapibaraModel."""

    def __init__(self):
        self._load_config()

    def _load_config(self):
        """Load configuration from TOML or use defaults."""
        config_root = Path(os.environ.get("CAPIBARA_CONFIG_ROOT", "capibara/config"))
        
        modular_config = {}
        dtypes_config = {"dtypes": {"model_dtype": "float32"}}
        
        if TOML_AVAILABLE:
            try:
                modular_config = toml.load(config_root / "configs_toml/production/modular_model.toml")
                dtypes_config = toml.load(config_root / "configs_toml/production/dtypes.toml")
            except Exception as e:
                logger.warning(f"Failed loading TOML config: {e}")

        # Model configuration
        model = modular_config.get("model", {})
        self.hidden_size: int = model.get("hidden_size", 768)
        self.vocab_size: int = model.get("vocab_size", 50257)
        self.max_context_length: int = model.get("max_context_length", 2048)
        self.num_router_experts: int = model.get("num_router_experts", 8)
        self.num_attention_heads: int = model.get("num_attention_heads", 12)
        
        # Data type
        dtype_str = dtypes_config.get("dtypes", {}).get("model_dtype", "float32")
        self.dtype = jnp.bfloat16 if dtype_str == "bfloat16" else jnp.float32

        # MoE configuration
        moe = modular_config.get("moe", {})
        self.enable_moe: bool = moe.get("enable_moe", False)
        self.moe_num_layers: int = moe.get("num_layers", 1)
        self.moe_num_experts: int = moe.get("num_experts", self.num_router_experts)
        self.moe_num_active_experts: int = moe.get("num_active_experts", min(4, self.moe_num_experts))
        self.moe_hidden_size: int = moe.get("hidden_size", self.hidden_size)

        # Active modules from config
        modules = modular_config.get("modules", {})
        self.active_module_names: List[str] = modules.get("active", [])


class ModularCapibaraModel:
    """Capibara model with modular capabilities."""
    
    def __init__(self, config: Optional[ModularConfig] = None):
        self.config = config or ModularConfig()
        
        # Core components
        self.registry: ModuleRegistry = ModuleRegistry()
        self.memory_monitor: CoreIntegratedMemoryMonitor = CoreIntegratedMemoryMonitor()
        self.metrics: MetricsCollector = MetricsCollector()
        
        # Router
        self.router: Router = Router(
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            dropout_rate=0.1,
            dtype=self.config.dtype,
        )

        # MoE system
        self.moe_layers: Dict[int, Any] = {}
        if self.config.enable_moe:
            self._init_moe()
        
        # Active modules
        self.active_modules: List[IModule] = []
        self._load_active_modules()

    def _init_moe(self):
        """Initialize MoE layers if available."""
        try:
            from capibara.core.moe import MoEConfig, create_moe_manager
            
            moe_config = MoEConfig(
                num_experts=self.config.moe_num_experts,
                num_active_experts=self.config.moe_num_active_experts,
                hidden_size=self.config.moe_hidden_size,
            )
            moe_manager = create_moe_manager(moe_config)
            
            for layer_id in range(self.config.moe_num_layers):
                self.moe_layers[layer_id] = moe_manager.create_layer(layer_id)
            
            logger.info(f"MoE initialized: {self.config.moe_num_layers} layers, {self.config.moe_num_experts} experts")
        except ImportError:
            logger.warning("MoE module not available")
        except Exception as e:
            logger.warning(f"MoE initialization failed: {e}")
        
    def _load_active_modules(self):
        """Load active modules from configuration."""
        # Available module classes
        available_modules = {}
        
        try:
            from capibara.sub_models.experimental.dual_process import DualProcessThinking
            available_modules["dual_process"] = DualProcessThinking
        except ImportError:
            pass
        
        try:
            from capibara.sub_models.experimental.adaptive_vq_submodel import AdaptiveVQSubModel
            available_modules["adaptive"] = AdaptiveVQSubModel
        except ImportError:
            pass
        
        try:
            from capibara.sub_models.semiotic.mnemosyne_semio_module import MnemosyneSemioModule
            available_modules["semiotic"] = MnemosyneSemioModule
        except ImportError:
            pass

        # Register and instantiate configured modules
        for name in self.config.active_module_names:
            if name in available_modules:
                try:
                    self.registry.register(name, available_modules[name])
                    module = self.registry.create_module(name, hidden_size=self.config.hidden_size)
                    self.active_modules.append(module)
                    logger.info(f"Loaded module: {name}")
                except Exception as e:
                    logger.error(f"Failed to load module {name}: {e}")
    
    def __call__(
        self, 
        inputs: jnp.ndarray, 
        context: Optional[jnp.ndarray] = None,
        training: bool = False
    ) -> Dict[str, Any]:
        """Forward pass."""
        # Validate inputs
        inputs = self._validate_inputs(inputs)
        
        # Memory check
        if self.memory_monitor.should_cleanup():
            self.memory_monitor.force_cleanup()
        
        outputs: Dict[str, Any] = {}

        # MoE processing
        if self.moe_layers:
            moe_input = inputs
            for layer_id in sorted(self.moe_layers.keys()):
                try:
                    moe_result = self.moe_layers[layer_id](moe_input, training=training)
                    moe_input = moe_result.get("output", moe_input)
                    outputs[f"moe_{layer_id}"] = moe_result
                except Exception as e:
                    logger.error(f"MoE layer {layer_id} failed: {e}")
                    break
            inputs = moe_input
        
        # Module forward passes
        for i, module in enumerate(self.active_modules):
            try:
                module_output = module(inputs, training=training)
                outputs[f"module_{i}"] = module_output
                
                if isinstance(module_output, dict) and "metrics" in module_output:
                    self.metrics.update(module_output["metrics"])
            except Exception as e:
                logger.error(f"Module {i} ({module.__class__.__name__}) failed: {e}")

        return {
            "outputs": outputs,
            "metrics": self.metrics.get_all(),
        }

    def _validate_inputs(self, inputs: Any) -> jnp.ndarray:
        """Validate and normalize inputs to proper tensor."""
        # Already a proper tensor
        if hasattr(inputs, 'shape') and len(inputs.shape) == 3:
            return inputs
        
        # Dict input - extract tensor
        if isinstance(inputs, dict):
            for key in ["hidden_states", "input_ids", "embeddings"]:
                if key in inputs and hasattr(inputs[key], 'shape'):
                    tensor = inputs[key]
                    if len(tensor.shape) == 3:
                        return tensor
                    if len(tensor.shape) == 2 and tensor.dtype in [jnp.int32, jnp.int64]:
                        # Token IDs - create dummy embeddings
                        batch_size, seq_len = tensor.shape
                        return jnp.ones((batch_size, seq_len, self.config.hidden_size), dtype=self.config.dtype)
        
        # Fallback
        logger.warning(f"Invalid input type {type(inputs)}, creating dummy tensor")
        return jnp.ones((1, 64, self.config.hidden_size), dtype=self.config.dtype)

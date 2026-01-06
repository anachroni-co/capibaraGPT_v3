"""Core Configuration Module for CapibaraGPT.

This module provides comprehensive configuration management for the CapibaraGPT system,
including model parameters, training settings, routing configuration, and distributed
training utilities.

Key Components:
    - Config: General-purpose configuration manager with file I/O
    - ModelConfig: Model-specific formeters (temperature, max_length, etc.)
    - RouterConfig: Router behavior and load balancing settings
    - ModularModelConfig: Configuration for modular model architectures
    - Distributed training decorators and utilities

Example:
    Basic configuration usage:

    >>> from capibara.core.config import Config, ModelConfig
    >>> config = Config(model_name="capibara-v2", max_length=2048)
    >>> config.load_from_file("config/production.json")
    >>> model_config = ModelConfig()
    >>> model_config.set_parameter("temperature", 0.8)

Note:
    Configuration can be loaded from JSON files or set programmatically.
    Environment variable CAPIBARA_ENV determines the runtime environment.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import os

logger = logging.getLogger(__name__)

class Config:
    """General-purpose configuration manager for CapibaraGPT.

    This class manages all system-wide configuration settings, providing methods for
    loading, saving, and validating configuration data. Supports both programmatic
    configuration and file-based configuration loading.

    Attributes:
        settings (Dict[str, Any]): Dictionary containing all configuration key-value pairs.
        config_file (Optional[str]): Path to the loaded configuration file, if any.
        environment (str): Current runtime environment (development, staging, production).
        model (Dict[str, Any]): Model-specific configuration dictionary.
        training_config (TrainingConfig): Training configuration object.
        dataset_name (str): Name of the dataset to use for training/evaluation.

    Example:
        >>> config = Config(model_name="capibara-v2", batch_size=32)
        >>> config.set("learning_rate", 0.001)
        >>> config.save_to_file("my_config.json")
        >>>
        >>> # Load existing config
        >>> config2 = Config()
        >>> config2.load_from_file("my_config.json")
    """

    def __init__(self, **kwargs):
        """Initialize configuration manager.

        Args:
            **kwargs: Arbitrary keyword arguments to initialize configuration settings.
                     Any key-value pairs provided will be stored in the settings dictionary.

        Example:
            >>> config = Config(
            ...     model_name="capibara-v2",
            ...     max_length=2048,
            ...     temperature=0.7
            ... )
        """
        self.settings = {}
        self.config_file = None
        self.environment = os.getenv("CAPIBARA_ENV", "development")
        
        # Load any provided configuration
        for key, value in kwargs.items():
            self.settings[key] = value
        
        # Set common attributes that might be expected
        self.model = self.settings.get('model', {})
        self.training_config = self._create_training_config()
        self.dataset_name = self.settings.get('dataset_name', 'openai/openai_humaneval')
        
    def _create_training_config(self):
        """Create training config object with common attributes."""
        training_data = self.settings.get('training', {})
        
        class TrainingConfig:
            def __init__(self, data):
                self.learning_rate = data.get('learning_rate', 0.001)
                self.batch_size = data.get('batch_size', 32)
                self.epochs = data.get('epochs', 10)
                self.num_epochs = data.get('num_epochs', data.get('epochs', 10))
                self.optimizer = data.get('optimizer', 'adamw')
                self.weight_decay = data.get('weight_decay', 0.01)
                self.seed = data.get('seed', 42)
                
        return TrainingConfig(training_data)
        
    def get(self, key: str, default=None):
        """Retrieve a configuration value by key.

        Args:
            key (str): The configuration key to retrieve.
            default (Any, optional): Default value to return if key is not found. Defaults to None.

        Returns:
            Any: The configuration value associated with the key, or default if not found.

        Example:
            >>> config = Config(model_name="capibara-v2")
            >>> config.get("model_name")
            'capibara-v2'
            >>> config.get("nonexistent_key", "default_value")
            'default_value'
        """
        return self.settings.get(key, default)

    def set(self, key: str, value: Any):
        """Set a configuration value.

        Updates the configuration settings with a new key-value pair. If the key
        already exists, its value is overwritten. Changes are logged for tracking.

        Args:
            key (str): The configuration key to set.
            value (Any): The value to assign to the key.

        Example:
            >>> config = Config()
            >>> config.set("learning_rate", 0.001)
            >>> config.set("batch_size", 32)
        """
        self.settings[key] = value
        logger.info(f"Config set: {key} = {value}")
    
    def load_from_file(self, config_path: str) -> bool:
        """Load configuration from a JSON file.

        Reads a JSON configuration file and updates the current settings dictionary.
        Existing settings are preserved and updated with values from the file.

        Args:
            config_path (str): Path to the JSON configuration file.

        Returns:
            bool: True if loading was successful, False otherwise.

        Raises:
            Exception: Catches all exceptions during file loading and logs errors.

        Example:
            >>> config = Config()
            >>> success = config.load_from_file("config/production.json")
            >>> if success:
            ...     print(f"Loaded config from: {config.config_file}")
        """
        try:
            with open(config_path, 'r') as f:
                self.settings.update(json.load(f))
            self.config_file = config_path
            logger.info(f"Loaded config from: {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False

    def save_to_file(self, config_path: str) -> bool:
        """Save current configuration to a JSON file.

        Writes all current settings to a JSON file with pretty printing (indent=2).
        Useful for persisting runtime configuration changes.

        Args:
            config_path (str): Path where the configuration file should be saved.

        Returns:
            bool: True if saving was successful, False otherwise.

        Raises:
            Exception: Catches all exceptions during file writing and logs errors.

        Example:
            >>> config = Config(model_name="capibara-v2")
            >>> config.set("temperature", 0.8)
            >>> config.save_to_file("config/my_config.json")
        """
        try:
            with open(config_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
            logger.info(f"Saved config to: {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False

    def get_all_settings(self) -> Dict[str, Any]:
        """Retrieve a copy of all configuration settings.

        Returns:
            Dict[str, Any]: A copy of the settings dictionary containing all configuration values.

        Note:
            Returns a copy to prevent external modification of the internal settings dictionary.

        Example:
            >>> config = Config(model_name="capibara-v2", batch_size=32)
            >>> all_settings = config.get_all_settings()
            >>> print(all_settings)
            {'model_name': 'capibara-v2', 'batch_size': 32}
        """
        return self.settings.copy()

    def validate_config(self) -> bool:
        """Validate that required configuration keys are present.

        Checks for the presence of essential configuration keys needed for model operation.
        Logs warnings for any missing required keys.

        Returns:
            bool: True if all required keys are present, False otherwise.

        Note:
            Required keys: model_name, max_length, temperature.
            This list can be extended based on system requirements.

        Example:
            >>> config = Config(model_name="capibara-v2", max_length=2048, temperature=0.7)
            >>> is_valid = config.validate_config()
            >>> print(f"Config is valid: {is_valid}")
            Config is valid: True
        """
        required_keys = ["model_name", "max_length", "temperature"]
        for key in required_keys:
            if key not in self.settings:
                logger.warning(f"Missing required config: {key}")
                return False
        return True

class ModelConfig:
    """Model-specific configuration for inference and generation parameters.

    Manages model-level parameters that control generation behavior, including
    sampling strategies, sequence lengths, and decoding methods.

    Attributes:
        model_name (str): Identifier for the model being configured.
        parameters (Dict[str, Any]): Dictionary of model parameters including:
            - max_length: Maximum sequence length for generation
            - temperature: Sampling temperature (higher = more random)
            - top_p: Nucleus sampling threshold
            - top_k: Top-k sampling threshold

    Example:
        >>> model_config = ModelConfig("capibara-v2")
        >>> model_config.set_parameter("temperature", 0.8)
        >>> model_config.set_parameter("max_length", 4096)
        >>> temp = model_config.get_parameter("temperature")
        >>> print(f"Temperature: {temp}")
        Temperature: 0.8
    """

    def __init__(self, model_name: str = "capibara-gpt"):
        """Initialize model configuration with default parameters.

        Args:
            model_name (str, optional): Name/identifier for the model.
                Defaults to "capibara-gpt".

        Note:
            Default parameters are suitable for general text generation.
            Adjust based on specific use cases (creative writing, code, etc.).
        """
        self.model_name = model_name
        self.parameters = {
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50
        }

    def get_parameter(self, key: str, default=None):
        """Retrieve a model parameter value.

        Args:
            key (str): Parameter name to retrieve.
            default (Any, optional): Default value if parameter not found. Defaults to None.

        Returns:
            Any: The parameter value, or default if not found.

        Example:
            >>> model_config = ModelConfig()
            >>> model_config.get_parameter("temperature")
            0.7
            >>> model_config.get_parameter("nonexistent", 1.0)
            1.0
        """
        return self.parameters.get(key, default)

    def set_parameter(self, key: str, value: Any):
        """Set a model parameter value.

        Args:
            key (str): Parameter name to set.
            value (Any): Value to assign to the parameter.

        Example:
            >>> model_config = ModelConfig()
            >>> model_config.set_parameter("temperature", 0.9)
            >>> model_config.set_parameter("max_length", 4096)
        """
        self.parameters[key] = value
        logger.info(f"Model parameter set: {key} = {value}")

# Global config instances
config = Config()
model_config = ModelConfig()

class RouterConfig:
    """Configuration for routing and load balancing behavior.

    Controls how requests are routed between different model variants,
    services, or computational backends. Includes timeout management,
    retry logic, and fallback strategies.

    Attributes:
        strategy (str): Routing strategy to use (e.g., "default", "round_robin", "least_loaded").
        load_balancing (bool): Whether to enable load balancing across multiple backends.
        timeout (float): Maximum time in seconds to wait for a response before timing out.
        max_retries (int): Maximum number of retry attempts for failed requests.
        fallback_enabled (bool): Whether to enable fallback to alternative backends on failure.

    Example:
        >>> router_config = RouterConfig()
        >>> router_config.strategy = "least_loaded"
        >>> router_config.timeout = 60.0
        >>> config_dict = router_config.to_dict()
    """

    def __init__(self):
        """Initialize router configuration with default values.

        Default configuration provides:
        - Standard routing strategy
        - Load balancing enabled
        - 30-second timeout
        - Up to 3 retry attempts
        - Fallback enabled for resilience
        """
        self.strategy = "default"
        self.load_balancing = True
        self.timeout = 30.0
        self.max_retries = 3
        self.fallback_enabled = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert router configuration to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the router configuration.

        Example:
            >>> router_config = RouterConfig()
            >>> config_dict = router_config.to_dict()
            >>> print(config_dict)
            {'strategy': 'default', 'load_balancing': True, ...}
        """
        return {
            "strategy": self.strategy,
            "load_balancing": self.load_balancing,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "fallback_enabled": self.fallback_enabled
        }

class ModularModelConfig:
    """Configuration manager for modular model architectures.

    Manages configurations for systems composed of multiple interchangeable modules,
    such as MoE (Mixture of Experts), multi-stage pipelines, or ensemble models.
    Provides module-level configuration with global optimization settings.

    Attributes:
        modules (List[Dict[str, Any]]): List of module configurations, each containing
            a module name and its specific configuration dictionary.
        enable_caching (bool): Whether to enable caching of module outputs for efficiency.
        parallel_execution (bool): Whether to execute independent modules in parallel.
        memory_optimization (bool): Whether to enable memory-saving optimizations.
        debug_mode (bool): Whether to enable verbose debugging and logging.

    Example:
        >>> model_config = ModularModelConfig()
        >>> model_config.add_module("moe", {"num_experts": 32, "top_k": 2})
        >>> model_config.add_module("router", {"strategy": "learned"})
        >>> model_config.parallel_execution = True
        >>> moe_config = model_config.get_module_config("moe")
    """

    def __init__(self):
        """Initialize modular model configuration with default settings.

        Default configuration optimizes for:
        - Caching enabled for faster repeated computations
        - Sequential execution (parallel disabled by default)
        - Memory optimization enabled
        - Debug mode disabled for production use
        """
        self.modules = []
        self.enable_caching = True
        self.parallel_execution = False
        self.memory_optimization = True
        self.debug_mode = False

    def add_module(self, module_name: str, config: Dict[str, Any]):
        """Add a module with its configuration to the modular model.

        Args:
            module_name (str): Unique identifier for the module.
            config (Dict[str, Any]): Module-specific configuration dictionary.

        Example:
            >>> model_config = ModularModelConfig()
            >>> model_config.add_module("attention", {
            ...     "num_heads": 12,
            ...     "hidden_size": 768,
            ...     "dropout": 0.1
            ... })
        """
        self.modules.append({"name": module_name, "config": config})

    def get_module_config(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve configuration for a specific module by name.

        Args:
            module_name (str): Name of the module to retrieve configuration for.

        Returns:
            Optional[Dict[str, Any]]: Module configuration dictionary if found, None otherwise.

        Example:
            >>> model_config = ModularModelConfig()
            >>> model_config.add_module("encoder", {"layers": 6})
            >>> encoder_config = model_config.get_module_config("encoder")
            >>> print(encoder_config)
            {'layers': 6}
        """
        for module in self.modules:
            if module["name"] == module_name:
                return module["config"]
        return None

# TPU and distributed training utilities
def distributed_jit(*args, **kwargs):
    """Decorator placeholder for distributed JIT compilation.

    This decorator is a placeholder for distributed just-in-time compilation
    functionality. In production, this would integrate with JAX's distributed
    compilation features for multi-device training.

    Args:
        *args: Positional arguments for the decorator.
        **kwargs: Keyword arguments for distributed compilation configuration.

    Returns:
        Callable: The decorated function (currently returns function unchanged).

    Note:
        This is a placeholder implementation. Full functionality requires
        JAX distributed runtime and TPU mesh configuration.

    Example:
        >>> @distributed_jit
        ... def train_step(params, batch):
        ...     return loss
    """
    def decorator(f):
        return f
    if args and callable(args[0]):
        return args[0]  # Direct decoration without parentheses
    return decorator

def model_sharded_jit(*args, **kwargs):
    """Decorator placeholder for model-sharded JIT compilation.

    Applies model parallelism sharding during JIT compilation. In production,
    this would distribute model parameters across multiple devices/TPUs.

    Args:
        *args: Positional arguments for the decorator.
        **kwargs: Keyword arguments for model sharding configuration.

    Returns:
        Callable: The decorated function (currently returns function unchanged).

    Note:
        Requires TPU mesh configuration and JAX sharding specifications.
        This is a placeholder for the full implementation.

    Example:
        >>> @model_sharded_jit
        ... def forward_pass(params, inputs):
        ...     return outputs
    """
    def decorator(f):
        return f
    if args and callable(args[0]):
        return args[0]
    return decorator

def batch_sharded_jit(*args, **kwargs):
    """Decorator placeholder for batch-sharded JIT compilation.

    Applies data parallelism by sharding batches across multiple devices.
    In production, this distributes batch processing for faster training.

    Args:
        *args: Positional arguments for the decorator.
        **kwargs: Keyword arguments for batch sharding configuration.

    Returns:
        Callable: The decorated function (currently returns function unchanged).

    Note:
        This is a placeholder. Full implementation requires JAX pmap
        or xmap for distributed batch processing.

    Example:
        >>> @batch_sharded_jit
        ... def process_batch(batch):
        ...     return processed_batch
    """
    def decorator(f):
        return f
    if args and callable(args[0]):
        return args[0]
    return decorator

def create_unified_mesh(*args, **kwargs):
    """Create a unified device mesh for distributed training.

    Creates a mesh of TPU/GPU devices for distributed model training and inference.
    This is a placeholder for JAX mesh creation functionality.

    Args:
        *args: Positional arguments for mesh configuration.
        **kwargs: Keyword arguments including:
            - devices: Number or list of devices to include in mesh
            - mesh_shape: Shape of the device mesh (e.g., (4, 2) for 4x2)
            - axis_names: Names for mesh axes (e.g., ('data', 'model'))

    Returns:
        None: Placeholder return value. Full implementation would return JAX Mesh object.

    Note:
        Requires JAX distributed runtime and proper device initialization.

    Example:
        >>> mesh = create_unified_mesh(devices=8, mesh_shape=(4, 2), axis_names=('data', 'model'))
    """
    return None

# TPU Sharding constants
BATCH_SHARDING = None  # Placeholder for batch sharding specification
MODEL_SHARDING = None  # Placeholder for model sharding specification
HYBRID_SHARDING = None  # Placeholder for hybrid (data + model) sharding
TPU_DTYPE = None  # Placeholder for TPU-optimized data type (typically bfloat16)

def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a JSON file or return current settings.

    Utility function to load configuration from a file path or retrieve
    current configuration settings if no path is provided.

    Args:
        path (Optional[str]): Path to JSON configuration file. If None, returns
            current settings from the global config instance.

    Returns:
        Dict[str, Any]: Configuration dictionary loaded from file or current settings.

    Raises:
        Exception: Silently catches file loading errors and returns empty dict.

    Example:
        >>> # Load from file
        >>> settings = load_config("config/production.json")
        >>>
        >>> # Get current settings
        >>> current = load_config()

    Note:
        This function accesses the global `config` instance. Errors during
        file loading are silently caught and return an empty dictionary.
    """
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return config.get_all_settings()

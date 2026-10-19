# Content of model_config.py
"""
CapibaraModel model configuration.

This module defines the model architecture specific configuration,
including layer parameters, attention, and other components.
"""

import os
import sys

import logging
from pathlib import Path
import toml  # type: ignore
from .config import TPUConfig
from dataclasses import dataclass
from .memory_config import MemoryOptimizationConfig
from .convexity_config import ConvexityTrainingConfig, ConvexityControllerConfig
from pydantic import BaseModel, Field, field_validator  # type: ignore


from typing import Optional, List, Dict, Any, Tuple, Callable, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ModelConfig(BaseModel):
    """Model architecture configuration."""

    # Basic configuration
    model_name: str = Field("capibara", description="Model name")
    model_type: str = Field("transformer", description="Architecture type")
    hidden_size: int = Field(768, description="Hidden layer size")
    num_layers: int = Field(12, description="Number of layers")
    num_heads: int = Field(12, description="Number of attention heads")

    # Attention configuration
    attention_dropout: float = Field(0.1, description="Attention dropout")
    attention_scale: bool = Field(True, description="Scale attention")
    use_rotary_embeddings: bool = Field(True, description="Use rotary embeddings")

    # Feed-forward configuration
    intermediate_size: int = Field(3072, description="Intermediate layer size")
    activation_function: str = Field("gelu", description="Activation function")
    dropout: float = Field(0.1, description="General dropout")

    # Embeddings configuration
    vocab_size: int = Field(50257, description="Vocabulary size")
    max_position_embeddings: int = Field(2048, description="Maximum sequence length")
    type_vocab_size: int = Field(2, description="Type vocabulary size")

    # Normalization configuration
    layer_norm_eps: float = Field(1e-12, description="Epsilon for LayerNorm")
    use_layer_norm: bool = Field(True, description="Use LayerNorm")

    # Initialization configuration
    initializer_range: float = Field(0.02, description="Initializer range")
    use_xavier_init: bool = Field(True, description="Use Xavier initialization")

    # Quantization configuration
    use_quantization: bool = Field(False, description="Use quantization")
    num_bits: int = Field(8, description="Number of bits for quantization")
    quant_min: float = Field(0.0, description="Minimum quantization value")
    quant_max: float = Field(255.0, description="Maximum quantization value")

    # Advanced configuration
    custom_layers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Custom layers"
    )

    @field_validator('hidden_size')
    def validate_hidden_size(cls, v):
        """Validates that hidden size is divisible by number of heads."""
        if v % 12 != 0:
            raise ValueError("Hidden size must be divisible by 12")
        return v

    @field_validator('num_bits')
    def validate_num_bits(cls, v):
        """Validates that the number of bits is valid."""
        if v not in [1, 2, 4, 8, 16]:
            raise ValueError("Number of bits must be 1, 2, 4, 8, or 16")
        return v

    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"

class CapibaraConfig(BaseModel):
    """
    Main configuration for CapibaraModel.
    """

    base_model_name: str
    tokenizer_name: str
    max_length: int = 512
    batch_size: int = 128  # Recommended batch size for TPUs
    learning_rate: float = 1e-3
    num_epochs: int = 5
    output_dir: str = 'gs://capibara_gpt/output'
    device: str = 'tpu'
    tpu_config: TPUConfig
    log_level: str = 'INFO'

    @field_validator('device')
    def validate_device(cls, v):
        """Validates that the device is configured as 'tpu'."""
        if v != 'tpu':
            raise ValueError("`device` must be configured as 'tpu' for exclusive TPU training.")
        return v

    @classmethod
    def from_yaml(cls, yaml_path: str, tpu_yaml_path: str) -> 'CapibaraConfig':
        """
        Loads model configuration from YAML files.

        Args:
            yaml_path: Path to main configuration file.
            tpu_yaml_path: Path to TPU configuration file.

        Returns:
            CapibaraConfig: Model configuration instance.

        Raises:
            FileNotFoundError: If any YAML file does not exist.
            yaml.YAMLError: If there is an error parsing any YAML file.
        """
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            tpu_config = TPUConfig.from_yaml(tpu_yaml_path)
            config_dict['tpu_config'] = tpu_config
            return cls(**config_dict)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {yaml_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error loading configuration from {yaml_path}: {e}")
            raise

@dataclass
class BitNetConfig:
    """Configuration for BitNet158."""
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_attention_heads: int = 32
    num_hidden_layers: int = 24
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    use_cache: bool = True
    gradient_checkpointing: bool = False
    remat_frequency: int = 2  # How many layers to apply remat

@dataclass
class NeuroAdaptiveConfig:
    """Configuration for NeuroAdaptiveStack."""
    num_layers: int = 24
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_attention_heads: int = 32
    dropout_rate: float = 0.1
    remat_frequency: int = 2
    use_gradient_checkpointing: bool = True
    layer_id: Optional[int] = None

@dataclass
class ModularModelConfig:
    """
    Merged configuration for ModularCapibaraModel v2.3.
    Reads configuration from centralized TOML file.
    """

    # Basic model configuration
    hidden_size: int = 768
    num_virtual_qubits: int = 512
    vocab_size: int = 50257
    max_context_length: int = 2048
    num_router_experts: int = 8
    router_capacity_factor: float = 1.25
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    activation_function: str = "gelu"

    # Memory and optimization configuration
    memory_config: MemoryOptimizationConfig = MemoryOptimizationConfig()

    # Training configuration with convexity control
    convexity_training_config: Optional[ConvexityTrainingConfig] = None
    enable_convexity_control: bool = False

    # TPU v4-32 configuration
    use_tpu_optimizations: bool = True
    tpu_cores: int = 32
    tpu_memory_limit_gb: float = 32.0
    tpu_cleanup_threshold: float = 0.9
    tpu_mesh_shape: Tuple[int, int] = (4, 8)
    tpu_axis_names: List[str] = None
    dtype: str = "bfloat16"
    param_dtype: str = "float32"
    enable_systolic_arrays: bool = True
    enable_xla_optimization: bool = True
    optimization_level: int = 3

    # Neuroadaptive configuration
    enable_neuroadaptive: bool = True
    neuroadaptive_rate: float = 0.01
    use_remat_neuroadaptive: bool = True
    neurogenesis_threshold: float = 0.8
    num_adaptive_layers: int = 2
    adaptation_interval: int = 100
    enable_neurogenesis: bool = True
    enable_adaptive_embedding: bool = True

    # Router configuration
    router_hidden_size: int = 768
    router_num_heads: int = 8
    router_cache_size: int = 1000
    routing_cache_size: int = 1000
    enable_adaptive_routing: bool = True
    enable_contextual_activation: bool = True
    enable_vision: bool = True
    enable_audio: bool = True
    enable_bio: bool = True

    # Agent configuration
    max_agents: int = 100
    agent_cleanup_interval: float = 300.0
    enable_hot_reload: bool = False
    agent_timeout: float = 30.0

    # Module configuration
    active_modules: List[str] = None
    enable_lazy_loading: bool = True
    lazy_loading_timeout: float = 30.0
    
    # Monitoring configuration
    enable_metrics: bool = True
    metrics_window_size: int = 1000
    log_interval: int = 100
    enable_visualization: bool = True
    visualization_cleanup_interval: float = 3600.0
    max_visualization_files: int = 50

    # Memory configuration
    memory_pressure_threshold: float = 0.85
    enable_memory_monitoring: bool = True
    cleanup_interval: float = 300.0

    # Paths configuration
    config_root: Path = None
    modules_root: Path = None
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"

    # === TTS SERVICES CONFIGURATION ===
    # General services configuration
    enable_tts: bool = True
    tts_backend: str = "capibara_tts"
    tts_fallback: str = "pyttsx3"
    websocket_enabled: bool = True
    websocket_host: str = "localhost"
    websocket_port: int = 8765
    ssl_enabled: bool = False
    max_concurrent_connections: int = 100
    audio_cache_enabled: bool = True
    audio_cache_size: int = 1000
    audio_cache_ttl: int = 3600
    
    # TTS specific configuration
    fastspeech_model_path: str = "models/fastspeech.onnx"
    hifigan_model_path: str = "models/hifigan.onnx"
    sample_rate: int = 22050
    audio_format: str = "wav"
    bit_depth: int = 16
    channels: int = 1
    voice_speed: int = 150
    voice_volume: float = 1.0
    spectrogram_clipping: List[float] = None
    enable_websocket_server: bool = True
    websocket_buffer_size: int = 8192
    websocket_timeout: float = 30.0
    fallback_engine: str = "pyttsx3"
    fallback_voice_rate: int = 150
    fallback_voice_volume: float = 1.0
    cert_file: str = ""
    key_file: str = ""
    enable_audio_preprocessing: bool = True
    enable_noise_reduction: bool = False
    enable_voice_enhancement: bool = True
    
    def __post_init__(self):
        """Post-creation initialization."""
        if self.tpu_axis_names is None:
            self.tpu_axis_names = ["data", "model"]
        
        if self.active_modules is None:
            self.active_modules = [
                "dual_process", "adaptive", "semiotic", 
                "semiotic_interaction", "ethics", "personality_manager"
            ]
        
        if self.config_root is None:
            self.config_root = Path("capibara/config")
        
        if self.modules_root is None:
            self.modules_root = self.config_root / "modules"
        
        # Initialize TTS configuration
        if self.spectrogram_clipping is None:
            self.spectrogram_clipping = [-4.0, 4.0]
        
        # Validations
        assert self.hidden_size % 64 == 0, "hidden_size must be a multiple of 64 for TPU"
        assert self.hidden_size % self.num_router_experts == 0, "hidden_size must be divisible by num_router_experts"
    
    @classmethod
    def from_toml(cls, config_path: Optional[str] = None) -> 'ModularModelConfig':
        """
        Load configuration from TOML file.

        Args:
            config_path: Path to TOML file. If None, uses the default file.

        Returns:
            ModularModelConfig: Instance with loaded configuration.
        """
        if config_path is None:
            config_path = "capibara/config/configs_toml/modular_model_fusionado.toml"
        
        try:
            with open(config_path, 'r') as f:
                config_dict = toml.load(f)
            
            # Map TOML configuration to class attributes
            kwargs = {}
            
            # Basic model configuration
            if 'model' in config_dict:
                model_config = config_dict['model']
                kwargs.update({
                    'hidden_size': model_config.get('hidden_size', 768),
                    'num_virtual_qubits': model_config.get('num_virtual_qubits', 512),
                    'vocab_size': model_config.get('vocab_size', 50257),
                    'max_context_length': model_config.get('max_context_length', 2048),
                    'num_router_experts': model_config.get('num_router_experts', 8),
                    'router_capacity_factor': model_config.get('router_capacity_factor', 1.25),
                    'num_layers': model_config.get('num_layers', 12),
                    'num_heads': model_config.get('num_heads', 12),
                    'intermediate_size': model_config.get('intermediate_size', 3072),
                    'dropout_rate': model_config.get('dropout_rate', 0.1),
                    'attention_dropout': model_config.get('attention_dropout', 0.1),
                    'layer_norm_eps': model_config.get('layer_norm_eps', 1e-12),
                    'activation_function': model_config.get('activation_function', 'gelu'),
                })
            
            # Memory and optimization configuration
            if 'memory_optimization' in config_dict:
                memory_config = config_dict['memory_optimization']
                kwargs['memory_config'] = MemoryOptimizationConfig(
                    use_gradient_checkpointing=memory_config.get('use_gradient_checkpointing', True),
                    checkpoint_policy=memory_config.get('checkpoint_policy', 'block_depth'),
                    checkpoint_blocks=memory_config.get('checkpoint_blocks', 2),
                    remat_layer_interval=memory_config.get('remat_layer_interval', 2),
                    preserve_rng_state=memory_config.get('preserve_rng_state', True),
                    memory_pressure_threshold=memory_config.get('memory_pressure_threshold', 0.85),
                    enable_memory_monitoring=memory_config.get('enable_memory_monitoring', True),
                    cleanup_interval=memory_config.get('cleanup_interval', 300.0),
                    max_memory_usage_gb=memory_config.get('max_memory_usage_gb', 32.0),
                    force_device_gc=memory_config.get('force_device_gc', True),
                    use_mixed_precision=memory_config.get('use_mixed_precision', True),
                    compute_dtype=memory_config.get('compute_dtype', 'bfloat16'),
                    param_dtype=memory_config.get('param_dtype', 'float32'),
                    output_dtype=memory_config.get('output_dtype', 'float32'),
                    use_model_parallelism=memory_config.get('use_model_parallelism', True),
                    model_parallel_submesh=tuple(memory_config.get('model_parallel_submesh', [2, 2])),
                    data_parallel_submesh=tuple(memory_config.get('data_parallel_submesh', [2, 1])),
                    shard_strategy=memory_config.get('shard_strategy', 'axis_0'),
                    dynamic_batch_size=memory_config.get('dynamic_batch_size', True),
                    min_batch_size=memory_config.get('min_batch_size', 4),
                    max_batch_size=memory_config.get('max_batch_size', 32),
                    batch_growth_factor=memory_config.get('batch_growth_factor', 1.5),
                )
            
            # TPU v4-32 configuration
            if 'tpu_v4' in config_dict:
                tpu_config = config_dict['tpu_v4']
                kwargs.update({
                    'use_tpu_optimizations': tpu_config.get('use_tpu_optimizations', True),
                    'tpu_cores': tpu_config.get('cores', 32),
                    'tpu_memory_limit_gb': tpu_config.get('memory_limit_gb', 32.0),
                    'tpu_cleanup_threshold': tpu_config.get('cleanup_threshold', 0.9),
                    'tpu_mesh_shape': tuple(tpu_config.get('mesh_shape', [4, 8])),
                    'tpu_axis_names': tpu_config.get('axis_names', ["data", "model"]),
                    'dtype': tpu_config.get('dtype', 'bfloat16'),
                    'param_dtype': tpu_config.get('param_dtype', 'float32'),
                    'enable_systolic_arrays': tpu_config.get('enable_systolic_arrays', True),
                    'enable_xla_optimization': tpu_config.get('enable_xla_optimization', True),
                    'optimization_level': tpu_config.get('optimization_level', 3),
                })
            
            # Neuroadaptive configuration
            if 'neuroadaptive' in config_dict:
                neuro_config = config_dict['neuroadaptive']
                kwargs.update({
                    'enable_neuroadaptive': neuro_config.get('enable_neuroadaptive', True),
                    'neuroadaptive_rate': neuro_config.get('neuroadaptive_rate', 0.01),
                    'use_remat_neuroadaptive': neuro_config.get('use_remat_neuroadaptive', True),
                    'neurogenesis_threshold': neuro_config.get('neurogenesis_threshold', 0.8),
                    'num_adaptive_layers': neuro_config.get('num_adaptive_layers', 2),
                    'adaptation_interval': neuro_config.get('adaptation_interval', 100),
                    'enable_neurogenesis': neuro_config.get('enable_neurogenesis', True),
                    'enable_adaptive_embedding': neuro_config.get('enable_adaptive_embedding', True),
                })
            
            # Router configuration
            if 'router' in config_dict:
                router_config = config_dict['router']
                kwargs.update({
                    'router_hidden_size': router_config.get('hidden_size', 768),
                    'router_num_heads': router_config.get('num_heads', 8),
                    'router_cache_size': router_config.get('cache_size', 1000),
                    'routing_cache_size': router_config.get('routing_cache_size', 1000),
                    'enable_adaptive_routing': router_config.get('enable_adaptive_routing', True),
                    'enable_contextual_activation': router_config.get('enable_contextual_activation', True),
                    'enable_vision': router_config.get('enable_vision', True),
                    'enable_audio': router_config.get('enable_audio', True),
                    'enable_bio': router_config.get('enable_bio', True),
                })
            
            # Agent configuration
            if 'agents' in config_dict:
                agents_config = config_dict['agents']
                kwargs.update({
                    'max_agents': agents_config.get('max_agents', 100),
                    'agent_cleanup_interval': agents_config.get('agent_cleanup_interval', 300.0),
                    'enable_hot_reload': agents_config.get('enable_hot_reload', False),
                    'agent_timeout': agents_config.get('agent_timeout', 30.0),
                })
            
            # Modules configuration
            if 'modules' in config_dict:
                modules_config = config_dict['modules']
                kwargs.update({
                    'active_modules': modules_config.get('active', []),
                    'enable_lazy_loading': modules_config.get('enable_lazy_loading', True),
                    'lazy_loading_timeout': modules_config.get('lazy_loading_timeout', 30.0),
                })
            
            # Monitoring configuration
            if 'monitoring' in config_dict:
                monitoring_config = config_dict['monitoring']
                kwargs.update({
                    'enable_metrics': monitoring_config.get('enable_metrics', True),
                    'metrics_window_size': monitoring_config.get('metrics_window_size', 1000),
                    'log_interval': monitoring_config.get('log_interval', 100),
                    'enable_visualization': monitoring_config.get('enable_visualization', True),
                    'visualization_cleanup_interval': monitoring_config.get('visualization_cleanup_interval', 3600.0),
                    'max_visualization_files': monitoring_config.get('max_visualization_files', 50),
                })
            
            # Memory configuration
            if 'memory' in config_dict:
                memory_config = config_dict['memory']
                kwargs.update({
                    'memory_pressure_threshold': memory_config.get('memory_pressure_threshold', 0.85),
                    'enable_memory_monitoring': memory_config.get('enable_memory_monitoring', True),
                    'cleanup_interval': memory_config.get('cleanup_interval', 300.0),
                })
            
            # Paths configuration
            if 'paths' in config_dict:
                paths_config = config_dict['paths']
                kwargs.update({
                    'config_root': Path(paths_config.get('config_root', 'capibara/config')),
                    'modules_root': Path(paths_config.get('modules_root', 'capibara/config/modules')),
                    'checkpoints_dir': paths_config.get('checkpoints_dir', 'checkpoints'),
                    'logs_dir': paths_config.get('logs_dir', 'logs'),
                })
            
            # === TTS SERVICES CONFIGURATION ===
            if 'services' in config_dict:
                services_config = config_dict['services']
                kwargs.update({
                    'enable_tts': services_config.get('enable_tts', True),
                    'tts_backend': services_config.get('tts_backend', 'capibara_tts'),
                    'tts_fallback': services_config.get('tts_fallback', 'pyttsx3'),
                    'websocket_enabled': services_config.get('websocket_enabled', True),
                    'websocket_host': services_config.get('websocket_host', 'localhost'),
                    'websocket_port': services_config.get('websocket_port', 8765),
                    'ssl_enabled': services_config.get('ssl_enabled', False),
                    'max_concurrent_connections': services_config.get('max_concurrent_connections', 100),
                    'audio_cache_enabled': services_config.get('audio_cache_enabled', True),
                    'audio_cache_size': services_config.get('audio_cache_size', 1000),
                    'audio_cache_ttl': services_config.get('audio_cache_ttl', 3600),
                })
            
            if 'tts' in config_dict:
                tts_config = config_dict['tts']
                kwargs.update({
                    'fastspeech_model_path': tts_config.get('fastspeech_model_path', 'models/fastspeech.onnx'),
                    'hifigan_model_path': tts_config.get('hifigan_model_path', 'models/hifigan.onnx'),
                    'sample_rate': tts_config.get('sample_rate', 22050),
                    'audio_format': tts_config.get('audio_format', 'wav'),
                    'bit_depth': tts_config.get('bit_depth', 16),
                    'channels': tts_config.get('channels', 1),
                    'voice_speed': tts_config.get('voice_speed', 150),
                    'voice_volume': tts_config.get('voice_volume', 1.0),
                    'spectrogram_clipping': tts_config.get('spectrogram_clipping', [-4.0, 4.0]),
                    'enable_websocket_server': tts_config.get('enable_websocket_server', True),
                    'websocket_buffer_size': tts_config.get('websocket_buffer_size', 8192),
                    'websocket_timeout': tts_config.get('websocket_timeout', 30.0),
                    'fallback_engine': tts_config.get('fallback_engine', 'pyttsx3'),
                    'fallback_voice_rate': tts_config.get('fallback_voice_rate', 150),
                    'fallback_voice_volume': tts_config.get('fallback_voice_volume', 1.0),
                    'cert_file': tts_config.get('cert_file', ''),
                    'key_file': tts_config.get('key_file', ''),
                    'enable_audio_preprocessing': tts_config.get('enable_audio_preprocessing', True),
                    'enable_noise_reduction': tts_config.get('enable_noise_reduction', False),
                    'enable_voice_enhancement': tts_config.get('enable_voice_enhancement', True),
                })
            
            return cls(**kwargs)
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            logger.info("Using default configuration")
            return cls()
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            logger.info("Using default configuration")
            return cls()
    
    def save_to_toml(self, output_path: str) -> None:
        """
        Save current configuration to a TOML file.

        Args:
            output_path: Path where to save the TOML file.
        """
        config_dict = {
            'metadata': {
                'version': '2.3',
                'name': 'ModularCapibaraModel-Fusionado',
                'description': 'Configuración integrada con optimizaciones TPU v4-32 y componentes neuroadaptativos'
            },
            'model': {
                'hidden_size': self.hidden_size,
                'num_virtual_qubits': self.num_virtual_qubits,
                'vocab_size': self.vocab_size,
                'max_context_length': self.max_context_length,
                'num_router_experts': self.num_router_experts,
                'router_capacity_factor': self.router_capacity_factor,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'intermediate_size': self.intermediate_size,
                'dropout_rate': self.dropout_rate,
                'attention_dropout': self.attention_dropout,
                'layer_norm_eps': self.layer_norm_eps,
                'activation_function': self.activation_function,
            },
            'memory_optimization': {
                'use_gradient_checkpointing': self.memory_config.use_gradient_checkpointing,
                'checkpoint_policy': self.memory_config.checkpoint_policy,
                'checkpoint_blocks': self.memory_config.checkpoint_blocks,
                'remat_layer_interval': self.memory_config.remat_layer_interval,
                'preserve_rng_state': self.memory_config.preserve_rng_state,
                'memory_pressure_threshold': self.memory_config.memory_pressure_threshold,
                'enable_memory_monitoring': self.memory_config.enable_memory_monitoring,
                'cleanup_interval': self.memory_config.cleanup_interval,
                'max_memory_usage_gb': self.memory_config.max_memory_usage_gb,
                'force_device_gc': self.memory_config.force_device_gc,
                'use_mixed_precision': self.memory_config.use_mixed_precision,
                'compute_dtype': self.memory_config.compute_dtype,
                'param_dtype': self.memory_config.param_dtype,
                'output_dtype': self.memory_config.output_dtype,
                'use_model_parallelism': self.memory_config.use_model_parallelism,
                'model_parallel_submesh': list(self.memory_config.model_parallel_submesh),
                'data_parallel_submesh': list(self.memory_config.data_parallel_submesh),
                'shard_strategy': self.memory_config.shard_strategy,
                'dynamic_batch_size': self.memory_config.dynamic_batch_size,
                'min_batch_size': self.memory_config.min_batch_size,
                'max_batch_size': self.memory_config.max_batch_size,
                'batch_growth_factor': self.memory_config.batch_growth_factor,
            },
            'tpu_v4': {
                'use_tpu_optimizations': self.use_tpu_optimizations,
                'cores': self.tpu_cores,
                'memory_limit_gb': self.tpu_memory_limit_gb,
                'cleanup_threshold': self.tpu_cleanup_threshold,
                'mesh_shape': list(self.tpu_mesh_shape),
                'axis_names': self.tpu_axis_names,
                'dtype': self.dtype,
                'param_dtype': self.param_dtype,
                'enable_systolic_arrays': self.enable_systolic_arrays,
                'enable_xla_optimization': self.enable_xla_optimization,
                'optimization_level': self.optimization_level,
            },
            'neuroadaptive': {
                'enable_neuroadaptive': self.enable_neuroadaptive,
                'neuroadaptive_rate': self.neuroadaptive_rate,
                'use_remat_neuroadaptive': self.use_remat_neuroadaptive,
                'neurogenesis_threshold': self.neurogenesis_threshold,
                'num_adaptive_layers': self.num_adaptive_layers,
                'adaptation_interval': self.adaptation_interval,
                'enable_neurogenesis': self.enable_neurogenesis,
                'enable_adaptive_embedding': self.enable_adaptive_embedding,
            },
            'router': {
                'hidden_size': self.router_hidden_size,
                'num_heads': self.router_num_heads,
                'cache_size': self.router_cache_size,
                'routing_cache_size': self.routing_cache_size,
                'enable_adaptive_routing': self.enable_adaptive_routing,
                'enable_contextual_activation': self.enable_contextual_activation,
                'enable_vision': self.enable_vision,
                'enable_audio': self.enable_audio,
                'enable_bio': self.enable_bio,
            },
            'agents': {
                'max_agents': self.max_agents,
                'agent_cleanup_interval': self.agent_cleanup_interval,
                'enable_hot_reload': self.enable_hot_reload,
                'agent_timeout': self.agent_timeout,
            },
            'modules': {
                'active': self.active_modules,
                'enable_lazy_loading': self.enable_lazy_loading,
                'lazy_loading_timeout': self.lazy_loading_timeout,
            },
            'monitoring': {
                'enable_metrics': self.enable_metrics,
                'metrics_window_size': self.metrics_window_size,
                'log_interval': self.log_interval,
                'enable_visualization': self.enable_visualization,
                'visualization_cleanup_interval': self.visualization_cleanup_interval,
                'max_visualization_files': self.max_visualization_files,
            },
            'memory': {
                'memory_pressure_threshold': self.memory_pressure_threshold,
                'enable_memory_monitoring': self.enable_memory_monitoring,
                'cleanup_interval': self.cleanup_interval,
            },
            'paths': {
                'config_root': str(self.config_root),
                'modules_root': str(self.modules_root),
                'checkpoints_dir': self.checkpoints_dir,
                'logs_dir': self.logs_dir,
            },
            'services': {
                'enable_tts': self.enable_tts,
                'tts_backend': self.tts_backend,
                'tts_fallback': self.tts_fallback,
                'websocket_enabled': self.websocket_enabled,
                'websocket_host': self.websocket_host,
                'websocket_port': self.websocket_port,
                'ssl_enabled': self.ssl_enabled,
                'max_concurrent_connections': self.max_concurrent_connections,
                'audio_cache_enabled': self.audio_cache_enabled,
                'audio_cache_size': self.audio_cache_size,
                'audio_cache_ttl': self.audio_cache_ttl,
            },
            'tts': {
                'fastspeech_model_path': self.fastspeech_model_path,
                'hifigan_model_path': self.hifigan_model_path,
                'sample_rate': self.sample_rate,
                'audio_format': self.audio_format,
                'bit_depth': self.bit_depth,
                'channels': self.channels,
                'voice_speed': self.voice_speed,
                'voice_volume': self.voice_volume,
                'spectrogram_clipping': self.spectrogram_clipping,
                'enable_websocket_server': self.enable_websocket_server,
                'websocket_buffer_size': self.websocket_buffer_size,
                'websocket_timeout': self.websocket_timeout,
                'fallback_engine': self.fallback_engine,
                'fallback_voice_rate': self.fallback_voice_rate,
                'fallback_voice_volume': self.fallback_voice_volume,
                'cert_file': self.cert_file,
                'key_file': self.key_file,
                'enable_audio_preprocessing': self.enable_audio_preprocessing,
                'enable_noise_reduction': self.enable_noise_reduction,
                'enable_voice_enhancement': self.enable_voice_enhancement,
            },
        }
        
        try:
            with open(output_path, 'w') as f:
                toml.dump(config_dict, f)
            logger.info(f"Configuration saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {output_path}: {e}")
    
    def get_tpu_dtype(self):
        """Convert string dtype to JAX type."""
        import jax.numpy as jnp
        dtype_map = {
            'float32': jnp.float32,
            'float16': jnp.float16,
            'bfloat16': jnp.bfloat16,
        }
        return dtype_map.get(self.dtype, jnp.bfloat16)
    
    def get_param_dtype(self):
        """Convert string param_dtype to JAX type."""
        import jax.numpy as jnp
        dtype_map = {
            'float32': jnp.float32,
            'float16': jnp.float16,
            'bfloat16': jnp.bfloat16,
        }
        return dtype_map.get(self.param_dtype, jnp.float32)



# Content from training_config.py
"""
CapibaraModel training configuration.

This module defines the specific configuration for model training,
including training strategies, validation and monitoring.
"""
from typing import Optional, List, Dict, Any, cast, TypeVar, Callable
try:
    from pydantic.v1 import BaseModel, Field, validator #type: ignore
except ImportError:
    raise ImportError("Please install pydantic with: pip install pydantic")

try:
    import yaml #type: ignore
except ImportError:
    raise ImportError("Please install PyYAML with: pip install PyYAML")

import os

T = TypeVar('T')

def typed_validator(field: str) -> Callable[[Callable[[Any, T], T]], Callable[[Any, T], T]]:
    """
    Typed decorator for Pydantic validators.

    Args:
        field: Name of the field to validate

    Returns:
        A decorator that wraps the validation function with correct types
    """
    def decorator(func: Callable[[Any, T], T]) -> Callable[[Any, T], T]:
        return cast(Callable[[Any, T], T], validator(field)(func))
    return decorator

class TrainingConfig(BaseModel):
    """
    Complete configuration for model training.

    This class defines all necessary parameters to configure the training process,
    including optimization, training strategies, validation and monitoring.

    Attributes:
        epochs: Total number of training epochs
        steps_per_epoch: Steps per epoch (optional, calculated automatically if None)
        validation_steps: Validation steps per epoch (optional)
        early_stopping_patience: Number of epochs without improvement before stopping training
        strategy: Distribution strategy (mirrored, multi_worker, etc.)
        use_tpu: Whether to use TPU for training
        use_mixed_precision: Enables mixed precision to accelerate training
        optimizer: Optimizer type (adamw, sgd, etc.)
        learning_rate: Initial learning rate
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay factor for regularization
        gradient_clip_norm: Maximum value for gradient clipping
        use_gradient_centralization: Enables gradient centralization to improve generalization
        batch_size: Training batch size
        per_replica_batch_size: Batch size per replica in distributed training
        save_checkpoints: Whether to save checkpoints
        checkpoint_frequency: Checkpoint save frequency
        keep_checkpoint_max: Maximum number of checkpoints to keep
        log_frequency: Logging frequency for metrics
        tensorboard_update_freq: TensorBoard update frequency
        validation_frequency: Validation frequency
        metrics: List of metrics to monitor
        progressive_training_config: Path to progressive training configuration
        component_strategy_config: Path to component strategy configuration
        custom_callbacks: List of custom callbacks
    """
    
    # Basic configuration
    epochs: int = Field(10, description="Number of training epochs")
    steps_per_epoch: Optional[int] = Field(None, description="Steps per epoch")
    validation_steps: Optional[int] = Field(None, description="Validation steps")
    early_stopping_patience: int = Field(3, description="Patience for early stopping")

    # Strategy configuration
    strategy: str = Field("mirrored", description="Distribution strategy")
    use_tpu: bool = Field(False, description="Use TPU for training")
    use_mixed_precision: bool = Field(True, description="Use mixed precision")

    # Optimization configuration
    optimizer: str = Field("adamw", description="Optimizer to use")
    learning_rate: float = Field(1e-4, description="Base learning rate")
    warmup_steps: int = Field(1000, description="Warmup steps")
    weight_decay: float = Field(0.01, description="Weight decay")
    gradient_clip_norm: float = Field(1.0, description="Gradient norm for clipping")
    use_gradient_centralization: bool = Field(
        True,
        description="Use Gradient Centralization to improve generalization"
    )

    # Batch configuration
    batch_size: int = Field(32, description="Batch size")
    per_replica_batch_size: Optional[int] = Field(None, description="Batch size per replica")

    # Checkpoints configuration
    save_checkpoints: bool = Field(True, description="Save checkpoints")
    checkpoint_frequency: int = Field(1000, description="Checkpoint save frequency")
    keep_checkpoint_max: int = Field(5, description="Maximum number of checkpoints to keep")

    # Logging configuration
    log_frequency: int = Field(100, description="Logging frequency")
    tensorboard_update_freq: int = Field(100, description="TensorBoard update frequency")

    # Validation configuration
    validation_frequency: int = Field(1000, description="Validation frequency")
    metrics: List[str] = Field(
        default_factory=lambda: ["loss", "accuracy"],
        description="Metrics to monitor"
    )

    # Strategies configuration
    progressive_training_config: Optional[str] = Field(
        None,
        description="Path to progressive training configuration file"
    )
    component_strategy_config: Optional[str] = Field(
        None,
        description="Path to component strategy configuration file"
    )

    # Advanced configuration
    custom_callbacks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Custom callbacks"
    )
    
    def load_progressive_config(self) -> Dict[str, Any]:
        """
        Load progressive training configuration from YAML file.

        Returns:
            Dict[str, Any]: Configuration loaded as dictionary

        Raises:
            FileNotFoundError: If configuration file does not exist
        """
        if not self.progressive_training_config:
            return {}

        if not os.path.exists(self.progressive_training_config):
            raise FileNotFoundError(
                f"Progressive configuration file not found: {self.progressive_training_config}"
            )
        
        with open(self.progressive_training_config, 'r') as f:
            return cast(Dict[str, Any], yaml.safe_load(f))
    
    def load_component_strategy(self) -> Dict[str, Any]:
        """
        Load component training strategy from YAML file.

        Returns:
            Dict[str, Any]: Strategy loaded as dictionary

        Raises:
            FileNotFoundError: If strategy file does not exist
        """
        if not self.component_strategy_config:
            return {}

        if not os.path.exists(self.component_strategy_config):
            raise FileNotFoundError(
                f"Component strategy file not found: {self.component_strategy_config}"
            )
        
        with open(self.component_strategy_config, 'r') as f:
            return cast(Dict[str, Any], yaml.safe_load(f))
    
    @typed_validator('learning_rate')
    def validate_learning_rate(cls, v: float) -> float:
        """
        Validate that learning rate is positive.

        Args:
            v: Learning rate value to validate

        Returns:
            float: Validated learning rate

        Raises:
            ValueError: If learning rate is not positive
        """
        if v <= 0:
            raise ValueError("Learning rate must be positive")
        return v
    
    @typed_validator('batch_size')
    def validate_batch_size(cls, v: int) -> int:
        """
        Validate that batch size is positive.

        Args:
            v: Batch size value to validate

        Returns:
            int: Validated batch size

        Raises:
            ValueError: If batch size is not positive
        """
        if v <= 0:
            raise ValueError("Batch size must be positive")
        return v

    class Config:
        """
        Pydantic configuration for TrainingConfig class.

        Attributes:
            validate_assignment: Enables validation on value assignment
            extra: Forbids additional fields not defined in the model
        """
        validate_assignment = True
        extra = "forbid" 

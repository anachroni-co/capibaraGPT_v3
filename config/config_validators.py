"""
Extended validators for CapibaraModel setup
"""

import os
import sys
import psutil
import logging

# Get the current directory path (scripts) -> /.../scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to obtain the project root -> /.../capibaraGPT-v2
project_root = os.path.dirname(script_dir)
# Add the project root to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from pathlib import Path
import shutil
from capibara.jax import jax
import numpy as np #type: ignore
from .config_types import CapibaraConfig
from capibara.jax.experimental import mesh_utils 
from typing import Dict, Any, List, Optional, Union
from pydantic import validator, ValidationError #type: ignore

# Importaciones de tpu v4-32
from capibara.jax.tpu_v4.backend import (
    TpuV4LinalgOps,
    TpuV4SparseOps,
    TpuV4NeuralOps,
    TpuV4RandomOps,
    TpuV4PerformanceUtils,
)
from capibara.jax.tpu_v4.optimizations import (
    create_tpu_mesh,
    TpuMemoryMonitor,
    tpu_optimized_gemm,
    create_jitted_forward,
)
from capibara.jax.tpu_v4.profiling import (
    TpuProfiler,
    _uniform_fallback_weights,
    _expert_weights_with_cache,
    checkpointed_transformer_block,
)

logger = logging.getLogger(__name__)

def validate_hardware_compatibility(config: Dict[str, Any]) -> List[str]:
    """Validates hardware compatibility with the configuration, including TPU v4-32."""
    warnings = []

    # Calculate memory required by the model
    model_config = config.get('model', {})
    hidden_size = model_config.get('hidden_size', 1024)
    num_layers = model_config.get('num_layers', 24)
    vocab_size = model_config.get('vocab_size', 32000)

    # Approximate memory requirement estimation
    model_mem = (
        hidden_size * hidden_size * 4 * num_layers +  # Attention parameters
        hidden_size * vocab_size * 2 +  # Embeddings
        hidden_size * 4 * num_layers  # FFN parameters
    ) * 4  # 4 bytes per parameter
    
    try:
        # Validate CPU memory
        cpu_mem = psutil.virtual_memory().total
        if model_mem > cpu_mem * 0.8:  # 80% of limit
            warnings.append(
                f"Configuration requires {model_mem/1e9:.2f}GB of RAM but only {cpu_mem/1e9:.2f}GB are available"
            )

        # Validate CPU cores
        cpu_cores = psutil.cpu_count(logical=False)
        if cpu_cores < 4:
            warnings.append(f"Low CPU cores ({cpu_cores}) may limit performance")
    except Exception as e:
        logger.error(f"Error validating CPU: {str(e)}")
        warnings.append(f"Could not validate CPU memory: {str(e)}")

    # Validate TPU v4-32 if enabled
    if config.get('use_tpu', False):
        try:
            tpu_devices = jax.devices('tpu')
            if not tpu_devices:
                warnings.append("TPU v4-32 configured but not detected in the system")
            else:
                # Validate sharding for TPU v4-32
                num_devices = len(tpu_devices)
                if config.get('model', {}).get('sharding', {}).get('enabled', False):
                    shard_size = config['model']['sharding']['size']
                    if shard_size > num_devices:
                        warnings.append(f"Sharding size ({shard_size}) exceeds number of available TPUs ({num_devices})")

                    # Validate TPU v4-32 topology
                    try:
                        topology = mesh_utils.create_device_mesh((num_devices,))
                        logger.info(f"TPU v4-32 topology detected: {topology}")

                        # Validate memory per chip
                        memory_per_chip = 32 * 1024**3  # 32GB HBM per chip
                        total_memory = memory_per_chip * num_devices
                        if model_mem > total_memory * 0.9:  # 90% of limit
                            warnings.append(
                                f"Configuration requires {model_mem/1e9:.2f}GB of TPU memory "
                                f"but only {total_memory/1e9:.2f}GB are available"
                            )
                    except Exception as e:
                        warnings.append(f"Error validating TPU v4-32 topology: {str(e)}")

                # Validate mixed precision configuration
                if config.get('training', {}).get('mixed_precision', False):
                    if not config.get('tpu', {}).get('enable_auto_mixed_precision', True):
                        warnings.append("Mixed precision enabled but auto_mixed_precision disabled in TPU")

                # Validate optimization configuration
                if config.get('tpu', {}).get('optimization_level', 0) < 3:
                    warnings.append("TPU v4-32 optimization level is suboptimal for the model")

        except Exception as e:
            logger.error(f"Error validating TPU v4-32: {str(e)}")
            warnings.append(f"Error validating TPU v4-32: {str(e)}")
    
    return warnings

def validate_training_compatibility(config: Dict[str, Any]) -> List[str]:
    """Validates compatibility of training parameters."""
    warnings = []

    try:
        # Validate batch size
        batch_size = config['training']['batch_size']
        seq_length = config['model']['max_seq_length']
        if batch_size * seq_length > 1e6:
            warnings.append("Batch size × sequence length may cause memory issues")

        # Validate learning rate
        lr = config['training']['learning_rate']
        if lr > 1e-3:
            warnings.append("Learning rate may be too high for stable training")
        elif lr < 1e-6:
            warnings.append("Learning rate may be too low for convergence")

        # Validate progressive training
        if config['progressive_training']['enabled']:
            if not config['training']['phase_ordering']:
                warnings.append("Progressive training enabled but phase_ordering is empty")
            else:
                # Validate phase ordering
                phases = config['training']['phase_ordering']
                if not all(isinstance(p, dict) and 'name' in p for p in phases):
                    warnings.append("Invalid format in phase_ordering")
        
        # Validate optimizations
        if config.get('optimization', {}).get('quantization', {}).get('enabled', False):
            if config['training']['mixed_precision']:
                warnings.append("Quantization and mixed precision may cause conflicts")
            if config.get('optimization', {}).get('gradient_checkpointing', False):
                warnings.append("Quantization and gradient checkpointing may reduce performance")

            # Validate quantization configuration
            bit_width = config.get('optimization', {}).get('quantization', {}).get('bit_width', 8)
            if bit_width not in [4, 8, 16]:
                warnings.append(f"Invalid bit width for quantization: {bit_width}")
    except Exception as e:
        logger.error(f"Error validating training: {str(e)}")
        warnings.append(f"Error in training validation: {str(e)}")
    
    return warnings

def validate_component_dependencies(config: Dict[str, Any]) -> List[str]:
    """Validates dependencies between components."""
    warnings = []
    components = config['model']['components']

    try:
        # Validate attention dependencies
        if components.get('self_attention', {}).get('enabled', False):
            if not components.get('embeddings', {}).get('enabled', False):
                warnings.append("Self-attention requires embeddings to be enabled")

            # Validate attention configuration
            num_heads = config['model']['num_heads']
            hidden_size = config['model']['hidden_size']
            if hidden_size % num_heads != 0:
                warnings.append(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        
        # Validate quantization dependencies
        if config.get('optimization', {}).get('quantization', {}).get('enabled', False):
            if config['training']['mixed_precision']:
                warnings.append("Quantization and mixed precision may cause conflicts")
            if not components.get('normalization', {}).get('enabled', False):
                warnings.append("Quantization requires normalization for stability")

            # Validate quantization configuration
            if config.get('optimization', {}).get('quantization', {}).get('symmetric', True):
                if not components.get('normalization', {}).get('enabled', False):
                    warnings.append("Symmetric quantization requires normalization")

        # Validate TPU dependencies
        if config.get('use_tpu', False):
            if not config.get('optimization', {}).get('sharding', {}).get('enabled', False):
                warnings.append("TPU requires sharding for maximum performance")

            # Validate sharding configuration
            if config.get('optimization', {}).get('sharding', {}).get('enabled', False):
                shard_size = config['optimization']['sharding']['size']
                if shard_size < 2:
                    warnings.append("Sharding size must be at least 2 for TPU")
    except Exception as e:
        logger.error(f"Error validating components: {str(e)}")
        warnings.append(f"Error in component validation: {str(e)}")
    
    return warnings

def validate_paths_and_permissions(config: Dict[str, Any]) -> List[str]:
    """Validates paths and file permissions."""
    warnings = []

    try:
        for key, path in config['paths'].items():
            path_obj = Path(path)

            # Verify existence and permissions
            if not path_obj.exists():
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Directory created: {path}")
                except Exception as e:
                    logger.error(f"Error creating directory {path}: {str(e)}")
                    warnings.append(f"Cannot create {key} at {path}: {str(e)}")
            else:
                if not os.access(path, os.W_OK):
                    warnings.append(f"No write permissions at {path}")
                if not os.access(path, os.R_OK):
                    warnings.append(f"No read permissions at {path}")

                # Verify available space
                try:
                    free_space = shutil.disk_usage(path).free
                    if free_space < 1e9:  # Expect 1GB
                        warnings.append(f"Low available space at {path}: {free_space/1e9:.2f}GB")
                except Exception:
                    pass  # Ignore disk space errors
    except Exception as e:
        logger.error(f"Error validating paths: {str(e)}")
        warnings.append(f"Error in path validation: {str(e)}")
    
    return warnings

def estimate_model_memory(config: Dict[str, Any]) -> int:
    """Estimates model memory usage in bytes."""
    try:
        hidden_size = config['model']['hidden_size']
        num_layers = config['model']['num_layers']
        vocab_size = config['model']['vocab_size']
        seq_length = config['model']['max_seq_length']
        batch_size = config['training']['batch_size']

        # Precision factor based on configuration
        precision_factor = 4  # float32 by default
        if config.get('optimization', {}).get('quantization', {}).get('enabled', False):
            precision_factor = 2  # bfloat16
        elif config['training']['mixed_precision']:
            precision_factor = 2  # mixed precision

        # Estimation of main parameters
        embedding_params = vocab_size * hidden_size * precision_factor
        attention_params = num_layers * (hidden_size * hidden_size * precision_factor) * 4  # 4 matrices per layer
        ffn_params = num_layers * (hidden_size * hidden_size * precision_factor) * 2  # 2 matrices per layer

        # Estimation of activations during forward pass
        activations = batch_size * seq_length * hidden_size * precision_factor * num_layers

        # Overhead for gradients during training
        training_overhead = (embedding_params + attention_params + ffn_params) * 2

        # Additional memory for optimizations
        optimization_overhead = 0
        if config.get('optimization', {}).get('gradient_checkpointing', False):
            optimization_overhead += activations * 0.5  # Memory reduction by checkpointing
        if config.get('optimization', {}).get('sharding', {}).get('enabled', False):
            optimization_overhead += (embedding_params + attention_params + ffn_params) * 0.1  # Sharding overhead

        total_memory = embedding_params + attention_params + ffn_params + activations + training_overhead + optimization_overhead
        return int(total_memory)
    except Exception as e:
        logger.error(f"Error estimating memory: {str(e)}")
        raise

def validate_full_config(config: Dict[str, Any]) -> List[str]:
    """Performs complete configuration validation."""
    all_warnings = []

    try:
        # Validate hardware
        all_warnings.extend(validate_hardware_compatibility(config))

        # Validate training parameters
        all_warnings.extend(validate_training_compatibility(config))

        # Validate component dependencies
        all_warnings.extend(validate_component_dependencies(config))

        # Validate paths and permissions
        all_warnings.extend(validate_paths_and_permissions(config))

        # Validate using Pydantic
        try:
            CapibaraConfig(**config)
        except ValidationError as e:
            all_warnings.extend(str(err) for err in e.errors())
    except Exception as e:
        logger.error(f"Error in complete validation: {str(e)}")
        all_warnings.append(f"Error in complete validation: {str(e)}")
    
    return all_warnings

def validate_tpu_config(config: Dict[str, Any]) -> List[str]:
    """Validates TPU v4-32 specific configuration."""
    warnings = []

    tpu_config = config.get('tpu', {})
    if not tpu_config.get('enabled', False):
        return warnings

    # Validate number of cores
    num_cores = tpu_config.get('num_cores', 8)
    valid_cores = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    if num_cores not in valid_cores:
        warnings.append(f"Invalid TPU v4-32 number of cores: {num_cores}")

    # Validate data type
    dtype = tpu_config.get('dtype', 'bfloat16')
    valid_dtypes = ['bfloat16', 'float32', 'float16']
    if dtype not in valid_dtypes:
        warnings.append(f"Invalid TPU v4-32 data type: {dtype}")

    # Validate memory configuration
    memory_fraction = tpu_config.get('memory_fraction', 0.9)
    if not 0 < memory_fraction <= 1:
        warnings.append(f"Invalid TPU v4-32 memory fraction: {memory_fraction}")

    # Validate batch configuration
    batch_size = tpu_config.get('batch_size_per_core', 16)
    if batch_size <= 0:
        warnings.append(f"Invalid TPU v4-32 batch size per core: {batch_size}")

    # Validate optimization configuration
    optimization_level = tpu_config.get('optimization_level', 3)
    if not 0 <= optimization_level <= 3:
        warnings.append(f"Invalid TPU v4-32 optimization level: {optimization_level}")
        
    return warnings

def validate_model_config(config: Dict[str, Any]) -> List[str]:
    """Validates model configuration with TPU v4-32 support."""
    warnings = []

    model_config = config.get('model', {})

    # Validate model dimensions
    hidden_size = model_config.get('hidden_size', 1024)
    if hidden_size % 128 != 0:
        warnings.append(f"hidden_size ({hidden_size}) is not a multiple of 128, which may affect TPU v4-32 performance")

    # Validate attention configuration
    num_heads = model_config.get('num_heads', 16)
    if hidden_size % num_heads != 0:
        warnings.append(f"hidden_size ({hidden_size}) is not divisible by num_heads ({num_heads})")

    # Validate mixed precision configuration
    if model_config.get('use_mixed_precision', False):
        if not config.get('tpu', {}).get('enable_auto_mixed_precision', True):
            warnings.append("Mixed precision enabled in model but disabled in TPU v4-32")
            
    return warnings

def validate_training_config(config: Dict[str, Any]) -> List[str]:
    """Validates training configuration with TPU v4-32 support."""
    warnings = []

    training_config = config.get('training', {})

    # Validate batch configuration
    batch_size = training_config.get('batch_size', 16)
    if batch_size <= 0:
        warnings.append(f"Invalid batch size: {batch_size}")

    # Validate gradient configuration
    gradient_clip_norm = training_config.get('gradient_clip_norm', 1.0)
    if gradient_clip_norm <= 0:
        warnings.append(f"Invalid gradient norm: {gradient_clip_norm}")

    # Validate optimizer configuration
    learning_rate = training_config.get('learning_rate', 1e-4)
    if learning_rate <= 0:
        warnings.append(f"Invalid learning rate: {learning_rate}")

    # Validate TPU v4-32 configuration
    if training_config.get('use_tpu', False):
        if not config.get('tpu', {}).get('enabled', False):
            warnings.append("Training configured for TPU v4-32 but TPU not enabled")

    return warnings 
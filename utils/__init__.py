"""
Utils Module - CapibaraGPT v3

Utility functions and tools including:
- cache_standalone: TPU-optimized caching
- checkpoint_manager: Training checkpoints
- monitoring: Resource monitoring
- config_utils: Configuration utilities
- data_utils: Data processing
- validation_utils: Input validation
- logging_utils: Logging configuration
"""

import logging

logger = logging.getLogger(__name__)

# Caching
try:
    from .cache_standalone import CacheManager, create_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    CacheManager = None
    create_cache = None

# Checkpointing
try:
    from .checkpoint_manager import CheckpointManager
    CHECKPOINT_AVAILABLE = True
except ImportError:
    CHECKPOINT_AVAILABLE = False
    CheckpointManager = None

# Monitoring
try:
    from .monitoring import ResourceMonitor
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    ResourceMonitor = None

# Config utilities
try:
    from .config_utils import load_config, save_config, merge_configs
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    load_config = None
    save_config = None
    merge_configs = None

# Data utilities
try:
    from .data_utils import DataProcessor
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False
    DataProcessor = None

# Validation
try:
    from .validation_utils import validate_input, ValidationError
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    validate_input = None
    ValidationError = None

# Logging
try:
    from .logging_utils import setup_logging, get_logger
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    setup_logging = None
    get_logger = None

# Memory Profiling
try:
    from .memory_profiler import (
        MemoryProfiler,
        get_profiler,
        profile_memory,
        memory_profile_block,
        check_for_leaks,
        print_memory_summary,
        TrainingMemoryTracker
    )
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    MemoryProfiler = None
    get_profiler = None
    profile_memory = None
    memory_profile_block = None
    check_for_leaks = None
    print_memory_summary = None
    TrainingMemoryTracker = None


__all__ = [
    # Cache
    "CacheManager",
    "create_cache",
    # Checkpoint
    "CheckpointManager",
    # Monitoring
    "ResourceMonitor",
    # Config
    "load_config",
    "save_config",
    "merge_configs",
    # Data
    "DataProcessor",
    # Validation
    "validate_input",
    "ValidationError",
    # Logging
    "setup_logging",
    "get_logger",
    # Memory Profiling
    "MemoryProfiler",
    "get_profiler",
    "profile_memory",
    "memory_profile_block",
    "check_for_leaks",
    "print_memory_summary",
    "TrainingMemoryTracker",
    # Flags
    "CACHE_AVAILABLE",
    "CHECKPOINT_AVAILABLE",
    "MONITORING_AVAILABLE",
    "CONFIG_AVAILABLE",
    "DATA_AVAILABLE",
    "VALIDATION_AVAILABLE",
    "LOGGING_AVAILABLE",
    "MEMORY_PROFILER_AVAILABLE",
]

"""
Data Preprocessing Integration for Capibara-6

Integrates the semantic deduplication and quality filtering system
with the existing UnifiedTrainingSystem and TPU infrastructure.

This module provides:
- Integration with UnifiedTrainingSystem
- Configuration management for preprocessing
- Metrics collection and monitoring
- Integration with existing training workflows
"""

import logging
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# Import preprocessing components
try:
    from .data_preprocessing import (
        DataPreprocessingPipeline,
        DEDUPLICATION_AVAILABLE,
        QUALITY_FILTER_AVAILABLE,
        TPU_PROCESSOR_AVAILABLE
    )
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Data preprocessing not available: {e}")
    PREPROCESSING_AVAILABLE = False

# Import configurations conditionally
if PREPROCESSING_AVAILABLE:
    try:
        from .data_preprocessing.semantic_deduplicator import DedupConfig
    except ImportError:
        DedupConfig = None
    
    try:
        from .data_preprocessing.quality_filter import QualityConfig
    except ImportError:
        QualityConfig = None
    
    try:
        from .data_preprocessing.tpu_optimized_processor import TPUOptimizationConfig
    except ImportError:
        TPUOptimizationConfig = None
else:
    DedupConfig = None
    QualityConfig = None
    TPUOptimizationConfig = None


@dataclass
class CapibaraPreprocessingConfig:
    """Unified preprocessing configuration for Capibara-6."""
    
    # Global settings
    enabled: bool = True
    profile_name: str = "default"
    output_path: Optional[str] = None
    save_filtered_docs: bool = True
    
    # Pipeline components
    use_deduplication: bool = True
    use_quality_filter: bool = True
    use_tpu_optimization: bool = True
    
    # Deduplication configuration
    deduplication: Optional[Dict[str, Any]] = None
    
    # Quality filter configuration
    quality: Optional[Dict[str, Any]] = None
    
    # TPU optimization configuration
    tpu_optimization: Optional[Dict[str, Any]] = None
    
    # Performance settings
    batch_processing: bool = True
    parallel_workers: int = 8
    memory_limit_gb: float = 32.0
    
    # Monitoring and logging
    log_level: str = "INFO"
    metrics_enabled: bool = True
    detailed_stats: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CapibaraPreprocessingConfig':
        """Creates from dictionary."""
        return cls(**data)
    
    def get_dedup_config(self):
        """Get deduplication configuration."""
        if not PREPROCESSING_AVAILABLE or DedupConfig is None:
            raise ImportError("Deduplication components not available")
        
        dedup_params = self.deduplication or {}
        return DedupConfig(**dedup_params)
    
    def get_quality_config(self):
        """Get quality filter configuration."""
        if not PREPROCESSING_AVAILABLE or QualityConfig is None:
            raise ImportError("Quality filter components not available")
        
        quality_params = self.quality or {}
        return QualityConfig(**quality_params)
    
    def get_tpu_config(self):
        """Get TPU optimization configuration."""
        if not PREPROCESSING_AVAILABLE or TPUOptimizationConfig is None:
            raise ImportError("TPU optimization components not available")
        
        tpu_params = self.tpu_optimization or {}
        return TPUOptimizationConfig(**tpu_params)


class PreprocessingProfileManager:
    """Manages preprocessing configuration profiles."""
    
    # Predefined profiles for different use cases
    PROFILES = {
        "aggressive": CapibaraPreprocessingConfig(
            profile_name="aggressive",
            deduplication={
                "jaccard_threshold": 0.85,
                "cos_threshold": 0.90,
                "quality_threshold": 0.45
            },
            quality={
                "min_quality_score": 0.45,
                "max_digit_ratio": 0.2,
                "max_punct_ratio": 0.15,
                "use_toxicity_filter": True
            }
        ),
        
        "balanced": CapibaraPreprocessingConfig(
            profile_name="balanced",
            deduplication={
                "jaccard_threshold": 0.90,
                "cos_threshold": 0.94,
                "quality_threshold": 0.40
            },
            quality={
                "min_quality_score": 0.35,
                "max_digit_ratio": 0.3,
                "max_punct_ratio": 0.2,
                "use_toxicity_filter": False
            }
        ),
        
        "conservative": CapibaraPreprocessingConfig(
            profile_name="conservative",
            deduplication={
                "jaccard_threshold": 0.95,
                "cos_threshold": 0.97,
                "quality_threshold": 0.30
            },
            quality={
                "min_quality_score": 0.25,
                "max_digit_ratio": 0.4,
                "max_punct_ratio": 0.3,
                "use_toxicity_filter": False
            }
        ),
        
        "tpu_optimized": CapibaraPreprocessingConfig(
            profile_name="tpu_optimized",
            use_tpu_optimization=True,
            deduplication={
                "tpu_optimized": True,
                "batch_size_embedding": 1024,
                "parallel_processing": True
            },
            tpu_optimization={
                "batch_size": 2048,
                "max_workers": 32,
                "memory_efficient_mode": True,
                "adaptive_batching": True
            }
        )
    }
    
    @classmethod
    def get_profile(cls, profile_name: str) -> CapibaraPreprocessingConfig:
        """Get a predefined profile."""
        if profile_name not in cls.PROFILES:
            raise ValueError(f"Unknown profile: {profile_name}. Available: {list(cls.PROFILES.keys())}")
        
        return cls.PROFILES[profile_name]
    
    @classmethod
    def list_profiles(cls) -> List[str]:
        """List available profiles."""
        return list(cls.PROFILES.keys())
    
    @classmethod
    def create_custom_profile(cls, base_profile: str, overrides: Dict[str, Any]) -> CapibaraPreprocessingConfig:
        """Creates a custom profile based on an existing one."""
        base_config = cls.get_profile(base_profile)
        config_dict = base_config.to_dict()
        
        # Apply overrides
        for key, value in overrides.items():
            if key in config_dict:
                config_dict[key] = value
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        return CapibaraPreprocessingConfig.from_dict(config_dict)


class PreprocessingMetrics:
    """Collects and manages preprocessing metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.start_time = time.time()
        self.end_time = None
        
        self.input_stats = {
            'total_documents': 0,
            'total_tokens': 0,
            'avg_doc_length': 0.0,
            'size_mb': 0.0
        }
        
        self.output_stats = {
            'total_documents': 0,
            'total_tokens': 0,
            'avg_doc_length': 0.0,
            'size_mb': 0.0
        }
        
        self.processing_stats = {
            'reduction_rate': 0.0,
            'tokens_saved': 0,
            'processing_time': 0.0,
            'throughput_docs_per_sec': 0.0
        }
        
        self.component_stats = {}
    
    def record_input(self, docs: List[Dict]):
        """Record input statistics."""
        self.input_stats['total_documents'] = len(docs)
        
        if docs:
            doc_lengths = [len(doc.get('text', '')) for doc in docs]
            self.input_stats['total_tokens'] = sum(doc_lengths)
            self.input_stats['avg_doc_length'] = sum(doc_lengths) / len(doc_lengths)
            self.input_stats['size_mb'] = self.input_stats['total_tokens'] / 1024 / 1024
    
    def record_output(self, docs: List[Dict]):
        """Record output statistics."""
        self.end_time = time.time()
        
        self.output_stats['total_documents'] = len(docs)
        
        if docs:
            doc_lengths = [len(doc.get('text_norm', doc.get('text', ''))) for doc in docs]
            self.output_stats['total_tokens'] = sum(doc_lengths)
            self.output_stats['avg_doc_length'] = sum(doc_lengths) / len(doc_lengths)
            self.output_stats['size_mb'] = self.output_stats['total_tokens'] / 1024 / 1024
        
        # Calculate derived metrics
        self._calculate_processing_stats()
    
    def record_component_stats(self, component: str, stats: Dict[str, Any]):
        """Record statistics from a processing component."""
        self.component_stats[component] = stats
    
    def _calculate_processing_stats(self):
        """Calculate processing statistics."""
        if self.input_stats['total_documents'] > 0:
            self.processing_stats['reduction_rate'] = 1.0 - (
                self.output_stats['total_documents'] / self.input_stats['total_documents']
            )
        
        self.processing_stats['tokens_saved'] = (
            self.input_stats['total_tokens'] - self.output_stats['total_tokens']
        )
        
        if self.end_time and self.start_time:
            self.processing_stats['processing_time'] = self.end_time - self.start_time
            
            if self.processing_stats['processing_time'] > 0:
                self.processing_stats['throughput_docs_per_sec'] = (
                    self.input_stats['total_documents'] / self.processing_stats['processing_time']
                )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            'input': self.input_stats,
            'output': self.output_stats,
            'processing': self.processing_stats,
            'components': self.component_stats,
            'reduction_summary': {
                'documents_removed': self.input_stats['total_documents'] - self.output_stats['total_documents'],
                'tokens_saved': self.processing_stats['tokens_saved'],
                'size_reduction_mb': self.input_stats['size_mb'] - self.output_stats['size_mb'],
                'reduction_percentage': self.processing_stats['reduction_rate'] * 100
            }
        }


class CapibaraPreprocessor:
    """
    Main preprocessing interface for Capibara-6.
    
    Integrates with the existing training system and provides
    a unified interface for all data preprocessing operations.
    """
    
    def __init__(self, config: Optional[Union[CapibaraPreprocessingConfig, Dict[str, Any], str]] = None):
        # Handle different config types
        if isinstance(config, str):
            # Profile name
            self.config = PreprocessingProfileManager.get_profile(config)
        elif isinstance(config, dict):
            # Dictionary config
            self.config = CapibaraPreprocessingConfig.from_dict(config)
        elif isinstance(config, CapibaraPreprocessingConfig):
            # Already a config object
            self.config = config
        else:
            # Default config
            self.config = CapibaraPreprocessingConfig()
        
        # Initialize components
        self.pipeline = None
        self.metrics = PreprocessingMetrics()
        
        # Check availability
        self.available = PREPROCESSING_AVAILABLE
        
        if self.available:
            self._initialize_pipeline()
        else:
            logger.warning("Preprocessing components not available - running in passthrough mode")
        
        logger.info(f"🔄 CapibaraPreprocessor initialized with profile: {self.config.profile_name}")
    
    def _initialize_pipeline(self):
        """Initialize the preprocessing pipeline."""
        if not self.available:
            return
        
        try:
            pipeline_config = {
                'deduplication': self.config.deduplication or {},
                'quality': self.config.quality or {},
                'tpu_optimization': self.config.tpu_optimization or {}
            }
            
            self.pipeline = DataPreprocessingPipeline(pipeline_config)
            logger.info("✅ Preprocessing pipeline initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize preprocessing pipeline: {e}")
            self.available = False
    
    def preprocess_dataset(self, docs: List[Dict], 
                          custom_config: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Preprocess a dataset using the configured pipeline.
        
        Args:
            docs: List of documents to preprocess
            custom_config: Optional configuration overrides
            
        Returns:
            Preprocessed documents
        """
        logger.info(f"🚀 Starting preprocessing of {len(docs)} documents")
        
        # Record input metrics
        self.metrics.reset()
        self.metrics.record_input(docs)
        
        # Check if preprocessing is available and enabled
        if not self.available or not self.config.enabled:
            logger.info("Preprocessing disabled or unavailable - returning original documents")
            self.metrics.record_output(docs)
            return docs
        
        try:
            # Merge custom config with default
            pipeline_config = {
                'use_deduplication': self.config.use_deduplication,
                'use_quality_filter': self.config.use_quality_filter,
                'use_tpu_optimization': self.config.use_tpu_optimization
            }
            
            if custom_config:
                pipeline_config.update(custom_config)
            
            # Process documents
            processed_docs = self.pipeline.process_dataset(docs, pipeline_config)
            
            # Collect component metrics
            pipeline_stats = self.pipeline.get_pipeline_stats()
            self.metrics.record_component_stats('pipeline', pipeline_stats)
            
            # Record output metrics
            self.metrics.record_output(processed_docs)
            
            # Save results if configured
            if self.config.save_filtered_docs and self.config.output_path:
                self._save_results(processed_docs)
            
            logger.info(f"✅ Preprocessing completed: {len(processed_docs)} documents")
            self._log_metrics()
            
            return processed_docs
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            # Return original documents on failure
            self.metrics.record_output(docs)
            return docs
    
    def _save_results(self, docs: List[Dict]):
        """Save preprocessing results."""
        try:
            import json
            from pathlib import Path
            
            output_path = Path(self.config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save documents
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(docs, f, ensure_ascii=False, indent=2)
            
            # Save metrics
            metrics_path = output_path.with_suffix('.metrics.json')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(self.metrics.get_summary(), f, indent=2)
            
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")
    
    def _log_metrics(self):
        """Log preprocessing metrics."""
        summary = self.metrics.get_summary()
        
        logger.info("📊 Preprocessing Summary:")
        logger.info(f"  Input documents: {summary['input']['total_documents']:,}")
        logger.info(f"  Output documents: {summary['output']['total_documents']:,}")
        logger.info(f"  Documents removed: {summary['reduction_summary']['documents_removed']:,}")
        logger.info(f"  Reduction rate: {summary['reduction_summary']['reduction_percentage']:.1f}%")
        logger.info(f"  Tokens saved: {summary['reduction_summary']['tokens_saved']:,}")
        logger.info(f"  Processing time: {summary['processing']['processing_time']:.2f}s")
        logger.info(f"  Throughput: {summary['processing']['throughput_docs_per_sec']:.1f} docs/sec")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get preprocessing metrics."""
        return self.metrics.get_summary()
    
    def get_status(self) -> Dict[str, Any]:
        """Get preprocessor status."""
        return {
            'available': self.available,
            'enabled': self.config.enabled,
            'profile': self.config.profile_name,
            'components_available': {
                'deduplication': DEDUPLICATION_AVAILABLE,
                'quality_filter': QUALITY_FILTER_AVAILABLE,
                'tpu_processor': TPU_PROCESSOR_AVAILABLE
            },
            'pipeline_ready': self.pipeline is not None
        }


# Integration functions for existing training system
def integrate_with_unified_training(training_system, preprocessor_config: Optional[Dict[str, Any]] = None):
    """
    Integrate preprocessing with the UnifiedTrainingSystem.
    
    Args:
        training_system: Instance of UnifiedTrainingSystem
        preprocessor_config: Optional preprocessor configuration
    """
    try:
        # Add preprocessing capability to training system
        preprocessor = CapibaraPreprocessor(preprocessor_config)
        
        # Monkey patch the training system to include preprocessing
        original_create_session = training_system.create_training_session
        
        def create_training_session_with_preprocessing(session_config):
            # Check if preprocessing is requested
            if session_config.get('use_preprocessing', True):
                logger.info("🔄 Preprocessing enabled for training session")
                
                # Add preprocessor to session config
                session_config['preprocessor'] = preprocessor
            
            return original_create_session(session_config)
        
        training_system.create_training_session = create_training_session_with_preprocessing
        training_system.preprocessor = preprocessor
        
        logger.info("✅ Preprocessing integrated with UnifiedTrainingSystem")
        
    except Exception as e:
        logger.error(f"Failed to integrate preprocessing with training system: {e}")


def create_preprocessing_config_for_training(training_type: str = "balanced") -> Dict[str, Any]:
    """Creates preprocessing configuration optimized for specific training types."""
    
    configs = {
        "tpu_v6e": {
            "profile": "tpu_optimized",
            "use_tpu_optimization": True,
            "tpu_optimization": {
                "batch_size": 2048,
                "tpu_cores": 64,
                "memory_per_core_gb": 32.0,
                "adaptive_batching": True
            }
        },
        
        "ultra_training": {
            "profile": "aggressive",
            "use_deduplication": True,
            "use_quality_filter": True,
            "deduplication": {
                "jaccard_threshold": 0.85,
                "cos_threshold": 0.90,
                "tpu_optimized": True
            }
        },
        
        "balanced": {
            "profile": "balanced",
            "use_deduplication": True,
            "use_quality_filter": True,
            "use_tpu_optimization": True
        },
        
        "conservative": {
            "profile": "conservative",
            "use_deduplication": False,
            "use_quality_filter": True,
            "quality": {
                "min_quality_score": 0.2
            }
        }
    }
    
    return configs.get(training_type, configs["balanced"])


# Module exports and convenience functions
__all__ = [
    'CapibaraPreprocessingConfig',
    'PreprocessingProfileManager', 
    'CapibaraPreprocessor',
    'PreprocessingMetrics',
    'integrate_with_unified_training',
    'create_preprocessing_config_for_training',
    'PREPROCESSING_AVAILABLE'
]
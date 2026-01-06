"""
Data Preprocessing Module for Capibara-6

This module provides advanced data preprocessing capabilities including:
- Semantic deduplication with LSH and embeddings
- Quality filtering with heuristics and language detection
- TPU v6e-64 optimized processing
- Integration with Capibara-6 training pipeline

Components:
- SemanticDeduplicator: Main deduplication engine
- QualityFilter: Content quality assessment
- TPUOptimizedProcessor: TPU-specific optimizations
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Component availability flags
DEDUPLICATION_AVAILABLE = False
QUALITY_FILTER_AVAILABLE = False
TPU_PROCESSOR_AVAILABLE = False

# Try to import components
try:
    from .semantic_deduplicator import SemanticDeduplicator, DedupConfig
    DEDUPLICATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Semantic deduplicator not available: {e}")

try:
    from .quality_filter import QualityFilter, QualityConfig
    QUALITY_FILTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Quality filter not available: {e}")

try:
    from .tpu_optimized_processor import TPUOptimizedProcessor
    TPU_PROCESSOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TPU processor not available: {e}")


class DataPreprocessingPipeline:
    """
    Unified data preprocessing pipeline for Capibara-6.
    
    Integrates semantic deduplication, quality filtering, and TPU optimizations
    into a cohesive preprocessing system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.deduplicator = None
        self.quality_filter = None
        self.tpu_processor = None
        
        self.available_components = {
            'deduplication': DEDUPLICATION_AVAILABLE,
            'quality_filter': QUALITY_FILTER_AVAILABLE,
            'tpu_processor': TPU_PROCESSOR_AVAILABLE
        }
        
        self._initialize_components()
        
        logger.info("🔄 Data Preprocessing Pipeline initialized")
        logger.info(f"Available components: {sum(self.available_components.values())}/3")
    
    def _initialize_components(self):
        """Initialize available preprocessing components."""
        
        if DEDUPLICATION_AVAILABLE:
            try:
                dedup_config = DedupConfig(**self.config.get('deduplication', {}))
                self.deduplicator = SemanticDeduplicator(dedup_config)
                logger.info("✅ Semantic deduplicator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize deduplicator: {e}")
        
        if QUALITY_FILTER_AVAILABLE:
            try:
                quality_config = QualityConfig(**self.config.get('quality', {}))
                self.quality_filter = QualityFilter(quality_config)
                logger.info("✅ Quality filter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize quality filter: {e}")
        
        if TPU_PROCESSOR_AVAILABLE:
            try:
                tpu_config = self.config.get('tpu_optimization', {})
                self.tpu_processor = TPUOptimizedProcessor(tpu_config)
                logger.info("✅ TPU processor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize TPU processor: {e}")
    
    def process_dataset(self, docs: List[Dict], pipeline_config: Optional[Dict] = None) -> List[Dict]:
        """
        Process a dataset through the complete preprocessing pipeline.
        
        Args:
            docs: List of documents with 'text' field
            pipeline_config: Optional configuration overrides
            
        Returns:
            Processed and deduplicated documents
        """
        config = pipeline_config or {}
        
        logger.info(f"🚀 Starting preprocessing pipeline with {len(docs)} documents")
        
        # Step 1: Quality filtering (if available)
        if self.quality_filter and config.get('use_quality_filter', True):
            docs = self.quality_filter.filter_documents(docs)
            logger.info(f"📊 After quality filtering: {len(docs)} documents")
        
        # Step 2: Semantic deduplication (if available)
        if self.deduplicator and config.get('use_deduplication', True):
            docs = self.deduplicator.deduplicate_documents(docs)
            logger.info(f"🔄 After deduplication: {len(docs)} documents")
        
        # Step 3: TPU optimization (if available)
        if self.tpu_processor and config.get('use_tpu_optimization', True):
            docs = self.tpu_processor.optimize_for_tpu(docs)
            logger.info(f"⚡ After TPU optimization: {len(docs)} documents")
        
        logger.info(f"✅ Preprocessing completed: {len(docs)} documents ready for training")
        
        return docs
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        
        stats = {
            'available_components': self.available_components,
            'pipeline_health': self._get_pipeline_health()
        }
        
        if self.deduplicator:
            stats['deduplication'] = self.deduplicator.get_stats()
        
        if self.quality_filter:
            stats['quality_filter'] = self.quality_filter.get_stats()
        
        if self.tpu_processor:
            stats['tpu_processor'] = self.tpu_processor.get_stats()
        
        return stats
    
    def _get_pipeline_health(self) -> str:
        """Get overall pipeline health status."""
        available_count = sum(self.available_components.values())
        
        if available_count == 3:
            return 'excellent'
        elif available_count == 2:
            return 'good'
        elif available_count == 1:
            return 'limited'
        else:
            return 'unavailable'


# Module exports
__all__ = [
    'DataPreprocessingPipeline',
    'SemanticDeduplicator',
    'QualityFilter',
    'TPUOptimizedProcessor',
    'DedupConfig',
    'QualityConfig',
    'DEDUPLICATION_AVAILABLE',
    'QUALITY_FILTER_AVAILABLE',
    'TPU_PROCESSOR_AVAILABLE'
]

# Version info
__version__ = "1.0.0"
__author__ = "Capibara-6 Team"
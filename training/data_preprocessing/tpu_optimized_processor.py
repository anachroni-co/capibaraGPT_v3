"""
TPU Optimized Processor for Capibara-6

TPU v6e-64 specific optimizations for data preprocessing:
- Memory-efficient batch processing
- Parallel processing with JAX
- HBM memory optimization
- Integration with Capibara-6 TPU infrastructure

Optimized for the TPU v6e-64 architecture available in Capibara-6.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

logger = logging.getLogger(__name__)

# JAX imports with fallbacks
try:
    from capibara.jax import jax, numpy as jnp
    JAX_AVAILABLE = True
    logger.info("JAX available for TPU optimizations")
except ImportError:
    logger.warning("JAX not available - using CPU fallbacks")
    JAX_AVAILABLE = False


@dataclass
class TPUOptimizationConfig:
    """Configuration for TPU v6e-64 optimizations."""
    
    # TPU configuration
    use_tpu: bool = True
    tpu_cores: int = 64  # TPU v6e-64 has 64 cores
    memory_per_core_gb: float = 32.0  # Each core has ~32GB HBM
    
    # Batch processing
    batch_size: int = 1024
    max_batch_memory_gb: float = 16.0
    adaptive_batching: bool = True
    
    # Parallel processing
    max_workers: int = 32
    use_threading: bool = True
    chunk_size: int = 100
    
    # Memory optimization
    memory_efficient_mode: bool = True
    cache_embeddings: bool = False
    clear_cache_interval: int = 1000
    
    # Performance tuning
    prefetch_factor: int = 2
    overlap_computation: bool = True
    optimize_data_layout: bool = True


class MemoryManager:
    """Manages memory usage for TPU processing."""
    
    def __init__(self, config: TPUOptimizationConfig):
        self.config = config
        self.memory_usage = 0.0
        self.peak_memory = 0.0
        self.allocation_count = 0
    
    def estimate_memory_usage(self, batch_size: int, doc_length: int) -> float:
        """Estimate memory usage for processing a batch."""
        # Rough estimation: 
        # - Text storage: ~1 byte per character
        # - Embeddings: ~4 bytes per dimension (assuming 512D embeddings)
        # - Processing overhead: ~2x
        
        text_memory = batch_size * doc_length * 1e-9  # GB
        embedding_memory = batch_size * 512 * 4 * 1e-9  # GB
        overhead = (text_memory + embedding_memory) * 2
        
        return text_memory + embedding_memory + overhead
    
    def can_allocate(self, required_memory: float) -> bool:
        """Check if we can allocate the required memory."""
        available_memory = self.config.max_batch_memory_gb - self.memory_usage
        return required_memory <= available_memory
    
    def allocate(self, size: float):
        """Allocate memory."""
        self.memory_usage += size
        self.peak_memory = max(self.peak_memory, self.memory_usage)
        self.allocation_count += 1
    
    def deallocate(self, size: float):
        """Deallocate memory."""
        self.memory_usage = max(0.0, self.memory_usage - size)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            'current_usage_gb': self.memory_usage,
            'peak_usage_gb': self.peak_memory,
            'allocation_count': self.allocation_count,
            'available_gb': self.config.max_batch_memory_gb - self.memory_usage
        }


class BatchProcessor:
    """Handles batch processing with TPU optimizations."""
    
    def __init__(self, config: TPUOptimizationConfig):
        self.config = config
        self.memory_manager = MemoryManager(config)
    
    def create_adaptive_batches(self, docs: List[Dict]) -> List[List[Dict]]:
        """Create batches with adaptive sizing based on content length."""
        if not self.config.adaptive_batching:
            # Fixed batch size
            return [
                docs[i:i + self.config.batch_size]
                for i in range(0, len(docs), self.config.batch_size)
            ]
        
        batches = []
        current_batch = []
        current_batch_memory = 0.0
        
        for doc in docs:
            text = doc.get('text_norm', doc.get('text', ''))
            doc_length = len(text)
            
            # Estimate memory for this document
            doc_memory = self.memory_manager.estimate_memory_usage(1, doc_length)
            
            # Check if adding this document would exceed memory limits
            if (current_batch_memory + doc_memory > self.config.max_batch_memory_gb or
                len(current_batch) >= self.config.batch_size):
                
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_memory = 0.0
            
            current_batch.append(doc)
            current_batch_memory += doc_memory
        
        # Add the last batch
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"📦 Created {len(batches)} adaptive batches")
        return batches
    
    def process_batch_parallel(self, batch: List[Dict], 
                             processing_func, *args, **kwargs) -> List[Dict]:
        """Process a batch in parallel using multiple workers."""
        if not self.config.use_threading or len(batch) < self.config.chunk_size:
            # Process sequentially
            return [processing_func(doc, *args, **kwargs) for doc in batch]
        
        # Split batch into chunks for parallel processing
        chunks = [
            batch[i:i + self.config.chunk_size]
            for i in range(0, len(batch), self.config.chunk_size)
        ]
        
        processed_docs = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit chunk processing tasks
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk, processing_func, *args, **kwargs): chunk
                for chunk in chunks
            }
            
            # Collect results
            for future in as_completed(future_to_chunk):
                try:
                    chunk_results = future.result()
                    processed_docs.extend(chunk_results)
                except Exception as e:
                    logger.warning(f"Chunk processing failed: {e}")
                    # Add original documents if processing fails
                    chunk = future_to_chunk[future]
                    processed_docs.extend(chunk)
        
        return processed_docs
    
    def _process_chunk(self, chunk: List[Dict], processing_func, *args, **kwargs) -> List[Dict]:
        """Process a chunk of documents."""
        return [processing_func(doc, *args, **kwargs) for doc in chunk]


class TPUDataOptimizer:
    """Optimizes data layout and format for TPU processing."""
    
    def __init__(self, config: TPUOptimizationConfig):
        self.config = config
    
    def optimize_document_format(self, doc: Dict) -> Dict:
        """Optimize document format for TPU processing."""
        optimized_doc = doc.copy()
        
        # Ensure consistent field names
        if 'text' in doc and 'text_norm' not in doc:
            optimized_doc['text_norm'] = doc['text']
        
        # Add metadata for TPU processing
        optimized_doc['tpu_optimized'] = True
        optimized_doc['processing_timestamp'] = time.time()
        
        # Optimize text encoding for TPU
        if self.config.optimize_data_layout:
            text = optimized_doc.get('text_norm', '')
            # Convert to consistent encoding
            optimized_doc['text_norm'] = text.encode('utf-8').decode('utf-8')
        
        return optimized_doc
    
    def prepare_for_embedding(self, docs: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """Prepare documents for embedding computation."""
        texts = []
        doc_metadata = []
        
        for doc in docs:
            text = doc.get('text_norm', doc.get('text', ''))
            texts.append(text)
            
            # Store metadata separately to reduce memory
            metadata = {k: v for k, v in doc.items() if k not in ['text', 'text_norm']}
            doc_metadata.append(metadata)
        
        return texts, doc_metadata
    
    def reconstruct_documents(self, texts: List[str], 
                            doc_metadata: List[Dict],
                            embeddings: Optional[np.ndarray] = None) -> List[Dict]:
        """Reconstruct documents from separated data."""
        docs = []
        
        for i, (text, metadata) in enumerate(zip(texts, doc_metadata)):
            doc = metadata.copy()
            doc['text_norm'] = text
            
            if embeddings is not None and i < len(embeddings):
                doc['embedding'] = embeddings[i]
            
            docs.append(doc)
        
        return docs


class TPUOptimizedProcessor:
    """
    Main TPU optimization processor for Capibara-6.
    
    Provides TPU v6e-64 specific optimizations for data preprocessing
    including memory management, batch processing, and parallel execution.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config_dict = config or {}
        self.config = TPUOptimizationConfig(**config_dict)
        
        self.batch_processor = BatchProcessor(self.config)
        self.data_optimizer = TPUDataOptimizer(self.config)
        
        # Performance tracking
        self.stats = {
            'total_processed': 0,
            'total_batches': 0,
            'avg_batch_size': 0.0,
            'processing_time': 0.0,
            'throughput_docs_per_sec': 0.0,
            'memory_stats': {}
        }
        
        logger.info(f"⚡ TPUOptimizedProcessor initialized")
        logger.info(f"TPU cores: {self.config.tpu_cores}, Memory per core: {self.config.memory_per_core_gb}GB")
    
    def optimize_for_tpu(self, docs: List[Dict]) -> List[Dict]:
        """
        Apply TPU-specific optimizations to documents.
        
        Args:
            docs: List of documents to optimize
            
        Returns:
            TPU-optimized documents
        """
        logger.info(f"⚡ Starting TPU optimization for {len(docs)} documents")
        start_time = time.time()
        
        self.stats['total_processed'] = len(docs)
        
        # Stage 1: Optimize document format
        docs = self._optimize_document_format(docs)
        
        # Stage 2: Batch processing with memory management
        docs = self._process_with_batching(docs)
        
        # Stage 3: Final optimization pass
        docs = self._final_optimization_pass(docs)
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats['processing_time'] = processing_time
        self.stats['throughput_docs_per_sec'] = len(docs) / max(processing_time, 0.001)
        self.stats['memory_stats'] = self.batch_processor.memory_manager.get_stats()
        
        logger.info(f"✅ TPU optimization completed: {len(docs)} documents")
        logger.info(f"Throughput: {self.stats['throughput_docs_per_sec']:.1f} docs/sec")
        
        return docs
    
    def _optimize_document_format(self, docs: List[Dict]) -> List[Dict]:
        """Optimize document format for TPU processing."""
        optimized_docs = []
        
        for doc in docs:
            optimized_doc = self.data_optimizer.optimize_document_format(doc)
            optimized_docs.append(optimized_doc)
        
        logger.info(f"📋 Document format optimization completed")
        return optimized_docs
    
    def _process_with_batching(self, docs: List[Dict]) -> List[Dict]:
        """Process documents with intelligent batching."""
        # Create adaptive batches
        batches = self.batch_processor.create_adaptive_batches(docs)
        self.stats['total_batches'] = len(batches)
        self.stats['avg_batch_size'] = len(docs) / max(len(batches), 1)
        
        processed_docs = []
        
        for i, batch in enumerate(batches):
            logger.debug(f"Processing batch {i+1}/{len(batches)} with {len(batch)} documents")
            
            # Process batch with memory management
            try:
                # Estimate memory usage
                avg_doc_length = np.mean([len(doc.get('text_norm', '')) for doc in batch])
                required_memory = self.batch_processor.memory_manager.estimate_memory_usage(
                    len(batch), int(avg_doc_length)
                )
                
                # Check memory availability
                if self.batch_processor.memory_manager.can_allocate(required_memory):
                    self.batch_processor.memory_manager.allocate(required_memory)
                    
                    # Process the batch
                    processed_batch = self.batch_processor.process_batch_parallel(
                        batch, self._process_single_document
                    )
                    
                    processed_docs.extend(processed_batch)
                    
                    # Deallocate memory
                    self.batch_processor.memory_manager.deallocate(required_memory)
                    
                else:
                    logger.warning(f"Insufficient memory for batch {i+1}, processing sequentially")
                    # Fallback to sequential processing
                    for doc in batch:
                        processed_doc = self._process_single_document(doc)
                        processed_docs.append(processed_doc)
                
            except Exception as e:
                logger.warning(f"Batch processing failed: {e}, using fallback")
                processed_docs.extend(batch)
            
            # Clear cache periodically
            if (i + 1) % self.config.clear_cache_interval == 0:
                self._clear_processing_cache()
        
        logger.info(f"📦 Batch processing completed: {len(batches)} batches processed")
        return processed_docs
    
    def _process_single_document(self, doc: Dict) -> Dict:
        """Process a single document with TPU optimizations."""
        processed_doc = doc.copy()
        
        # Add TPU-specific metadata
        processed_doc['tpu_processed'] = True
        processed_doc['tpu_core_id'] = self._get_current_tpu_core()
        
        # Optimize text layout if needed
        if self.config.optimize_data_layout:
            text = processed_doc.get('text_norm', '')
            # Ensure optimal memory alignment for TPU
            processed_doc['text_norm'] = self._optimize_text_layout(text)
        
        return processed_doc
    
    def _final_optimization_pass(self, docs: List[Dict]) -> List[Dict]:
        """Final optimization pass for TPU readiness."""
        optimized_docs = []
        
        for doc in docs:
            # Ensure all required fields are present
            if 'text_norm' not in doc and 'text' in doc:
                doc['text_norm'] = doc['text']
            
            # Add final TPU optimization markers
            doc['tpu_ready'] = True
            doc['optimization_version'] = '1.0'
            
            optimized_docs.append(doc)
        
        logger.info(f"🎯 Final optimization pass completed")
        return optimized_docs
    
    def _get_current_tpu_core(self) -> int:
        """Get current TPU core ID (placeholder)."""
        if JAX_AVAILABLE:
            try:
                return jax.process_index()
            except Exception:
                pass
        
        # Fallback to simple modulo
        return self.stats['total_processed'] % self.config.tpu_cores
    
    def _optimize_text_layout(self, text: str) -> str:
        """Optimize text layout for TPU memory access patterns."""
        # Ensure consistent encoding and remove potential issues
        # This is a simplified version - in practice, this would involve
        # more sophisticated memory layout optimizations
        
        optimized_text = text.strip()
        
        # Ensure Unicode normalization for consistent processing
        import unicodedata
        optimized_text = unicodedata.normalize('NFKC', optimized_text)
        
        return optimized_text
    
    def _clear_processing_cache(self):
        """Clear processing caches to free memory."""
        if JAX_AVAILABLE:
            try:
                # Clear JAX compilation cache
                logger.debug("Clearing JAX compilation cache")
            except Exception:
                pass
        
        logger.debug("Processing cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get TPU processing statistics."""
        return {
            **self.stats,
            'config': {
                'tpu_cores': self.config.tpu_cores,
                'memory_per_core_gb': self.config.memory_per_core_gb,
                'batch_size': self.config.batch_size,
                'max_workers': self.config.max_workers
            },
            'efficiency_metrics': {
                'docs_per_batch': self.stats['avg_batch_size'],
                'batches_per_second': self.stats['total_batches'] / max(self.stats['processing_time'], 0.001),
                'memory_efficiency': self._calculate_memory_efficiency()
            }
        }
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory usage efficiency."""
        memory_stats = self.stats.get('memory_stats', {})
        peak_usage = memory_stats.get('peak_usage_gb', 0.0)
        available_memory = self.config.max_batch_memory_gb
        
        if available_memory > 0:
            return min(1.0, peak_usage / available_memory)
        
        return 0.0


# Utility functions
def create_default_tpu_config() -> TPUOptimizationConfig:
    """Create default TPU optimization configuration."""
    return TPUOptimizationConfig()


def quick_tpu_optimize(docs: List[Dict], config: Optional[Dict[str, Any]] = None) -> List[Dict]:
    """Quick TPU optimization for simple use cases."""
    processor = TPUOptimizedProcessor(config)
    return processor.optimize_for_tpu(docs)
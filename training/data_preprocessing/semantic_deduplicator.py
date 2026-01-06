"""
Semantic Deduplicator for Capibara-6

Advanced deduplication system with:
- Exact deduplication (hash-based)
- Near-duplicate detection (LSH MinHash)
- Semantic deduplication (embeddings + FAISS)
- TPU v6e-64 optimizations
- Integration with Capibara-6 systems

Adapted from the provided deduplication pipeline with Capibara-6 specific optimizations.
"""

import re
import regex as re2
import xxhash
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Conditional imports with fallbacks
try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    logger.warning("datasketch not available - LSH deduplication disabled")
    DATASKETCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available - semantic deduplication disabled")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("faiss not available - semantic search disabled")
    FAISS_AVAILABLE = False

try:
    import langid
    LANGID_AVAILABLE = True
except ImportError:
    logger.warning("langid not available - language detection disabled")
    LANGID_AVAILABLE = False


@dataclass
class DedupConfig:
    """Configuration for semantic deduplication adapted for Capibara-6."""
    
    # Deduplication thresholds
    jaccard_threshold: float = 0.90
    cos_threshold: float = 0.94  # Slightly higher for better precision
    
    # Content filtering
    min_len_chars: int = 200
    max_len_chars: int = 40000
    keep_langs: Tuple[str, ...] = ("es", "en", "pt", "ca")  # Extended for Capibara-6
    
    # Model configuration
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size_embedding: int = 512
    
    # TPU optimizations
    tpu_optimized: bool = True
    parallel_processing: bool = True
    memory_efficient: bool = True
    
    # Capibara-6 specific
    use_moe_filtering: bool = True
    quality_threshold: float = 0.40
    preserve_high_quality: bool = True
    
    # Advanced options
    shingle_size: int = 13
    num_perm: int = 128
    percentile_threshold: float = 99.9


class TextNormalizer:
    """Text normalization utilities for consistent processing."""
    
    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for deduplication."""
        # Remove HTML tags
        text = re2.sub(r"<[^>]+>", " ", text)
        
        # Normalize whitespace
        text = re2.sub(r"\s+", " ", text).strip()
        
        # Optional: normalize quotes and special characters
        text = re2.sub(r"[""''`´]", '"', text)
        text = re2.sub(r"[–—−]", "-", text)
        
        return text
    
    @staticmethod
    def doc_hash(text: str) -> str:
        """Generate hash for exact deduplication."""
        return xxhash.xxh3_128_hexdigest(text.encode('utf-8'))
    
    @staticmethod
    def shingles(text: str, k: int = 13) -> Iterable[str]:
        """Generate character shingles for LSH."""
        if len(text) <= k:
            yield text
        else:
            for i in range(len(text) - k + 1):
                yield text[i:i+k]


class QualityScorer:
    """Content quality assessment for filtering."""
    
    @staticmethod
    def quality_score(text: str) -> float:
        """Calculate quality score using multiple heuristics."""
        if not text:
            return 0.0
        
        n = max(len(text), 1)
        
        # Punctuation ratio (penalize excessive punctuation)
        punc = sum(ch in r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""" for ch in text) / n
        punc_penalty = 0.8 * max(0, punc - 0.15)
        
        # Digit ratio (penalize excessive digits)
        digits = sum(ch.isdigit() for ch in text) / n
        digit_penalty = 0.5 * max(0, digits - 0.1)
        
        # Uppercase ratio (penalize excessive uppercase)
        upper = sum(ch.isupper() for ch in text) / n
        upper_penalty = 0.3 * max(0, upper - 0.2)
        
        # Trigram diversity (reward lexical diversity)
        trigrams = [text[i:i+3] for i in range(len(text)-2)]
        diversity = len(set(trigrams)) / max(len(trigrams), 1) if trigrams else 0
        diversity_bonus = 0.5 * diversity
        
        # Line repetition penalty
        lines = text.split('\n')
        unique_lines = len(set(lines))
        line_diversity = unique_lines / max(len(lines), 1)
        line_bonus = 0.2 * line_diversity
        
        # Final score calculation
        score = 1.0 - (punc_penalty + digit_penalty + upper_penalty) + diversity_bonus + line_bonus
        
        return float(np.clip(score, 0.0, 1.0))


class LanguageDetector:
    """Language detection with fallbacks."""
    
    @staticmethod
    def detect_language(text: str, allowed_langs: Tuple[str, ...] = ("es", "en")) -> bool:
        """Detect if text is in allowed languages."""
        if not LANGID_AVAILABLE:
            # Fallback: simple heuristics
            return LanguageDetector._heuristic_language_check(text, allowed_langs)
        
        try:
            # Use first 5000 chars for detection
            sample = text[:5000]
            lang, confidence = langid.classify(sample)
            
            # Accept if language is allowed and confidence is reasonable
            return lang in allowed_langs and confidence > 0.8
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return LanguageDetector._heuristic_language_check(text, allowed_langs)
    
    @staticmethod
    def _heuristic_language_check(text: str, allowed_langs: Tuple[str, ...]) -> bool:
        """Simple heuristic language check."""
        if not text:
            return False
        
        # Basic checks for Spanish/English
        if "es" in allowed_langs or "en" in allowed_langs:
            # Check for common patterns
            alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
            return alpha_ratio > 0.6  # At least 60% alphabetic characters
        
        return True  # Default to accept if unsure


class SemanticDeduplicator:
    """
    Main semantic deduplication engine for Capibara-6.
    
    Implements a multi-stage deduplication pipeline:
    1. Exact deduplication (hash-based)
    2. Near-duplicate detection (LSH MinHash)
    3. Semantic deduplication (embeddings + FAISS)
    4. Quality filtering
    """
    
    def __init__(self, config: DedupConfig):
        self.config = config
        
        # Initialize components
        self.normalizer = TextNormalizer()
        self.quality_scorer = QualityScorer()
        self.language_detector = LanguageDetector()
        
        # LSH for near-duplicate detection
        self.lsh = None
        self.mh_map: Dict[str, MinHash] = {}
        
        # Embedding model for semantic deduplication
        self.embedding_model = None
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'exact_duplicates_removed': 0,
            'near_duplicates_removed': 0,
            'semantic_duplicates_removed': 0,
            'quality_filtered': 0,
            'language_filtered': 0,
            'final_count': 0
        }
        
        self._initialize_components()
        
        logger.info(f"🔄 SemanticDeduplicator initialized with config: {self.config}")
    
    def _initialize_components(self):
        """Initialize deduplication components."""
        
        # Initialize LSH if available
        if DATASKETCH_AVAILABLE:
            try:
                self.lsh = MinHashLSH(
                    threshold=self.config.jaccard_threshold,
                    num_perm=self.config.num_perm
                )
                logger.info("✅ LSH MinHash initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize LSH: {e}")
        
        # Initialize embedding model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(self.config.embed_model)
                logger.info(f"✅ Embedding model loaded: {self.config.embed_model}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.embedding_model = None
    
    def deduplicate_documents(self, docs: List[Dict]) -> List[Dict]:
        """
        Main deduplication pipeline.
        
        Args:
            docs: List of documents with 'text' field
            
        Returns:
            Deduplicated documents
        """
        logger.info(f"🚀 Starting deduplication of {len(docs)} documents")
        
        self.stats['total_processed'] = len(docs)
        
        # Stage 1: Basic filtering and normalization
        docs = self._normalize_and_filter(docs)
        
        # Stage 2: Exact deduplication
        docs = self._exact_deduplication(docs)
        
        # Stage 3: Near-duplicate detection (LSH)
        if self.lsh is not None:
            docs = self._lsh_deduplication(docs)
        
        # Stage 4: Semantic deduplication
        if self.embedding_model is not None and FAISS_AVAILABLE:
            docs = self._semantic_deduplication(docs)
        
        # Stage 5: Final quality filtering
        docs = self._final_quality_filter(docs)
        
        self.stats['final_count'] = len(docs)
        
        logger.info(f"✅ Deduplication completed: {len(docs)} documents remaining")
        self._log_stats()
        
        return docs
    
    def _normalize_and_filter(self, docs: List[Dict]) -> List[Dict]:
        """Normalize text and apply basic filters."""
        filtered_docs = []
        
        for doc in docs:
            text = doc.get('text', '')
            
            # Normalize text
            normalized_text = self.normalizer.normalize(text)
            
            # Length filtering
            if not (self.config.min_len_chars <= len(normalized_text) <= self.config.max_len_chars):
                continue
            
            # Language filtering
            if not self.language_detector.detect_language(normalized_text, self.config.keep_langs):
                self.stats['language_filtered'] += 1
                continue
            
            # Add normalized text and metadata
            doc['text_norm'] = normalized_text
            doc['hash'] = self.normalizer.doc_hash(normalized_text)
            doc['quality_score'] = self.quality_scorer.quality_score(normalized_text)
            
            filtered_docs.append(doc)
        
        logger.info(f"📊 After normalization and basic filtering: {len(filtered_docs)} documents")
        return filtered_docs
    
    def _exact_deduplication(self, docs: List[Dict]) -> List[Dict]:
        """Remove exact duplicates using hash comparison."""
        seen_hashes = set()
        unique_docs = []
        
        for doc in docs:
            hash_val = doc['hash']
            if hash_val not in seen_hashes:
                seen_hashes.add(hash_val)
                unique_docs.append(doc)
            else:
                self.stats['exact_duplicates_removed'] += 1
        
        logger.info(f"🔍 After exact deduplication: {len(unique_docs)} documents")
        return unique_docs
    
    def _lsh_deduplication(self, docs: List[Dict]) -> List[Dict]:
        """Remove near-duplicates using LSH MinHash."""
        if not DATASKETCH_AVAILABLE:
            return docs
        
        # Generate MinHash signatures
        for doc in docs:
            mh = MinHash(num_perm=self.config.num_perm)
            for shingle in self.normalizer.shingles(doc['text_norm'], self.config.shingle_size):
                mh.update(shingle.encode('utf-8'))
            
            self.mh_map[doc['hash']] = mh
            self.lsh.insert(doc['hash'], mh)
        
        # Group documents by LSH clusters
        clusters: Dict[str, List[Dict]] = {}
        
        for doc in docs:
            # Find similar documents
            candidates = list(self.lsh.query(self.mh_map[doc['hash']]))
            cluster_key = min(candidates)  # Use minimum hash as cluster key
            
            clusters.setdefault(cluster_key, []).append(doc)
        
        # Select best representative from each cluster
        deduplicated_docs = []
        
        for cluster_docs in clusters.values():
            if len(cluster_docs) == 1:
                deduplicated_docs.append(cluster_docs[0])
            else:
                # Select best document by quality score and length
                best_doc = max(
                    cluster_docs,
                    key=lambda d: (d['quality_score'], len(d['text_norm']))
                )
                deduplicated_docs.append(best_doc)
                self.stats['near_duplicates_removed'] += len(cluster_docs) - 1
        
        logger.info(f"🔗 After LSH deduplication: {len(deduplicated_docs)} documents")
        return deduplicated_docs
    
    def _semantic_deduplication(self, docs: List[Dict]) -> List[Dict]:
        """Remove semantic duplicates using embeddings and FAISS."""
        if not self.embedding_model or not FAISS_AVAILABLE:
            return docs
        
        if len(docs) < 2:
            return docs
        
        # Generate embeddings
        texts = [doc['text_norm'] for doc in docs]
        
        try:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=self.config.batch_size_embedding,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        except Exception as e:
            logger.warning(f"Failed to generate embeddings: {e}")
            return docs
        
        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
        index.add(embeddings.astype(np.float32))
        
        # Find semantic duplicates
        keep_mask = np.ones(len(docs), dtype=bool)
        
        for i in range(len(docs)):
            if not keep_mask[i]:
                continue
            
            # Search for similar documents
            similarities, indices = index.search(
                embeddings[i:i+1].astype(np.float32), 
                min(20, len(docs))  # Search top 20 similar docs
            )
            
            for sim, j in zip(similarities[0][1:], indices[0][1:]):  # Skip self
                if j == -1 or not keep_mask[j]:
                    continue
                
                if sim >= self.config.cos_threshold:
                    # Decide which document to keep
                    doc_i, doc_j = docs[i], docs[j]
                    
                    # Prefer higher quality, then longer content
                    keep_i = (
                        doc_i['quality_score'], 
                        len(doc_i['text_norm'])
                    ) >= (
                        doc_j['quality_score'], 
                        len(doc_j['text_norm'])
                    )
                    
                    if keep_i:
                        keep_mask[j] = False
                        self.stats['semantic_duplicates_removed'] += 1
                    else:
                        keep_mask[i] = False
                        self.stats['semantic_duplicates_removed'] += 1
                        break
        
        deduplicated_docs = [doc for i, doc in enumerate(docs) if keep_mask[i]]
        
        logger.info(f"🧠 After semantic deduplication: {len(deduplicated_docs)} documents")
        return deduplicated_docs
    
    def _final_quality_filter(self, docs: List[Dict]) -> List[Dict]:
        """Apply final quality filtering."""
        filtered_docs = []
        
        for doc in docs:
            if doc['quality_score'] >= self.config.quality_threshold:
                filtered_docs.append(doc)
            else:
                self.stats['quality_filtered'] += 1
        
        logger.info(f"✨ After final quality filtering: {len(filtered_docs)} documents")
        return filtered_docs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        total_removed = (
            self.stats['exact_duplicates_removed'] + 
            self.stats['near_duplicates_removed'] + 
            self.stats['semantic_duplicates_removed'] + 
            self.stats['quality_filtered'] + 
            self.stats['language_filtered']
        )
        
        reduction_rate = total_removed / max(self.stats['total_processed'], 1)
        
        return {
            **self.stats,
            'total_removed': total_removed,
            'reduction_rate': reduction_rate,
            'components_available': {
                'lsh': self.lsh is not None,
                'embeddings': self.embedding_model is not None,
                'faiss': FAISS_AVAILABLE,
                'langid': LANGID_AVAILABLE
            }
        }
    
    def _log_stats(self):
        """Log detailed statistics."""
        stats = self.get_stats()
        
        logger.info("📊 Deduplication Statistics:")
        logger.info(f"  Total processed: {stats['total_processed']:,}")
        logger.info(f"  Exact duplicates removed: {stats['exact_duplicates_removed']:,}")
        logger.info(f"  Near duplicates removed: {stats['near_duplicates_removed']:,}")
        logger.info(f"  Semantic duplicates removed: {stats['semantic_duplicates_removed']:,}")
        logger.info(f"  Quality filtered: {stats['quality_filtered']:,}")
        logger.info(f"  Language filtered: {stats['language_filtered']:,}")
        logger.info(f"  Final count: {stats['final_count']:,}")
        logger.info(f"  Reduction rate: {stats['reduction_rate']:.1%}")


# Utility functions for external use
def create_default_config() -> DedupConfig:
    """Create default deduplication configuration for Capibara-6."""
    return DedupConfig()


def quick_deduplicate(docs: List[Dict], config: Optional[DedupConfig] = None) -> List[Dict]:
    """Quick deduplication function for simple use cases."""
    if config is None:
        config = create_default_config()
    
    deduplicator = SemanticDeduplicator(config)
    return deduplicator.deduplicate_documents(docs)
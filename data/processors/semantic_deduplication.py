#!/usr/bin/env python3
"""
Semantic Deduplication Processor for CapibaraGPT Data Pipeline

This module provides advanced semantic deduplication capabilities using MinHash LSH and
sentence embeddings, integrated with the existing Capibara data processing pipeline.
"""

import os
import re
import gzip
import json
import logging
import numpy as np
import xxhash
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Optional, Union, Generator
from pathlib import Path
from tqdm import tqdm

# Lazy imports for heavy dependencies
_MINHASH = None
_LSH = None
_SENTENCE_MODEL = None
_FAISS = None
_LANGID = None

logger = logging.getLogger(__name__)

def _lazy_imports():
    """Lazy import heavy dependencies to avoid startup overhead."""
    global _MINHASH, _LSH, _SENTENCE_MODEL, _FAISS, _LANGID
    try:
        from datasketch import MinHash, MinHashLSH
        from sentence_transformers import SentenceTransformer
        import faiss
        import langid
        _MINHASH, _LSH = MinHash, MinHashLSH
        _SENTENCE_MODEL = SentenceTransformer
        _FAISS = faiss
        _LANGID = langid
        logger.info("Successfully imported heavy deduplication dependencies")
    except ImportError as e:
        logger.error(f"Failed to import deduplication dependencies: {e}")
        raise ImportError("Please install required dependencies: datasketch, sentence-transformers, faiss-cpu, langid")

@dataclass
class DeduplicationConfig:
    """Configuration for semantic deduplication process."""
    jaccard_threshold: float = 0.90
    cosine_threshold: float = 0.94
    quality_min_score: float = 0.40
    min_length_chars: int = 200
    max_length_chars: int = 40000
    allowed_languages: Tuple[str, ...] = ("es", "en")
    embedding_model: str = "intfloat/multilingual-e5-base"
    batch_size: int = 256
    num_permutations: int = 128
    shingle_size: int = 13
    workdir: str = "./dedup_work"

class SemanticDeduplicator:
    """
    Advanced semantic deduplication processor for text datasets.
    
    This class implements a multi-stage deduplication pipeline:
    1. Exact hash-based deduplication
    2. MinHash LSH for approximate duplicates
    3. Semantic similarity using sentence embeddings
    4. Quality filtering based on linguistic features
    """
    
    def __init__(self, config: DeduplicationConfig):
        """Initialize the deduplicator with the given configuration."""
        self.config = config
        self.exact_hashes = set()
        self.lsh = None
        self.embedding_model = None
        self.discarded_docs = []
        
        # Ensure workdir exists
        os.makedirs(config.workdir, exist_ok=True)
        
        # Initialize lazy imports
        _lazy_imports()
    
    def _open_file(self, path: Union[str, Path], mode: str = "rt", encoding: str = "utf-8"):
        """Open file with automatic gzip detection."""
        path = str(path)
        if path.endswith(".gz"):
            return gzip.open(path, mode, encoding=encoding)
        return open(path, mode, encoding=encoding)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing HTML tags and excess whitespace."""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def _generate_shingles(self, text: str, k: int = None) -> Iterable[str]:
        """Generates k-shingles from text."""
        k = k or self.config.shingle_size
        if len(text) <= k:
            yield text
        else:
            for i in range(len(text) - k + 1):
                yield text[i:i+k]
    
    def _compute_minhash(self, text: str) -> 'MinHash':
        """Compute MinHash signature for text."""
        mh = _MINHASH(num_perm=self.config.num_permutations)
        for shingle in self._generate_shingles(text):
            mh.update(shingle.encode("utf-8"))
        return mh
    
    def _compute_exact_hash(self, text: str) -> str:
        """Compute exact hash for text using xxhash."""
        return xxhash.xxh3_128_hexdigest(text)
    
    def _is_valid_language(self, text: str) -> bool:
        """Check if text is in an allowed language."""
        if not _LANGID:
            return True  # Skip language check if not available
        
        # Use first 5000 chars for language detection
        sample = text[:5000]
        lang, confidence = _LANGID.classify(sample)
        return lang in self.config.allowed_languages
    
    def _compute_quality_score(self, text: str) -> float:
        """Compute text quality score based on linguistic features."""
        n = max(len(text), 1)
        
        # Punctuation ratio
        punct_chars = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        punc_ratio = sum(ch in punct_chars for ch in text) / n
        
        # Digits ratio
        digit_ratio = sum(ch.isdigit() for ch in text) / n
        
        # Uppercase ratio
        upper_ratio = sum(ch.isupper() for ch in text) / n
        
        # Unique trigram ratio (diversity measure)
        trigrams = [text[i:i+3] for i in range(len(text)-2)] if len(text) >= 3 else []
        unique_tri_ratio = len(set(trigrams)) / max(len(trigrams), 1)
        
        # Compute composite score
        score = 1.0 - (
            0.8 * max(0, punc_ratio - 0.15) +
            0.5 * max(0, digit_ratio - 0.1) +
            0.3 * max(0, upper_ratio - 0.2)
        ) + 0.5 * unique_tri_ratio
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _is_high_quality(self, text: str) -> bool:
        """Check if text meets quality standards."""
        # Length check
        if not (self.config.min_length_chars <= len(text) <= self.config.max_length_chars):
            return False
        
        # Language check
        if not self._is_valid_language(text):
            return False
        
        # Quality score check
        quality_score = self._compute_quality_score(text)
        return quality_score >= self.config.quality_min_score
    
    def _initialize_lsh(self):
        """Initialize the LSH index for approximate duplicate detection."""
        if self.lsh is None:
            self.lsh = _LSH(
                threshold=self.config.jaccard_threshold,
                num_perm=self.config.num_permutations
            )
            logger.info(f"Initialized LSH with threshold {self.config.jaccard_threshold}")
    
    def _initialize_embedding_model(self):
        """Initialize the sentence embedding model."""
        if self.embedding_model is None:
            self.embedding_model = _SENTENCE_MODEL(self.config.embedding_model)
            logger.info(f"Loaded embedding model: {self.config.embedding_model}")
    
    def _load_documents(self, input_paths: List[str]) -> Generator[Dict, None, None]:
        """Load documents from input files."""
        for path in input_paths:
            logger.info(f"Loading documents from {path}")
            with self._open_file(path) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        doc = json.loads(line.strip())
                        if "text" not in doc:
                            logger.warning(f"Document at {path}:{line_num} missing 'text' field")
                            continue
                        yield doc
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON at {path}:{line_num}: {e}")
                        continue
    
    def deduplicate(self, input_paths: List[str], output_path: str, discards_csv_path: str) -> Dict[str, int]:
        """
        Run the complete deduplication pipeline.
        
        Args:
            input_paths: List of input file paths (JSONL/JSONL.gz)
            output_path: Output path for deduplicated data
            discards_csv_path: Path for discarded documents CSV
            
        Returns:
            Dictionary with deduplication statistics
        """
        logger.info("Starting semantic deduplication pipeline")
        
        stats = {
            "total_docs": 0,
            "exact_duplicates": 0,
            "lsh_duplicates": 0,
            "semantic_duplicates": 0,
            "quality_filtered": 0,
            "kept_docs": 0
        }
        
        # Stage 1: Exact deduplication and quality filtering
        logger.info("Stage 1: Exact hash deduplication and quality filtering")
        filtered_docs = []
        
        for doc in tqdm(self._load_documents(input_paths), desc="Processing documents"):
            stats["total_docs"] += 1
            
            text = self._normalize_text(doc["text"])
            exact_hash = self._compute_exact_hash(text)
            
            # Check for exact duplicates
            if exact_hash in self.exact_hashes:
                stats["exact_duplicates"] += 1
                self.discarded_docs.append({
                    "id": doc.get("id", f"doc_{stats['total_docs']}"),
                    "reason": "exact_duplicate",
                    "text_preview": text[:100]
                })
                continue
            
            self.exact_hashes.add(exact_hash)
            
            # Quality filtering
            if not self._is_high_quality(text):
                stats["quality_filtered"] += 1
                self.discarded_docs.append({
                    "id": doc.get("id", f"doc_{stats['total_docs']}"),
                    "reason": "low_quality",
                    "quality_score": self._compute_quality_score(text),
                    "text_preview": text[:100]
                })
                continue
            
            doc["_normalized_text"] = text
            doc["_exact_hash"] = exact_hash
            filtered_docs.append(doc)
        
        logger.info(f"After stage 1: {len(filtered_docs)} documents remaining")
        
        # Stage 2: LSH-based approximate deduplication
        logger.info("Stage 2: LSH-based approximate deduplication")
        self._initialize_lsh()
        
        lsh_filtered_docs = []
        for doc in tqdm(filtered_docs, desc="LSH deduplication"):
            text = doc["_normalized_text"]
            minhash = self._compute_minhash(text)
            
            # Check for LSH duplicates
            candidates = self.lsh.query(minhash)
            if candidates:
                stats["lsh_duplicates"] += 1
                self.discarded_docs.append({
                    "id": doc.get("id", "unknown"),
                    "reason": "lsh_duplicate",
                    "candidates": len(candidates),
                    "text_preview": text[:100]
                })
                continue
            
            # Add to LSH index
            doc_id = doc.get("id", doc["_exact_hash"])
            self.lsh.insert(doc_id, minhash)
            doc["_minhash"] = minhash
            lsh_filtered_docs.append(doc)
        
        logger.info(f"After stage 2: {len(lsh_filtered_docs)} documents remaining")
        
        # Stage 3: Semantic similarity deduplication
        logger.info("Stage 3: Semantic similarity deduplication")
        final_docs = self._semantic_deduplication(lsh_filtered_docs, stats)
        
        stats["kept_docs"] = len(final_docs)
        
        # Save results
        self._save_results(final_docs, output_path, discards_csv_path)
        
        logger.info(f"Deduplication complete. Kept {stats['kept_docs']} out of {stats['total_docs']} documents")
        return stats
    
    def _semantic_deduplication(self, docs: List[Dict], stats: Dict[str, int]) -> List[Dict]:
        """Perform semantic deduplication using sentence embeddings."""
        if not docs:
            return []
        
        self._initialize_embedding_model()
        
        # Extract texts for embedding
        texts = [doc["_normalized_text"] for doc in docs]
        
        # Compute embeddings in batches
        logger.info("Computing embeddings for semantic deduplication")
        embeddings = []
        for i in tqdm(range(0, len(texts), self.config.batch_size), desc="Computing embeddings"):
            batch = texts[i:i + self.config.batch_size]
            batch_embeddings = self.embedding_model.encode(batch, convert_to_numpy=True)
            embeddings.extend(batch_embeddings)
        
        embeddings = np.array(embeddings)
        
        # Build FAISS index for similarity search
        logger.info("Building FAISS index for semantic similarity")
        dimension = embeddings.shape[1]
        index = _FAISS.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        _FAISS.normalize_L2(embeddings)
        index.add(embeddings)
        
        # Find semantic duplicates
        kept_indices = set(range(len(docs)))
        
        for i, embedding in enumerate(tqdm(embeddings, desc="Finding semantic duplicates")):
            if i not in kept_indices:
                continue
            
            # Search for similar documents
            similarities, indices = index.search(
                embedding.reshape(1, -1), 
                min(50, len(embeddings))  # Search top 50 similar
            )
            
            for sim, idx in zip(similarities[0], indices[0]):
                if idx != i and idx in kept_indices and sim >= self.config.cosine_threshold:
                    # Remove the duplicate
                    kept_indices.discard(idx)
                    stats["semantic_duplicates"] += 1
                    
                    self.discarded_docs.append({
                        "id": docs[idx].get("id", "unknown"),
                        "reason": "semantic_duplicate",
                        "similarity": float(sim),
                        "similar_to": docs[i].get("id", "unknown"),
                        "text_preview": docs[idx]["_normalized_text"][:100]
                    })
        
        # Return kept documents
        final_docs = [docs[i] for i in sorted(kept_indices)]
        
        # Clean up temporary fields
        for doc in final_docs:
            doc.pop("_normalized_text", None)
            doc.pop("_exact_hash", None)
            doc.pop("_minhash", None)
        
        return final_docs
    
    def _save_results(self, final_docs: List[Dict], output_path: str, discards_csv_path: str):
        """Save deduplicated documents and discard information."""
        # Save deduplicated documents
        logger.info(f"Saving {len(final_docs)} deduplicated documents to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with self._open_file(output_path, "wt") as f:
            for doc in final_docs:
                json.dump(doc, f, ensure_ascii=False)
                f.write("\n")
        
        # Save discarded documents info
        logger.info(f"Saving {len(self.discarded_docs)} discard records to {discards_csv_path}")
        os.makedirs(os.path.dirname(discards_csv_path), exist_ok=True)
        
        import csv
        with open(discards_csv_path, "w", newline="", encoding="utf-8") as f:
            if self.discarded_docs:
                fieldnames = list(self.discarded_docs[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.discarded_docs)

def create_deduplication_processor(config_dict: Optional[Dict] = None) -> SemanticDeduplicator:
    """
    Factory function to create a semantic deduplicator.
    
    Args:
        config_dict: Optional configuration dictionary
        
    Returns:
        Configured SemanticDeduplicator instance
    """
    config_dict = config_dict or {}
    config = DeduplicationConfig(**config_dict)
    return SemanticDeduplicator(config)

# Integration with existing data pipeline
def process_datasets_with_deduplication(
    input_paths: List[str],
    output_path: str,
    config: Optional[Dict] = None
) -> Dict[str, int]:
    """
    Process datasets with semantic deduplication.
    
    This function integrates with the existing CapibaraGPT data pipeline
    to provide semantic deduplication capabilities.
    
    Args:
        input_paths: List of input dataset paths
        output_path: Output path for processed dataset
        config: Optional deduplication configuration
        
    Returns:
        Processing statistics
    """
    logger.info("Starting dataset processing with semantic deduplication")
    
    # Create output directory structure
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate paths for outputs
    discards_csv_path = os.path.join(output_dir, "deduplication_discards.csv")
    
    # Initialize deduplicator
    deduplicator = create_deduplication_processor(config)
    
    # Run deduplication
    stats = deduplicator.deduplicate(input_paths, output_path, discards_csv_path)
    
    logger.info("Dataset processing with deduplication completed")
    return stats
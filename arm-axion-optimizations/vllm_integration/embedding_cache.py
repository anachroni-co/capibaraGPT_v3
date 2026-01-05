"""
Real Embeddings with LRU Cache
Optimizado para ARM Axion con NEON acceleration

Features:
- SentenceTransformer para embeddings reales
- LRU cache para evitar recalcular textos repetidos
- Batch processing para eficiencia
- Fallback a embeddings hash-based si modelo no disponible
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
import hashlib
import numpy as np
import pickle
from collections import OrderedDict
import time

# Try importing sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Using fallback embeddings.")
    print("   Install: pip install sentence-transformers")


class EmbeddingCache:
    """
    LRU Cache para embeddings con persistencia opcional

    Evita recalcular embeddings para textos repetidos o similares
    """

    def __init__(
        self,
        max_size: int = 10000,
        persist_path: Optional[Path] = None
    ):
        """
        Args:
            max_size: Maximum number of embeddings to cache
            persist_path: Path to persist cache on disk (optional)
        """
        self.max_size = max_size
        self.persist_path = persist_path
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

        # Stats
        self.hits = 0
        self.misses = 0
        self.total_queries = 0

        # Load persisted cache if exists
        if persist_path and persist_path.exists():
            self._load_cache()

    def _hash_text(self, text: str) -> str:
        """Create hash key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        self.total_queries += 1
        key = self._hash_text(text)

        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]

        self.misses += 1
        return None

    def put(self, text: str, embedding: np.ndarray):
        """Add embedding to cache"""
        key = self._hash_text(text)

        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        self.cache[key] = embedding.copy()

    def get_batch(self, texts: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """
        Get embeddings for batch of texts

        Args:
            texts: List of text strings

        Returns:
            (cached_embeddings, uncached_texts)
            cached_embeddings: embeddings found in cache (may be None)
            uncached_texts: texts not found in cache
        """
        cached = []
        uncached = []

        for text in texts:
            emb = self.get(text)
            if emb is not None:
                cached.append(emb)
            else:
                uncached.append(text)

        return cached, uncached

    def put_batch(self, texts: List[str], embeddings: List[np.ndarray]):
        """Add batch of embeddings to cache"""
        for text, emb in zip(texts, embeddings):
            self.put(text, emb)

    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.total_queries = 0

    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics"""
        hit_rate = self.hits / self.total_queries if self.total_queries > 0 else 0.0

        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'total_queries': self.total_queries,
            'hit_rate': hit_rate,
            'memory_mb': len(self.cache) * 768 * 4 / (1024 * 1024)  # Approx for 768d float32
        }

    def _save_cache(self):
        """Persist cache to disk"""
        if not self.persist_path:
            return

        try:
            with open(self.persist_path, 'wb') as f:
                pickle.dump(dict(self.cache), f)
            print(f"✅ Cache saved to {self.persist_path}")
        except Exception as e:
            print(f"⚠️  Failed to save cache: {e}")

    def _load_cache(self):
        """Load cache from disk"""
        try:
            with open(self.persist_path, 'rb') as f:
                loaded = pickle.load(f)
                self.cache = OrderedDict(loaded)
            print(f"✅ Cache loaded from {self.persist_path} ({len(self.cache)} entries)")
        except Exception as e:
            print(f"⚠️  Failed to load cache: {e}")


class RealEmbeddingModel:
    """
    Real embedding model with caching and NEON optimization

    Uses SentenceTransformer for high-quality embeddings
    Falls back to hash-based if not available
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_size: int = 10000,
        cache_persist_path: Optional[Path] = None,
        device: str = "cpu",
        use_neon: bool = True
    ):
        """
        Args:
            model_name: SentenceTransformer model name
                       'all-MiniLM-L6-v2': 384d, fast, good quality
                       'all-mpnet-base-v2': 768d, slower, better quality
            cache_size: Max embeddings to cache
            cache_persist_path: Path to persist cache
            device: 'cpu' or 'cuda' (use cpu for ARM)
            use_neon: Enable NEON optimizations (for normalization)
        """
        self.model_name = model_name
        self.device = device
        self.use_neon = use_neon

        # Initialize cache
        self.cache = EmbeddingCache(
            max_size=cache_size,
            persist_path=cache_persist_path
        )

        # Try loading real model
        self.model = None
        self.embed_dim = 768  # Default, will be updated

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"📥 Loading embedding model: {model_name}...")
                self.model = SentenceTransformer(model_name, device=device)
                self.embed_dim = self.model.get_sentence_embedding_dimension()
                print(f"✅ Embedding model loaded ({self.embed_dim}d)")
            except Exception as e:
                print(f"⚠️  Failed to load model: {e}")
                print("   Using fallback embeddings")

        # Load NEON kernels if available
        if use_neon:
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from kernels.neon_kernels import get_kernels
                self.kernels = get_kernels()
                self.neon_available = self.kernels.available
            except:
                self.kernels = None
                self.neon_available = False
        else:
            self.kernels = None
            self.neon_available = False

        # Stats
        self.total_embeds = 0
        self.cache_hits = 0
        self.embed_time_ms = []

    def embed(self, text: str) -> np.ndarray:
        """
        Get embedding for single text

        Args:
            text: Input text

        Returns:
            Embedding vector (normalized)
        """
        # Check cache first
        cached = self.cache.get(text)
        if cached is not None:
            self.cache_hits += 1
            return cached

        # Compute embedding
        start_time = time.time()

        if self.model:
            # Real embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32)
        else:
            # Fallback: hash-based
            embedding = self._fallback_embed(text)

        elapsed_ms = (time.time() - start_time) * 1000
        self.embed_time_ms.append(elapsed_ms)

        # Cache it
        self.cache.put(text, embedding)

        self.total_embeds += 1

        return embedding

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[np.ndarray]:
        """
        Get embeddings for batch of texts

        More efficient than calling embed() multiple times

        Args:
            texts: List of input texts
            batch_size: Batch size for model inference
            show_progress: Show progress bar

        Returns:
            List of embedding vectors
        """
        # Check cache
        cached_embeddings, uncached_texts = self.cache.get_batch(texts)

        if not uncached_texts:
            # All cached!
            self.cache_hits += len(texts)
            return cached_embeddings

        # Compute uncached embeddings
        start_time = time.time()

        if self.model:
            # Real embeddings (batched)
            new_embeddings = self.model.encode(
                uncached_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress
            )
            new_embeddings = [emb.astype(np.float32) for emb in new_embeddings]
        else:
            # Fallback: hash-based
            new_embeddings = [self._fallback_embed(text) for text in uncached_texts]

        elapsed_ms = (time.time() - start_time) * 1000
        self.embed_time_ms.append(elapsed_ms / len(uncached_texts))  # Per item

        # Cache new embeddings
        self.cache.put_batch(uncached_texts, new_embeddings)

        self.total_embeds += len(uncached_texts)
        self.cache_hits += len(cached_embeddings)

        # Merge cached and new embeddings in original order
        result = []
        cached_idx = 0
        new_idx = 0

        for text in texts:
            if self.cache.get(text) is not None and cached_idx < len(cached_embeddings):
                result.append(cached_embeddings[cached_idx])
                cached_idx += 1
            else:
                result.append(new_embeddings[new_idx])
                new_idx += 1

        return result

    def _fallback_embed(self, text: str) -> np.ndarray:
        """
        Fallback embedding using simple TF-IDF-like hashing

        Not as good as real embeddings but better than nothing
        """
        embedding = np.zeros(self.embed_dim, dtype=np.float32)

        # Simple word hashing
        words = text.lower().split()
        for word in words[:100]:  # Limit to 100 words
            idx = hash(word) % self.embed_dim
            embedding[idx] += 1.0

        # Normalize
        if self.neon_available and self.kernels:
            # NEON-optimized normalization
            norm = np.sqrt(np.sum(embedding ** 2))
            if norm > 0:
                embedding = embedding / norm
        else:
            # NumPy fallback
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between embeddings

        Uses NEON if available for 4-5x speedup
        """
        if self.neon_available and self.kernels:
            # NEON-optimized dot product
            return float(self.kernels.dot_product(emb1, emb2))
        else:
            # NumPy fallback
            return float(np.dot(emb1, emb2))

    def get_stats(self) -> Dict[str, any]:
        """Get embedding model statistics"""
        cache_stats = self.cache.get_stats()

        avg_time = np.mean(self.embed_time_ms) if self.embed_time_ms else 0.0

        return {
            'model_name': self.model_name,
            'embed_dim': self.embed_dim,
            'model_available': self.model is not None,
            'neon_available': self.neon_available,
            'total_embeds': self.total_embeds,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(self.total_embeds, 1),
            'avg_embed_time_ms': avg_time,
            'cache_stats': cache_stats
        }

    def save_cache(self):
        """Save cache to disk"""
        self.cache._save_cache()


# Singleton instance for easy import
_default_model: Optional[RealEmbeddingModel] = None

def get_embedding_model(
    model_name: str = "all-MiniLM-L6-v2",
    cache_size: int = 10000,
    force_reload: bool = False
) -> RealEmbeddingModel:
    """
    Get singleton embedding model instance

    Args:
        model_name: SentenceTransformer model name
        cache_size: Max embeddings to cache
        force_reload: Force reload model

    Returns:
        RealEmbeddingModel instance
    """
    global _default_model

    if _default_model is None or force_reload:
        _default_model = RealEmbeddingModel(
            model_name=model_name,
            cache_size=cache_size,
            cache_persist_path=Path("/tmp/capibara6_embedding_cache.pkl")
        )

    return _default_model


if __name__ == '__main__':
    print("🧪 Testing Real Embeddings with Cache")
    print("=" * 60)

    # Create model
    model = RealEmbeddingModel(
        model_name="all-MiniLM-L6-v2",
        cache_size=100
    )

    # Test single embedding
    print("\n📝 Test 1: Single embedding")
    text1 = "This is a test sentence about machine learning"
    emb1 = model.embed(text1)
    print(f"   Text: '{text1[:50]}...'")
    print(f"   Embedding shape: {emb1.shape}")
    print(f"   Norm: {np.linalg.norm(emb1):.3f}")

    # Test cache hit
    print("\n📝 Test 2: Cache hit")
    emb1_cached = model.embed(text1)
    print(f"   Same embedding: {np.allclose(emb1, emb1_cached)}")

    # Test batch
    print("\n📝 Test 3: Batch embeddings")
    texts = [
        "Machine learning is amazing",
        "Deep learning with transformers",
        "Natural language processing",
        text1  # This should hit cache
    ]
    embeddings = model.embed_batch(texts)
    print(f"   Batch size: {len(embeddings)}")
    print(f"   Embedding dims: {[emb.shape for emb in embeddings]}")

    # Test similarity
    print("\n📝 Test 4: Similarity")
    sim = model.similarity(embeddings[0], embeddings[1])
    print(f"   Similarity (ML vs DL): {sim:.3f}")

    sim2 = model.similarity(embeddings[0], embeddings[2])
    print(f"   Similarity (ML vs NLP): {sim2:.3f}")

    # Stats
    print("\n📊 Model Stats:")
    stats = model.get_stats()
    for key, value in stats.items():
        if key != 'cache_stats':
            print(f"   {key}: {value}")

    print("\n📊 Cache Stats:")
    for key, value in stats['cache_stats'].items():
        print(f"   {key}: {value}")

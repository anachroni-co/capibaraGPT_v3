"""
RAG Parallel Fetcher for vLLM Integration
Optimized for ARM Axion

Provides:
- RAG query detection (identify queries that need retrieval)
- Parallel context fetching from Milvus/Nebula during routing
- Context injection before generation
- Expected impact: RAG queries -40% latency
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import time
import re
import httpx
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class RAGQuery:
    """Detected RAG query with metadata"""
    query: str
    is_rag_query: bool
    confidence: float
    detected_intent: str  # 'factual', 'technical', 'contextual', 'general'
    top_k: int = 5


@dataclass
class RAGContext:
    """Retrieved context from RAG systems"""
    context_text: str
    sources: List[Dict[str, Any]]
    fetch_time: float
    source_type: str  # 'milvus', 'nebula', 'hybrid'
    tokens_count: int


class RAGQueryDetector:
    """
    Fast RAG query detection using keyword patterns

    Detects queries that would benefit from retrieval:
    - Factual questions (who, what, when, where)
    - Technical queries (code, documentation, API)
    - Contextual queries (reference to previous data)
    """

    def __init__(self):
        """Initialize detector with keyword patterns"""

        # Patterns that indicate RAG is needed
        self.rag_patterns = {
            'factual': [
                r'\b(who|what|when|where|which|whose)\b',
                r'\b(tell me about|explain|describe|define)\b',
                r'\b(fact|information|data|details|source)\b',
            ],
            'technical': [
                r'\b(code|function|api|documentation|docs|error|bug)\b',
                r'\b(implementation|how to|tutorial|guide)\b',
                r'\b(example|sample|snippet)\b',
            ],
            'contextual': [
                r'\b(previous|last time|mentioned|discussed|earlier)\b',
                r'\b(remember|recall|in the conversation|we talked)\b',
                r'\b(according to|based on|reference)\b',
            ]
        }

        # Patterns that indicate NO RAG needed (chat, creative)
        self.non_rag_patterns = [
            r'\b(write|generate|create|imagine|compose)\b.*\b(story|poem|essay|letter)\b',
            r'\b(hello|hi|hey|good morning|good evening)\b',
            r'\b(thank|thanks|appreciate)\b',
            r'\b(joke|fun|funny)\b',
        ]

    def detect(self, query: str) -> RAGQuery:
        """
        Detect if query needs RAG

        Args:
            query: User query

        Returns:
            RAGQuery with detection results
        """
        query_lower = query.lower()

        # Check non-RAG patterns first (faster rejection)
        for pattern in self.non_rag_patterns:
            if re.search(pattern, query_lower):
                return RAGQuery(
                    query=query,
                    is_rag_query=False,
                    confidence=0.9,
                    detected_intent='general',
                    top_k=0
                )

        # Check RAG patterns
        max_confidence = 0.0
        detected_intent = 'general'

        for intent, patterns in self.rag_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    confidence = 0.7 + (len(re.findall(pattern, query_lower)) * 0.1)
                    if confidence > max_confidence:
                        max_confidence = confidence
                        detected_intent = intent

        # Heuristics: questions usually need RAG
        if '?' in query:
            max_confidence = max(max_confidence, 0.6)

        # Long queries (>100 chars) likely need context
        if len(query) > 100:
            max_confidence = max(max_confidence, 0.5)

        is_rag = max_confidence > 0.5

        # Determine top_k based on intent
        top_k = 0
        if is_rag:
            if detected_intent == 'factual':
                top_k = 5
            elif detected_intent == 'technical':
                top_k = 8
            elif detected_intent == 'contextual':
                top_k = 3
            else:
                top_k = 5

        return RAGQuery(
            query=query,
            is_rag_query=is_rag,
            confidence=max_confidence,
            detected_intent=detected_intent,
            top_k=top_k
        )


class MilvusClient:
    """
    Async client for Milvus vector search
    Connects via capibara6-api bridge
    """

    def __init__(
        self,
        bridge_url: str = "http://localhost:8001",
        collection_name: str = "capibara_docs",
        timeout: float = 3.0
    ):
        """
        Initialize Milvus client

        Args:
            bridge_url: URL of capibara6-api bridge
            collection_name: Milvus collection name
            timeout: Request timeout in seconds
        """
        self.bridge_url = bridge_url
        self.collection_name = collection_name
        self.timeout = timeout

        # Create async HTTP client
        self.client = httpx.AsyncClient(timeout=timeout)

        # Stats
        self.searches = 0
        self.cache_hits = 0
        self.total_time = 0.0

    async def search(
        self,
        query: str,
        top_k: int = 5,
        nprobe: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Search Milvus for relevant context

        Args:
            query: Search query
            top_k: Number of results
            nprobe: Search parameter (higher = more accurate, slower)

        Returns:
            List of search results with scores
        """
        start_time = time.time()
        self.searches += 1

        try:
            # Get embedding from bridge
            embed_response = await self.client.post(
                f"{self.bridge_url}/api/v1/embeddings",
                json={"text": query}
            )
            embed_response.raise_for_status()
            embedding = embed_response.json()["embedding"]

            # Search Milvus
            search_response = await self.client.post(
                f"{self.bridge_url}/api/v1/milvus/search",
                json={
                    "collection_name": self.collection_name,
                    "vector": embedding,
                    "top_k": top_k,
                    "nprobe": nprobe,
                    "output_fields": ["id", "text", "metadata", "timestamp"]
                }
            )
            search_response.raise_for_status()
            results = search_response.json().get("results", [])

            self.total_time += time.time() - start_time

            return results

        except Exception as e:
            print(f"⚠️  Milvus search failed: {e}")
            return []

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            'searches': self.searches,
            'cache_hits': self.cache_hits,
            'avg_search_time': self.total_time / self.searches if self.searches > 0 else 0.0
        }


class RAGParallelFetcher:
    """
    Parallel RAG context fetcher

    Workflow:
    1. Detect if query needs RAG (fast keyword check)
    2. If yes: Fetch context from Milvus in parallel with routing
    3. Inject context into prompt before generation

    Expected impact:
    - RAG queries: -40% latency (parallel fetch vs sequential)
    - Non-RAG queries: No overhead (fast rejection)
    """

    def __init__(
        self,
        bridge_url: str = "http://localhost:8001",
        collection_name: str = "capibara_docs",
        enable_rag: bool = True,
        detection_threshold: float = 0.5,
        max_context_tokens: int = 1000
    ):
        """
        Initialize RAG parallel fetcher

        Args:
            bridge_url: URL of capibara6-api bridge
            collection_name: Milvus collection name
            enable_rag: Enable RAG integration
            detection_threshold: Confidence threshold for RAG detection
            max_context_tokens: Max tokens for context (approx)
        """
        self.enable_rag = enable_rag
        self.detection_threshold = detection_threshold
        self.max_context_tokens = max_context_tokens

        # Initialize components
        if enable_rag:
            self.detector = RAGQueryDetector()
            self.milvus_client = MilvusClient(
                bridge_url=bridge_url,
                collection_name=collection_name
            )
        else:
            self.detector = None
            self.milvus_client = None

        # Stats
        self.total_queries = 0
        self.rag_queries = 0
        self.non_rag_queries = 0
        self.total_fetch_time = []
        self.context_cache = {}  # Simple in-memory cache

        print(f"✅ RAG Parallel Fetcher initialized")
        print(f"   Enabled: {enable_rag}")
        print(f"   Bridge: {bridge_url}")

    async def detect_and_fetch(
        self,
        query: str,
        request_id: str
    ) -> Tuple[bool, Optional[RAGContext]]:
        """
        Detect if query needs RAG and fetch context in parallel

        This runs concurrently with routing to minimize latency

        Args:
            query: User query
            request_id: Request ID for tracking

        Returns:
            Tuple of (is_rag_query, context or None)
        """
        self.total_queries += 1

        if not self.enable_rag:
            return False, None

        # Phase 1: Fast detection (< 1ms)
        start_time = time.time()
        rag_query = self.detector.detect(query)

        if not rag_query.is_rag_query or rag_query.confidence < self.detection_threshold:
            self.non_rag_queries += 1
            return False, None

        self.rag_queries += 1
        print(f"🔍 [{request_id}] RAG query detected: {rag_query.detected_intent} (conf: {rag_query.confidence:.2f})")

        # Check cache
        cache_key = f"{query[:100]}"  # First 100 chars as key
        if cache_key in self.context_cache:
            cached_context = self.context_cache[cache_key]
            if time.time() - cached_context['timestamp'] < 300:  # 5 min TTL
                print(f"✅ [{request_id}] RAG cache hit")
                return True, cached_context['context']

        # Phase 2: Fetch context from Milvus (parallel with routing)
        try:
            results = await self.milvus_client.search(
                query=query,
                top_k=rag_query.top_k,
                nprobe=16
            )

            fetch_time = time.time() - start_time
            self.total_fetch_time.append(fetch_time)

            if not results:
                print(f"⚠️  [{request_id}] No RAG results found")
                return True, None

            # Format context
            context = self._format_context(results, query, rag_query.detected_intent)

            print(f"✅ [{request_id}] RAG context fetched: {len(results)} sources ({fetch_time:.3f}s)")

            # Cache result
            self.context_cache[cache_key] = {
                'context': context,
                'timestamp': time.time()
            }

            return True, context

        except Exception as e:
            print(f"❌ [{request_id}] RAG fetch failed: {e}")
            return True, None  # Continue without context

    def _format_context(
        self,
        results: List[Dict[str, Any]],
        query: str,
        intent: str
    ) -> RAGContext:
        """
        Format retrieved results into context

        Args:
            results: Milvus search results
            query: Original query
            intent: Detected intent

        Returns:
            Formatted RAGContext
        """
        # Build context text
        context_parts = [
            f"[RETRIEVED CONTEXT for: \"{query}\"]",
            f"[Intent: {intent}]",
            ""
        ]

        sources = []
        total_chars = 0
        max_chars = self.max_context_tokens * 4  # Rough estimate: 1 token ≈ 4 chars

        for i, result in enumerate(results):
            # Extract fields
            text = result.get('text', '')
            score = result.get('score', 0.0)
            metadata = result.get('metadata', {})

            # Format source
            source_text = f"{i+1}. {text}\n   (Score: {score:.3f})"

            if total_chars + len(source_text) > max_chars:
                break  # Reached max tokens

            context_parts.append(source_text)
            total_chars += len(source_text)

            sources.append({
                'id': result.get('id'),
                'text': text,
                'score': score,
                'metadata': metadata
            })

        context_parts.append("\n[END CONTEXT]\n")

        context_text = "\n".join(context_parts)

        return RAGContext(
            context_text=context_text,
            sources=sources,
            fetch_time=0.0,  # Set by caller
            source_type='milvus',
            tokens_count=len(context_text) // 4  # Rough estimate
        )

    def inject_context(
        self,
        prompt: str,
        context: Optional[RAGContext]
    ) -> str:
        """
        Inject RAG context into prompt

        Args:
            prompt: Original user prompt
            context: Retrieved context (or None)

        Returns:
            Prompt with injected context
        """
        if not context:
            return prompt

        # Inject context before prompt
        enhanced_prompt = f"""{context.context_text}

User query: {prompt}

Instructions: Use the retrieved context above to answer the user's query. If the context is relevant, cite it in your response."""

        return enhanced_prompt

    async def close(self):
        """Cleanup resources"""
        if self.milvus_client:
            await self.milvus_client.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get fetcher statistics"""
        stats = {
            'total_queries': self.total_queries,
            'rag_queries': self.rag_queries,
            'non_rag_queries': self.non_rag_queries,
            'rag_rate': f"{(self.rag_queries / self.total_queries * 100):.1f}%" if self.total_queries > 0 else "0%",
            'cache_size': len(self.context_cache)
        }

        if self.total_fetch_time:
            import numpy as np
            stats['fetch_time'] = {
                'mean': np.mean(self.total_fetch_time),
                'median': np.median(self.total_fetch_time),
                'p95': np.percentile(self.total_fetch_time, 95)
            }

        if self.milvus_client:
            stats['milvus'] = self.milvus_client.get_stats()

        return stats


if __name__ == '__main__':
    print("🧪 Testing RAG Parallel Fetcher")
    print("=" * 60)

    # Test queries
    test_queries = [
        ("What is vLLM and how does it work?", True, "factual"),
        ("Tell me about PagedAttention", True, "factual"),
        ("Show me code examples for vLLM", True, "technical"),
        ("Hello, how are you?", False, "general"),
        ("Write a poem about AI", False, "general"),
        ("What did we discuss earlier?", True, "contextual"),
    ]

    # Create detector
    detector = RAGQueryDetector()

    print("\n📝 Testing RAG Query Detection:")
    for query, expected_rag, expected_intent in test_queries:
        result = detector.detect(query)
        match = "✅" if result.is_rag_query == expected_rag else "❌"
        print(f"{match} \"{query}\"")
        print(f"   RAG: {result.is_rag_query} (conf: {result.confidence:.2f}), Intent: {result.detected_intent}, Top-K: {result.top_k}")

    # Test fetcher (async)
    async def test_fetcher():
        print("\n📝 Testing RAG Parallel Fetcher:")

        fetcher = RAGParallelFetcher(
            bridge_url="http://localhost:8001",
            enable_rag=True
        )

        query = "What is vLLM?"
        is_rag, context = await fetcher.detect_and_fetch(query, "test_001")

        print(f"\nQuery: \"{query}\"")
        print(f"Is RAG: {is_rag}")
        if context:
            print(f"Context tokens: {context.tokens_count}")
            print(f"Sources: {len(context.sources)}")

        # Get stats
        print(f"\n📊 Stats:")
        stats = fetcher.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")

        await fetcher.close()

    # Run async test
    try:
        asyncio.run(test_fetcher())
    except Exception as e:
        print(f"⚠️  Fetcher test skipped (bridge not running): {e}")

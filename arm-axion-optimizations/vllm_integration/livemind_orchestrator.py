"""
LiveMind Orchestrator for vLLM Multi-Expert System
Optimized for ARM Axion

Combines:
- Incremental chunk processing
- Semantic routing with NEON acceleration
- Multi-expert vLLM inference
- Speculative consensus generation
- Continuous batching (handled by vLLM)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass
import asyncio
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm_integration.vllm_axion_backend import (
    AxionVLLMEngine,
    AxionMultiExpertVLLM,
    AxionVLLMConfig
)
from vllm_integration.semantic_router import (
    IncrementalSemanticRouter,
    FastDomainClassifier,
    RoutingPrediction
)
from vllm_integration.rag_parallel_fetcher import (
    RAGParallelFetcher,
    RAGContext
)
from vllm_integration.speculative_router import (
    SpeculativeRouter,
    SpeculativeDecision
)

try:
    from vllm import SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


@dataclass
class Chunk:
    """Text chunk for incremental processing"""
    text: str
    position: int
    token_count: int
    semantic_hint: Optional[str] = None
    confidence: float = 0.0


@dataclass
class GenerationRequest:
    """Generation request with metadata"""
    request_id: str
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = True
    created_at: float = 0.0
    expert_id: Optional[str] = None  # Specific expert to use (bypasses routing)


@dataclass
class GenerationResult:
    """Result from generation"""
    request_id: str
    text: str
    expert_id: str
    tokens_generated: int
    time_to_first_token: float
    total_time: float
    chunks_processed: int


class ChunkedTokenizer:
    """
    Tokenizer that processes text in chunks for incremental routing

    In production: Use actual tokenizer from model
    For now: Simple word-based chunking
    """

    def __init__(self, chunk_size: int = 64):
        """
        Args:
            chunk_size: Approximate tokens per chunk
        """
        self.chunk_size = chunk_size

    def chunk_text(self, text: str) -> List[Chunk]:
        """
        Split text into chunks

        Args:
            text: Input text

        Returns:
            List of chunks
        """
        # Simple word-based chunking
        # In production: Use actual tokenizer
        words = text.split()

        chunks = []
        position = 0

        for i in range(0, len(words), self.chunk_size):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            chunk = Chunk(
                text=chunk_text,
                position=position,
                token_count=len(chunk_words)  # Approximate
            )

            chunks.append(chunk)
            position += len(chunk_words)

        return chunks

    async def chunk_stream(self, text_stream: AsyncIterator[str]) -> AsyncIterator[Chunk]:
        """
        Process streaming text into chunks

        Args:
            text_stream: Async iterator of text chunks

        Yields:
            Chunk objects
        """
        buffer = []
        position = 0

        async for text_piece in text_stream:
            words = text_piece.split()
            buffer.extend(words)

            # Emit chunk when buffer is large enough
            while len(buffer) >= self.chunk_size:
                chunk_words = buffer[:self.chunk_size]
                buffer = buffer[self.chunk_size:]

                chunk = Chunk(
                    text=' '.join(chunk_words),
                    position=position,
                    token_count=len(chunk_words)
                )

                yield chunk
                position += len(chunk_words)

        # Emit remaining buffer
        if buffer:
            chunk = Chunk(
                text=' '.join(buffer),
                position=position,
                token_count=len(buffer)
            )
            yield chunk


class LiveMindOrchestrator:
    """
    Main orchestrator for LiveMind + vLLM system

    Workflow:
    1. Receive request (streaming or complete)
    2. Process chunks incrementally
    3. Route to experts when confident
    4. Generate from experts in parallel (vLLM handles batching)
    5. Optionally: Consensus from multiple experts
    6. Stream results back
    """

    def __init__(
        self,
        expert_system: AxionMultiExpertVLLM,
        enable_consensus: bool = True,
        consensus_model: Optional[str] = None,
        chunk_size: int = 64,
        routing_threshold: float = 0.7,
        use_fast_classifier: bool = True,
        enable_rag: bool = True,
        rag_bridge_url: str = "http://localhost:8001",
        rag_collection: str = "capibara_docs"
    ):
        """
        Args:
            expert_system: Multi-expert vLLM system
            enable_consensus: Enable consensus from multiple experts
            consensus_model: Model path for consensus (if enabled)
            chunk_size: Tokens per chunk
            routing_threshold: Confidence threshold for routing
            use_fast_classifier: Use fast keyword-based classifier
            enable_rag: Enable RAG parallel fetching
            rag_bridge_url: URL for RAG bridge API
            rag_collection: Milvus collection name
        """
        self.expert_system = expert_system
        self.enable_consensus = enable_consensus
        self.chunk_size = chunk_size

        # Chunking
        self.chunker = ChunkedTokenizer(chunk_size=chunk_size)

        # Routing
        expert_domains = {
            expert_id: info['domain']
            for expert_id, info in expert_system.experts.items()
        }

        self.router = IncrementalSemanticRouter(
            expert_domains=expert_domains,
            use_neon=True,
            chunk_size=chunk_size,
            routing_threshold=routing_threshold,
            top_k_experts=2 if enable_consensus else 1
        )

        # Fast classifier for early hints
        if use_fast_classifier:
            self.fast_classifier = FastDomainClassifier()
        else:
            self.fast_classifier = None

        # RAG parallel fetcher
        if enable_rag:
            self.rag_fetcher = RAGParallelFetcher(
                bridge_url=rag_bridge_url,
                collection_name=rag_collection,
                enable_rag=True
            )
        else:
            self.rag_fetcher = None

        # Speculative router (for early generation start)
        self.speculative_router = SpeculativeRouter(
            speculation_threshold=0.85,
            enable_speculation=True  # Can be controlled via config
        )

        # Consensus engine (if enabled)
        if enable_consensus and consensus_model:
            consensus_config = AxionVLLMConfig(
                model_path=consensus_model,
                enable_neon=True,
                max_num_seqs=64
            )
            self.consensus_engine = AxionVLLMEngine(
                consensus_config,
                engine_id="consensus"
            )
        else:
            self.consensus_engine = None

        # Metrics
        self.total_requests = 0
        self.total_ttft = []
        self.total_latency = []

        print(f"✅ LiveMind Orchestrator initialized")
        print(f"   Experts: {len(expert_system.experts)}")
        print(f"   Consensus: {enable_consensus}")
        print(f"   Chunk size: {chunk_size}")
        print(f"   RAG enabled: {enable_rag}")

    async def generate(
        self,
        request: GenerationRequest
    ) -> GenerationResult:
        """
        Generate completion for request (non-streaming)

        Optimized with PARALLEL RAG fetching:
        - RAG detection and fetch happens concurrently with routing
        - Expected impact: RAG queries -40% latency

        Args:
            request: Generation request

        Returns:
            Generation result
        """
        start_time = time.time()
        request.created_at = start_time

        # If expert_id is specified, bypass routing and use it directly
        if request.expert_id:
            print(f"✅ [{request.request_id}] Using specified expert: {request.expert_id}")
            result = await self._generate_single_expert(
                request,
                request.expert_id
            )
            return result

        # Phase 1: PARALLEL processing - routing AND RAG fetch
        # Start RAG fetch in parallel (if enabled)
        rag_task = None
        if self.rag_fetcher:
            rag_task = asyncio.create_task(
                self.rag_fetcher.detect_and_fetch(request.prompt, request.request_id)
            )

        # Process chunks and route (runs in parallel with RAG fetch)
        chunks = self.chunker.chunk_text(request.prompt)

        routing_prediction = None
        ttft = None

        for i, chunk in enumerate(chunks):
            # Fast classification hint (optional)
            if i == 0 and self.fast_classifier:
                domain, confidence = self.fast_classifier.classify(chunk.text)
                chunk.semantic_hint = domain
                chunk.confidence = confidence

            # Update routing
            routing_prediction = self.router.process_chunk(
                request.request_id,
                chunk.text
            )

            # Can we route?
            if routing_prediction.can_route:
                ttft = time.time() - start_time
                print(f"✅ [{request.request_id}] Routing after {i+1} chunks (TTFT: {ttft:.3f}s)")
                break

        # If we didn't route yet, finalize now
        if not routing_prediction or not routing_prediction.can_route:
            routing_prediction = self.router.finalize_routing(request.request_id)
            ttft = time.time() - start_time

        # Wait for RAG fetch to complete (if started)
        is_rag_query = False
        rag_context = None
        if rag_task:
            is_rag_query, rag_context = await rag_task
            if rag_context:
                # Inject context into prompt
                request.prompt = self.rag_fetcher.inject_context(request.prompt, rag_context)
                print(f"✅ [{request.request_id}] RAG context injected ({rag_context.tokens_count} tokens)")

        # Phase 2: Generate from expert(s)
        expert_ids = routing_prediction.expert_ids
        expert_probs = routing_prediction.probabilities

        if self.enable_consensus and len(expert_ids) > 1:
            # Multi-expert consensus
            result = await self._generate_consensus(
                request,
                expert_ids,
                routing_prediction
            )
        else:
            # Single expert (pass probability for lazy loading priority)
            result = await self._generate_single_expert(
                request,
                expert_ids[0],
                expert_probability=expert_probs[0] if expert_probs else 1.0
            )

        # Metrics
        total_time = time.time() - start_time
        result.time_to_first_token = ttft
        result.total_time = total_time
        result.chunks_processed = routing_prediction.chunks_processed

        self.total_requests += 1
        self.total_ttft.append(ttft)
        self.total_latency.append(total_time)

        return result

    async def _generate_single_expert(
        self,
        request: GenerationRequest,
        expert_id: str,
        expert_probability: float = 1.0
    ) -> GenerationResult:
        """Generate from single expert (lazy loads if necessary)"""
        # Get expert (will lazy load if not already loaded)
        expert = await self.expert_system.get_expert(
            expert_id,
            predicted_probability=expert_probability
        )

        if not expert:
            raise ValueError(f"Expert not found: {expert_id}")

        # Prepare prompt
        full_prompt = request.prompt
        if request.system_prompt:
            full_prompt = f"{request.system_prompt}\n\n{request.prompt}"

        # Sampling params
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )

        # Generate
        results = expert.generate([full_prompt], sampling_params)

        return GenerationResult(
            request_id=request.request_id,
            text=results[0]['text'],
            expert_id=expert_id,
            tokens_generated=len(results[0]['tokens']),
            time_to_first_token=0.0,  # Will be set by caller
            total_time=0.0,  # Will be set by caller
            chunks_processed=0  # Will be set by caller
        )

    async def _generate_consensus(
        self,
        request: GenerationRequest,
        expert_ids: List[str],
        routing_prediction: RoutingPrediction
    ) -> GenerationResult:
        """
        Generate from multiple experts and create consensus

        Args:
            request: Generation request
            expert_ids: Expert IDs to use
            routing_prediction: Routing prediction

        Returns:
            Consensus result
        """
        # Generate from all experts in parallel
        expert_results = await self.expert_system.generate_parallel(
            request.prompt,
            expert_ids,
            SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens
            )
        )

        if not self.consensus_engine:
            # No consensus engine, return first expert
            best_result = expert_results[0]['result']
            return GenerationResult(
                request_id=request.request_id,
                text=best_result['text'],
                expert_id=expert_results[0]['expert_id'],
                tokens_generated=len(best_result['tokens']),
                time_to_first_token=0.0,
                total_time=0.0,
                chunks_processed=0
            )

        # Create consensus prompt
        consensus_prompt = self._create_consensus_prompt(
            request.prompt,
            expert_results,
            routing_prediction
        )

        # Generate consensus
        consensus_sampling = SamplingParams(
            temperature=0.7,
            max_tokens=request.max_tokens
        )

        consensus_result = self.consensus_engine.generate(
            [consensus_prompt],
            consensus_sampling
        )[0]

        return GenerationResult(
            request_id=request.request_id,
            text=consensus_result['text'],
            expert_id='consensus',
            tokens_generated=len(consensus_result['tokens']),
            time_to_first_token=0.0,
            total_time=0.0,
            chunks_processed=0
        )

    def _create_consensus_prompt(
        self,
        original_prompt: str,
        expert_results: List[Dict[str, Any]],
        routing_prediction: RoutingPrediction
    ) -> str:
        """
        Create prompt for consensus generation

        Args:
            original_prompt: Original user prompt
            expert_results: Results from experts
            routing_prediction: Routing prediction with probabilities

        Returns:
            Consensus prompt
        """
        prompt = f"""You are a meta-expert that synthesizes responses from multiple specialist experts.

Original question:
{original_prompt}

Expert responses:

"""

        for i, expert_result in enumerate(expert_results):
            expert_id = expert_result['expert_id']
            result = expert_result['result']
            prob = routing_prediction.probabilities[i] if i < len(routing_prediction.probabilities) else 0.0

            prompt += f"""Expert {i+1} ({expert_id}, confidence: {prob:.2f}):
{result['text']}

"""

        prompt += """Your task: Synthesize the expert responses into a single, coherent answer that:
1. Combines the best insights from each expert
2. Resolves any contradictions
3. Provides a clear, comprehensive answer

Synthesized response:"""

        return prompt

    async def generate_streaming(
        self,
        request: GenerationRequest
    ) -> AsyncIterator[str]:
        """
        Generate with TRUE streaming output using vLLM's AsyncLLMEngine

        Optimized with PARALLEL RAG fetching:
        - RAG detection and fetch happens concurrently with routing
        - Real token-by-token streaming with minimal TTFT (~0.05-0.1s)
        - Expected impact: RAG queries -40% latency

        Args:
            request: Generation request

        Yields:
            Token strings as they're generated in real-time
        """
        start_time = time.time()
        request.created_at = start_time

        # Phase 1: PARALLEL processing - routing AND RAG fetch
        # Start RAG fetch in parallel (if enabled)
        rag_task = None
        if self.rag_fetcher:
            rag_task = asyncio.create_task(
                self.rag_fetcher.detect_and_fetch(request.prompt, request.request_id)
            )

        # Fast routing (process minimal chunks) - runs in parallel with RAG fetch
        chunks = self.chunker.chunk_text(request.prompt)

        routing_prediction = None
        first_token_time = None

        # Process chunks incrementally for routing
        for i, chunk in enumerate(chunks):
            # Fast classification hint (optional)
            if i == 0 and self.fast_classifier:
                domain, confidence = self.fast_classifier.classify(chunk.text)
                chunk.semantic_hint = domain
                chunk.confidence = confidence

            # Update routing
            routing_prediction = self.router.process_chunk(
                request.request_id,
                chunk.text
            )

            # Route as soon as we're confident
            if routing_prediction.can_route:
                print(f"✅ [{request.request_id}] Fast routing after {i+1} chunks")
                break

        # If we didn't route yet, finalize now
        if not routing_prediction or not routing_prediction.can_route:
            routing_prediction = self.router.finalize_routing(request.request_id)

        # Wait for RAG fetch to complete (if started)
        is_rag_query = False
        rag_context = None
        if rag_task:
            is_rag_query, rag_context = await rag_task
            if rag_context:
                # Inject context into prompt
                request.prompt = self.rag_fetcher.inject_context(request.prompt, rag_context)
                print(f"✅ [{request.request_id}] RAG context injected ({rag_context.tokens_count} tokens)")

        # Phase 2: Stream from expert
        expert_ids = routing_prediction.expert_ids
        expert_probs = routing_prediction.probabilities

        # Get expert (lazy loads if necessary)
        expert_id = expert_ids[0]
        expert_prob = expert_probs[0] if expert_probs else 1.0

        expert = await self.expert_system.get_expert(
            expert_id,
            predicted_probability=expert_prob
        )

        if not expert:
            raise ValueError(f"Expert not found: {expert_id}")

        # Prepare prompt
        full_prompt = request.prompt
        if request.system_prompt:
            full_prompt = f"{request.system_prompt}\n\n{request.prompt}"

        # Sampling params
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )

        # Stream tokens from expert using REAL vLLM streaming
        total_tokens = 0
        async for token in expert.generate_streaming(
            full_prompt,
            sampling_params,
            request_id=request.request_id
        ):
            # Track first token time
            if first_token_time is None:
                first_token_time = time.time()
                ttft = first_token_time - start_time
                print(f"✅ [{request.request_id}] TTFT: {ttft:.3f}s (expert: {expert_id})")

            yield token
            total_tokens += 1

        # Update metrics
        total_time = time.time() - start_time
        if first_token_time:
            self.total_ttft.append(first_token_time - start_time)
        self.total_latency.append(total_time)
        self.total_requests += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        stats = {
            'total_requests': self.total_requests,
            'router_stats': self.router.get_routing_stats(),
            'experts': self.expert_system.list_experts()
        }

        if self.total_ttft:
            stats['ttft'] = {
                'mean': np.mean(self.total_ttft),
                'median': np.median(self.total_ttft),
                'p95': np.percentile(self.total_ttft, 95),
                'p99': np.percentile(self.total_ttft, 99)
            }

        if self.total_latency:
            stats['latency'] = {
                'mean': np.mean(self.total_latency),
                'median': np.median(self.total_latency),
                'p95': np.percentile(self.total_latency, 95),
                'p99': np.percentile(self.total_latency, 99)
            }

        # Add RAG stats if enabled
        if self.rag_fetcher:
            stats['rag'] = self.rag_fetcher.get_stats()

        # Add speculative routing stats
        if self.speculative_router:
            stats['speculative_routing'] = self.speculative_router.get_stats()

        return stats


if __name__ == '__main__':
    print("🧪 Testing LiveMind Orchestrator")
    print("=" * 60)

    if not VLLM_AVAILABLE:
        print("❌ vLLM not installed. Install: pip install vllm")
        sys.exit(1)

    # Mock expert system for testing
    print("\n📝 Creating mock expert system...")

    expert_configs = [
        {
            'expert_id': 'expert_general',
            'model_path': 'facebook/opt-125m',
            'domain': 'general',
            'quantization': None,
            'enable_neon': True
        }
    ]

    try:
        expert_system = AxionMultiExpertVLLM(expert_configs)

        # Create orchestrator
        orchestrator = LiveMindOrchestrator(
            expert_system=expert_system,
            enable_consensus=True,
            chunk_size=64,
            routing_threshold=0.7
        )

        # Test request
        request = GenerationRequest(
            request_id='test_001',
            prompt='Explain the concept of PagedAttention in large language models.',
            max_tokens=100,
            temperature=0.7
        )

        print(f"\n🚀 Processing request: {request.request_id}")
        print(f"   Prompt: {request.prompt[:50]}...")

        # Generate
        async def run_test():
            result = await orchestrator.generate(request)

            print(f"\n✅ Generation complete!")
            print(f"   Expert: {result.expert_id}")
            print(f"   Tokens: {result.tokens_generated}")
            print(f"   TTFT: {result.time_to_first_token:.3f}s")
            print(f"   Total time: {result.total_time:.3f}s")
            print(f"   Chunks processed: {result.chunks_processed}")
            print(f"\n   Response: {result.text[:200]}...")

            # Stats
            stats = orchestrator.get_stats()
            print(f"\n📊 Orchestrator stats:")
            print(f"   Total requests: {stats['total_requests']}")

        asyncio.run(run_test())

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

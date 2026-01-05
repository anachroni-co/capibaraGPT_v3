"""
Lazy Expert Loading Manager
Gestiona carga/descarga dinámica de expertos vLLM para optimizar memoria

Features:
- Lazy loading: Solo carga expertos cuando se necesitan
- LRU eviction: Descarga expertos menos usados cuando memoria es alta
- Priority-based loading: Expertos más probables se cargan primero
- Memory monitoring: Track de uso de memoria en tiempo real
- Warmup pool: Mantiene N expertos más comunes siempre cargados
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import time
import threading
import psutil
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm_integration.vllm_axion_backend import AxionVLLMEngine, AxionVLLMConfig

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


@dataclass
class ExpertState:
    """State of an expert model"""
    expert_id: str
    config: AxionVLLMConfig
    domain: str
    priority: int = 0  # Higher = more important

    # Runtime state
    engine: Optional[AxionVLLMEngine] = None
    is_loaded: bool = False
    last_used: float = 0.0
    total_requests: int = 0
    total_load_time_s: float = 0.0
    load_count: int = 0

    # Memory estimation
    estimated_memory_gb: float = 0.0


@dataclass
class MemoryStats:
    """Current memory statistics"""
    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float
    expert_memory_gb: float  # Estimated memory used by loaded experts


class LazyExpertManager:
    """
    Manages lazy loading/unloading of vLLM expert engines

    Strategy:
    1. Start with warmup_pool_size experts loaded (highest priority)
    2. Load other experts on-demand when requested
    3. Unload LRU experts when memory usage > memory_threshold
    4. Track usage stats to optimize future loading decisions
    """

    def __init__(
        self,
        expert_configs: List[Dict[str, Any]],
        warmup_pool_size: int = 2,
        max_loaded_experts: int = 4,
        memory_threshold: float = 0.80,  # Unload when >80% memory used
        auto_unload_after_s: float = 300.0,  # Auto-unload after 5min idle
        enable_auto_unload: bool = True
    ):
        """
        Args:
            expert_configs: List of expert configurations
            warmup_pool_size: Number of experts to keep loaded always
            max_loaded_experts: Maximum experts loaded simultaneously
            memory_threshold: Unload LRU when memory > this (0-1)
            auto_unload_after_s: Auto-unload after this many seconds idle
            enable_auto_unload: Enable automatic unloading of idle experts
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM not installed. Install: pip install vllm")

        self.warmup_pool_size = warmup_pool_size
        self.max_loaded_experts = max_loaded_experts
        self.memory_threshold = memory_threshold
        self.auto_unload_after_s = auto_unload_after_s
        self.enable_auto_unload = enable_auto_unload

        # Initialize expert states
        self.experts: Dict[str, ExpertState] = {}
        self._initialize_experts(expert_configs)

        # LRU tracking (OrderedDict maintains insertion order)
        self.lru_order: OrderedDict[str, bool] = OrderedDict()

        # Memory tracking
        self.system_memory_gb = psutil.virtual_memory().total / (1024**3)

        # Lock for thread safety
        self.lock = threading.Lock()

        # Stats
        self.total_loads = 0
        self.total_unloads = 0
        self.total_evictions = 0
        self.cache_hits = 0  # Expert already loaded
        self.cache_misses = 0  # Had to load expert

        print(f"✅ LazyExpertManager initialized")
        print(f"   Total experts: {len(self.experts)}")
        print(f"   Warmup pool: {warmup_pool_size}")
        print(f"   Max loaded: {max_loaded_experts}")
        print(f"   Memory threshold: {memory_threshold * 100:.0f}%")
        print(f"   System memory: {self.system_memory_gb:.1f} GB")

        # Start warmup loading
        self._warmup_experts()

        # Start auto-unload thread if enabled
        if enable_auto_unload:
            self._start_auto_unload_thread()

    def _initialize_experts(self, expert_configs: List[Dict[str, Any]]):
        """Initialize expert states from configs"""
        for i, config_dict in enumerate(expert_configs):
            expert_id = config_dict.get('expert_id', f'expert_{i}')
            domain = config_dict.get('domain', 'general')
            priority = config_dict.get('priority', i)  # Default: order in list

            # Create config
            config = AxionVLLMConfig(
                model_path=config_dict['model_path'],
                quantization=config_dict.get('quantization', None),
                tensor_parallel_size=config_dict.get('tensor_parallel_size', 1),
                gpu_memory_utilization=config_dict.get('gpu_memory_utilization', 0.85),
                max_num_seqs=config_dict.get('max_num_seqs', 128),
                enable_neon=config_dict.get('enable_neon', True),
                enable_chunked_prefill=config_dict.get('enable_chunked_prefill', True),
            )

            # Estimate memory based on quantization
            estimated_memory = self._estimate_model_memory(
                config_dict['model_path'],
                config_dict.get('quantization', None)
            )

            expert_state = ExpertState(
                expert_id=expert_id,
                config=config,
                domain=domain,
                priority=priority,
                estimated_memory_gb=estimated_memory
            )

            self.experts[expert_id] = expert_state

        # Sort by priority for warmup
        self.experts = dict(
            sorted(self.experts.items(), key=lambda x: x[1].priority, reverse=True)
        )

    def _estimate_model_memory(self, model_path: str, quantization: Optional[str]) -> float:
        """
        Estimate model memory usage in GB

        Based on model name and quantization
        """
        # Simple heuristic based on model size in name
        model_lower = model_path.lower()

        # Extract parameter count from path
        if '125m' in model_lower:
            base_params = 0.125
        elif '1b' in model_lower or '1.5b' in model_lower:
            base_params = 1.5
        elif '3b' in model_lower:
            base_params = 3
        elif '7b' in model_lower:
            base_params = 7
        elif '13b' in model_lower:
            base_params = 13
        elif '20b' in model_lower:
            base_params = 20
        elif '70b' in model_lower:
            base_params = 70
        else:
            # Default: assume 7B
            base_params = 7

        # Calculate memory based on quantization
        # FP16: 2 bytes per parameter
        # AWQ/GPTQ: ~0.5 bytes per parameter (4-bit)
        # Q4_0: ~0.5 bytes per parameter
        # Q8_0: ~1 byte per parameter

        if quantization in ['awq', 'gptq', 'q4_0', 'squeezellm']:
            bytes_per_param = 0.5
        elif quantization == 'q8_0':
            bytes_per_param = 1.0
        else:
            # FP16 default
            bytes_per_param = 2.0

        memory_gb = base_params * bytes_per_param

        # Add overhead for KV cache, activations (~30%)
        memory_gb *= 1.3

        return memory_gb

    def _warmup_experts(self):
        """Load warmup pool experts on startup"""
        print(f"\n🔥 Warming up {self.warmup_pool_size} experts...")

        # Get top priority experts
        warmup_experts = list(self.experts.values())[:self.warmup_pool_size]

        for expert_state in warmup_experts:
            try:
                print(f"   Loading {expert_state.expert_id} (priority {expert_state.priority})...")
                self._load_expert(expert_state.expert_id)
            except Exception as e:
                print(f"   ⚠️  Failed to load {expert_state.expert_id}: {e}")

        print(f"✅ Warmup complete: {self._get_loaded_count()}/{self.warmup_pool_size} experts loaded")

    async def get_expert(
        self,
        expert_id: str,
        predicted_probability: float = 1.0
    ) -> Optional[AxionVLLMEngine]:
        """
        Get expert engine, loading if necessary

        Args:
            expert_id: Expert identifier
            predicted_probability: Routing probability (for eviction decisions)

        Returns:
            Loaded expert engine or None if failed
        """
        if expert_id not in self.experts:
            print(f"❌ Unknown expert: {expert_id}")
            return None

        expert_state = self.experts[expert_id]

        with self.lock:
            # Update last used time
            expert_state.last_used = time.time()
            expert_state.total_requests += 1

            # Update LRU order
            if expert_id in self.lru_order:
                self.lru_order.move_to_end(expert_id)
                self.cache_hits += 1
            else:
                self.cache_misses += 1

            # If already loaded, return it
            if expert_state.is_loaded and expert_state.engine:
                return expert_state.engine

            # Need to load - check memory first
            mem_stats = self._get_memory_stats()

            # If memory is tight and we're at max loaded, evict LRU
            if (mem_stats.percent_used / 100 > self.memory_threshold or
                self._get_loaded_count() >= self.max_loaded_experts):

                # Evict LRU expert (but not from warmup pool)
                evicted = self._evict_lru_expert()
                if evicted:
                    print(f"♻️  Evicted {evicted} to make room for {expert_id}")

            # Load expert
            print(f"📥 Loading expert {expert_id} (prob: {predicted_probability:.2f})...")
            success = self._load_expert(expert_id)

            if success:
                return expert_state.engine
            else:
                return None

    def _load_expert(self, expert_id: str) -> bool:
        """
        Load expert engine

        Returns:
            True if loaded successfully
        """
        expert_state = self.experts[expert_id]

        if expert_state.is_loaded:
            return True

        try:
            start_time = time.time()

            # Create engine
            engine = AxionVLLMEngine(
                config=expert_state.config,
                engine_id=expert_id
            )

            load_time = time.time() - start_time

            # Update state
            expert_state.engine = engine
            expert_state.is_loaded = True
            expert_state.last_used = time.time()
            expert_state.total_load_time_s += load_time
            expert_state.load_count += 1

            # Add to LRU
            self.lru_order[expert_id] = True
            self.lru_order.move_to_end(expert_id)

            # Stats
            self.total_loads += 1

            print(f"✅ Expert {expert_id} loaded in {load_time:.1f}s")
            return True

        except Exception as e:
            print(f"❌ Failed to load expert {expert_id}: {e}")
            return False

    def _unload_expert(self, expert_id: str, is_eviction: bool = False) -> bool:
        """
        Unload expert engine

        Args:
            expert_id: Expert to unload
            is_eviction: Whether this is an eviction (vs manual unload)

        Returns:
            True if unloaded successfully
        """
        if expert_id not in self.experts:
            return False

        expert_state = self.experts[expert_id]

        if not expert_state.is_loaded:
            return False

        try:
            # Delete engine (vLLM will cleanup)
            expert_state.engine = None
            expert_state.is_loaded = False

            # Remove from LRU
            if expert_id in self.lru_order:
                del self.lru_order[expert_id]

            # Stats
            self.total_unloads += 1
            if is_eviction:
                self.total_evictions += 1

            print(f"🗑️  Expert {expert_id} unloaded ({'eviction' if is_eviction else 'manual'})")
            return True

        except Exception as e:
            print(f"⚠️  Error unloading {expert_id}: {e}")
            return False

    def _evict_lru_expert(self) -> Optional[str]:
        """
        Evict least recently used expert

        Will not evict from warmup pool

        Returns:
            Expert ID that was evicted, or None
        """
        if len(self.lru_order) == 0:
            return None

        # Get warmup expert IDs (protected from eviction)
        warmup_ids = set(list(self.experts.keys())[:self.warmup_pool_size])

        # Find LRU expert not in warmup pool
        for expert_id in self.lru_order:
            if expert_id not in warmup_ids:
                # Evict this one
                success = self._unload_expert(expert_id, is_eviction=True)
                if success:
                    return expert_id

        # All loaded experts are in warmup pool - can't evict
        return None

    def _get_loaded_count(self) -> int:
        """Get number of currently loaded experts"""
        return sum(1 for e in self.experts.values() if e.is_loaded)

    def _get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        vm = psutil.virtual_memory()

        # Estimate expert memory
        expert_memory_gb = sum(
            e.estimated_memory_gb
            for e in self.experts.values()
            if e.is_loaded
        )

        return MemoryStats(
            total_gb=vm.total / (1024**3),
            available_gb=vm.available / (1024**3),
            used_gb=vm.used / (1024**3),
            percent_used=vm.percent,
            expert_memory_gb=expert_memory_gb
        )

    def _start_auto_unload_thread(self):
        """Start background thread to auto-unload idle experts"""
        def auto_unload_worker():
            while True:
                time.sleep(60)  # Check every minute

                with self.lock:
                    current_time = time.time()
                    warmup_ids = set(list(self.experts.keys())[:self.warmup_pool_size])

                    for expert_id, expert_state in self.experts.items():
                        # Skip warmup pool
                        if expert_id in warmup_ids:
                            continue

                        # Skip if not loaded
                        if not expert_state.is_loaded:
                            continue

                        # Check if idle for too long
                        idle_time = current_time - expert_state.last_used
                        if idle_time > self.auto_unload_after_s:
                            print(f"⏰ Auto-unloading idle expert {expert_id} (idle {idle_time:.0f}s)")
                            self._unload_expert(expert_id, is_eviction=False)

        thread = threading.Thread(target=auto_unload_worker, daemon=True)
        thread.start()
        print(f"✅ Auto-unload thread started (idle threshold: {self.auto_unload_after_s}s)")

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        mem_stats = self._get_memory_stats()
        loaded_count = self._get_loaded_count()

        # Per-expert stats
        expert_stats = []
        for expert_id, expert_state in self.experts.items():
            expert_stats.append({
                'expert_id': expert_id,
                'domain': expert_state.domain,
                'priority': expert_state.priority,
                'is_loaded': expert_state.is_loaded,
                'total_requests': expert_state.total_requests,
                'load_count': expert_state.load_count,
                'avg_load_time_s': (
                    expert_state.total_load_time_s / expert_state.load_count
                    if expert_state.load_count > 0 else 0
                ),
                'estimated_memory_gb': expert_state.estimated_memory_gb,
                'idle_time_s': time.time() - expert_state.last_used if expert_state.last_used > 0 else 0
            })

        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)

        return {
            'total_experts': len(self.experts),
            'loaded_experts': loaded_count,
            'warmup_pool_size': self.warmup_pool_size,
            'max_loaded_experts': self.max_loaded_experts,
            'total_loads': self.total_loads,
            'total_unloads': self.total_unloads,
            'total_evictions': self.total_evictions,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'memory': {
                'total_gb': mem_stats.total_gb,
                'used_gb': mem_stats.used_gb,
                'available_gb': mem_stats.available_gb,
                'percent_used': mem_stats.percent_used,
                'expert_memory_gb': mem_stats.expert_memory_gb,
                'threshold_percent': self.memory_threshold * 100
            },
            'experts': expert_stats
        }

    def list_loaded_experts(self) -> List[str]:
        """Get list of currently loaded expert IDs"""
        return [
            expert_id
            for expert_id, expert_state in self.experts.items()
            if expert_state.is_loaded
        ]


if __name__ == '__main__':
    print("🧪 Testing Lazy Expert Manager")
    print("=" * 60)

    if not VLLM_AVAILABLE:
        print("❌ vLLM not installed")
        sys.exit(1)

    # Test config
    test_configs = [
        {
            'expert_id': 'phi4_fast',
            'model_path': 'facebook/opt-125m',  # Small model for testing
            'domain': 'general',
            'quantization': None,
            'priority': 3,
            'max_num_seqs': 4
        },
        {
            'expert_id': 'mistral_balanced',
            'model_path': 'facebook/opt-125m',
            'domain': 'technical',
            'quantization': None,
            'priority': 2,
            'max_num_seqs': 4
        },
        {
            'expert_id': 'qwen_multilingual',
            'model_path': 'facebook/opt-125m',
            'domain': 'multilingual',
            'quantization': None,
            'priority': 1,
            'max_num_seqs': 4
        }
    ]

    # Create manager
    manager = LazyExpertManager(
        expert_configs=test_configs,
        warmup_pool_size=2,
        max_loaded_experts=2,
        memory_threshold=0.80,
        enable_auto_unload=False  # Disable for test
    )

    print("\n📊 Initial Stats:")
    stats = manager.get_stats()
    print(f"   Loaded: {stats['loaded_experts']}/{stats['total_experts']}")
    print(f"   Memory: {stats['memory']['expert_memory_gb']:.1f} GB")

    # Test loading
    async def test_loading():
        print("\n📝 Test: Get expert (should be warmup hit)")
        expert1 = await manager.get_expert('phi4_fast')
        print(f"   Got expert: {expert1 is not None}")

        print("\n📝 Test: Get expert (should trigger load)")
        expert2 = await manager.get_expert('qwen_multilingual')
        print(f"   Got expert: {expert2 is not None}")

        print("\n📝 Test: Get expert again (cache hit)")
        expert1_again = await manager.get_expert('phi4_fast')
        print(f"   Same instance: {expert1 is expert1_again}")

        print("\n📊 Final Stats:")
        stats = manager.get_stats()
        print(f"   Loaded: {stats['loaded_experts']}/{stats['total_experts']}")
        print(f"   Cache hits: {stats['cache_hits']}")
        print(f"   Cache misses: {stats['cache_misses']}")
        print(f"   Hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"   Evictions: {stats['total_evictions']}")

    asyncio.run(test_loading())

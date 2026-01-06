"""
Continuum Memory System for CapibaraGPT

This module implements a continuous memory system inspired by the HOPE architecture
from Nested Learning (Behrouz et al., NeurIPS 2025). The core idea is to manage
memory at multiple time scales with different update frequencies to avoid
catastrophic forgetting.

Key Features:
- Multi-scale memory banks (ultra-short, short, medium, long-term)
- Different update frequencies per memory bank
- Memory consolidation from fast to slow banks
- Cross-temporal attention across memory banks
- Decay mechanisms for graceful forgetting

Memory Banks:
- Ultra-Short: 16 tokens, updates every step (immediate context)
- Short: 256 tokens, updates every 10 steps (recent context)
- Medium: 2048 tokens, updates every 100 steps (session context)
- Long: 8192 tokens, updates every 1000 steps (knowledge base)

This implements the key Nested Learning principle of different update frequencies
to separate concerns across time scales.

Reference:
    Behrouz, A.; Razaviyayn, M.; Mirrokni, V.; Zhong, P. (2025).
    "Nested Learning: The Illusion of Deep Learning Architectures"
    NeurIPS 2025 - HOPE Architecture
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import json

try:
    import numpy as np
except ImportError:
    # Fallback for systems without numpy
    class np:  # type: ignore
        @staticmethod
        def mean(x):
            return sum(x) / len(x) if x else 0

        @staticmethod
        def clip(x, min_val, max_val):
            return max(min_val, min(max_val, x))

        @staticmethod
        def exp(x):
            import math
            return math.exp(x)

        @staticmethod
        def array(x):
            return list(x)

logger = logging.getLogger(__name__)


@dataclass
class MemoryBankConfig:
    """Configuration for a single memory bank."""

    name: str
    capacity: int  # Number of memory slots
    update_freq: int  # Update every N steps
    decay_rate: float  # Decay rate per update (0.0 = no decay, 1.0 = full decay)
    importance_threshold: float = 0.5  # Minimum importance to keep
    enable_consolidation: bool = True  # Allow consolidation to slower banks


@dataclass
class MemoryEntry:
    """Single entry in a memory bank."""

    content: Any  # The actual memory content (could be embeddings, text, etc.)
    importance: float  # Importance score [0, 1]
    timestamp: int  # Step when created
    access_count: int = 0  # How many times accessed
    last_access: int = 0  # Last access step
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_access(self, current_step: int):
        """Update access statistics."""
        self.access_count += 1
        self.last_access = current_step

    def calculate_relevance(self, current_step: int, recency_weight: float = 0.3) -> float:
        """
        Calculate relevance score based on importance, recency, and access.

        Args:
            current_step: Current training step
            recency_weight: Weight for recency vs importance

        Returns:
            Relevance score [0, 1]
        """
        # Recency score (more recent = higher score)
        age = current_step - self.timestamp
        recency = np.exp(-age * 0.001)  # Exponential decay

        # Access frequency score
        access_score = min(self.access_count / 10.0, 1.0)  # Normalize to [0, 1]

        # Combined relevance
        relevance = (
            (1 - recency_weight) * self.importance +
            recency_weight * recency +
            0.1 * access_score
        )

        return np.clip(relevance, 0, 1)


class MemoryBank:
    """
    A single memory bank with specific capacity and update frequency.

    Implements the core Nested Learning concept of different update rates
    for different memory components.
    """

    def __init__(self, config: MemoryBankConfig):
        self.config = config
        self.memories: deque = deque(maxlen=config.capacity)
        self.last_update_step = 0
        self.total_writes = 0
        self.total_reads = 0

        logger.info(
            f"📝 Memory Bank '{config.name}' initialized: "
            f"capacity={config.capacity}, freq={config.update_freq}"
        )

    def should_update(self, current_step: int) -> bool:
        """Check if bank should update at current step."""
        return current_step % self.config.update_freq == 0

    def write(self, entry: MemoryEntry, current_step: int) -> bool:
        """
        Write a memory entry to the bank.

        Args:
            entry: Memory entry to write
            current_step: Current step

        Returns:
            True if write succeeded
        """
        if not self.should_update(current_step):
            return False

        # Check importance threshold
        if entry.importance < self.config.importance_threshold:
            logger.debug(
                f"Memory rejected in '{self.config.name}': "
                f"importance {entry.importance:.3f} < threshold {self.config.importance_threshold}"
            )
            return False

        self.memories.append(entry)
        self.total_writes += 1
        self.last_update_step = current_step

        logger.debug(
            f"Memory written to '{self.config.name}': "
            f"importance={entry.importance:.3f}, step={current_step}"
        )

        return True

    def read(self, query: Optional[Any] = None, current_step: int = 0, top_k: int = 5) -> List[MemoryEntry]:
        """
        Read relevant memories from the bank.

        Args:
            query: Optional query for retrieval (not implemented in basic version)
            current_step: Current step for relevance calculation
            top_k: Number of top memories to return

        Returns:
            List of most relevant memory entries
        """
        if not self.memories:
            return []

        self.total_reads += 1

        # Calculate relevance for all memories
        memories_with_relevance = [
            (mem, mem.calculate_relevance(current_step))
            for mem in self.memories
        ]

        # Sort by relevance
        memories_with_relevance.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        top_memories = [mem for mem, _ in memories_with_relevance[:top_k]]

        # Update access statistics
        for mem in top_memories:
            mem.update_access(current_step)

        return top_memories

    def apply_decay(self, current_step: int):
        """Apply decay to all memories based on decay rate."""
        if not self.should_update(current_step):
            return

        decayed_count = 0
        new_memories = deque(maxlen=self.config.capacity)

        for mem in self.memories:
            # Apply decay to importance
            mem.importance *= (1 - self.config.decay_rate)

            # Keep if still above threshold
            if mem.importance >= self.config.importance_threshold:
                new_memories.append(mem)
            else:
                decayed_count += 1

        self.memories = new_memories

        if decayed_count > 0:
            logger.debug(
                f"Decay in '{self.config.name}': {decayed_count} memories dropped "
                f"(step={current_step})"
            )

    def get_important_memories(self, threshold: float = 0.7, current_step: int = 0) -> List[MemoryEntry]:
        """
        Get memories above a certain importance threshold for consolidation.

        Args:
            threshold: Importance threshold
            current_step: Current step for relevance

        Returns:
            List of important memories
        """
        important = [
            mem for mem in self.memories
            if mem.importance >= threshold
        ]

        # Sort by relevance
        important.sort(
            key=lambda m: m.calculate_relevance(current_step),
            reverse=True
        )

        return important

    def consolidate(self, memories: List[MemoryEntry]):
        """
        Consolidate memories from a faster bank.

        Args:
            memories: Memories to consolidate
        """
        for mem in memories:
            # Increase importance for consolidated memories
            mem.importance = min(mem.importance * 1.2, 1.0)

            # Add to this bank
            self.memories.append(mem)

        logger.info(
            f"Consolidated {len(memories)} memories into '{self.config.name}'"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about this memory bank."""
        return {
            'name': self.config.name,
            'capacity': self.config.capacity,
            'current_size': len(self.memories),
            'utilization': len(self.memories) / self.config.capacity,
            'total_writes': self.total_writes,
            'total_reads': self.total_reads,
            'last_update_step': self.last_update_step,
            'avg_importance': np.mean([m.importance for m in self.memories]) if self.memories else 0.0,
        }


class CrossTemporalAttention:
    """
    Cross-temporal attention mechanism that attends to multiple memory banks.

    This allows the model to leverage context from different time scales
    simultaneously, a key feature of Nested Learning.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.attention_history = []

    def attend(
        self,
        query: Any,
        memory_contents: Dict[str, List[MemoryEntry]],
        bank_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Apply cross-temporal attention across memory banks.

        Args:
            query: Query (could be embeddings, not implemented in basic version)
            memory_contents: Dict of bank_name -> list of memory entries
            bank_weights: Optional weights for each bank

        Returns:
            Combined attention output
        """
        if not memory_contents:
            return {'combined_memories': [], 'attention_weights': {}}

        # Default weights (could be learned)
        if bank_weights is None:
            bank_weights = {
                'ultra_short': 0.4,  # High weight on recent context
                'short': 0.3,
                'medium': 0.2,
                'long': 0.1,  # Low weight on long-term
            }

        combined_memories = []
        attention_weights = {}

        for bank_name, memories in memory_contents.items():
            if not memories:
                continue

            weight = bank_weights.get(bank_name, 0.25)  # Default equal weight

            # Simple weighted aggregation (in practice, would use embeddings)
            weighted_memories = [
                {
                    'content': mem.content,
                    'importance': mem.importance * weight,
                    'source_bank': bank_name
                }
                for mem in memories
            ]

            combined_memories.extend(weighted_memories)
            attention_weights[bank_name] = weight

        # Sort by weighted importance
        combined_memories.sort(key=lambda x: x['importance'], reverse=True)

        # Record attention pattern
        self.attention_history.append({
            'weights': attention_weights.copy(),
            'num_memories': len(combined_memories)
        })

        return {
            'combined_memories': combined_memories[:20],  # Top 20
            'attention_weights': attention_weights
        }


@dataclass
class ContinuumMemoryConfig:
    """Configuration for the continuum memory system."""

    # Bank configurations
    ultra_short_capacity: int = 16
    short_capacity: int = 256
    medium_capacity: int = 2048
    long_capacity: int = 8192

    # Update frequencies (key to Nested Learning)
    ultra_short_freq: int = 1  # Every step
    short_freq: int = 10
    medium_freq: int = 100
    long_freq: int = 1000

    # Decay rates
    ultra_short_decay: float = 0.1  # Fast decay
    short_decay: float = 0.05
    medium_decay: float = 0.01
    long_decay: float = 0.001  # Slow decay

    # Consolidation
    consolidation_interval: int = 1000  # Consolidate every N steps
    consolidation_threshold: float = 0.7  # Minimum importance for consolidation
    enable_consolidation: bool = True


class ContinuumMemorySystem:
    """
    Continuum Memory System implementing multi-scale temporal memory.

    This is the core implementation of the HOPE-inspired memory system from
    Nested Learning. It maintains multiple memory banks that update at
    different frequencies, allowing the model to:

    1. Maintain immediate context (ultra-short)
    2. Track recent conversation/session (short)
    3. Remember important session information (medium)
    4. Retain long-term knowledge (long)

    The different update frequencies prevent catastrophic forgetting by
    separating fast adaptation from slow knowledge accumulation.
    """

    def __init__(self, config: Optional[ContinuumMemoryConfig] = None):
        self.config = config or ContinuumMemoryConfig()

        # Initialize memory banks
        self.memory_banks = {
            'ultra_short': MemoryBank(MemoryBankConfig(
                name='ultra_short',
                capacity=self.config.ultra_short_capacity,
                update_freq=self.config.ultra_short_freq,
                decay_rate=self.config.ultra_short_decay,
                importance_threshold=0.0,  # Accept all for immediate context
            )),
            'short': MemoryBank(MemoryBankConfig(
                name='short',
                capacity=self.config.short_capacity,
                update_freq=self.config.short_freq,
                decay_rate=self.config.short_decay,
                importance_threshold=0.3,
            )),
            'medium': MemoryBank(MemoryBankConfig(
                name='medium',
                capacity=self.config.medium_capacity,
                update_freq=self.config.medium_freq,
                decay_rate=self.config.medium_decay,
                importance_threshold=0.5,
            )),
            'long': MemoryBank(MemoryBankConfig(
                name='long',
                capacity=self.config.long_capacity,
                update_freq=self.config.long_freq,
                decay_rate=self.config.long_decay,
                importance_threshold=0.7,
            )),
        }

        # Cross-temporal attention
        self.cross_temporal_attention = CrossTemporalAttention()

        # State
        self.current_step = 0
        self.last_consolidation_step = 0
        self.consolidation_history = []

        logger.info("🧠 Continuum Memory System initialized")
        logger.info(f"   Ultra-Short: {self.config.ultra_short_capacity} slots, freq={self.config.ultra_short_freq}")
        logger.info(f"   Short: {self.config.short_capacity} slots, freq={self.config.short_freq}")
        logger.info(f"   Medium: {self.config.medium_capacity} slots, freq={self.config.medium_freq}")
        logger.info(f"   Long: {self.config.long_capacity} slots, freq={self.config.long_freq}")

    def write_memory(
        self,
        content: Any,
        importance: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """
        Write content to appropriate memory banks based on update frequencies.

        Args:
            content: Content to store
            importance: Importance score [0, 1]
            metadata: Optional metadata

        Returns:
            Dict mapping bank names to write success
        """
        self.current_step += 1

        # Create memory entry
        entry = MemoryEntry(
            content=content,
            importance=importance,
            timestamp=self.current_step,
            metadata=metadata or {}
        )

        # Write to banks based on their update frequency
        write_results = {}
        for bank_name, bank in self.memory_banks.items():
            success = bank.write(entry, self.current_step)
            write_results[bank_name] = success

            # Apply decay
            bank.apply_decay(self.current_step)

        # Periodic consolidation
        if self.config.enable_consolidation:
            if (self.current_step - self.last_consolidation_step) >= self.config.consolidation_interval:
                self.consolidate_memory()

        return write_results

    def read_memory(
        self,
        query: Optional[Any] = None,
        bank_weights: Optional[Dict[str, float]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Read from all memory banks with cross-temporal attention.

        Args:
            query: Optional query for retrieval
            bank_weights: Optional weights for each bank
            top_k: Number of memories to retrieve per bank

        Returns:
            Combined memory output with attention weights
        """
        # Read from each bank
        memory_contents = {}
        for bank_name, bank in self.memory_banks.items():
            if bank.should_update(self.current_step) or bank_name == 'ultra_short':
                memories = bank.read(query, self.current_step, top_k)
                if memories:
                    memory_contents[bank_name] = memories

        # Apply cross-temporal attention
        attended_output = self.cross_temporal_attention.attend(
            query,
            memory_contents,
            bank_weights
        )

        return attended_output

    def consolidate_memory(self):
        """
        Consolidate important memories from faster banks to slower banks.

        This implements the key Nested Learning concept of memory consolidation
        across time scales, preventing catastrophic forgetting.
        """
        logger.info(f"🔄 Consolidating memory at step {self.current_step}")

        consolidation_report = {
            'step': self.current_step,
            'transfers': {}
        }

        # Consolidate: ultra_short -> short
        important_ultra = self.memory_banks['ultra_short'].get_important_memories(
            threshold=self.config.consolidation_threshold,
            current_step=self.current_step
        )
        if important_ultra:
            self.memory_banks['short'].consolidate(important_ultra[:50])  # Top 50
            consolidation_report['transfers']['ultra_short->short'] = len(important_ultra[:50])

        # Consolidate: short -> medium
        important_short = self.memory_banks['short'].get_important_memories(
            threshold=self.config.consolidation_threshold,
            current_step=self.current_step
        )
        if important_short:
            self.memory_banks['medium'].consolidate(important_short[:20])  # Top 20
            consolidation_report['transfers']['short->medium'] = len(important_short[:20])

        # Consolidate: medium -> long
        important_medium = self.memory_banks['medium'].get_important_memories(
            threshold=self.config.consolidation_threshold + 0.1,  # Higher threshold
            current_step=self.current_step
        )
        if important_medium:
            self.memory_banks['long'].consolidate(important_medium[:10])  # Top 10
            consolidation_report['transfers']['medium->long'] = len(important_medium[:10])

        self.last_consolidation_step = self.current_step
        self.consolidation_history.append(consolidation_report)

        logger.info(f"   Consolidation report: {consolidation_report['transfers']}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the memory system."""
        bank_stats = {
            name: bank.get_statistics()
            for name, bank in self.memory_banks.items()
        }

        total_capacity = sum(b['capacity'] for b in bank_stats.values())
        total_memories = sum(b['current_size'] for b in bank_stats.values())

        return {
            'current_step': self.current_step,
            'total_capacity': total_capacity,
            'total_memories': total_memories,
            'global_utilization': total_memories / total_capacity if total_capacity > 0 else 0,
            'banks': bank_stats,
            'consolidations': len(self.consolidation_history),
            'last_consolidation': self.last_consolidation_step,
        }

    def reset(self):
        """Reset all memory banks."""
        for bank in self.memory_banks.values():
            bank.memories.clear()
            bank.total_writes = 0
            bank.total_reads = 0

        self.current_step = 0
        self.last_consolidation_step = 0
        self.consolidation_history.clear()

        logger.info("🔄 Continuum Memory System reset")


# Factory function
def create_continuum_memory(config: Optional[ContinuumMemoryConfig] = None) -> ContinuumMemorySystem:
    """Create a continuum memory system instance."""
    return ContinuumMemorySystem(config)


# Global instance
_global_continuum_memory: Optional[ContinuumMemorySystem] = None


def get_global_continuum_memory() -> ContinuumMemorySystem:
    """Get the global continuum memory instance."""
    global _global_continuum_memory
    if _global_continuum_memory is None:
        _global_continuum_memory = create_continuum_memory()
    return _global_continuum_memory


def main():
    """Test the continuum memory system."""
    logging.basicConfig(level=logging.INFO)
    logger.info("🧠 Continuum Memory System - Testing Mode")

    # Create system
    config = ContinuumMemoryConfig(
        consolidation_interval=50  # Faster for testing
    )
    memory_system = create_continuum_memory(config)

    # Simulate memory operations
    for step in range(200):
        # Write different types of memories
        if step % 5 == 0:
            # Important memory
            memory_system.write_memory(
                content=f"Important fact {step}",
                importance=0.8 + (step % 10) * 0.02,
                metadata={'type': 'important', 'step': step}
            )
        else:
            # Regular memory
            memory_system.write_memory(
                content=f"Regular info {step}",
                importance=0.3 + (step % 5) * 0.05,
                metadata={'type': 'regular', 'step': step}
            )

        # Periodic reads
        if step % 25 == 0:
            output = memory_system.read_memory()
            logger.info(f"\nStep {step} - Memory Read:")
            logger.info(f"  Retrieved {len(output['combined_memories'])} memories")
            logger.info(f"  Attention weights: {output['attention_weights']}")

    # Final statistics
    stats = memory_system.get_statistics()
    logger.info("\n📊 Final Statistics:")
    logger.info(f"  Total memories: {stats['total_memories']}/{stats['total_capacity']}")
    logger.info(f"  Global utilization: {stats['global_utilization']:.1%}")
    logger.info(f"  Consolidations: {stats['consolidations']}")

    for bank_name, bank_stats in stats['banks'].items():
        logger.info(f"\n  {bank_name}:")
        logger.info(f"    Size: {bank_stats['current_size']}/{bank_stats['capacity']}")
        logger.info(f"    Utilization: {bank_stats['utilization']:.1%}")
        logger.info(f"    Writes: {bank_stats['total_writes']}, Reads: {bank_stats['total_reads']}")
        logger.info(f"    Avg importance: {bank_stats['avg_importance']:.3f}")


if __name__ == "__main__":
    main()

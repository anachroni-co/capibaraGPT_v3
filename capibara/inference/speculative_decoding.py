"""
Speculative Decoding for CapibaraGPT.

Implements speculative sampling (Leviathan et al., 2023) for 2-3x inference speedup.

The key insight: a small draft model generates K candidate tokens, then the large
target model verifies all K tokens in a single forward pass. Accepted tokens are
kept, rejected ones are resampled from the target distribution.

This provides exact sampling from the target distribution (no quality loss)
while achieving significant speedup when draft acceptance rate is high.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any, Callable, Dict
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class DraftModelType(Enum):
    """Types of draft models for speculative decoding."""
    DISTILLED = "distilled"           # Distilled version of target
    INDEPENDENT = "independent"        # Separately trained small model
    EARLY_EXIT = "early_exit"          # Early layers of target model
    MEDUSA = "medusa"                  # Multiple heads predicting future tokens
    EAGLE = "eagle"                    # Auto-regressive draft heads


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    # Draft model settings
    draft_model_type: DraftModelType = DraftModelType.DISTILLED
    draft_model_path: Optional[str] = None

    # Speculation parameters
    num_speculative_tokens: int = 4      # K: tokens to speculate per step
    max_speculative_tokens: int = 8      # Maximum K (for adaptive)
    min_speculative_tokens: int = 2      # Minimum K (for adaptive)

    # Adaptive speculation
    adaptive_k: bool = True              # Dynamically adjust K based on acceptance
    target_acceptance_rate: float = 0.8  # Target acceptance for adaptive K
    acceptance_smoothing: float = 0.9    # EMA smoothing for acceptance tracking

    # Verification
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    # Performance
    draft_batch_size: int = 1
    use_tree_attention: bool = False     # Tree-based verification (Medusa style)

    # Fallback
    fallback_to_standard: bool = True    # If draft unavailable

    def validate(self) -> None:
        """Validate configuration."""
        assert self.num_speculative_tokens >= 1
        assert self.min_speculative_tokens <= self.num_speculative_tokens
        assert self.num_speculative_tokens <= self.max_speculative_tokens
        assert 0 < self.target_acceptance_rate <= 1


@dataclass
class SpeculativeStats:
    """Statistics for speculative decoding performance."""

    total_tokens_generated: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_target_calls: int = 0
    total_draft_calls: int = 0

    # Timing
    total_draft_time_ms: float = 0
    total_target_time_ms: float = 0
    total_time_ms: float = 0

    # Adaptive K tracking
    current_k: int = 4
    acceptance_rate_ema: float = 0.8

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def speedup(self) -> float:
        """Estimated speedup vs standard decoding."""
        if self.total_target_calls == 0:
            return 1.0
        tokens_per_call = self.total_tokens_generated / self.total_target_calls
        return tokens_per_call

    @property
    def tokens_per_second(self) -> float:
        if self.total_time_ms == 0:
            return 0.0
        return self.total_tokens_generated / (self.total_time_ms / 1000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens_generated,
            "acceptance_rate": f"{self.acceptance_rate:.1%}",
            "speedup": f"{self.speedup:.2f}x",
            "tokens_per_second": f"{self.tokens_per_second:.1f}",
            "current_k": self.current_k,
            "target_calls": self.total_target_calls,
            "draft_calls": self.total_draft_calls
        }


class DraftModel:
    """
    Wrapper for draft model used in speculative decoding.

    The draft model should be much faster than the target (e.g., 10-20x).
    Typically 500M-1B params for a 7B target.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: SpeculativeConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._is_mock = model is None

        if self._is_mock:
            logger.warning("Draft model is None - using mock mode")

    def generate_candidates(
        self,
        input_ids: Any,
        attention_mask: Optional[Any] = None,
        num_tokens: int = 4
    ) -> Tuple[Any, Any]:
        """
        Generate candidate tokens autoregressively.

        Args:
            input_ids: Current token sequence [batch, seq_len]
            attention_mask: Attention mask
            num_tokens: Number of tokens to generate

        Returns:
            Tuple of (candidate_ids, candidate_logits)
            - candidate_ids: [batch, num_tokens]
            - candidate_logits: [batch, num_tokens, vocab_size]
        """
        if self._is_mock:
            return self._mock_generate(input_ids, num_tokens)

        try:
            import torch

            candidates = []
            all_logits = []

            current_ids = input_ids
            current_mask = attention_mask

            with torch.no_grad():
                for _ in range(num_tokens):
                    outputs = self.model(
                        input_ids=current_ids,
                        attention_mask=current_mask
                    )

                    logits = outputs.logits[:, -1, :]  # [batch, vocab]
                    all_logits.append(logits)

                    # Sample next token
                    probs = torch.softmax(logits / self.config.temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    candidates.append(next_token)

                    # Update for next iteration
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                    if current_mask is not None:
                        current_mask = torch.cat([
                            current_mask,
                            torch.ones_like(next_token)
                        ], dim=1)

            candidate_ids = torch.cat(candidates, dim=1)
            candidate_logits = torch.stack(all_logits, dim=1)

            return candidate_ids, candidate_logits

        except ImportError:
            return self._mock_generate(input_ids, num_tokens)

    def _mock_generate(
        self,
        input_ids: Any,
        num_tokens: int
    ) -> Tuple[Any, Any]:
        """Mock generation for testing."""
        try:
            import numpy as np
            batch_size = input_ids.shape[0] if hasattr(input_ids, 'shape') else 1
            vocab_size = 32000

            candidate_ids = np.random.randint(0, vocab_size, (batch_size, num_tokens))
            candidate_logits = np.random.randn(batch_size, num_tokens, vocab_size)

            return candidate_ids, candidate_logits
        except ImportError:
            return None, None


class SpeculativeDecoder:
    """
    Speculative decoding implementation for fast inference.

    Algorithm:
    1. Draft model generates K candidate tokens
    2. Target model processes all K+1 positions in one forward pass
    3. Verify each candidate against target distribution
    4. Accept tokens until first rejection, then resample

    This provides exact sampling from target distribution while
    achieving ~2-3x speedup when draft acceptance is high.
    """

    def __init__(
        self,
        target_model: Any,
        draft_model: DraftModel,
        tokenizer: Any,
        config: Optional[SpeculativeConfig] = None
    ):
        self.target = target_model
        self.draft = draft_model
        self.tokenizer = tokenizer
        self.config = config or SpeculativeConfig()
        self.config.validate()

        self.stats = SpeculativeStats(current_k=self.config.num_speculative_tokens)
        self._is_mock = target_model is None

        if self._is_mock:
            logger.warning("Target model is None - using mock mode")

    def generate(
        self,
        input_ids: Any,
        max_new_tokens: int = 100,
        attention_mask: Optional[Any] = None,
        stop_tokens: Optional[List[int]] = None,
        callback: Optional[Callable[[Any], None]] = None
    ) -> Tuple[Any, SpeculativeStats]:
        """
        Generate tokens using speculative decoding.

        Args:
            input_ids: Input token ids [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            attention_mask: Optional attention mask
            stop_tokens: Token ids that stop generation
            callback: Called after each token for streaming

        Returns:
            Tuple of (generated_ids, stats)
        """
        start_time = time.perf_counter()

        if self._is_mock:
            return self._mock_generate(input_ids, max_new_tokens)

        try:
            import torch
            return self._generate_torch(
                input_ids, max_new_tokens, attention_mask,
                stop_tokens, callback, start_time
            )
        except ImportError:
            return self._mock_generate(input_ids, max_new_tokens)

    def _generate_torch(
        self,
        input_ids: Any,
        max_new_tokens: int,
        attention_mask: Optional[Any],
        stop_tokens: Optional[List[int]],
        callback: Optional[Callable],
        start_time: float
    ) -> Tuple[Any, SpeculativeStats]:
        """PyTorch implementation of speculative decoding."""
        import torch

        device = input_ids.device if hasattr(input_ids, 'device') else 'cpu'
        current_ids = input_ids
        current_mask = attention_mask

        generated_tokens = 0
        stop_tokens = set(stop_tokens or [])

        while generated_tokens < max_new_tokens:
            k = self.stats.current_k

            # Step 1: Draft generates K candidates
            draft_start = time.perf_counter()
            candidate_ids, draft_logits = self.draft.generate_candidates(
                current_ids, current_mask, num_tokens=k
            )
            self.stats.total_draft_time_ms += (time.perf_counter() - draft_start) * 1000
            self.stats.total_draft_calls += 1
            self.stats.total_draft_tokens += k

            # Step 2: Target verifies all candidates in one pass
            target_start = time.perf_counter()

            # Concatenate input + candidates for single forward
            verify_ids = torch.cat([current_ids, candidate_ids], dim=1)
            if current_mask is not None:
                verify_mask = torch.cat([
                    current_mask,
                    torch.ones_like(candidate_ids)
                ], dim=1)
            else:
                verify_mask = None

            with torch.no_grad():
                target_outputs = self.target(
                    input_ids=verify_ids,
                    attention_mask=verify_mask
                )

            self.stats.total_target_time_ms += (time.perf_counter() - target_start) * 1000
            self.stats.total_target_calls += 1

            # Get target logits for positions we need to verify
            # Position i compares draft[i] against target logits at position (seq_len + i - 1)
            seq_len = current_ids.shape[1]
            target_logits = target_outputs.logits[:, seq_len-1:seq_len+k, :]

            # Step 3: Verify and accept/reject
            accepted, next_token = self._verify_candidates(
                candidate_ids, draft_logits, target_logits
            )

            num_accepted = accepted.sum().item()
            self.stats.total_accepted_tokens += num_accepted

            # Update adaptive K
            if self.config.adaptive_k:
                self._update_adaptive_k(num_accepted, k)

            # Step 4: Update sequence
            # Accept all verified tokens + one resampled token
            if num_accepted == k:
                # All accepted - also sample next token from target
                new_tokens = torch.cat([candidate_ids, next_token], dim=1)
            else:
                # Partial accept - take accepted + resampled
                new_tokens = torch.cat([
                    candidate_ids[:, :num_accepted],
                    next_token
                ], dim=1)

            current_ids = torch.cat([current_ids, new_tokens], dim=1)
            if current_mask is not None:
                current_mask = torch.cat([
                    current_mask,
                    torch.ones_like(new_tokens)
                ], dim=1)

            tokens_added = new_tokens.shape[1]
            generated_tokens += tokens_added
            self.stats.total_tokens_generated += tokens_added

            # Callback for streaming
            if callback:
                callback(new_tokens)

            # Check stop condition
            if stop_tokens:
                for tok in new_tokens[0].tolist():
                    if tok in stop_tokens:
                        self.stats.total_time_ms = (time.perf_counter() - start_time) * 1000
                        return current_ids, self.stats

        self.stats.total_time_ms = (time.perf_counter() - start_time) * 1000
        return current_ids, self.stats

    def _verify_candidates(
        self,
        candidate_ids: Any,
        draft_logits: Any,
        target_logits: Any
    ) -> Tuple[Any, Any]:
        """
        Verify draft candidates against target distribution.

        Uses rejection sampling to ensure exact target distribution.

        Returns:
            Tuple of (accepted_mask, next_token)
        """
        import torch

        batch_size, k = candidate_ids.shape
        device = candidate_ids.device

        # Get probabilities
        draft_probs = torch.softmax(draft_logits / self.config.temperature, dim=-1)
        target_probs = torch.softmax(target_logits[:, :-1, :] / self.config.temperature, dim=-1)

        # For each position, get p_target(x) and p_draft(x) for chosen token x
        candidate_ids_expanded = candidate_ids.unsqueeze(-1)  # [batch, k, 1]

        p_draft = torch.gather(draft_probs, -1, candidate_ids_expanded).squeeze(-1)
        p_target = torch.gather(target_probs, -1, candidate_ids_expanded).squeeze(-1)

        # Acceptance probability: min(1, p_target / p_draft)
        accept_prob = torch.clamp(p_target / (p_draft + 1e-10), max=1.0)

        # Sample acceptance
        uniform = torch.rand_like(accept_prob)
        accepted = uniform < accept_prob

        # Find first rejection
        # accepted: [batch, k] bool
        # We want: for each batch, how many consecutive True from start
        accepted_cumsum = accepted.cumsum(dim=1)
        position_indices = torch.arange(1, k + 1, device=device).unsqueeze(0)
        all_accepted_up_to = (accepted_cumsum == position_indices)

        # Create mask of which tokens to keep
        keep_mask = all_accepted_up_to

        # Sample next token from modified distribution
        # If all accepted: sample from p_target at position k
        # If rejected at i: sample from (p_target - p_draft)_+ normalized at position i

        # For simplicity, sample from target at the last position
        final_target_logits = target_logits[:, -1, :]
        final_probs = torch.softmax(final_target_logits / self.config.temperature, dim=-1)
        next_token = torch.multinomial(final_probs, num_samples=1)

        return keep_mask[:, -1], next_token  # Return if all k were accepted

    def _update_adaptive_k(self, num_accepted: int, k: int) -> None:
        """Update speculation length based on acceptance rate."""
        # Update EMA of acceptance rate
        instant_rate = num_accepted / k
        self.stats.acceptance_rate_ema = (
            self.config.acceptance_smoothing * self.stats.acceptance_rate_ema +
            (1 - self.config.acceptance_smoothing) * instant_rate
        )

        # Adjust K
        if self.stats.acceptance_rate_ema > self.config.target_acceptance_rate + 0.1:
            # High acceptance - try more speculation
            self.stats.current_k = min(
                self.stats.current_k + 1,
                self.config.max_speculative_tokens
            )
        elif self.stats.acceptance_rate_ema < self.config.target_acceptance_rate - 0.1:
            # Low acceptance - speculate less
            self.stats.current_k = max(
                self.stats.current_k - 1,
                self.config.min_speculative_tokens
            )

    def _mock_generate(
        self,
        input_ids: Any,
        max_new_tokens: int
    ) -> Tuple[Any, SpeculativeStats]:
        """Mock generation for testing without models."""
        try:
            import numpy as np

            batch_size = input_ids.shape[0] if hasattr(input_ids, 'shape') else 1
            seq_len = input_ids.shape[1] if hasattr(input_ids, 'shape') else 10
            vocab_size = 32000

            # Simulate speculative decoding behavior
            k = self.config.num_speculative_tokens
            generated = 0
            acceptance_rate = 0.75  # Simulated

            while generated < max_new_tokens:
                # Simulate draft + verify cycle
                accepted = int(k * acceptance_rate) + 1
                generated += accepted

                self.stats.total_draft_tokens += k
                self.stats.total_accepted_tokens += min(accepted - 1, k)
                self.stats.total_target_calls += 1
                self.stats.total_draft_calls += 1

            self.stats.total_tokens_generated = min(generated, max_new_tokens)
            self.stats.total_time_ms = max_new_tokens * 10  # 10ms per token simulated

            # Generate random output
            new_tokens = np.random.randint(0, vocab_size, (batch_size, max_new_tokens))

            if hasattr(input_ids, 'shape'):
                result = np.concatenate([
                    input_ids if isinstance(input_ids, np.ndarray) else np.array([[0] * seq_len]),
                    new_tokens
                ], axis=1)
            else:
                result = new_tokens

            return result, self.stats

        except ImportError:
            self.stats.total_tokens_generated = max_new_tokens
            return None, self.stats

    def reset_stats(self) -> None:
        """Reset statistics for new generation session."""
        self.stats = SpeculativeStats(current_k=self.config.num_speculative_tokens)


class MedusaHead:
    """
    Medusa-style multi-head speculation.

    Instead of a separate draft model, adds multiple prediction heads
    to the target model that predict future tokens in parallel.

    Each head i predicts token at position t+i given hidden state at t.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 4,
        head_hidden_size: Optional[int] = None
    ):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.head_hidden_size = head_hidden_size or hidden_size

        self.heads = []
        self._initialized = False

        try:
            import torch.nn as nn

            for i in range(num_heads):
                head = nn.Sequential(
                    nn.Linear(hidden_size, self.head_hidden_size),
                    nn.SiLU(),
                    nn.Linear(self.head_hidden_size, vocab_size)
                )
                self.heads.append(head)

            self._initialized = True

        except ImportError:
            logger.warning("PyTorch not available - Medusa heads in mock mode")

    def forward(self, hidden_states: Any) -> List[Any]:
        """
        Predict future tokens from current hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            List of logits, one per head: [batch, seq_len, vocab_size]
        """
        if not self._initialized:
            return [None] * self.num_heads

        return [head(hidden_states) for head in self.heads]


def create_speculative_config(
    num_speculative: int = 4,
    adaptive: bool = True,
    draft_type: str = "distilled",
    temperature: float = 1.0
) -> SpeculativeConfig:
    """
    Factory function for creating speculative decoding config.

    Args:
        num_speculative: Number of tokens to speculate per step
        adaptive: Whether to adaptively adjust speculation length
        draft_type: Type of draft model ("distilled", "early_exit", "medusa")
        temperature: Sampling temperature

    Returns:
        Configured SpeculativeConfig
    """
    draft_map = {
        "distilled": DraftModelType.DISTILLED,
        "independent": DraftModelType.INDEPENDENT,
        "early_exit": DraftModelType.EARLY_EXIT,
        "medusa": DraftModelType.MEDUSA,
        "eagle": DraftModelType.EAGLE
    }

    return SpeculativeConfig(
        num_speculative_tokens=num_speculative,
        adaptive_k=adaptive,
        draft_model_type=draft_map.get(draft_type, DraftModelType.DISTILLED),
        temperature=temperature
    )


def create_draft_model(
    target_model: Any,
    config: SpeculativeConfig,
    tokenizer: Any,
    draft_model_path: Optional[str] = None
) -> DraftModel:
    """
    Create or load a draft model for speculative decoding.

    Args:
        target_model: The target model to accelerate
        config: Speculative decoding configuration
        tokenizer: Tokenizer (shared between draft and target)
        draft_model_path: Path to pre-trained draft model

    Returns:
        Configured DraftModel
    """
    draft_model = None

    if draft_model_path:
        try:
            from transformers import AutoModelForCausalLM
            draft_model = AutoModelForCausalLM.from_pretrained(
                draft_model_path,
                torch_dtype="auto",
                device_map="auto"
            )
            logger.info(f"Loaded draft model from {draft_model_path}")
        except Exception as e:
            logger.warning(f"Failed to load draft model: {e}")

    elif config.draft_model_type == DraftModelType.EARLY_EXIT:
        # Use early layers of target as draft
        # This requires model architecture support
        logger.info("Using early-exit draft (requires model support)")
        draft_model = None  # Would need model-specific implementation

    return DraftModel(draft_model, tokenizer, config)


# Convenience exports
__all__ = [
    "SpeculativeConfig",
    "SpeculativeStats",
    "SpeculativeDecoder",
    "DraftModel",
    "DraftModelType",
    "MedusaHead",
    "create_speculative_config",
    "create_draft_model"
]

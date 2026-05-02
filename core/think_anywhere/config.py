"""
ThinkAnywhere configuration.

Based on: "Think Anywhere in Code Generation" (Jiang et al., 2026)
arXiv:2603.29957
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ThinkAnywhereConfig:
    """Configuration for the Think-Anywhere reasoning mechanism.

    Mirrors the training setup from the paper (Section 3):
    - Text-based variant uses multi-token <thinkanywhere> delimiters.
    - Special-token variant (*) uses single <ta>/<ta_end> vocabulary entries
      initialized with the semantic-aware formula from Eq. (5-6).
    """

    # --- token strings ---------------------------------------------------
    # Standard (text-based) delimiters — paper default
    think_open: str = "<think>"
    think_close: str = "</think>"
    ta_open: str = "<thinkanywhere>"
    ta_close: str = "</thinkanywhere>"

    # Special-token variant delimiters (ThinkAnywhere*)
    use_special_tokens: bool = False
    ta_special_open: str = "<ta>"
    ta_special_close: str = "</ta>"

    # --- reward weights (Eq. 9) ------------------------------------------
    structure_reward_weight: float = 0.1   # alpha in the paper
    correctness_reward_weight: float = 0.9  # (1 - alpha)

    # --- cold-start / GRPO training params --------------------------------
    grpo_group_size: int = 8          # G rollout samples per problem
    grpo_clip_epsilon: float = 0.2    # epsilon for PPO-clip
    grpo_kl_weight: float = 0.001     # beta KL penalty coefficient
    max_generation_tokens: int = 4096
    cold_start_samples: int = 5000    # target cold-start dataset size

    # --- embedding initialization (Eq. 5-6) --------------------------------
    # Tokens whose embeddings are averaged for semantic content
    semantic_seed_tokens: List[str] = field(
        default_factory=lambda: ["think", "any", "where"]
    )
    # Delimiter tokens whose embeddings provide structural role
    delimiter_open_token: str = "<im_start>"
    delimiter_close_token: str = "<im_end>"
    semantic_mix_alpha: float = 0.5   # weight between semantic / delimiter

    @property
    def open_tag(self) -> str:
        return self.ta_special_open if self.use_special_tokens else self.ta_open

    @property
    def close_tag(self) -> str:
        return self.ta_special_close if self.use_special_tokens else self.ta_close

"""
Think-Anywhere — on-demand inline reasoning for code generation.

Reference: "Think Anywhere in Code Generation", Jiang et al. 2026
           arXiv:2603.29957

Public surface:
    ThinkAnywhereConfig      — configuration dataclass
    ThinkAnywhereProcessor   — format / parse / strip thinking blocks
    ThinkAnywhereReward      — hierarchical GRPO reward (R_struct + R_correct)
    ParsedResponse           — result datatype from ThinkAnywhereProcessor.parse()
    RewardResult             — result datatype from ThinkAnywhereReward.__call__()
    ThinkAnywhereStreamFilter — real-time streaming filter (gate-aware)
    ThinkAnywhereGate        — binary MLP gate: should model think here?
    GateConfig               — gate hyper-parameters
    PositionalFeatures       — lightweight positional proxy for streaming gate
"""

from .config import ThinkAnywhereConfig
from .token_processor import ThinkAnywhereProcessor, ParsedResponse
from .rewards import ThinkAnywhereReward, RewardResult
from .streaming import ThinkAnywhereStreamFilter
from .activation_gate import ThinkAnywhereGate, GateConfig, PositionalFeatures

__all__ = [
    "ThinkAnywhereConfig",
    "ThinkAnywhereProcessor",
    "ParsedResponse",
    "ThinkAnywhereReward",
    "RewardResult",
    "ThinkAnywhereStreamFilter",
    "ThinkAnywhereGate",
    "GateConfig",
    "PositionalFeatures",
]

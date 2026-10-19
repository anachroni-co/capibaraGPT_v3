"""
<plan> / </plan> — structured task-decomposition blocks.

The model inserts a <plan> block before tackling a multi-step problem —
outlining the approach (data structures, algorithm, edge cases) before
writing any code. Analogous to chain-of-thought but position-flexible.

Seed tokens: "step", "outline", "first", "plan", "approach"
Stripped from final output (strip_from_output=True).
"""

from .base import SpecialTokenConfig
from .registry import register_token

PLAN_TOKEN = SpecialTokenConfig(
    name="plan",
    open_tag="<plan>",
    close_tag="</plan>",
    seed_tokens=["step", "outline", "first", "plan", "approach"],
    alpha=0.5,
    boundary_token="<im_start>",
    strip_from_output=True,
    description="Planning block: structured task decomposition before multi-step generation",
)

register_token(PLAN_TOKEN)

"""
<verify> / </verify> — self-verification blocks.

The model inserts a <verify> block at any point to check its own
output before continuing — e.g. confirming that a function's logic
is correct before writing the next statement.

Seed tokens: "check", "assert", "confirm", "correct", "valid"
Stripped from final output (strip_from_output=True).
"""

from .base import SpecialTokenConfig
from .registry import register_token

VERIFY_TOKEN = SpecialTokenConfig(
    name="verify",
    open_tag="<verify>",
    close_tag="</verify>",
    seed_tokens=["check", "assert", "confirm", "correct", "valid"],
    alpha=0.5,
    boundary_token="<im_start>",
    strip_from_output=True,
    description="Self-verification block: model checks its own output before continuing",
)

register_token(VERIFY_TOKEN)

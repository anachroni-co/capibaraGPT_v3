"""
<debug> / </debug> — code-debugging reasoning blocks.

The model inserts a <debug> block when it detects that previously
generated code might be wrong — diagnosing the error (wrong index,
off-by-one, missing edge case) before writing the corrected version.

Complements <verify> (which checks correctness proactively) by
addressing already-suspected errors reactively.

Seed tokens: "error", "trace", "why", "bug", "fix", "wrong"
Stripped from final output (strip_from_output=True).
"""

from .base import SpecialTokenConfig
from .registry import register_token

DEBUG_TOKEN = SpecialTokenConfig(
    name="debug",
    open_tag="<debug>",
    close_tag="</debug>",
    seed_tokens=["error", "trace", "why", "bug", "fix", "wrong"],
    alpha=0.5,
    boundary_token="<im_start>",
    strip_from_output=True,
    description="Debug reasoning block: model diagnoses why code is wrong before fixing it",
)

register_token(DEBUG_TOKEN)

"""
<uncertain> / </uncertain> — low-confidence region markers.

Unlike the other tokens, <uncertain> blocks are NOT stripped from output
by default (strip_from_output=False). They are preserved as structured
metadata so callers can detect where the model is unsure and apply
post-processing (e.g. highlight, re-sample, flag for human review).

Seed tokens: "maybe", "unclear", "unsure", "perhaps", "might"
Kept in final output (strip_from_output=False).
"""

from .base import SpecialTokenConfig, SpecialTokenProcessor, ParseResult
from .registry import register_token

UNCERTAIN_TOKEN = SpecialTokenConfig(
    name="uncertain",
    open_tag="<uncertain>",
    close_tag="</uncertain>",
    seed_tokens=["maybe", "unclear", "unsure", "perhaps", "might"],
    alpha=0.5,
    boundary_token="<im_end>",
    strip_from_output=False,
    description="Uncertainty marker: flags low-confidence regions; kept in output as metadata",
)

register_token(UNCERTAIN_TOKEN)


class UncertainTokenExtractor:
    """Extract uncertainty markers and the text they wrap."""

    def __init__(self) -> None:
        self._proc = SpecialTokenProcessor(UNCERTAIN_TOKEN)

    def extract(self, text: str) -> ParseResult:
        """Return (text_with_markers_intact, list_of_uncertain_spans)."""
        return self._proc.parse(text)

    def has_uncertainty(self, text: str) -> bool:
        return UNCERTAIN_TOKEN.open_tag in text

    def strip_markers(self, text: str) -> str:
        """Remove tags but keep the wrapped text."""
        import re
        text = re.sub(re.escape(UNCERTAIN_TOKEN.open_tag), "", text)
        text = re.sub(re.escape(UNCERTAIN_TOKEN.close_tag), "", text)
        return text

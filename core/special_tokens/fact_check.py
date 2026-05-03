"""
<fact_check> / </fact_check> — contradiction / misinformation signal.

The model inserts <fact_check>claim</fact_check> when it detects that
a statement might be false, outdated, or contradicted by the user's
framing ("I thought X was Y…"). Unlike <search> (which fetches new info),
<fact_check> flags an existing claim for external verification.

Behaviour differs from <search>:
  <search>   → "I need information I don't have"
  <fact_check> → "this claim may be wrong — verify before trusting it"

strip_from_output=False: the marker is kept visible so the caller / UI
can show the user that a fact-check was triggered on that span.

Seed tokens: "verify", "false", "wrong", "contradict", "source", "check"
"""

from __future__ import annotations

import re
from typing import List, Tuple

from .base import SpecialTokenConfig, SpecialTokenProcessor
from .registry import register_token

FACT_CHECK_TOKEN = SpecialTokenConfig(
    name="fact_check",
    open_tag="<fact_check>",
    close_tag="</fact_check>",
    seed_tokens=["verify", "false", "wrong", "contradict", "source", "check"],
    alpha=0.5,
    boundary_token="<im_end>",
    strip_from_output=False,
    description=(
        "Contradiction / misinformation signal: flags a claim that may be "
        "false or outdated — kept in output so the UI can surface it"
    ),
)

register_token(FACT_CHECK_TOKEN)

_FULL_PATTERN = re.compile(
    re.escape(FACT_CHECK_TOKEN.open_tag) + r"(.*?)" + re.escape(FACT_CHECK_TOKEN.close_tag),
    re.DOTALL,
)


class FactCheckHandler:
    """
    Extract fact-check spans and optionally verify them via a web search.

    Without a verifier the handler returns the claims as-is so the caller
    can decide what to do (highlight in UI, queue for async verification…).
    With a verifier it replaces each span with a verdict inline.
    """

    def __init__(self, verifier=None) -> None:
        self._verifier = verifier
        self._proc = SpecialTokenProcessor(FACT_CHECK_TOKEN)

    def extract_claims(self, text: str) -> List[str]:
        """Return all claims wrapped in <fact_check> blocks."""
        return [m.strip() for m in _FULL_PATTERN.findall(text)]

    def has_claims(self, text: str) -> bool:
        return bool(_FULL_PATTERN.search(text))

    def verify(self, text: str) -> Tuple[str, List[dict]]:
        """
        Replace <fact_check>claim</fact_check> with inline verdicts.

        Returns (processed_text, [{"claim": …, "verdict": …, "source": …}]).
        If no verifier is set, claims are stripped and returned as metadata.
        """
        verdicts = []

        def _replace(match: re.Match) -> str:
            claim = match.group(1).strip()
            if self._verifier is None:
                verdicts.append({"claim": claim, "verdict": "unverified", "source": None})
                return claim
            try:
                result = self._verifier.verify(claim)
                verdicts.append(result)
                label = "✓" if result.get("correct") else "⚠ possibly incorrect"
                return f"{claim} [{label}]"
            except Exception:
                verdicts.append({"claim": claim, "verdict": "error", "source": None})
                return claim

        processed = _FULL_PATTERN.sub(_replace, text)
        return processed, verdicts

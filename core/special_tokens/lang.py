"""
<lang:XX> / </lang> — inline language-switching blocks.

The model inserts <lang:gl>…</lang> (or pt, es, en, …) to reason
internally in a different language — useful for low-resource models
like CUNCA-Hybrid where Galician reasoning may be more accurate than
English reasoning for certain domains.

The tag is dynamic: the language code is embedded in the opening tag.
LangTokenStreamFilter handles partial-tag detection for the variable
<lang:XX> opener.

Seed tokens: "language", "translate", "speak", "text", "lingua"
Stripped from final output (strip_from_output=True).
"""

from __future__ import annotations

import re
from typing import List, Tuple

from .base import SpecialTokenConfig, SpecialTokenStreamFilter
from .registry import register_token

# Base config used for registry + embedding init of the generic <lang:XX> token.
LANG_TOKEN = SpecialTokenConfig(
    name="lang",
    open_tag="<lang:XX>",
    close_tag="</lang>",
    seed_tokens=["language", "translate", "speak", "text", "lingua"],
    alpha=0.5,
    boundary_token="<im_start>",
    strip_from_output=True,
    description="Language-switching block: inline reasoning in a different language",
)

register_token(LANG_TOKEN)

_FULL_PATTERN = re.compile(
    r"<lang:([a-z]{2,3})>(.*?)</lang>", re.DOTALL | re.IGNORECASE
)
_OPEN_PATTERN = re.compile(r"<lang:([a-z]{2,3})>", re.IGNORECASE)


class LangTokenProcessor:
    """Parser for dynamic <lang:XX>…</lang> blocks."""

    def parse(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Return (clean_text, [(lang_code, block_content), …])."""
        blocks = [(m.group(1).lower(), m.group(2).strip()) for m in _FULL_PATTERN.finditer(text)]
        clean = _FULL_PATTERN.sub("", text).strip()
        return clean, blocks

    def strip(self, text: str) -> str:
        return _FULL_PATTERN.sub("", text).strip()

    def get_languages_used(self, text: str) -> List[str]:
        return [m.group(1).lower() for m in _OPEN_PATTERN.finditer(text)]

    def has_lang_block(self, text: str, lang_code: str) -> bool:
        return any(code == lang_code.lower() for code, _ in self.parse(text)[1])


class LangTokenStreamFilter(SpecialTokenStreamFilter):
    """
    Streaming filter for <lang:XX>…</lang> with variable open tags.

    Uses regex instead of exact string matching to handle the dynamic
    language-code portion of the opener (<lang:gl>, <lang:pt>, …).
    _safe_boundary_lang holds back any buffer suffix that could be the
    start of a <lang:XX> tag, including the variable lang-code chars.
    """

    _OPEN_RE = re.compile(r"<lang:[a-z]{2,3}>", re.IGNORECASE)
    # Matches any suffix that could be a partial <lang:XX> opener
    _PARTIAL_RE = re.compile(
        r"<(?:l(?:a(?:n(?:g(?::(?:[a-z]{0,3}>?)?)?)?)?)?)?$",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        super().__init__(LANG_TOKEN)
        self._open_tags = []   # disable base class exact matching
        self._max_open_len = 12

    def feed(self, token: str) -> str:
        self._buf += token
        out: List[str] = []

        while True:
            if self._depth == 0:
                m = self._OPEN_RE.search(self._buf)
                if m:
                    out.append(self._buf[:m.start()])
                    self._buf = self._buf[m.end():]
                    self._depth += 1
                else:
                    boundary = self._safe_boundary_lang()
                    out.append(self._buf[:boundary])
                    self._buf = self._buf[boundary:]
                    break
            else:
                pos = self._buf.find("</lang>")
                if pos != -1:
                    self._buf = self._buf[pos + len("</lang>"):]
                    self._depth -= 1
                else:
                    break

        return "".join(out)

    def _safe_boundary_lang(self) -> int:
        """Hold back any suffix that could be the start of <lang:XX>."""
        m = self._PARTIAL_RE.search(self._buf)
        return m.start() if m else len(self._buf)

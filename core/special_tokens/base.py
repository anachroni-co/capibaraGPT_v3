"""
Base classes for the special-token framework.

Pattern mirrors Think-Anywhere (arXiv:2603.29957): each token type has
a config, a processor (parse / strip / embedding-init), and a streaming
filter that suppresses blocks in real-time without buffering the full response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False


@dataclass
class SpecialTokenConfig:
    """Configuration for one special-token type."""
    name: str
    open_tag: str
    close_tag: str
    seed_tokens: List[str]
    alpha: float = 0.5
    boundary_token: str = "<im_start>"
    strip_from_output: bool = True
    description: str = ""


@dataclass
class ParseResult:
    clean_text: str
    blocks: List[str]

    @property
    def block_count(self) -> int:
        return len(self.blocks)


class SpecialTokenProcessor:
    """Parse, strip, and embedding-initialize for a single token type."""

    def __init__(self, config: SpecialTokenConfig) -> None:
        self.config = config
        open_esc = re.escape(config.open_tag)
        close_esc = re.escape(config.close_tag)
        self._pattern = re.compile(f"{open_esc}(.*?){close_esc}", re.DOTALL)

    def parse(self, text: str) -> ParseResult:
        blocks = self._pattern.findall(text)
        clean = self._pattern.sub("", text).strip()
        return ParseResult(clean_text=clean, blocks=blocks)

    def strip(self, text: str) -> str:
        return self._pattern.sub("", text).strip()

    def initialize_embedding(
        self,
        embedding_matrix: "np.ndarray",
        token_ids: dict,
    ) -> "np.ndarray":
        """Semantic-aware embedding init (Eqs. 5-6 from arXiv:2603.29957)."""
        if not _NP:
            raise ImportError("numpy required for embedding initialization")

        seed_ids = [token_ids[t] for t in self.config.seed_tokens if t in token_ids]
        if not seed_ids:
            return embedding_matrix[0].copy()

        semantic = embedding_matrix[seed_ids].mean(axis=0)
        boundary_id = token_ids.get(self.config.boundary_token)
        if boundary_id is not None:
            boundary = embedding_matrix[boundary_id]
            return self.config.alpha * semantic + (1.0 - self.config.alpha) * boundary
        return semantic


class SpecialTokenStreamFilter:
    """
    Real-time streaming filter for a single special-token type.

    Suppresses open_tag…close_tag blocks token-by-token without buffering
    the full response. Handles partial tags at chunk boundaries.
    """

    def __init__(self, config: SpecialTokenConfig) -> None:
        self._open_tags = [config.open_tag]
        self._close_tags = [config.close_tag]
        self._max_open_len = max(len(t) for t in self._open_tags)
        self._buf = ""
        self._depth = 0

    def feed(self, token: str) -> str:
        self._buf += token
        out: List[str] = []

        while True:
            if self._depth == 0:
                earliest, earliest_tag = self._find_earliest(self._open_tags)
                if earliest != -1:
                    out.append(self._buf[:earliest])
                    self._buf = self._buf[earliest + len(earliest_tag):]
                    self._depth += 1
                else:
                    boundary = self._safe_boundary()
                    out.append(self._buf[:boundary])
                    self._buf = self._buf[boundary:]
                    break
            else:
                earliest, earliest_tag = self._find_earliest(self._close_tags)
                if earliest != -1:
                    self._buf = self._buf[earliest + len(earliest_tag):]
                    self._depth -= 1
                else:
                    break

        return "".join(out)

    def flush(self) -> str:
        if self._depth == 0:
            out = self._buf
            self._buf = ""
            return out
        self._buf = ""
        self._depth = 0
        return ""

    def _find_earliest(self, tags: List[str]):
        earliest, earliest_tag = -1, ""
        for tag in tags:
            pos = self._buf.find(tag)
            if pos != -1 and (earliest == -1 or pos < earliest):
                earliest, earliest_tag = pos, tag
        return earliest, earliest_tag

    def _safe_boundary(self) -> int:
        buf = self._buf
        for suffix_len in range(min(self._max_open_len - 1, len(buf)), 0, -1):
            suffix = buf[len(buf) - suffix_len:]
            for tag in self._open_tags:
                if tag.startswith(suffix):
                    return len(buf) - suffix_len
        return len(buf)

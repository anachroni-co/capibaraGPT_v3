"""
Think-Anywhere token processor.

Handles three concerns:
1. Formatting: wraps a coding prompt in the Think-Anywhere template (Table 1).
2. Parsing: extracts clean executable code from a Think-Anywhere response by
   stripping all <think> and <thinkanywhere> blocks (Eq. 4).
3. Validation: verifies that a generated response conforms to the
   Think-Anywhere format (used by the structure reward, Eq. 10).

Reference: "Think Anywhere in Code Generation", Jiang et al. 2026
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import ThinkAnywhereConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template (Table 1 from the paper)
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = (
    "You are a coding assistant that generates both code and inline "
    "self-guidance signals. First output <think>...</think> with brief "
    "reasoning, then output the final code.\n"
    "MUST FOLLOW Rules for <thinkanywhere>...</thinkanywhere> tags:\n"
    "1. You MUST use <thinkanywhere>...</thinkanywhere> tags for "
    "self-guidance or intermediate reasoning.\n"
    "2. <thinkanywhere>...</thinkanywhere> MUST be embedded within an "
    "existing program statement token sequence.\n"
    "3. The code must remain valid and executable after removing all "
    "<thinkanywhere>...</thinkanywhere> segments."
)


@dataclass
class ParsedResponse:
    """Result of parsing a Think-Anywhere model response."""

    raw: str
    upfront_thinking: str      # content inside <think>...</think>
    think_anywhere_blocks: List[str]  # content of each <thinkanywhere> block
    clean_code: str            # executable code with all thinking stripped
    is_valid: bool             # passes structural validation
    validation_errors: List[str]


class ThinkAnywhereProcessor:
    """Format, parse, and validate Think-Anywhere model responses."""

    def __init__(self, config: Optional[ThinkAnywhereConfig] = None) -> None:
        self.cfg = config or ThinkAnywhereConfig()
        self._build_patterns()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_patterns(self) -> None:
        cfg = self.cfg
        # DOTALL so blocks can span multiple lines; non-greedy to avoid
        # merging adjacent blocks.
        self._think_pattern = re.compile(
            re.escape(cfg.think_open) + r"(.*?)" + re.escape(cfg.think_close),
            re.DOTALL,
        )
        self._ta_pattern = re.compile(
            re.escape(cfg.open_tag) + r"(.*?)" + re.escape(cfg.close_tag),
            re.DOTALL,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format_prompt(self, problem: str, system_override: Optional[str] = None) -> str:
        """Wrap a coding problem in the Think-Anywhere system prompt."""
        system = system_override or _SYSTEM_TEMPLATE
        return f"{system}\n\nUser: {problem}\nAssistant:"

    def parse(self, response: str) -> ParsedResponse:
        """Extract thinking blocks and clean code from a model response."""
        # 1. Extract upfront <think> block (should appear once at the start)
        think_matches = self._think_pattern.findall(response)
        upfront = think_matches[0].strip() if think_matches else ""

        # 2. Remove the upfront <think> block from the text
        without_upfront = self._think_pattern.sub("", response, count=1).strip()

        # 3. Extract all <thinkanywhere> blocks
        ta_matches = self._ta_pattern.findall(without_upfront)
        ta_blocks = [m.strip() for m in ta_matches]

        # 4. Strip all <thinkanywhere> blocks to produce executable code
        clean = self._ta_pattern.sub("", without_upfront).strip()

        # 5. Validate structure
        is_valid, errors = self.validate(response)

        return ParsedResponse(
            raw=response,
            upfront_thinking=upfront,
            think_anywhere_blocks=ta_blocks,
            clean_code=clean,
            is_valid=is_valid,
            validation_errors=errors,
        )

    def validate(self, response: str) -> Tuple[bool, List[str]]:
        """Check whether a response satisfies Think-Anywhere structural rules.

        Returns (is_valid, list_of_error_strings).
        """
        errors: List[str] = []

        # Rule 1 – must contain an upfront <think> block
        if not self._think_pattern.search(response):
            errors.append(
                f"Missing upfront thinking block "
                f"({self.cfg.think_open}...{self.cfg.think_close})"
            )

        # Rule 2 – must contain at least one <thinkanywhere> block
        if not self._ta_pattern.search(response):
            errors.append(
                f"Missing at least one inline thinking block "
                f"({self.cfg.open_tag}...{self.cfg.close_tag})"
            )

        # Rule 3 – opening and closing tags must be balanced
        n_open = response.count(self.cfg.open_tag)
        n_close = response.count(self.cfg.close_tag)
        if n_open != n_close:
            errors.append(
                f"Unbalanced tags: {n_open} opening vs {n_close} closing"
            )

        return len(errors) == 0, errors

    def strip_thinking(self, text: str) -> str:
        """Remove all thinking blocks from text, returning clean code only.

        Suitable as a post-processing step in the inference engine before
        returning the response to the caller.
        """
        # Strip upfront <think> block
        cleaned = self._think_pattern.sub("", text)
        # Strip all <thinkanywhere> blocks
        cleaned = self._ta_pattern.sub("", cleaned)
        return cleaned.strip()

    def initialize_special_token_embedding(
        self,
        tokenizer_embeddings,  # array-like of shape [vocab_size, hidden]
        token_ids: dict,        # must contain keys for semantic seeds + delimiters
    ):
        """Compute semantic-aware embedding for the <ta>/<ta_end> special tokens.

        Implements Eq. (5-6) from the paper:
            e_<ta>     = alpha * mean(e_think, e_any, e_where) + (1-alpha) * e_<im_start>
            e_</ta>    = alpha * mean(e_think, e_any, e_where) + (1-alpha) * e_<im_end>

        Args:
            tokenizer_embeddings: numpy/jax array of token embeddings.
            token_ids: mapping from token string to vocabulary index, must
                include keys for cfg.semantic_seed_tokens and
                cfg.delimiter_open_token / cfg.delimiter_close_token.

        Returns:
            Tuple (open_embedding, close_embedding) as numpy arrays.
        """
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy required for embedding initialization") from exc

        alpha = self.cfg.semantic_mix_alpha
        E = np.asarray(tokenizer_embeddings)

        # Semantic component: mean of seed token embeddings
        seed_ids = [token_ids[t] for t in self.cfg.semantic_seed_tokens if t in token_ids]
        if not seed_ids:
            raise ValueError(
                f"None of the semantic seed tokens {self.cfg.semantic_seed_tokens} "
                "found in token_ids."
            )
        semantic = E[seed_ids].mean(axis=0)

        # Structural component: delimiter embeddings
        open_delim_id = token_ids.get(self.cfg.delimiter_open_token)
        close_delim_id = token_ids.get(self.cfg.delimiter_close_token)
        if open_delim_id is None or close_delim_id is None:
            raise ValueError(
                f"Delimiter tokens {self.cfg.delimiter_open_token!r} / "
                f"{self.cfg.delimiter_close_token!r} not found in token_ids."
            )

        e_open = alpha * semantic + (1 - alpha) * E[open_delim_id]
        e_close = alpha * semantic + (1 - alpha) * E[close_delim_id]

        logger.info(
            "Initialized Think-Anywhere special token embeddings "
            "(alpha=%.2f, %d seed tokens).",
            alpha,
            len(seed_ids),
        )
        return e_open, e_close

"""
Streaming filter for Think-Anywhere inference.

Suppresses <think> and <thinkanywhere> blocks in real-time token streams so
callers receive only clean code/text without buffering the entire response.
"""
from __future__ import annotations


class ThinkAnywhereStreamFilter:
    """Stateful streaming filter that hides thinking blocks token-by-token.

    Handles partial tags correctly: if the end of the current buffer is a
    prefix of an opening tag, those characters are held back until the full
    tag (or a non-matching continuation) arrives.

    Usage::

        filt = ThinkAnywhereStreamFilter()
        for token in model_stream:
            chunk = filt.feed(token)
            if chunk:
                print(chunk, end="", flush=True)
        tail = filt.flush()   # remaining safe text after EOS
        if tail:
            print(tail)
    """

    _OPEN_TAGS = ("<think>", "<thinkanywhere>")
    _CLOSE_TAGS = ("</think>", "</thinkanywhere>")
    _MAX_OPEN_LEN = max(len(t) for t in _OPEN_TAGS)
    _MAX_CLOSE_LEN = max(len(t) for t in _CLOSE_TAGS)

    def __init__(self) -> None:
        self._buf = ""
        self._depth = 0  # nesting depth inside thinking blocks

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def feed(self, token: str) -> str:
        """Append a token fragment; returns the safe portion to yield."""
        self._buf += token
        out_parts: list[str] = []

        while True:
            if self._depth == 0:
                # Look for the earliest *complete* opening tag
                earliest_pos = len(self._buf)
                for tag in self._OPEN_TAGS:
                    idx = self._buf.find(tag)
                    if 0 <= idx < earliest_pos:
                        earliest_pos = idx

                if earliest_pos < len(self._buf):
                    # Found a complete opening tag: yield safe prefix, consume tag
                    out_parts.append(self._buf[:earliest_pos])
                    tag_end = earliest_pos + self._buf[earliest_pos:].index(">") + 1
                    self._buf = self._buf[tag_end:]
                    self._depth += 1
                    # Loop continues to handle subsequent tags / closing tags
                else:
                    # No complete opening tag: yield up to the last safe position
                    safe = self._safe_boundary_normal()
                    out_parts.append(self._buf[:safe])
                    self._buf = self._buf[safe:]
                    break

            else:  # self._depth > 0 — inside a thinking block
                earliest_close = len(self._buf)
                chosen_close: str | None = None
                for tag in self._CLOSE_TAGS:
                    idx = self._buf.find(tag)
                    if 0 <= idx < earliest_close:
                        earliest_close = idx
                        chosen_close = tag

                if chosen_close is not None:
                    # Found closing tag: discard content up through close tag
                    close_end = earliest_close + len(chosen_close)
                    self._buf = self._buf[close_end:]
                    self._depth -= 1
                    # Loop continues to handle text after closing tag
                else:
                    # Still inside block: hold entire buffer
                    break

        return "".join(out_parts)

    def flush(self) -> str:
        """Return remaining non-thinking text at end of stream."""
        if self._depth == 0:
            remaining = self._buf
            self._buf = ""
            return remaining
        # Unclosed thinking block at EOS — discard
        self._buf = ""
        return ""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _safe_boundary_normal(self) -> int:
        """Return index up to which buf can be safely yielded.

        buf[safe:] is held back because it is a *proper* non-empty prefix
        of some opening tag (i.e., more input might complete it).
        """
        buf = self._buf
        max_check = min(self._MAX_OPEN_LEN - 1, len(buf))
        # Walk backwards through possible suffix lengths
        for suffix_len in range(max_check, 0, -1):
            suffix = buf[-suffix_len:]
            if any(tag.startswith(suffix) for tag in self._OPEN_TAGS):
                return len(buf) - suffix_len
        return len(buf)

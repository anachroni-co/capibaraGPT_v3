"""Interactive REPL for Capibara Slim (improvements 5, 6, 7).

Features:
  5. History picker  — persistent readline history with arrow-key navigation
  6. Token budget    — per-turn and cumulative token counter displayed inline
  7. Context compaction — summarise old turns when the conversation grows long

Usage:
    from capibara.repl import CapibaraREPL
    repl = CapibaraREPL(generate_fn=my_fn)
    repl.run()

Or from the CLI:
    python -m capibara.repl
"""
from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# readline setup (improvement 5 — history picker)
# ---------------------------------------------------------------------------

_HISTORY_FILE = Path(os.environ.get("CAPIBARA_HISTORY_FILE",
                                    Path.home() / ".capibara_history"))
_HISTORY_MAX = int(os.environ.get("CAPIBARA_HISTORY_MAX", "500"))

try:
    import readline as _readline
    _READLINE_AVAILABLE = True
except ImportError:
    _readline = None  # type: ignore[assignment]
    _READLINE_AVAILABLE = False


class HistoryManager:
    """Persist and restore readline history across sessions."""

    def __init__(
        self,
        path: Path = _HISTORY_FILE,
        max_entries: int = _HISTORY_MAX,
    ) -> None:
        self.path = path
        self.max_entries = max_entries

    def load(self) -> None:
        if not _READLINE_AVAILABLE:
            return
        if self.path.exists():
            try:
                _readline.read_history_file(str(self.path))
            except OSError:
                pass
        _readline.set_history_length(self.max_entries)

    def save(self) -> None:
        if not _READLINE_AVAILABLE:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _readline.write_history_file(str(self.path))
        except OSError as exc:
            logger.warning("Could not save history: %s", exc)

    def prompt(self, text: str) -> str:
        """Read a line from stdin with history support."""
        try:
            return input(text)
        except (EOFError, KeyboardInterrupt):
            raise


# ---------------------------------------------------------------------------
# Token budget (improvement 6)
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Estimate token count.

    Uses the HF tokenizer when available, otherwise falls back to a
    whitespace splitter (1 word ≈ 1.3 tokens, rounded).
    """
    try:
        from utils.tokenizer import SlimTokenizer
        tok = SlimTokenizer.default()
        return len(tok.encode(text))
    except Exception:
        words = len(re.findall(r"\S+", text))
        return max(1, round(words * 1.3))


@dataclass
class TokenBudget:
    """Track token usage across conversation turns."""

    max_tokens: int = 0         # 0 = unlimited
    _total_in: int = field(default=0, init=False, repr=False)
    _total_out: int = field(default=0, init=False, repr=False)
    _turn_in: int = field(default=0, init=False, repr=False)
    _turn_out: int = field(default=0, init=False, repr=False)

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        self._turn_in = _count_tokens(user_text)
        self._turn_out = _count_tokens(assistant_text)
        self._total_in += self._turn_in
        self._total_out += self._turn_out

    @property
    def total(self) -> int:
        return self._total_in + self._total_out

    @property
    def last_turn_tokens(self) -> tuple[int, int]:
        return self._turn_in, self._turn_out

    def budget_exceeded(self) -> bool:
        return self.max_tokens > 0 and self.total >= self.max_tokens

    def status_line(self) -> str:
        tin, tout = self.last_turn_tokens
        if self.max_tokens > 0:
            remaining = max(0, self.max_tokens - self.total)
            return (f"[tokens: +{tin}↑ +{tout}↓ | "
                    f"total={self.total} | remaining={remaining}]")
        return f"[tokens: +{tin}↑ +{tout}↓ | total={self.total}]"


# ---------------------------------------------------------------------------
# Context compaction (improvement 7)
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    role: str       # "user" | "assistant"
    text: str
    token_count: int = 0

    def __post_init__(self) -> None:
        if self.token_count == 0:
            self.token_count = _count_tokens(self.text)


def _summarise(turns: list[Turn], max_tokens: int = 256) -> str:
    """Produce a brief bullet-point summary of a turn list.

    Uses a simple extractive approach: collect first sentence from each
    assistant turn (no external model needed for the fallback).
    """
    try:
        _summarise_with_model(turns, max_tokens)
    except Exception:
        pass

    lines: list[str] = []
    for t in turns:
        if t.role == "assistant":
            first = re.split(r"(?<=[.!?])\s", t.text.strip())[0][:120]
            if first:
                lines.append(f"• {first}")
    summary = "\n".join(lines) if lines else "[conversation compacted]"
    return f"[Summary of earlier conversation]\n{summary}"


def _summarise_with_model(turns: list[Turn], max_tokens: int) -> str:
    raise NotImplementedError


class ContextCompactor:
    """Summarise old turns when the conversation exceeds max_turns."""

    def __init__(self, max_turns: int = 20, summary_max_tokens: int = 256) -> None:
        self.max_turns = max_turns
        self.summary_max_tokens = summary_max_tokens
        self.compactions: int = 0

    def maybe_compact(self, turns: list[Turn]) -> list[Turn]:
        """Return a (possibly compacted) copy of the turn list."""
        if len(turns) <= self.max_turns:
            return turns

        keep_count = self.max_turns // 2
        old = turns[:-keep_count]
        recent = turns[-keep_count:]

        summary_text = _summarise(old, self.summary_max_tokens)
        summary_turn = Turn(role="assistant", text=summary_text)

        self.compactions += 1
        logger.info(
            "context_compactor: compacted %d turns into summary (compaction #%d)",
            len(old),
            self.compactions,
        )
        return [summary_turn] + recent


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

_QUIT_COMMANDS = frozenset({"quit", "exit", "q", ":q", "/quit", "/exit"})
_HELP_TEXT = """\
Capibara Slim REPL
  Type your prompt and press Enter.
  Commands:
    /help       — this message
    /history    — show session history
    /flags      — show active feature flags
    /clear      — clear conversation context
    /tokens     — show token usage
    quit | exit — exit the REPL
"""


GenerateFn = Callable[[str, list[Turn]], str]


def _stub_generate(query: str, history: list[Turn]) -> str:
    return f"[stub] Echo: {query}"


class CapibaraREPL:
    """Interactive REPL combining history, token budget, and context compaction."""

    def __init__(
        self,
        generate_fn: GenerateFn = _stub_generate,
        history_file: Path = _HISTORY_FILE,
        max_budget_tokens: int = 0,
        max_turns: int = 20,
        summary_max_tokens: int = 256,
        show_tokens: bool = True,
        prompt: str = "you> ",
        assistant_prefix: str = "capibara> ",
    ) -> None:
        self.generate_fn = generate_fn
        self.history_mgr = HistoryManager(path=history_file)
        self.budget = TokenBudget(max_tokens=max_budget_tokens)
        self.compactor = ContextCompactor(
            max_turns=max_turns,
            summary_max_tokens=summary_max_tokens,
        )
        self.show_tokens = show_tokens
        self.prompt = prompt
        self.assistant_prefix = assistant_prefix
        self._turns: list[Turn] = []

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _cmd_help(self) -> None:
        print(_HELP_TEXT)

    def _cmd_history(self) -> None:
        for i, turn in enumerate(self._turns, 1):
            tag = "U" if turn.role == "user" else "A"
            print(f"  {i:3d} [{tag}] {turn.text[:80]}")

    def _cmd_flags(self) -> None:
        try:
            from config.feature_flags import all_flags
            for name, enabled in all_flags().items():
                status = "ON " if enabled else "off"
                print(f"  {status}  {name}")
        except Exception as exc:
            print(f"  (could not load feature flags: {exc})")

    def _cmd_clear(self) -> None:
        self._turns.clear()
        self.budget._total_in = 0
        self.budget._total_out = 0
        print("  Context cleared.")

    def _cmd_tokens(self) -> None:
        print(f"  {self.budget.status_line()}")

    def _handle_command(self, line: str) -> bool:
        """Return True if the line was a command (consumed), False otherwise."""
        stripped = line.strip()
        if stripped in _QUIT_COMMANDS:
            raise SystemExit(0)
        if stripped == "/help":
            self._cmd_help()
            return True
        if stripped == "/history":
            self._cmd_history()
            return True
        if stripped == "/flags":
            self._cmd_flags()
            return True
        if stripped == "/clear":
            self._cmd_clear()
            return True
        if stripped == "/tokens":
            self._cmd_tokens()
            return True
        return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the interactive REPL loop."""
        self.history_mgr.load()
        print("Capibara Slim REPL — type /help for commands, quit to exit.")
        try:
            self._loop()
        finally:
            self.history_mgr.save()

    def _loop(self) -> None:
        while True:
            try:
                line = self.history_mgr.prompt(self.prompt)
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not line.strip():
                continue
            if self._handle_command(line):
                continue

            if self.budget.budget_exceeded():
                print("  [token budget exhausted — use /clear to reset]")
                continue

            # Compact context if needed
            self._turns = self.compactor.maybe_compact(self._turns)

            # Generate response
            try:
                response = self.generate_fn(line, list(self._turns))
            except Exception as exc:
                print(f"  [error: {exc}]")
                continue

            print(f"{self.assistant_prefix}{response}")

            # Record turn + budget
            self._turns.append(Turn(role="user", text=line))
            self._turns.append(Turn(role="assistant", text=response))
            self.budget.record_turn(line, response)

            if self.show_tokens:
                print(f"  {self.budget.status_line()}", flush=True)


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    repl = CapibaraREPL()
    repl.run()

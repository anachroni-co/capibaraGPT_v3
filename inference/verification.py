"""Post-generation verification step (improvement 3/7).

Validates model output before returning it to the caller.  Each check is
a predicate that returns (passed: bool, reason: str).  The verifier runs
all enabled checks and attaches a ``verification`` key to the result dict.

Checks (all opt-in via VerificationConfig):
  - non_empty      — output is not blank after stripping whitespace
  - min_length     — output meets a minimum character length
  - max_length     — output does not exceed a maximum character length
  - no_repetition  — output does not repeat the same n-gram excessively
  - no_truncation  — output does not end mid-sentence (heuristic)
  - coherence      — output contains at least one real word token

Checks are skipped individually when they are disabled; the overall result
is ``passed = all individual checks passed``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Tuple


CheckResult = Tuple[bool, str]  # (passed, reason)
CheckFn = Callable[[str], CheckResult]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_non_empty(text: str) -> CheckResult:
    passed = bool(text.strip())
    return passed, "" if passed else "output is empty"


def check_min_length(min_chars: int) -> CheckFn:
    def _check(text: str) -> CheckResult:
        n = len(text.strip())
        passed = n >= min_chars
        return passed, "" if passed else f"output too short ({n} < {min_chars} chars)"
    return _check


def check_max_length(max_chars: int) -> CheckFn:
    def _check(text: str) -> CheckResult:
        n = len(text.strip())
        passed = n <= max_chars
        return passed, "" if passed else f"output too long ({n} > {max_chars} chars)"
    return _check


def check_no_repetition(ngram: int = 4, max_ratio: float = 0.5) -> CheckFn:
    """Fail if any n-gram makes up more than max_ratio of all n-grams."""
    def _check(text: str) -> CheckResult:
        tokens = text.lower().split()
        if len(tokens) < ngram * 2:
            return True, ""
        grams = [tuple(tokens[i:i + ngram]) for i in range(len(tokens) - ngram + 1)]
        if not grams:
            return True, ""
        from collections import Counter
        most_common_count = Counter(grams).most_common(1)[0][1]
        ratio = most_common_count / len(grams)
        passed = ratio <= max_ratio
        return passed, "" if passed else f"excessive repetition (ratio={ratio:.2f})"
    return _check


def check_no_truncation(text: str) -> CheckResult:
    """Heuristic: output probably truncated if it ends without punctuation."""
    stripped = text.rstrip()
    if not stripped:
        return True, ""
    passed = stripped[-1] in ".!?\"')\n"
    return passed, "" if passed else "output may be truncated (no closing punctuation)"


def check_coherence(text: str) -> CheckResult:
    """At least one alphabetic word of length >= 2 must be present."""
    words = re.findall(r"[a-zA-Z]{2,}", text)
    passed = len(words) > 0
    return passed, "" if passed else "output contains no recognisable words"


# ---------------------------------------------------------------------------
# Config + Verifier
# ---------------------------------------------------------------------------

@dataclass
class VerificationConfig:
    non_empty: bool = True
    min_length: int = 0           # 0 = disabled
    max_length: int = 0           # 0 = disabled
    no_repetition: bool = True
    repetition_ngram: int = 4
    repetition_max_ratio: float = 0.5
    no_truncation: bool = False   # disabled by default (too strict for short outputs)
    coherence: bool = True


@dataclass
class VerificationReport:
    passed: bool
    failures: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"passed": self.passed, "failures": self.failures}


class OutputVerifier:
    """Run configured checks against a generated text string."""

    def __init__(self, config: VerificationConfig | None = None):
        self.config = config or VerificationConfig()

    def _build_checks(self) -> list[tuple[str, CheckFn]]:
        cfg = self.config
        checks: list[tuple[str, CheckFn]] = []
        if cfg.non_empty:
            checks.append(("non_empty", check_non_empty))
        if cfg.min_length > 0:
            checks.append(("min_length", check_min_length(cfg.min_length)))
        if cfg.max_length > 0:
            checks.append(("max_length", check_max_length(cfg.max_length)))
        if cfg.no_repetition:
            checks.append(("no_repetition", check_no_repetition(
                cfg.repetition_ngram, cfg.repetition_max_ratio)))
        if cfg.no_truncation:
            checks.append(("no_truncation", check_no_truncation))
        if cfg.coherence:
            checks.append(("coherence", check_coherence))
        return checks

    def verify(self, text: str) -> VerificationReport:
        """Run all enabled checks and return a VerificationReport."""
        failures: list[str] = []
        for name, fn in self._build_checks():
            passed, reason = fn(text)
            if not passed:
                failures.append(f"{name}: {reason}")
        return VerificationReport(passed=len(failures) == 0, failures=failures)


def verify_output(
    text: str,
    config: VerificationConfig | None = None,
) -> VerificationReport:
    """Convenience function — verify text with optional config."""
    return OutputVerifier(config).verify(text)

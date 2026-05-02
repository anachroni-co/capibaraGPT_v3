"""
Think-Anywhere reward functions.

Implements the hierarchical reward from Section 3.3 of the paper:

    R(y) = alpha * R_struct(y) + (1 - alpha) * R_correct(y)

where:
    R_struct  (Eq. 10) — binary: checks upfront <think> + at least one
              <thinkanywhere> block are present in the response.
    R_correct (Eq. 11) — binary: code extracted from response passes
              all provided test cases when executed.

Reference: "Think Anywhere in Code Generation", Jiang et al. 2026
"""
from __future__ import annotations

import ast
import contextlib
import io
import logging
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .config import ThinkAnywhereConfig
from .token_processor import ThinkAnywhereProcessor

logger = logging.getLogger(__name__)


@dataclass
class RewardResult:
    """Detailed reward breakdown for a single model response."""

    combined: float          # R(y) — final scalar fed to GRPO
    structure: float         # R_struct in {0, 1}
    correctness: float       # R_correct in [0, 1]
    passed_tests: int
    total_tests: int
    validation_errors: List[str] = field(default_factory=list)
    execution_errors: List[str] = field(default_factory=list)


class ThinkAnywhereReward:
    """Compute Think-Anywhere rewards for GRPO training.

    Usage::

        reward_fn = ThinkAnywhereReward(config)
        result = reward_fn(response_text, test_cases=["assert f(1) == 2"])
        print(result.combined)  # 0.0 – 1.0
    """

    def __init__(self, config: Optional[ThinkAnywhereConfig] = None) -> None:
        self.cfg = config or ThinkAnywhereConfig()
        self.processor = ThinkAnywhereProcessor(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        response: str,
        test_cases: Optional[List[str]] = None,
        timeout: float = 5.0,
    ) -> RewardResult:
        """Compute combined reward for a Think-Anywhere model response.

        Args:
            response: Full model output (including <think> and <thinkanywhere> blocks).
            test_cases: List of assertion strings used to evaluate code correctness.
                        If None or empty, R_correct = 0.
            timeout: Maximum seconds allowed for code execution per test case.
        """
        r_struct, validation_errors = self._structure_reward(response)
        r_correct, passed, total, exec_errors = self._correctness_reward(
            response, test_cases or [], timeout
        )

        alpha = self.cfg.structure_reward_weight
        combined = alpha * r_struct + (1 - alpha) * r_correct

        return RewardResult(
            combined=combined,
            structure=r_struct,
            correctness=r_correct,
            passed_tests=passed,
            total_tests=total,
            validation_errors=validation_errors,
            execution_errors=exec_errors,
        )

    def batch(
        self,
        responses: List[str],
        test_cases: Optional[List[str]] = None,
        timeout: float = 5.0,
    ) -> List[RewardResult]:
        """Compute rewards for a group of G GRPO rollout responses."""
        return [self(r, test_cases, timeout) for r in responses]

    def group_normalized_advantages(self, results: List[RewardResult]) -> List[float]:
        """Compute GRPO group-normalized advantages (Eq. 7).

        Returns the standardised advantage for each response in the group.
        """
        rewards = [r.combined for r in results]
        mean_r = sum(rewards) / max(len(rewards), 1)
        variance = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1)
        std_r = variance ** 0.5 + 1e-8
        return [(r - mean_r) / std_r for r in rewards]

    # ------------------------------------------------------------------
    # Internal reward components
    # ------------------------------------------------------------------

    def _structure_reward(self, response: str):
        """R_struct (Eq. 10): 1 if format is valid, 0 otherwise."""
        is_valid, errors = self.processor.validate(response)
        return float(is_valid), errors

    def _correctness_reward(
        self,
        response: str,
        test_cases: List[str],
        timeout: float,
    ):
        """R_correct (Eq. 11): fraction of test cases passed by clean code."""
        if not test_cases:
            return 0.0, 0, 0, ["No test cases provided"]

        parsed = self.processor.parse(response)
        code = parsed.clean_code
        if not code.strip():
            return 0.0, 0, len(test_cases), ["Extracted code is empty"]

        # Validate Python syntax before executing
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return 0.0, 0, len(test_cases), [f"SyntaxError: {exc}"]

        passed = 0
        exec_errors: List[str] = []
        for i, test in enumerate(test_cases):
            ok, err = self._run_test(code, test, timeout)
            if ok:
                passed += 1
            else:
                exec_errors.append(f"Test {i}: {err}")

        correctness = passed / len(test_cases)
        return correctness, passed, len(test_cases), exec_errors

    @staticmethod
    def _run_test(code: str, test_case: str, timeout: float):
        """Execute code + assertion in a subprocess sandbox.

        Returns (passed: bool, error_message: str).
        Using a subprocess avoids polluting the current process namespace.
        """
        script = textwrap.dedent(f"""
import sys
try:
{textwrap.indent(code, '    ')}
{textwrap.indent(test_case, '    ')}
    sys.exit(0)
except AssertionError as e:
    print(f"AssertionError: {{e}}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"{{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(2)
""")
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as tmp:
                tmp.write(script)
                tmp_path = tmp.name

            result = subprocess.run(  # nosec B603 — isolated tmp script
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            Path(tmp_path).unlink(missing_ok=True)

            if result.returncode == 0:
                return True, ""
            return False, (result.stderr or "non-zero exit").strip()

        except subprocess.TimeoutExpired:
            return False, f"Timeout after {timeout}s"
        except Exception as exc:
            return False, str(exc)
        finally:
            with contextlib.suppress(Exception):
                Path(tmp_path).unlink(missing_ok=True)

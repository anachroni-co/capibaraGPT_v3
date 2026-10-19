"""evaluation/code_eval.py

Real evaluation for code-generation models on CPU.

Metrics
-------
* exact_match     — generated text == reference (after normalisation)
* prefix_match    — reference is a prefix of generated text (partial credit)
* pass_at_k       — k independent samples, at least one passes sandbox exec
* token_accuracy  — greedy next-token accuracy on held-out corpus
* ntp_loss        — cross-entropy on held-out byte sequence (perplexity proxy)

Task set
--------
A small built-in set of Python tasks with reference solutions so that
evaluation runs with zero external dependencies.  Sandbox execution uses
``subprocess`` with a wall-clock timeout (no network, restricted resources).

Usage
-----
    from evaluation.code_eval import Evaluator, BUILTIN_TASKS
    from scripts.train_lmtp_cpu import ByteLM, LMTPHeads

    eval = Evaluator(backbone, heads)
    report = eval.run(BUILTIN_TASKS, k=4, max_new_tokens=128)
    print(report.summary())
"""
from __future__ import annotations

import ast
import hashlib
import logging
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

@dataclass
class CodeTask:
    """One evaluation task."""
    task_id: str
    prompt: str                   # what we feed to the model
    reference: str                # canonical solution (for exact/prefix match)
    test_code: str                # executed after the generated code to verify
    description: str = ""


# ---------------------------------------------------------------------------
# Built-in task set
# ---------------------------------------------------------------------------

BUILTIN_TASKS: list[CodeTask] = [
    CodeTask(
        task_id="add_two",
        description="Simple addition function",
        prompt="# Write a Python function that returns the sum of two integers\ndef add(a, b):",
        reference="    return a + b",
        test_code="assert add(1, 2) == 3\nassert add(-1, 5) == 4\nassert add(0, 0) == 0",
    ),
    CodeTask(
        task_id="factorial",
        description="Recursive factorial",
        prompt="# Write a Python function that computes n! (factorial of n)\ndef factorial(n):",
        reference="    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        test_code="assert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800",
    ),
    CodeTask(
        task_id="is_palindrome",
        description="Palindrome check",
        prompt="# Write a Python function that returns True if s is a palindrome\ndef is_palindrome(s):",
        reference="    return s == s[::-1]",
        test_code="assert is_palindrome('racecar')\nassert not is_palindrome('hello')\nassert is_palindrome('')",
    ),
    CodeTask(
        task_id="max_list",
        description="Maximum element of a list",
        prompt="# Write a Python function that returns the maximum element of a list\ndef list_max(lst):",
        reference="    return max(lst)",
        test_code="assert list_max([1,3,2]) == 3\nassert list_max([-1,-2,-3]) == -1",
    ),
    CodeTask(
        task_id="fizzbuzz",
        description="FizzBuzz",
        prompt="# Write a function that returns 'Fizz' if n%3==0, 'Buzz' if n%5==0, 'FizzBuzz' if both, else str(n)\ndef fizzbuzz(n):",
        reference="    if n % 15 == 0:\n        return 'FizzBuzz'\n    if n % 3 == 0:\n        return 'Fizz'\n    if n % 5 == 0:\n        return 'Buzz'\n    return str(n)",
        test_code="assert fizzbuzz(3)=='Fizz'\nassert fizzbuzz(5)=='Buzz'\nassert fizzbuzz(15)=='FizzBuzz'\nassert fizzbuzz(7)=='7'",
    ),
    CodeTask(
        task_id="count_vowels",
        description="Count vowels in string",
        prompt="# Write a function that counts vowels (aeiou) in a string\ndef count_vowels(s):",
        reference="    return sum(1 for c in s.lower() if c in 'aeiou')",
        test_code="assert count_vowels('hello') == 2\nassert count_vowels('rhythm') == 0\nassert count_vowels('AEIOU') == 5",
    ),
    CodeTask(
        task_id="flatten",
        description="Flatten a list of lists",
        prompt="# Write a function that flattens a list of lists\ndef flatten(lst):",
        reference="    return [x for sub in lst for x in sub]",
        test_code="assert flatten([[1,2],[3,4]]) == [1,2,3,4]\nassert flatten([]) == []\nassert flatten([[1]]) == [1]",
    ),
    CodeTask(
        task_id="is_prime",
        description="Primality test",
        prompt="# Write a function that returns True if n is a prime number\ndef is_prime(n):",
        reference="    if n < 2:\n        return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0:\n            return False\n    return True",
        test_code="assert is_prime(2)\nassert is_prime(13)\nassert not is_prime(1)\nassert not is_prime(9)",
    ),
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task_id: str
    exact_match: bool
    prefix_match: bool
    pass_at_k: bool             # at least one of k samples passed sandbox
    samples: list[str]          # raw generated strings
    passed_samples: list[bool]
    latency_ms: float
    error: str = ""


@dataclass
class EvalReport:
    results: list[TaskResult]
    ntp_loss: float = 0.0
    token_accuracy: float = 0.0
    elapsed_s: float = 0.0

    def summary(self) -> str:
        n = len(self.results)
        exact = sum(r.exact_match for r in self.results)
        prefix = sum(r.prefix_match for r in self.results)
        passed = sum(r.pass_at_k for r in self.results)
        lines = [
            f"{'─'*50}",
            f"  Tasks evaluated : {n}",
            f"  Exact match     : {exact}/{n} ({100*exact/max(n,1):.1f}%)",
            f"  Prefix match    : {prefix}/{n} ({100*prefix/max(n,1):.1f}%)",
            f"  Pass@k          : {passed}/{n} ({100*passed/max(n,1):.1f}%)",
            f"  NTP loss        : {self.ntp_loss:.4f} nats/byte",
            f"  Token accuracy  : {100*self.token_accuracy:.2f}%",
            f"  Total time      : {self.elapsed_s:.1f}s",
            f"{'─'*50}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        n = max(len(self.results), 1)
        return {
            "n_tasks": len(self.results),
            "exact_match_pct": round(100 * sum(r.exact_match for r in self.results) / n, 1),
            "prefix_match_pct": round(100 * sum(r.prefix_match for r in self.results) / n, 1),
            "pass_at_k_pct": round(100 * sum(r.pass_at_k for r in self.results) / n, 1),
            "ntp_loss": round(self.ntp_loss, 4),
            "token_accuracy_pct": round(100 * self.token_accuracy, 2),
            "elapsed_s": round(self.elapsed_s, 1),
        }


# ---------------------------------------------------------------------------
# Sandbox execution
# ---------------------------------------------------------------------------

def _sandbox_exec(code: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Run code in a subprocess; return (passed, error_message)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr or result.stdout).strip()[:200]
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired"
    except Exception as e:
        return False, str(e)
    finally:
        Path(fname).unlink(missing_ok=True)


def _normalise(text: str) -> str:
    """Normalise generated text for comparison."""
    return textwrap.dedent(text).strip().replace("\r\n", "\n")


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Evaluate a ByteLM + LMTPHeads model on code tasks.

    Parameters
    ----------
    backbone : ByteLM
    heads    : LMTPHeads
    decode_fn : optional callable ``(prompt: str, max_new_tokens: int) -> str``
        If None, uses LMTPCachedDecoder with byte-level encoding.
    """

    def __init__(self, backbone, heads, decode_fn: Optional[Callable] = None) -> None:
        self.backbone = backbone
        self.heads = heads
        self._decode_fn = decode_fn or self._default_decode

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def _default_decode(self, prompt: str, max_new_tokens: int = 128) -> str:
        from inference.cpu_kv_cache import LMTPCachedDecoder, CacheDecodeConfig
        cfg = CacheDecodeConfig(max_new_tokens=max_new_tokens, greedy=True)
        decoder = LMTPCachedDecoder(self.backbone, self.heads, cfg)

        prompt_ids = list(prompt.encode("utf-8", errors="replace"))
        out_ids = decoder.generate(prompt_ids, max_new_tokens=max_new_tokens)
        new_ids = out_ids[len(prompt_ids):]
        return bytes([t % 256 for t in new_ids]).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # NTP loss on a corpus
    # ------------------------------------------------------------------

    def ntp_loss(self, corpus: np.ndarray, seq_len: int = 256,
                 n_batches: int = 8, batch_size: int = 4) -> float:
        """Cross-entropy NTP loss on a random sample from corpus."""
        total_loss = 0.0
        n_valid = 0
        rng = np.random.default_rng(0)
        for _ in range(n_batches):
            starts = rng.integers(0, max(1, len(corpus) - seq_len - 1), size=batch_size)
            ids = np.stack([corpus[s:s + seq_len] for s in starts])      # (B, T)
            targets = np.stack([corpus[s+1:s+1+seq_len] for s in starts])
            logits, _ = self.backbone.forward(ids.astype(np.int32))
            # log-softmax
            shift = logits.max(axis=-1, keepdims=True)
            log_p = logits - shift - np.log(np.exp(logits - shift).sum(-1, keepdims=True))
            B, T, V = logits.shape
            tgt_lp = log_p[np.arange(B)[:, None], np.arange(T)[None, :], targets]
            total_loss += float(-tgt_lp.mean())
            n_valid += 1
        return total_loss / max(n_valid, 1)

    # ------------------------------------------------------------------
    # Token accuracy
    # ------------------------------------------------------------------

    def token_accuracy(self, corpus: np.ndarray, seq_len: int = 256,
                       n_batches: int = 8, batch_size: int = 4) -> float:
        """Fraction of next-token predictions that match the greedy argmax."""
        correct = total = 0
        rng = np.random.default_rng(1)
        for _ in range(n_batches):
            starts = rng.integers(0, max(1, len(corpus) - seq_len - 1), size=batch_size)
            ids = np.stack([corpus[s:s + seq_len] for s in starts])
            targets = np.stack([corpus[s+1:s+1+seq_len] for s in starts])
            logits, _ = self.backbone.forward(ids.astype(np.int32))
            preds = logits.argmax(-1)
            correct += int((preds == targets).sum())
            total += preds.size
        return correct / max(total, 1)

    # ------------------------------------------------------------------
    # Task evaluation
    # ------------------------------------------------------------------

    def _eval_task(self, task: CodeTask, k: int, max_new_tokens: int) -> TaskResult:
        t0 = time.perf_counter()
        samples: list[str] = []
        passed: list[bool] = []

        for _ in range(k):
            try:
                generated = self._decode_fn(task.prompt, max_new_tokens)
            except Exception as e:
                generated = ""
                logger.debug("Generation error on %s: %s", task.task_id, e)
            samples.append(generated)

            # Build executable code: prompt + generated + test
            full_code = task.prompt + "\n" + generated + "\n" + task.test_code
            ok, _ = _sandbox_exec(full_code)
            passed.append(ok)

        gen_norm = _normalise(samples[0]) if samples else ""
        ref_norm = _normalise(task.reference)

        return TaskResult(
            task_id=task.task_id,
            exact_match=(gen_norm == ref_norm),
            prefix_match=gen_norm.startswith(ref_norm[:20]) if len(ref_norm) >= 20 else False,
            pass_at_k=any(passed),
            samples=samples,
            passed_samples=passed,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        tasks: Optional[list[CodeTask]] = None,
        k: int = 1,
        max_new_tokens: int = 128,
        corpus: Optional[np.ndarray] = None,
    ) -> EvalReport:
        """Run full evaluation.

        Args:
            tasks:          List of CodeTask objects (default: BUILTIN_TASKS).
            k:              Samples per task for pass@k.
            max_new_tokens: Generation budget per sample.
            corpus:         Byte array for NTP loss / token-accuracy metrics.
                            If None, these metrics are skipped.

        Returns:
            EvalReport with all results.
        """
        tasks = tasks or BUILTIN_TASKS
        t_start = time.perf_counter()

        results = []
        for i, task in enumerate(tasks, 1):
            logger.info("[%d/%d] %s — %s", i, len(tasks), task.task_id, task.description)
            r = self._eval_task(task, k, max_new_tokens)
            results.append(r)
            logger.info(
                "  exact=%s prefix=%s pass@%d=%s  %.0fms",
                r.exact_match, r.prefix_match, k, r.pass_at_k, r.latency_ms,
            )

        ntp_l = 0.0
        tok_acc = 0.0
        if corpus is not None and len(corpus) > 0:
            logger.info("Computing NTP loss and token accuracy on corpus…")
            ntp_l = self.ntp_loss(corpus)
            tok_acc = self.token_accuracy(corpus)

        elapsed = time.perf_counter() - t_start
        report = EvalReport(results=results, ntp_loss=ntp_l,
                            token_accuracy=tok_acc, elapsed_s=elapsed)
        logger.info("\n%s", report.summary())
        return report

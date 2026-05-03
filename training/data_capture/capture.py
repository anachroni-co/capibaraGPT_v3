"""
TrainingDataCapture — intercepts inference outputs and saves them as
(prompt, response) pairs for fine-tuning.

Signal sources:
  1. web_search  — model fetched live data; response is grounded
  2. fact_check  — a claim was verified; verdict + source attached
  3. api_routing — response came from an external model (distillation)
  4. uncertain   — model flagged low confidence; pair queued for review

Storage format: newline-delimited JSON (jsonl) one file per source type.
Each record is a self-contained training sample ready for SFT or DPO.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CaptureConfig:
    output_dir: str = "data/captured"
    max_file_size_mb: int = 100
    flush_every: int = 10          # write to disk every N samples
    min_response_tokens: int = 10  # discard very short responses
    sources: List[str] = field(default_factory=lambda: [
        "web_search", "fact_check", "api_routing", "uncertain"
    ])


@dataclass
class CapturedSample:
    prompt: str
    response: str
    source: str                        # web_search | fact_check | api_routing | uncertain
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional DPO fields
    chosen: Optional[str] = None       # preferred response (e.g. verified answer)
    rejected: Optional[str] = None     # dispreferred response (e.g. original hallucination)


class TrainingDataCapture:
    """
    Thread-safe writer that accumulates CapturedSamples and flushes to disk.

    Usage:
        capture = TrainingDataCapture(CaptureConfig(output_dir="data/captured"))

        # From WebSearchHandler side-effect:
        capture.log_web_search(query="...", result=web_result, context={...})

        # From FactCheckHandler side-effect:
        capture.log_fact_check(prompt="...", original="...", verdict={...})

        # From ConfidenceRouter side-effect:
        capture.log_api_routing(prompt="...", local_response="...",
                                api_response="...", model="gpt-4o-mini")

        # Flush remaining buffer to disk:
        capture.flush()
    """

    def __init__(self, config: Optional[CaptureConfig] = None) -> None:
        self.config = config or CaptureConfig()
        self._output_dir = Path(self.config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._buffers: Dict[str, List[CapturedSample]] = {
            s: [] for s in self.config.sources
        }
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = {s: 0 for s in self.config.sources}

    # ------------------------------------------------------------------
    # Public logging methods (called as side-effects from token handlers)
    # ------------------------------------------------------------------

    def log_web_search(
        self,
        query: str,
        result: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a web-search grounded (query, snippet) pair."""
        if not result or not getattr(result, "top_snippet", ""):
            return
        prompt = context.get("prompt", query) if context else query
        response = result.top_snippet
        self._add(CapturedSample(
            prompt=prompt,
            response=response,
            source="web_search",
            metadata={
                "query": query,
                "urls": getattr(result, "urls", []),
                "engine": getattr(result, "engine", ""),
                "latency_ms": getattr(result, "latency_ms", 0),
            },
        ))

    def log_fact_check(
        self,
        prompt: str,
        original_response: str,
        verdict: Dict[str, Any],
    ) -> None:
        """
        Log a fact-check event as a DPO pair:
          chosen  = corrected / verified response
          rejected = original (potentially wrong) response
        """
        corrected = verdict.get("corrected") or verdict.get("source")
        if not corrected:
            return
        self._add(CapturedSample(
            prompt=prompt,
            response=corrected,
            source="fact_check",
            chosen=corrected,
            rejected=original_response,
            metadata=verdict,
        ))

    def log_api_routing(
        self,
        prompt: str,
        local_response: str,
        api_response: str,
        model: str = "unknown",
        api_latency_ms: float = 0.0,
    ) -> None:
        """
        Log an API-routed response as a distillation pair:
          chosen  = api_response (higher-quality teacher)
          rejected = local_response (student, may be weaker)
        Only saved when api_response is meaningfully different.
        """
        if not api_response or api_response.strip() == local_response.strip():
            return
        self._add(CapturedSample(
            prompt=prompt,
            response=api_response,
            source="api_routing",
            chosen=api_response,
            rejected=local_response,
            metadata={"model": model, "api_latency_ms": api_latency_ms},
        ))

    def log_uncertain(
        self,
        prompt: str,
        response: str,
        uncertain_spans: List[str],
    ) -> None:
        """Queue a response that contains <uncertain> spans for human review."""
        if not uncertain_spans:
            return
        self._add(CapturedSample(
            prompt=prompt,
            response=response,
            source="uncertain",
            metadata={"uncertain_spans": uncertain_spans},
        ))

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def flush(self) -> None:
        """Force-write all buffered samples to disk."""
        with self._lock:
            for source, buf in self._buffers.items():
                if buf:
                    self._write(source, buf)
                    self._buffers[source] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add(self, sample: CapturedSample) -> None:
        if len(sample.response.split()) < self.config.min_response_tokens:
            return
        with self._lock:
            self._buffers[sample.source].append(sample)
            self._counts[sample.source] += 1
            if len(self._buffers[sample.source]) >= self.config.flush_every:
                buf = self._buffers[sample.source]
                self._buffers[sample.source] = []
                self._write(sample.source, buf)

    def _write(self, source: str, samples: List[CapturedSample]) -> None:
        path = self._output_dir / f"{source}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

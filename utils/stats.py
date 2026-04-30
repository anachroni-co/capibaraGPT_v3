"""Capibara Slim — request statistics collector.

Thread-safe counters updated on every /generate call.
Exposed via GET /metrics.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Stats:
    total_requests: int = 0
    total_tokens: int = 0
    total_errors: int = 0
    total_blocked: int = 0
    latency_sum_ms: float = 0.0
    latency_count: int = 0
    _started: float = field(default_factory=time.monotonic, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        tokens: int = 0,
        latency_ms: float = 0.0,
        error: bool = False,
        blocked: bool = False,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.total_tokens += tokens
            self.latency_sum_ms += latency_ms
            self.latency_count += 1
            if error:
                self.total_errors += 1
            if blocked:
                self.total_blocked += 1

    def snapshot(self) -> dict:
        with self._lock:
            avg_lat = (
                round(self.latency_sum_ms / self.latency_count, 1)
                if self.latency_count
                else 0.0
            )
            uptime = round(time.monotonic() - self._started, 1)
            return {
                "uptime_seconds": uptime,
                "total_requests": self.total_requests,
                "total_tokens": self.total_tokens,
                "total_errors": self.total_errors,
                "total_blocked": self.total_blocked,
                "avg_latency_ms": avg_lat,
            }


_global = _Stats()


def record(**kwargs) -> None:
    _global.record(**kwargs)


def snapshot() -> dict:
    return _global.snapshot()

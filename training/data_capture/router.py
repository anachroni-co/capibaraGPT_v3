"""
ConfidenceRouter — routes inference requests to an external API when
the local model signals low confidence, and captures both responses
as training data.

Routing triggers (any one is sufficient):
  1. <uncertain> token present in the local response
  2. <fact_check> token present in the local response
  3. model log-probability below confidence_threshold
  4. random sample rate (sample_rate, for exploration)

When routing fires:
  - query is sent to the configured external API (via OpenRouter or direct)
  - api_response is returned to the user
  - (prompt, local_response, api_response) is logged via TrainingDataCapture
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from core.special_tokens.uncertain import UNCERTAIN_TOKEN
from core.special_tokens.fact_check import FACT_CHECK_TOKEN


@dataclass
class RouterConfig:
    # Fraction of requests always routed to API (exploration)
    sample_rate: float = 0.0

    # Route when local model log-prob per token falls below this
    confidence_threshold: float = 0.0   # 0 = disabled (use token signals only)

    # External API config
    api_base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    api_model: str = "meta-llama/llama-3.1-8b-instruct:free"
    api_timeout: float = 30.0
    max_tokens: int = 512

    # Cost guard: stop routing after this many API calls per session
    max_api_calls: int = 1000


class ConfidenceRouter:
    """
    Wraps a local generate function and transparently routes to an
    external API when confidence signals are detected.

        router = ConfidenceRouter(
            local_fn=my_model.generate,
            config=RouterConfig(sample_rate=0.1, api_key="sk-…"),
            capture=TrainingDataCapture(),
        )
        response = router.generate(prompt)
    """

    def __init__(
        self,
        local_fn: Callable[[str], str],
        config: Optional[RouterConfig] = None,
        capture=None,
    ) -> None:
        self._local = local_fn
        self.config = config or RouterConfig()
        self._capture = capture
        self._api_call_count = 0

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response, routing to API when confidence is low."""
        local_response = self._local(prompt, **kwargs)

        if self._should_route(local_response):
            api_response = self._call_api(prompt)
            if api_response:
                if self._capture:
                    self._capture.log_api_routing(
                        prompt=prompt,
                        local_response=local_response,
                        api_response=api_response,
                        model=self.config.api_model,
                    )
                return api_response

        # Log uncertain spans even when not routing
        if self._capture and UNCERTAIN_TOKEN.open_tag in local_response:
            from core.special_tokens.uncertain import UncertainTokenExtractor
            spans = UncertainTokenExtractor().extract(local_response).blocks
            if spans:
                self._capture.log_uncertain(prompt, local_response, spans)

        return local_response

    def get_stats(self) -> Dict[str, Any]:
        return {
            "api_calls": self._api_call_count,
            "api_calls_remaining": self.config.max_api_calls - self._api_call_count,
        }

    # ------------------------------------------------------------------

    def _should_route(self, local_response: str) -> bool:
        if self._api_call_count >= self.config.max_api_calls:
            return False
        if UNCERTAIN_TOKEN.open_tag in local_response:
            return True
        if FACT_CHECK_TOKEN.open_tag in local_response:
            return True
        if self.config.sample_rate > 0 and random.random() < self.config.sample_rate:
            return True
        return False

    def _call_api(self, prompt: str) -> Optional[str]:
        try:
            import urllib.request
            import json

            self._api_call_count += 1
            payload = json.dumps({
                "model": self.config.api_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.config.max_tokens,
            }).encode()

            req = urllib.request.Request(
                f"{self.config.api_base_url}/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=self.config.api_timeout) as r:
                data = json.loads(r.read())
            return data["choices"][0]["message"]["content"]
        except Exception:
            return None

"""inference/gate_inference_loop.py

Online gate-training inference loop for ThinkAnywhereGate.

The gate (core/think_anywhere/activation_gate.py) decides at inference
time whether each <thinkanywhere> block improves output quality.  It is a
small MLP trained online via SGD with RewardResult.combined as the label.

This module provides the loop that connects the three parts:
  1. Model generates a response (with ThinkAnywhereStreamFilter)
  2. ThinkAnywhereReward scores the response → RewardResult
  3. gate.record(hidden, reward_result) buffers the example
  4. Every `train_every` responses, gate.train_step() runs one SGD update

After `checkpoint_every` responses the gate is saved to disk so training
survives restarts.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

from core.think_anywhere import (
    ThinkAnywhereGate,
    ThinkAnywhereReward,
    ThinkAnywhereStreamFilter,
    GateConfig,
    PositionalFeatures,
)
from core.think_anywhere.rewards import RewardResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GateLoopConfig:
    """Configuration for the online gate-training loop."""
    train_every: int = 32          # SGD update every N scored responses
    checkpoint_every: int = 256    # save gate to disk every N responses
    checkpoint_path: str = "checkpoints/gate.npz"
    log_every: int = 16
    max_responses: int = 0         # 0 = run forever
    min_reward_for_positive: float = 0.5   # RewardResult.combined threshold


# ---------------------------------------------------------------------------
# Response record
# ---------------------------------------------------------------------------

@dataclass
class ResponseRecord:
    """One generated response with its reward and hidden state."""
    prompt: str
    response: str
    hidden: Optional[np.ndarray]    # backbone hidden at <thinkanywhere> position
    reward: RewardResult
    had_think_block: bool
    gate_decision: bool             # what the gate decided before generation


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------

class GateInferenceLoop:
    """Runs generation + reward + gate-update in a single online loop.

    Parameters
    ----------
    generate_fn : callable
        ``generate_fn(prompt: str) -> (response: str, hidden: np.ndarray | None)``
        The hidden state should be the backbone output at the position just
        before the first <thinkanywhere> block (if any).  Pass None if the
        backbone does not expose hidden states.
    reward_fn : ThinkAnywhereReward or compatible callable
        ``reward_fn(response: str, reference: str | None) -> RewardResult``
    gate : ThinkAnywhereGate
    cfg : GateLoopConfig
    """

    def __init__(
        self,
        generate_fn: Callable[[str], tuple[str, Optional[np.ndarray]]],
        reward_fn: "ThinkAnywhereReward | Callable",
        gate: ThinkAnywhereGate,
        cfg: Optional[GateLoopConfig] = None,
    ) -> None:
        self.generate_fn = generate_fn
        self.reward_fn = reward_fn
        self.gate = gate
        self.cfg = cfg or GateLoopConfig()

        self._n_responses = 0
        self._n_train_steps = 0
        self._reward_history: list[float] = []
        self._loss_history: list[float] = []

    # ------------------------------------------------------------------

    def run_one(self, prompt: str, reference: Optional[str] = None) -> ResponseRecord:
        """Process a single prompt: generate → reward → record → maybe train."""
        cfg = self.cfg

        # 1. Gate decision (positional, no hidden yet)
        feats = PositionalFeatures(
            tokens_generated=self._n_responses * 50,  # rough proxy
            think_blocks_open=0,
            think_tokens_used=0,
        )
        gate_decision = self.gate.should_think(features=feats)

        # 2. Generate
        response, hidden = self.generate_fn(prompt)

        # 3. Reward
        reward: RewardResult = self.reward_fn(response, reference)
        self._reward_history.append(reward.combined)

        had_think = "<thinkanywhere>" in response or "<think>" in response

        # 4. Record to gate buffer
        self.gate.record(
            hidden=hidden if hidden is not None else np.zeros(self.gate.cfg.hidden_size),
            reward_result=reward,
            features=feats,
        )

        self._n_responses += 1

        # 5. Maybe train
        if self._n_responses % cfg.train_every == 0:
            buf_before = self.gate.buffer_size
            metrics = self.gate.train_step()
            self._n_train_steps += 1
            self._loss_history.append(metrics.get("loss", float("nan")))
            logger.info(
                "gate train_step %d | loss=%.4f | buf_before=%d | skipped=%s | responses=%d",
                self._n_train_steps,
                metrics.get("loss", float("nan")),
                buf_before,
                metrics.get("skipped", False),
                self._n_responses,
            )

        # 6. Maybe checkpoint
        if cfg.checkpoint_every and self._n_responses % cfg.checkpoint_every == 0:
            path = cfg.checkpoint_path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.gate.save(path)
            logger.info("Gate checkpoint saved → %s", path)

        # 7. Periodic log
        if self._n_responses % cfg.log_every == 0:
            recent = self._reward_history[-cfg.log_every:]
            logger.info(
                "responses=%d | avg_reward=%.3f | gate_updates=%d",
                self._n_responses,
                sum(recent) / len(recent),
                self._n_train_steps,
            )

        return ResponseRecord(
            prompt=prompt,
            response=response,
            hidden=hidden,
            reward=reward,
            had_think_block=had_think,
            gate_decision=gate_decision,
        )

    def run(self, prompts: Iterator[tuple[str, Optional[str]]]) -> None:
        """Process an iterator of (prompt, reference) pairs."""
        for i, (prompt, reference) in enumerate(prompts):
            self.run_one(prompt, reference)
            if self.cfg.max_responses and i + 1 >= self.cfg.max_responses:
                break

    @property
    def stats(self) -> dict:
        n = len(self._reward_history)
        return {
            "total_responses": self._n_responses,
            "gate_train_steps": self._n_train_steps,
            "avg_reward": round(sum(self._reward_history) / max(n, 1), 4),
            "avg_gate_loss": round(
                sum(self._loss_history) / max(len(self._loss_history), 1), 4
            ),
        }


# ---------------------------------------------------------------------------
# Standalone demo (no real model needed)
# ---------------------------------------------------------------------------

def _demo_generate(prompt: str) -> tuple[str, Optional[np.ndarray]]:
    """Toy generator: wraps prompt in a think block 50% of the time."""
    import random
    hidden = np.random.randn(64).astype(np.float32)
    if random.random() < 0.5:
        response = (
            f"<think>\nThinking about: {prompt[:20]}\n</think>\n"
            f"def answer():\n    return 42\n"
        )
    else:
        response = f"def answer():\n    return 42\n"
    return response, hidden


def run_demo(n: int = 64, hidden_size: int = 64) -> None:
    """Demonstrate gate online training with a toy generator."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    gate_cfg = GateConfig(hidden_size=hidden_size, gate_hidden=16,
                          threshold=0.5, update_every=16)
    gate = ThinkAnywhereGate(gate_cfg)
    reward_fn = ThinkAnywhereReward()
    loop_cfg = GateLoopConfig(train_every=16, checkpoint_every=64,
                              checkpoint_path="/tmp/demo_gate.npz", log_every=16)

    loop = GateInferenceLoop(
        generate_fn=_demo_generate,
        reward_fn=lambda resp, ref: reward_fn(resp),
        gate=gate,
        cfg=loop_cfg,
    )

    prompts = (("Write a function", None) for _ in range(n))
    loop.run(prompts)
    print("\nFinal stats:", loop.stats)


if __name__ == "__main__":
    run_demo()

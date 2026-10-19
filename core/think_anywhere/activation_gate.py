"""core/think_anywhere/activation_gate.py

Think-Anywhere Activation Gate — decides whether to allow a <thinkanywhere>
block at the current position.

Design
------
A small 2-layer MLP (H → gate_hidden → 1, sigmoid output) trained with binary
cross-entropy on labels derived from ThinkAnywhereReward:

    label = 1  if RewardResult.combined >= reward_threshold   (thinking helped)
    label = 0  otherwise                                       (thinking hurt / wasted tokens)

The gate is pure NumPy — no JAX or Flax required.  It integrates into:

  1. Full-model path  — called with the backbone hidden state at the current
     token position before emitting <thinkanywhere>.
  2. Streaming path   — called with a lightweight PositionalFeatures proxy
     when hidden states are unavailable (e.g. ThinkAnywhereStreamFilter).

Usage
-----
    from core.think_anywhere.activation_gate import ThinkAnywhereGate, GateConfig
    from core.think_anywhere.rewards import ThinkAnywhereReward

    gate = ThinkAnywhereGate(GateConfig(hidden_size=2048))

    # Inference: given backbone hidden state (numpy, shape (H,) or (B, H))
    if gate.should_think(hidden_state):
        # allow <thinkanywhere> block
        ...

    # Training: record (hidden_state, reward_result) pairs, then update
    reward_fn = ThinkAnywhereReward()
    result = reward_fn(response, test_cases=[...])
    gate.record(hidden_state, result)
    if gate.buffer_size >= gate.cfg.update_every:
        metrics = gate.train_step()

    # Persistence
    gate.save("checkpoints/think_gate.npz")
    gate = ThinkAnywhereGate.load("checkpoints/think_gate.npz")
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GateConfig:
    """Hyper-parameters for ThinkAnywhereGate.

    Attributes:
        hidden_size:       Backbone hidden dimension (input to the gate).
        gate_hidden:       Hidden units in the gate MLP.
        threshold:         Score ≥ threshold → allow Think-Anywhere.
        reward_threshold:  Minimum RewardResult.combined to label as positive.
        lr:                SGD learning rate for gate training.
        momentum:          SGD momentum coefficient.
        update_every:      Accumulate this many samples before a gradient step.
        max_buffer:        Maximum replay buffer size (oldest entries dropped).
        max_think_ratio:   Hard cap: block Think-Anywhere if more than this
                           fraction of tokens so far are already thinking tokens.
        l2:                L2 weight regularisation.
        min_positive_frac: Minimum positive fraction required to perform a
                           train step (guards against all-negative mini-batches).
    """
    hidden_size: int = 2048
    gate_hidden: int = 64
    threshold: float = 0.5
    reward_threshold: float = 0.5
    lr: float = 1e-3
    momentum: float = 0.9
    update_every: int = 32
    max_buffer: int = 2048
    max_think_ratio: float = 0.30
    l2: float = 1e-4
    min_positive_frac: float = 0.1


# ---------------------------------------------------------------------------
# Positional features (streaming path — no hidden states available)
# ---------------------------------------------------------------------------

@dataclass
class PositionalFeatures:
    """Lightweight proxy used when backbone hidden states are unavailable.

    Attributes:
        tokens_generated:   Total tokens produced so far in this response.
        think_blocks_open:  Number of <thinkanywhere> blocks already opened.
        think_tokens_used:  Tokens consumed inside thinking blocks so far.
        response_entropy:   Optional proxy for response uncertainty (e.g.
                            mean negative log-prob of recent tokens, 0 if unknown).
    """
    tokens_generated: int = 0
    think_blocks_open: int = 0
    think_tokens_used: int = 0
    response_entropy: float = 0.0

    def to_vector(self, max_tokens: int = 4096) -> np.ndarray:
        """Convert to a fixed-length feature vector for the fallback gate."""
        think_ratio = self.think_tokens_used / max(self.tokens_generated, 1)
        blocks_norm = self.think_blocks_open / 10.0          # soft-cap at 10
        pos_norm    = self.tokens_generated / max_tokens
        return np.array(
            [pos_norm, blocks_norm, think_ratio, self.response_entropy],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Gate MLP (pure NumPy)
# ---------------------------------------------------------------------------

class ThinkAnywhereGate:
    """Binary gate that controls Think-Anywhere activation.

    Architecture:
        hidden_state (H,) → Linear(H, gate_hidden) → ReLU
                          → Linear(gate_hidden, 1)  → Sigmoid → score

    A PositionalFallbackGate (4 features → 1) is also maintained for the
    streaming path where hidden states are unavailable.
    """

    def __init__(self, cfg: Optional[GateConfig] = None) -> None:
        self.cfg = cfg or GateConfig()
        rng = np.random.default_rng(0)
        H, G = self.cfg.hidden_size, self.cfg.gate_hidden

        # Main gate (hidden-state path)
        s1 = math.sqrt(2.0 / H)
        self.W1 = (rng.standard_normal((H, G)) * s1).astype(np.float32)
        self.b1 = np.zeros(G, dtype=np.float32)
        s2 = math.sqrt(2.0 / G)
        self.W2 = (rng.standard_normal((G, 1)) * s2).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

        # Momentum buffers
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

        # Positional fallback gate (4 features)
        self.Wp = (rng.standard_normal((4, 1)) * 0.01).astype(np.float32)
        self.bp = np.zeros(1, dtype=np.float32)
        self.vWp = np.zeros_like(self.Wp)
        self.vbp = np.zeros_like(self.bp)

        # Replay buffers
        self._hidden_buf: List[np.ndarray] = []   # (H,) float32
        self._label_buf:  List[float]       = []   # 0.0 or 1.0
        self._pos_buf:    List[np.ndarray] = []    # (4,) float32
        self._pos_label:  List[float]       = []

        # Running metrics
        self.metrics: Dict[str, float] = {
            "activation_rate": 0.0,
            "positive_rate": 0.0,
            "last_loss": 0.0,
            "train_steps": 0.0,
            "total_decisions": 0.0,
            "total_activations": 0.0,
        }

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def score(self, hidden: np.ndarray) -> float:
        """Score ∈ [0,1]: probability that Think-Anywhere is useful here.

        Args:
            hidden: backbone hidden state, shape (H,) or (B, H).
                    If batched, returns the mean score.
        """
        h = np.asarray(hidden, dtype=np.float32)
        if h.ndim > 1:
            return float(np.mean([self._forward(row) for row in h]))
        return float(self._forward(h))

    def score_positional(self, features: PositionalFeatures) -> float:
        """Score using positional features only (streaming path)."""
        x = features.to_vector()
        return float(self._sigmoid(x @ self.Wp + self.bp)[0])

    def should_think(
        self,
        hidden: Optional[np.ndarray] = None,
        features: Optional[PositionalFeatures] = None,
    ) -> bool:
        """Return True if Think-Anywhere should be allowed.

        At least one of `hidden` or `features` must be provided.
        If both are provided, the hidden-state score takes priority.

        Hard caps applied regardless of score:
        - if features.think_ratio > max_think_ratio → always False
        """
        if features is not None:
            think_ratio = features.think_tokens_used / max(features.tokens_generated, 1)
            if think_ratio >= self.cfg.max_think_ratio:
                self._record_decision(activated=False)
                return False

        if hidden is not None:
            s = self.score(hidden)
        elif features is not None:
            s = self.score_positional(features)
        else:
            raise ValueError("Provide hidden state or PositionalFeatures")

        activated = s >= self.cfg.threshold
        self._record_decision(activated)
        return activated

    # ------------------------------------------------------------------
    # Training API
    # ------------------------------------------------------------------

    def record(
        self,
        hidden: np.ndarray,
        reward_result,                      # ThinkAnywhereReward.RewardResult
        features: Optional[PositionalFeatures] = None,
    ) -> None:
        """Store a (hidden_state, label) pair in the replay buffer.

        The label is 1 if reward_result.combined >= reward_threshold, else 0.
        """
        label = float(reward_result.combined >= self.cfg.reward_threshold)
        h = np.asarray(hidden, dtype=np.float32).flatten()[:self.cfg.hidden_size]

        self._hidden_buf.append(h)
        self._label_buf.append(label)

        if len(self._hidden_buf) > self.cfg.max_buffer:
            self._hidden_buf.pop(0)
            self._label_buf.pop(0)

        if features is not None:
            self._pos_buf.append(features.to_vector())
            self._pos_label.append(label)
            if len(self._pos_buf) > self.cfg.max_buffer:
                self._pos_buf.pop(0)
                self._pos_label.pop(0)

    @property
    def buffer_size(self) -> int:
        return len(self._hidden_buf)

    def train_step(self) -> Dict[str, float]:
        """One mini-batch gradient step on the accumulated buffer.

        Returns a dict of training metrics.
        Clears the buffer after the step.
        """
        if not self._hidden_buf:
            return {"loss": 0.0, "n": 0, "buffer_size": 0}

        X = np.stack(self._hidden_buf).astype(np.float32)  # (N, H)
        y = np.array(self._label_buf, dtype=np.float32)    # (N,)

        pos_frac = y.mean()
        if pos_frac < self.cfg.min_positive_frac:
            logger.debug("Gate train_step skipped: pos_frac=%.3f < %.3f",
                         pos_frac, self.cfg.min_positive_frac)
            self._hidden_buf.clear(); self._label_buf.clear()
            return {"loss": 0.0, "n": len(y), "buffer_size": len(y), "skipped": True}

        loss, dW1, db1, dW2, db2 = self._backward(X, y)

        # SGD + momentum + L2
        self.vW1 = self.cfg.momentum * self.vW1 + dW1 + self.cfg.l2 * self.W1
        self.W1 -= self.cfg.lr * self.vW1
        self.vb1 = self.cfg.momentum * self.vb1 + db1
        self.b1 -= self.cfg.lr * self.vb1
        self.vW2 = self.cfg.momentum * self.vW2 + dW2 + self.cfg.l2 * self.W2
        self.W2 -= self.cfg.lr * self.vW2
        self.vb2 = self.cfg.momentum * self.vb2 + db2
        self.b2 -= self.cfg.lr * self.vb2

        # Train positional fallback if buffer has data
        if self._pos_buf:
            Xp = np.stack(self._pos_buf).astype(np.float32)
            yp = np.array(self._pos_label, dtype=np.float32)
            _, dWp, dbp = self._backward_positional(Xp, yp)
            self.vWp = self.cfg.momentum * self.vWp + dWp
            self.Wp -= self.cfg.lr * self.vWp
            self.vbp = self.cfg.momentum * self.vbp + dbp
            self.bp -= self.cfg.lr * self.vbp
            self._pos_buf.clear(); self._pos_label.clear()

        self._hidden_buf.clear(); self._label_buf.clear()

        self.metrics["last_loss"] = float(loss)
        self.metrics["train_steps"] += 1
        self.metrics["positive_rate"] = float(pos_frac)

        return {"loss": float(loss), "n": len(y), "pos_frac": float(pos_frac),
                "buffer_size": len(y)}

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------

    def evaluate(
        self,
        hidden_states: np.ndarray,   # (N, H)
        labels: np.ndarray,          # (N,)  binary
    ) -> Dict[str, float]:
        """Compute accuracy, precision, recall, F1 on a labelled dataset."""
        scores = np.array([self._forward(h) for h in hidden_states])
        preds  = (scores >= self.cfg.threshold).astype(float)
        acc    = float((preds == labels).mean())
        tp = float(((preds == 1) & (labels == 1)).sum())
        fp = float(((preds == 1) & (labels == 0)).sum())
        fn = float(((preds == 0) & (labels == 1)).sum())
        prec = tp / max(tp + fp, 1e-8)
        rec  = tp / max(tp + fn, 1e-8)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "activation_rate": float(preds.mean())}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save gate weights to a .npz file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p,
            W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
            Wp=self.Wp, bp=self.bp,
            vW1=self.vW1, vb1=self.vb1, vW2=self.vW2, vb2=self.vb2,
            vWp=self.vWp, vbp=self.vbp,
            threshold=np.array([self.cfg.threshold]),
            hidden_size=np.array([self.cfg.hidden_size]),
            gate_hidden=np.array([self.cfg.gate_hidden]),
        )
        logger.info("Gate saved → %s", p)

    @classmethod
    def load(cls, path: str) -> "ThinkAnywhereGate":
        """Load gate weights from a .npz file."""
        d = np.load(path)
        cfg = GateConfig(
            hidden_size=int(d["hidden_size"][0]),
            gate_hidden=int(d["gate_hidden"][0]),
            threshold=float(d["threshold"][0]),
        )
        gate = cls(cfg)
        for name in ("W1", "b1", "W2", "b2", "Wp", "bp",
                     "vW1", "vb1", "vW2", "vb2", "vWp", "vbp"):
            setattr(gate, name, d[name])
        logger.info("Gate loaded ← %s", path)
        return gate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0,
                        1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))

    def _forward(self, h: np.ndarray) -> float:
        """Forward pass: (H,) → scalar score."""
        z1 = h @ self.W1 + self.b1           # (G,)
        a1 = np.maximum(z1, 0)               # ReLU
        z2 = a1 @ self.W2 + self.b2          # (1,)
        return float(self._sigmoid(z2)[0])

    def _backward(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Batched forward + backward; returns (loss, dW1, db1, dW2, db2)."""
        N = X.shape[0]
        Z1 = X @ self.W1 + self.b1           # (N, G)
        A1 = np.maximum(Z1, 0)               # (N, G)
        Z2 = A1 @ self.W2 + self.b2          # (N, 1)
        P  = self._sigmoid(Z2).squeeze(-1)   # (N,)

        # Binary cross-entropy loss
        eps = 1e-7
        loss = -float(np.mean(
            y * np.log(P + eps) + (1 - y) * np.log(1 - P + eps)
        ))

        # Backprop
        dZ2 = (P - y)[:, None] / N            # (N, 1)
        dW2 = A1.T @ dZ2                       # (G, 1)
        db2 = dZ2.sum(axis=0)                  # (1,)

        dA1 = dZ2 @ self.W2.T                  # (N, G)
        dZ1 = dA1 * (Z1 > 0)                  # ReLU gate
        dW1 = X.T @ dZ1                        # (H, G)
        db1 = dZ1.sum(axis=0)                  # (G,)

        return loss, dW1, db1, dW2, db2

    def _backward_positional(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Positional fallback gate backward pass."""
        N = X.shape[0]
        P = self._sigmoid(X @ self.Wp + self.bp).squeeze(-1)
        eps = 1e-7
        loss = -float(np.mean(
            y * np.log(P + eps) + (1 - y) * np.log(1 - P + eps)
        ))
        dZ = (P - y)[:, None] / N
        dWp = X.T @ dZ
        dbp = dZ.sum(axis=0)
        return loss, dWp, dbp

    def _record_decision(self, activated: bool) -> None:
        self.metrics["total_decisions"] += 1
        if activated:
            self.metrics["total_activations"] += 1
        total = self.metrics["total_decisions"]
        self.metrics["activation_rate"] = self.metrics["total_activations"] / max(total, 1)

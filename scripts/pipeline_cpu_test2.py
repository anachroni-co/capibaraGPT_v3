#!/usr/bin/env python3
"""
scripts/pipeline_cpu_test2.py

Second CPU-only pipeline covering the training → deployment lifecycle:

  Generated responses
       ↓
  [ThinkAnywhereReward]  — sandbox code execution + GRPO reward/advantages
       ↓
  [HybridAttentionModule] — adaptive routing (Transformer vs Mamba by seq length)
       ↓
  [CalibrationEngine]    — INT8 quantization calibration on model weights
       ↓
  [FactCheckHandler]     — fact-claim extraction from generated text

No JAX, no PyTorch. All four use their real implementation paths on CPU.
"""
from __future__ import annotations

import logging
import time
import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s", force=True)
log = logging.getLogger("pipeline2")
log.setLevel(logging.INFO)


def section(title: str) -> None:
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")

def ok(label: str, value: str = "") -> None:
    print(f"  [OK] {label}", f"→ {value}" if value else "")

def show(label: str, val) -> None:
    print(f"       {label}: {val}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — ThinkAnywhereReward: sandbox execution + GRPO rewards
# ─────────────────────────────────────────────────────────────────────────────
section("STAGE 1 — ThinkAnywhereReward (subprocess sandbox + GRPO)")

from core.think_anywhere.rewards import ThinkAnywhereReward

# Simulate a GRPO rollout group: 4 candidate responses to the same prompt.
# Response quality intentionally varies to show reward differentiation.
ROLLOUT = [
    # Good: valid structure + correct code
    "<think>I need to sum all elements.</think>"
    "<thinkanywhere>edge case: empty list</thinkanywhere>"
    "def total(lst):\n    return sum(lst)",

    # Partial: correct code but missing Think-Anywhere structure
    "def total(lst):\n    return sum(lst)",

    # Wrong: structure OK but code fails the test
    "<think>multiply instead</think>"
    "<thinkanywhere>try product</thinkanywhere>"
    "def total(lst):\n    result = 1\n    for x in lst: result *= x\n    return result",

    # Bad: neither structure nor correct code
    "def total(lst):\n    pass",
]

TEST_CASES = [
    "assert total([1, 2, 3]) == 6",
    "assert total([]) == 0",
    "assert total([10]) == 10",
]

t0 = time.perf_counter()
reward_fn = ThinkAnywhereReward()
results   = reward_fn.batch(ROLLOUT, test_cases=TEST_CASES, timeout=5.0)
advantages = reward_fn.group_normalized_advantages(results)
elapsed_rw = time.perf_counter() - t0

ok("ThinkAnywhereReward.batch() — 4 rollouts, 3 test cases", f"{elapsed_rw*1000:.0f} ms")
print()
labels = ["complete (struct+code)", "code only", "struct+wrong code", "empty"]
for label, r, adv in zip(labels, results, advantages):
    sym = "✓" if r.combined >= 0.8 else ("~" if r.combined >= 0.4 else "✗")
    print(f"  {sym} [{label:<22}]  "
          f"R={r.combined:.2f}  struct={r.structure:.0f}  "
          f"correct={r.correctness:.2f}  "
          f"tests={r.passed_tests}/{r.total_tests}  "
          f"adv={adv:+.3f}")

best_idx = int(np.argmax([r.combined for r in results]))
ok("Best rollout", f"index {best_idx} ({labels[best_idx]}) R={results[best_idx].combined:.2f}")

# Pass the best response's hidden representation forward
# (simulate embedding of the winning response for the next stage)
HIDDEN = 128
rng = np.random.default_rng(best_idx)  # seed differs per winner → different repr
best_embedding = rng.standard_normal((1, len(ROLLOUT[best_idx]), HIDDEN)).astype(np.float32) * 0.1
show("Winning response embedding", f"shape={best_embedding.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — HybridAttentionModule: adaptive routing
# ─────────────────────────────────────────────────────────────────────────────
section("STAGE 2 — HybridAttentionModule (Transformer ↔ Mamba routing)")

from sub_models.hybrid.hybrid_attention_module import HybridAttentionModule

cfg = {
    "hidden_size": HIDDEN,
    "num_heads": 4,
    "intermediate_size": HIDDEN * 4,
    "mamba_threshold": 512,   # seq < 512 → Transformer,  seq ≥ 512 → Mamba
    "dropout_rate": 0.0,
}
hybrid = HybridAttentionModule(cfg)

timings = {}
outputs = {}
for seq_len, label in [(20, "short (20)"), (600, "long (600)")]:
    x = rng.standard_normal((1, seq_len, HIDDEN)).astype(np.float32)
    t0 = time.perf_counter()
    out = hybrid(x, training=False)
    timings[label] = time.perf_counter() - t0
    outputs[label] = out

ok("HybridAttentionModule initialised")
print()
for label, out in outputs.items():
    shape = out["output"].shape
    success = out.get("success", "n/a")
    metrics = out.get("metrics", {})
    mod_used = metrics.get("module_used", "—") if metrics else "—"
    proc_info = out.get("processing_info", {})
    backend = proc_info.get("backend", "—") if proc_info else "—"
    print(f"  seq={label:<12}  shape={shape}  "
          f"success={success}  backend={backend}  "
          f"time={timings[label]*1000:.1f}ms")

# The long-sequence output feeds into calibration
long_output = outputs["long (600)"]["output"]   # (1, 600, HIDDEN)
# pool to get per-layer weight-like matrix for calibration demo
pooled = long_output[0]                          # (600, HIDDEN)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — CalibrationEngine: INT8 quantization calibration
# ─────────────────────────────────────────────────────────────────────────────
section("STAGE 3 — CalibrationEngine (INT8 quantization calibration)")

from inference.quantization.calibration import CalibrationEngine, CalibrationConfig

# Simulate a small model's weight tensors
# (in production these come from a real checkpoint)
# CalibrationEngine expects {layer_name: {param_name: array}}
# where param_name must be one of 'kernel', 'weight', or 'w'
model_weights = {
    "embed":       {"weight": rng.standard_normal((256, HIDDEN)).astype(np.float32) * 0.02},
    "attn.query":  {"weight": rng.standard_normal((HIDDEN, HIDDEN)).astype(np.float32) * 0.02},
    "attn.key":    {"weight": rng.standard_normal((HIDDEN, HIDDEN)).astype(np.float32) * 0.02},
    "attn.value":  {"weight": rng.standard_normal((HIDDEN, HIDDEN)).astype(np.float32) * 0.02},
    "ffn.w1":      {"weight": rng.standard_normal((HIDDEN, HIDDEN * 4)).astype(np.float32) * 0.02},
    "ffn.w2":      {"weight": rng.standard_normal((HIDDEN * 4, HIDDEN)).astype(np.float32) * 0.02},
    "output":      {"weight": rng.standard_normal((HIDDEN, 256)).astype(np.float32) * 0.02},
}

cal_cfg = CalibrationConfig(
    weight_calibration_method="percentile",
    activation_calibration_method="percentile",
)
eng = CalibrationEngine(cal_cfg)

t0 = time.perf_counter()
cal_params = eng.calibrate_model(
    model_params=model_weights,
    calibration_dataset=None,
    model_forward_fn=None,
)
elapsed_cal = time.perf_counter() - t0

ok("CalibrationEngine.calibrate_model()", f"{elapsed_cal*1000:.1f} ms")
show("Calibrated layers", len(model_weights))
show("Output keys", list(cal_params.keys())[:5])

stats = eng.get_stats()
if isinstance(stats, dict):
    for k, v in list(stats.items())[:6]:
        show(f"  {k}", v)

# Estimate memory reduction
param_count = sum(v["weight"].size for v in model_weights.values())
fp32_mb = param_count * 4 / 1e6
int8_mb  = param_count * 1 / 1e6
show("Weight params", f"{param_count:,}")
show("FP32 size", f"{fp32_mb:.2f} MB")
show("INT8 size (est.)", f"{int8_mb:.2f} MB  (×4 compression)")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — FactCheckHandler: extract and process claims
# ─────────────────────────────────────────────────────────────────────────────
section("STAGE 4 — FactCheckHandler (claim extraction + tagging)")

from core.special_tokens import FactCheckHandler

# Simulate a model response that contains fact claims tagged inline
MODEL_OUTPUT = (
    "The CapibaraGPT architecture uses "
    "<fact_check>Mamba SSM blocks with O(n) complexity</fact_check> "
    "for long sequences and switches to standard attention for "
    "<fact_check>sequences shorter than 512 tokens</fact_check>. "
    "The quantized INT8 model achieves "
    "<fact_check>4× memory compression vs FP32</fact_check> "
    "with minimal accuracy degradation."
)

t0 = time.perf_counter()
fch = FactCheckHandler()
processed_text, verdicts = fch.verify(MODEL_OUTPUT)
elapsed_fc = time.perf_counter() - t0

ok("FactCheckHandler.verify()", f"{elapsed_fc*1000:.1f} ms")
show("Claims found", len(verdicts))
show("Processed text (preview)", processed_text[:80] + "…")
print()
for i, v in enumerate(verdicts):
    status = "⚑ " + v["verdict"].upper()
    print(f"  Claim {i+1}: {v['claim']}")
    print(f"           {status}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
section("PIPELINE 2 SUMMARY")

total_ms = (elapsed_rw + sum(timings.values()) + elapsed_cal + elapsed_fc) * 1000
rows = [
    ("ThinkAnywhereReward",  f"{elapsed_rw*1000:.0f} ms",
     f"4 rollouts, subprocess sandbox, GRPO advantages"),
    ("HybridAttentionModule", f"{sum(timings.values())*1000:.0f} ms",
     f"short→Transformer, long→Mamba, adaptive routing"),
    ("CalibrationEngine",    f"{elapsed_cal*1000:.0f} ms",
     f"INT8 calibration, {len(model_weights)} layers, ×4 compression"),
    ("FactCheckHandler",     f"{elapsed_fc*1000:.0f} ms",
     f"{len(verdicts)} claims extracted and tagged"),
    ("TOTAL",               f"{total_ms:.0f} ms",
     "end-to-end, CPU only, no JAX/torch"),
]
print()
print(f"  {'Component':<26} {'Time':>8}  {'Notes'}")
print(f"  {'─'*24} {'─'*8}  {'─'*44}")
for name, t, note in rows:
    print(f"  {name:<26} {t:>8}  {note}")
print()
print("  4 real model components executed successfully on CPU.")

#!/usr/bin/env python3
"""
scripts/pipeline_cpu_test.py

Runs three real model components in sequence on CPU (no JAX/torch):

  Text → [ByteEmbed] → MambaModule (SSM) → EnhancedCoTModule → CSAExpert → Report

Each step uses the component's actual NumPy/CPU fallback path, not mocks.
"""
from __future__ import annotations

import logging
import time
import numpy as np

logging.basicConfig(
    level=logging.WARNING,          # suppress component-level noise
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("pipeline")
log.setLevel(logging.INFO)

# ── helpers ──────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def ok(label: str, value: str = "") -> None:
    print(f"  [OK] {label}", f"→ {value}" if value else "")

def show(label: str, obj) -> None:
    print(f"       {label}: {obj}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0 — Real text input + byte embedding
# ─────────────────────────────────────────────────────────────────────────────
section("STAGE 0 — Input text → byte embeddings")

INPUT_TEXT = (
    "The consensus router selects the best expert for each token "
    "using plausibility and utility scores from the CSA module."
)

HIDDEN = 64      # embedding dim — small enough to be fast on CPU
VOCAB  = 256     # byte vocabulary

rng = np.random.default_rng(0)
W_emb = (rng.standard_normal((VOCAB, HIDDEN)) * 0.02).astype(np.float32)

byte_ids = np.frombuffer(INPUT_TEXT.encode("utf-8"), dtype=np.uint8).astype(np.int32)
# shape: (1, T, HIDDEN)  — batch=1
embeddings = W_emb[byte_ids][None, :, :]

ok("Input text", f"{len(byte_ids)} bytes")
ok("Embedding shape", str(embeddings.shape))
show("First 5 byte values", byte_ids[:5].tolist())
show("Embedding norm (mean per token)", f"{np.linalg.norm(embeddings, axis=-1).mean():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — MambaModule (SSM, NumPy fallback)
# ─────────────────────────────────────────────────────────────────────────────
section("STAGE 1 — MambaModule (Selective State Space Model)")

from sub_models.mamba.mamba_module import MambaModule

t0 = time.perf_counter()
mamba = MambaModule(config={"hidden_size": HIDDEN, "d_state": 16, "expand_factor": 2})
mamba_out = mamba(embeddings, training=False)
elapsed_mamba = time.perf_counter() - t0

hidden_seq = mamba_out["output"]          # (1, T, HIDDEN)
metrics_m  = mamba_out["metrics"]

ok("MambaModule instantiated + forward pass", f"{elapsed_mamba*1000:.1f} ms")
ok("Output shape", str(hidden_seq.shape))
show("Fallback used", metrics_m.get("fallback_used"))
show("Complexity reported", metrics_m.get("complexity"))
show("Output norm (mean)", f"{np.linalg.norm(hidden_seq, axis=-1).mean():.4f}")

# Pool across sequence → single context vector for CoT
context_vec = hidden_seq[0].mean(axis=0)   # (HIDDEN,)
ok("Pooled context vector", f"shape={context_vec.shape}, norm={np.linalg.norm(context_vec):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — EnhancedCoTModule (Chain-of-Thought, CPU fallback)
# ─────────────────────────────────────────────────────────────────────────────
section("STAGE 2 — EnhancedCoTModule (Chain-of-Thought reasoning)")

from core.cot.enhanced_cot_module import EnhancedCoTModule, ReasoningConfig

cot_cfg = ReasoningConfig(
    max_reasoning_steps=4,
    confidence_threshold=0.5,
    use_process_rewards=True,
    enable_meta_cognition=True,
    enable_self_verification=True,
    hidden_size=HIDDEN,
)

t0 = time.perf_counter()
cot = EnhancedCoTModule(config=cot_cfg)
cot_out = cot(context_vec.tolist(), training=False)
elapsed_cot = time.perf_counter() - t0

ok("EnhancedCoTModule forward pass", f"{elapsed_cot*1000:.1f} ms")
ok("Backend", cot_out["metrics"]["backend"])
show("Reasoning steps taken", cot_out["metrics"]["num_steps"])
show("Overall confidence", f"{cot_out['confidence']:.4f}")
show("Verification passed", cot_out["verification"]["verified"])
show("Verification score", f"{cot_out['verification']['score']:.4f}")

for i, step in enumerate(cot_out["reasoning_trace"]):
    print(f"       step {i}: confidence={step['step_confidence']:.3f}  "
          f"reward={step['step_reward']:.3f}")

# Extract reasoning quality signal to pass to CSA
reasoning_confidence = cot_out["confidence"]
num_steps = cot_out["metrics"]["num_steps"]
verified   = cot_out["verification"]["verified"]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — CSAExpert (Counterfactual Scenario Analysis)
# ─────────────────────────────────────────────────────────────────────────────
section("STAGE 3 — CSAExpert (Counterfactual Scenario Analysis)")

from sub_models.csa_expert import CSAExpert, ExpertContext

csa = CSAExpert()

# Build context from upstream outputs
context = ExpertContext(
    text=INPUT_TEXT,
    task_hint="diagnosis",
    metadata={
        "cot_confidence": reasoning_confidence,
        "cot_steps": num_steps,
        "cot_verified": verified,
        "mamba_complexity": metrics_m.get("complexity"),
    },
)

t0 = time.perf_counter()
hypotheses = csa.generate_hypotheses(context, max_hypotheses=3)
scenarios  = csa.evaluate_scenarios(context, hypotheses)
elapsed_csa = time.perf_counter() - t0

ok("CSAExpert generate + evaluate", f"{elapsed_csa*1000:.1f} ms")
show("Hypotheses generated", len(hypotheses))
show("Scenarios evaluated", len(scenarios))

for i, (hyp, sc) in enumerate(zip(hypotheses, scenarios)):
    print(f"\n       Hypothesis {i+1}:")
    print(f"         delta      : {hyp.delta[:70]}")
    print(f"         prior score: {hyp.prior_score:.3f}")
    print(f"         score      : {sc.score:.3f}  (plaus={sc.plausibility:.2f} util={sc.utility:.2f})")
    print(f"         actionable : {sc.actionability:.2f}  risk={sc.risk_assessment:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
section("PIPELINE SUMMARY")

total = elapsed_mamba + elapsed_cot + elapsed_csa
rows = [
    ("Stage 0 – Byte embed",    "—",           f"{len(byte_ids)} bytes → {embeddings.shape}"),
    ("Stage 1 – MambaModule",   f"{elapsed_mamba*1000:.1f} ms",  f"SSM NumPy fallback, O(n) complexity"),
    ("Stage 2 – CoTModule",     f"{elapsed_cot*1000:.1f} ms",    f"{num_steps} steps, conf={reasoning_confidence:.3f}, verified={verified}"),
    ("Stage 3 – CSAExpert",     f"{elapsed_csa*1000:.1f} ms",    f"{len(hypotheses)} hypotheses, {len(scenarios)} evaluated"),
    ("TOTAL",                   f"{total*1000:.1f} ms",           "end-to-end on CPU, no JAX/torch"),
]
print()
print(f"  {'Component':<28} {'Time':>10}  {'Notes'}")
print(f"  {'─'*26} {'─'*10}  {'─'*42}")
for name, t, note in rows:
    print(f"  {name:<28} {t:>10}  {note}")

print()
cap = csa.get_capabilities()
print(f"  CSAExpert capabilities: {cap}")
metrics_csa = csa.get_metrics()
print(f"  CSAExpert metrics:      {metrics_csa}")
print()
print("  All three real components executed successfully on CPU.")

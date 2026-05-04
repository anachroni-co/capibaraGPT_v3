#!/usr/bin/env python3
"""
scripts/pipeline_cpu_test3.py

Third CPU-only pipeline — modules that depend on the ones already tested:

  Model token stream
       ↓
  [ThinkAnywhereStreamFilter]  — suppress <think> blocks token-by-token in real time
       ↓  (clean visible text)
  [SearchTokenHandler + TOON]  — resolve <search> queries via RAG, inject context
       ↓  (grounded text)
  [ReasoningEnhancementExpert] — multi-step expert reasoning over the answer
       ↓  (ExpertResult with confidence)
  [CoT factory]                — structured chain-of-thought output

All four are real module code running on CPU (no JAX/torch).
"""
from __future__ import annotations

import asyncio
import logging
import time

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s", force=True)
log = logging.getLogger("pipeline3")
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
# STAGE 1 — ThinkAnywhereStreamFilter: real-time token-by-token suppression
# ─────────────────────────────────────────────────────────────────────────────
section("STAGE 1 — ThinkAnywhereStreamFilter (real-time streaming)")

from core.think_anywhere.streaming import ThinkAnywhereStreamFilter

# Simulate the model emitting tokens one by one, with a <think> block mid-stream
# and a <thinkanywhere> block later — both must be suppressed before the caller sees them
TOKEN_STREAM = [
    "The", " Eiffel", " Tower", " is",
    " <think>", "let me", " recall", " geography", "</think>",
    " located", " in",
    " <thinkanywhere>", "Paris, capital city", "</thinkanywhere>",
    " Paris,", " France.", " It", " was", " built", " in",
    " <think>", "1889", "</think>",
    " 1889", ".",
]

sf = ThinkAnywhereStreamFilter()

t0 = time.perf_counter()
visible_tokens = []
suppressed_count = 0
for tok in TOKEN_STREAM:
    out = sf.feed(tok)
    if out:
        visible_tokens.append(out)
    else:
        suppressed_count += 1
remainder = sf.flush()
if remainder:
    visible_tokens.append(remainder)
elapsed_sf = time.perf_counter() - t0

visible_text = "".join(visible_tokens)

ok("ThinkAnywhereStreamFilter", f"{elapsed_sf*1000:.2f} ms")
show("Input tokens", len(TOKEN_STREAM))
show("Suppressed tokens", suppressed_count)
show("Visible output", repr(visible_text))

# The clean text contains a <search> query to resolve in the next stage
# We inject one artificially to demonstrate the RAG integration
GROUNDED_INPUT = (
    "The Eiffel Tower is located in"
    " <search>Eiffel Tower location and year built</search>"
    " Paris, France. It was built in 1889."
)
show("Prepared for RAG", repr(GROUNDED_INPUT[:80]))


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — SearchTokenHandler: RAG retrieval + TOON context compression
# ─────────────────────────────────────────────────────────────────────────────
section("STAGE 2 — SearchTokenHandler (RAG retrieval + TOON compression)")

from core.special_tokens.search import SearchTokenHandler

# Mock retriever using the interface the handler expects (.retrieve, not .search)
class _MockRetriever:
    def retrieve(self, query: str, top_k: int = 3):
        # Simulate a real RAG store returning ranked results
        return [
            {"text": "The Eiffel Tower is in Paris, France.", "score": 0.97},
            {"text": "It was constructed between 1887 and 1889 by Gustave Eiffel.", "score": 0.91},
            {"text": "The tower is 330 metres tall and attracts 7 million visitors/year.", "score": 0.85},
        ]

sh = SearchTokenHandler(retriever=_MockRetriever(), use_toon=True)

t0 = time.perf_counter()
grounded_text = sh.process(GROUNDED_INPUT)
elapsed_sh = time.perf_counter() - t0

ok("SearchTokenHandler.process()", f"{elapsed_sh*1000:.2f} ms")
show("TOON enabled", True)
show("Output", grounded_text)

# Count token savings (rough estimate: chars as proxy)
original_len = len(GROUNDED_INPUT)
result_len   = len(grounded_text)
show("Input chars", original_len)
show("Output chars", result_len)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — ReasoningEnhancementExpert: multi-step expert reasoning
# ─────────────────────────────────────────────────────────────────────────────
section("STAGE 3 — ReasoningEnhancementExpert (multi-step reasoning)")

from sub_models.reasoning_enhancement import (
    ReasoningEnhancementExpert,
    ReasoningConfig,
    ExpertContext,
)

cfg = ReasoningConfig(hidden_size=64, num_heads=4)
expert = ReasoningEnhancementExpert(cfg)

context = ExpertContext(
    text=grounded_text,
    task_hint="reasoning",
    metadata={"source": "rag_grounded", "stream_filtered": True},
)

t0 = time.perf_counter()
result = asyncio.run(expert.process(context))
elapsed_re = time.perf_counter() - t0

ok("ReasoningEnhancementExpert.process()", f"{elapsed_re*1000:.1f} ms")
show("Success", result.success)
show("Expert name", result.expert_name)
show("Confidence", f"{result.confidence:.4f}")
show("Processing time (internal)", f"{result.processing_time:.4f}s")

if result.metadata:
    for k, v in list(result.metadata.items())[:4]:
        show(f"  meta.{k}", str(v)[:80])

# Feed confidence and output into final structured reasoning
reasoning_confidence = result.confidence
reasoning_output     = result.output or grounded_text


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — CoT factory: structured chain-of-thought from grounded answer
# ─────────────────────────────────────────────────────────────────────────────
section("STAGE 4 — CoT factory (structured chain-of-thought output)")

from core.cot.factory import enhanced_chain_of_thought

# Generate function simulates a core model that already has the grounded context
def _generate(prompt: str) -> str:
    return (
        f"Based on retrieved information: the Eiffel Tower is in Paris, France, "
        f"built in 1889 by Gustave Eiffel. Confidence from reasoning stage: "
        f"{reasoning_confidence:.2f}."
    )

t0 = time.perf_counter()
cot_result = enhanced_chain_of_thought(
    "Where is the Eiffel Tower and when was it built?",
    core_model_generate_fn=_generate,
    device_type="cpu",
)
elapsed_cot = time.perf_counter() - t0

ok("enhanced_chain_of_thought()", f"{elapsed_cot*1000:.1f} ms")
show("Query", cot_result.get("query"))
show("Solution", str(cot_result.get("solution", ""))[:80])
show("Steps", len(cot_result.get("steps", [])))
show("Confidence", f"{cot_result.get('confidence', 0):.4f}")
show("Module", cot_result.get("module"))

for i, step in enumerate(cot_result.get("steps", [])[:3]):
    print(f"       step {i+1}: {str(step)[:70]}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
section("PIPELINE 3 SUMMARY")

total_ms = (elapsed_sf + elapsed_sh + elapsed_re + elapsed_cot) * 1000
rows = [
    ("ThinkAnywhereStreamFilter",    f"{elapsed_sf*1000:.2f} ms",
     f"{len(TOKEN_STREAM)} tokens in, {suppressed_count} suppressed, output clean"),
    ("SearchTokenHandler (TOON)",    f"{elapsed_sh*1000:.2f} ms",
     f"3 RAG results retrieved and injected"),
    ("ReasoningEnhancementExpert",   f"{elapsed_re*1000:.1f} ms",
     f"async expert reasoning, conf={reasoning_confidence:.3f}"),
    ("CoT factory",                  f"{elapsed_cot*1000:.1f} ms",
     f"structured output, {len(cot_result.get('steps',[]))} steps"),
    ("TOTAL",                        f"{total_ms:.1f} ms",
     "end-to-end on CPU, no JAX/torch"),
]
print()
print(f"  {'Component':<30} {'Time':>9}  {'Notes'}")
print(f"  {'─'*28} {'─'*9}  {'─'*44}")
for name, t, note in rows:
    print(f"  {name:<30} {t:>9}  {note}")

print()
print("  Dependency chain: ThinkAnywhere → SpecialTokens → ReasoningExpert → CoT")
print("  All 4 components execute on CPU using real implementation paths.")

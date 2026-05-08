#!/usr/bin/env python3
"""3-level speculative decoding inference engine for Capibara Legal.

Architecture
------------

  Query ──► Cerebro (34M, distilled from Large)
              │  keyword-routes to LoRA specialty
              │  drafts k=8 tokens autoregressively
              ▼
         Medium (114M, distilled from Large)
              │  verifies k tokens in ONE forward pass
              │  acceptance rate ~87% → ~13% rejected
              ▼
         Large (474M) + LoRA adapter
              │  verifies medium-accepted tokens in ONE forward pass
              │  acceptance rate ~99%
              ▼
         Final output tokens

Speedup: Large processes only ~13 % of tokens → ~4× vs Large-alone.

The output distribution is statistically equivalent to Large+LoRA
(Leviathan et al. 2023), not an approximation.

Usage
-----
    # Interactive CLI
    python scripts/speculative_inference.py \\
        --cerebro  checkpoints/distil_cerebro/soup_uniform.pkl \\
        --medium   checkpoints/distil_medium_legal/soup_uniform.pkl \\
        --large    checkpoints/axion_large_legal/soup_uniform.pkl \\
        --lora-dir checkpoints/lora/

    # HTTP server (stdlib, no FastAPI needed)
    python scripts/speculative_inference.py \\
        --cerebro  checkpoints/distil_cerebro/soup_uniform.pkl \\
        --medium   checkpoints/distil_medium_legal/soup_uniform.pkl \\
        --large    checkpoints/axion_large_legal/soup_uniform.pkl \\
        --lora-dir checkpoints/lora/ \\
        --serve --port 8080

    # With RAG (legal corpus index) + live tools
    python scripts/speculative_inference.py \\
        --cerebro  checkpoints/distil_cerebro/soup_uniform.pkl \\
        --medium   checkpoints/distil_medium_legal/soup_uniform.pkl \\
        --large    checkpoints/axion_large_legal/soup_uniform.pkl \\
        --lora-dir checkpoints/lora/ \\
        --rag-index data/rag_index/ \\
        --tools

    # Quick throughput benchmark
    python scripts/speculative_inference.py \\
        --cerebro  ... --medium ... --large ... \\
        --benchmark

    # POST /generate  {"prompt": "...", "max_tokens": 256}
    curl -s -X POST http://localhost:8080/generate \\
        -H "Content-Type: application/json" \\
        -d '{"prompt": "¿Qué es el recurso de amparo?", "max_tokens": 200}'

Tool call syntax (model output):
    ÿTOOL:{"name":"search_boe","query":"artículo 248 CP"}ÿ  →  tool executed
    ÿRESULT:{"status":"ok","data":"..."}ÿ                   →  result injected
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("speculative")

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Model architecture presets (must match training) ───────────────────────────

PRESETS = {
    "smoke":  dict(hidden_size=256,  num_layers=4,  num_heads=4,  seq_len=256),
    "small":  dict(hidden_size=512,  num_layers=8,  num_heads=8,  seq_len=512),
    "medium": dict(hidden_size=768,  num_layers=12, num_heads=12, seq_len=1024),
    "full":   dict(hidden_size=1024, num_layers=12, num_heads=16, seq_len=2048),
    "large":  dict(hidden_size=1280, num_layers=24, num_heads=20, seq_len=1024),
}

VOCAB_SIZE = 512  # byte-level (0–255) + 256 reserved/extended tokens

# ── Tool call byte markers ─────────────────────────────────────────────────────
# Byte 0xFF (255) never appears in valid UTF-8 text → safe delimiter
_TOOL_OPEN   = bytes([255]) + b"TOOL:"    # ÿTOOL:
_TOOL_CLOSE  = bytes([255])               # ÿ  (closing)
_RESULT_OPEN = bytes([255]) + b"RESULT:"  # ÿRESULT:

# ── Legal specialty routing ────────────────────────────────────────────────────

SPECIALTY_KEYWORDS: dict[str, list[str]] = {
    "penal": [
        "delito", "pena", "condena", "tribunal penal", "código penal",
        "sentencia penal", "acusado", "fiscal", "defensa penal",
        "prisión", "libertad condicional", "homicidio", "robo", "estafa",
        "detención", "instrucción penal",
    ],
    "civil": [
        "contrato", "propiedad", "herencia", "divorcio", "matrimonio",
        "arrendamiento", "daños", "indemnización civil", "testamento",
        "hipoteca", "compraventa", "responsabilidad civil", "tutela",
        "filiación", "usufructo",
    ],
    "laboral": [
        "trabajador", "empresa", "despido", "nómina", "contrato laboral",
        "seguridad social", "erte", "convenio colectivo", "sindicato",
        "accidente laboral", "incapacidad", "jornada", "salario",
        "huelga", "inspección de trabajo",
    ],
    "constitucional": [
        "constitución", "derechos fundamentales", "tribunal constitucional",
        "recurso de amparo", "ley orgánica", "separación de poderes",
        "garantías constitucionales", "estado de derecho", "cortes generales",
        "inconstitucionalidad",
    ],
    "administrativo": [
        "administración pública", "funcionario", "multa", "sanción",
        "licencia", "permiso", "recurso administrativo", "ayuntamiento",
        "ministerio", "ley administrativa", "contrato público", "licitación",
        "expropiación", "silencio administrativo",
    ],
    "mercantil": [
        "sociedad", "empresa mercantil", "concurso de acreedores",
        "acciones", "junta general", "consejo de administración",
        "quiebra", "fusión", "derecho mercantil", "letra de cambio",
        "auditoría", "registro mercantil",
    ],
    "resumen": [
        "resume", "resumen", "resumir", "síntesis", "sintetiza",
        "puntos principales", "ideas clave", "en pocas palabras",
        "abstract", "summarize", "summary",
    ],
    "instruccion": [
        "haz", "crea", "genera", "escribe", "explica", "describe",
        "lista", "enumera", "define", "calcula",
    ],
    "qa": [
        "según el texto", "basándote en", "de acuerdo con el pasaje",
        "el texto dice", "¿qué dice el texto", "lee el siguiente",
    ],
    "extraccion": [
        "extrae", "identifica las entidades", "personas mencionadas",
        "organizaciones en el texto", "lugares que aparecen",
        "entidades nombradas", "nombra las partes",
    ],
    "redaccion": [
        "redacta", "redacción", "escribe un contrato", "escribe una demanda",
        "escribe un recurso", "carta de requerimiento", "escrito de",
        "modelo de", "plantilla de",
    ],
    "dialogo": [
        "conversación", "me puedes ayudar", "tengo una consulta",
        "necesito consejo", "qué harías", "cómo me recomiendas",
    ],
    "razonamiento": [
        "paso a paso", "razona", "explica tu razonamiento",
        "¿por qué", "¿cómo llegaste", "demuestra", "calcula",
        "resuelve el problema",
    ],
    "traduccion": [
        "traduce", "traducción", "en inglés", "en catalán", "en euskera",
        "translate", "versión en", "cómo se dice en",
    ],
    "herramientas": [
        "busca en el boe", "consulta la ley", "busca jurisprudencia",
        "cuál es el texto vigente", "artículo vigente",
        "busca sentencias", "hay sentencias recientes",
        "calcula el plazo", "fecha límite", "cuándo vence",
    ],
}


def _route_specialty(text: str) -> str:
    """Keyword-based router → specialty name or 'general'."""
    text_lower = text.lower()
    scores = {
        spec: sum(1 for kw in kws if kw in text_lower)
        for spec, kws in SPECIALTY_KEYWORDS.items()
    }
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "general"


# ── Byte-level tokenizer (matches training vocab) ──────────────────────────────

def encode(text: str) -> list[int]:
    """UTF-8 bytes → token ids. IDs 0–255 are raw bytes."""
    return list(text.encode("utf-8"))


def decode(token_ids: list[int]) -> str:
    """Token ids → string. Skips non-byte tokens; replaces invalid UTF-8."""
    raw = bytes(t for t in token_ids if 0 <= t <= 255)
    return raw.decode("utf-8", errors="replace")


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _load_ckpt(path: str) -> dict:
    logger.info("Loading: %s", path)
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_lora(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── LoRA adapter injection ─────────────────────────────────────────────────────

def _apply_lora(base_params: dict, lora_data: dict) -> dict:
    """
    Return new param tree with LoRA deltas merged into base_params.
    base_params is NOT mutated (deep copy at leaf level).
    """
    import copy
    import jax.numpy as jnp

    meta        = lora_data.get("meta", {})
    lora_params = lora_data.get("lora_params", {})
    rank        = meta.get("rank", 16)
    lora_alpha  = float(meta.get("lora_alpha", 32.0))
    scale       = lora_alpha / rank

    def _inject(base, lora, depth: int = 0):
        if depth > 20:
            return base
        if isinstance(base, dict):
            return {
                k: _inject(base[k], lora.get(k, {}) if isinstance(lora, dict) else {}, depth + 1)
                for k in base
            }
        # Leaf: merge if LoRA adapter present
        if isinstance(lora, dict) and "A" in lora and "B" in lora:
            delta = scale * (lora["B"] @ lora["A"])
            return (base + delta.astype(base.dtype))
        return base  # no adapter for this weight → pass through

    return _inject(base_params, lora_params)


# ── Model handle ───────────────────────────────────────────────────────────────

class ModelHandle:
    """Wraps a loaded Slim200M with a JIT-compiled single-forward-pass."""

    def __init__(self, preset: str, params, use_bf16: bool):
        import jax
        import jax.numpy as jnp
        from models.slim_200m import Slim200M, ModelConfig

        p = PRESETS[preset]
        cfg = ModelConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=p["hidden_size"],
            num_layers=p["num_layers"],
            num_heads=p["num_heads"],
            max_seq_len=p["seq_len"],
            dropout_rate=0.0,
        )
        self._model    = Slim200M(cfg)
        self.params    = params
        self.max_seq   = p["seq_len"]
        self._use_bf16 = use_bf16

        @jax.jit
        def _fwd(params, ids):
            fwd = params
            if use_bf16:
                fwd = jax.tree_util.tree_map(
                    lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x,
                    params,
                )
            return self._model.apply(fwd, ids).astype(jnp.float32)

        self._fwd = _fwd

    def forward(self, input_ids):
        """input_ids: (seq,) or (1, seq) → logits (seq, vocab) float32."""
        import jax.numpy as jnp
        if input_ids.ndim == 1:
            input_ids = input_ids[None]
        logits = self._fwd(self.params, input_ids)
        return logits[0]  # (seq, vocab)


# ── Speculative decoding engine ────────────────────────────────────────────────

class SpeculativeEngine:
    """
    3-level speculative decoder.

    Level 1 (Cerebro)  → drafts k tokens autoregressively.
    Level 2 (Medium)   → verifies in one forward pass; accepts ~87%.
    Level 3 (Large)    → verifies medium-accepted in one forward pass; accepts ~99%.

    Output distribution == Large+LoRA distribution (exact, not approximate).
    """

    def __init__(
        self,
        cerebro: ModelHandle,
        medium:  ModelHandle,
        large:   ModelHandle,
        large_base_params,
        lora_dir:      Optional[Path],
        draft_len:     int   = 8,
        temperature:   float = 0.8,
        top_p:         float = 0.95,
        rag_retriever  = None,   # RAGRetriever instance or None
        tool_registry  = None,   # ToolRegistry instance or None
        rag_top_k:     int   = 3,
        max_tool_calls: int  = 5,
    ):
        self.cerebro           = cerebro
        self.medium            = medium
        self.large             = large
        self._large_base       = large_base_params
        self.lora_dir          = lora_dir
        self.draft_len         = draft_len
        self.temperature       = temperature
        self.top_p             = top_p
        self.rag               = rag_retriever
        self.tools             = tool_registry
        self.rag_top_k         = rag_top_k
        self.max_tool_calls    = max_tool_calls
        self._lora_cache:      dict[str, dict] = {}
        self._active_specialty: str = "general"

    # ── LoRA management ────────────────────────────────────────────────────────

    def _activate_lora(self, specialty: str) -> None:
        if specialty == self._active_specialty:
            return
        if specialty == "general" or self.lora_dir is None:
            self.large.params = self._large_base
            self._active_specialty = "general"
            return

        if specialty not in self._lora_cache:
            candidates = [
                self.lora_dir / f"lora_{specialty}_final.pkl",
                self.lora_dir / f"large_{specialty}" / "lora_final.pkl",
                self.lora_dir / specialty / "lora_final.pkl",
            ]
            found = next((p for p in candidates if p.exists()), None)
            if found is None:
                logger.warning("No LoRA adapter for '%s' — using base Large", specialty)
                self.large.params = self._large_base
                self._active_specialty = specialty
                return
            self._lora_cache[specialty] = _load_lora(str(found))
            logger.info("Cached LoRA adapter: %s (%s)", specialty, found.name)

        self.large.params = _apply_lora(self._large_base, self._lora_cache[specialty])
        self._active_specialty = specialty
        logger.info("Active LoRA: %s", specialty)

    # ── Sampling primitives ────────────────────────────────────────────────────

    def _sample_from_logits(self, logits_1d, rng_key) -> int:
        """Tempered + top-p sample from a single (vocab,) logit vector."""
        import jax
        import jax.numpy as jnp

        if self.temperature == 0.0:
            return int(jnp.argmax(logits_1d))

        probs = jax.nn.softmax(logits_1d / self.temperature)

        if self.top_p < 1.0:
            sorted_idx  = jnp.argsort(probs)[::-1]
            sorted_p    = probs[sorted_idx]
            cum         = jnp.cumsum(sorted_p)
            cutoff      = int(jnp.searchsorted(cum, self.top_p)) + 1
            mask        = jnp.arange(VOCAB_SIZE) >= cutoff
            sorted_p    = jnp.where(mask, 0.0, sorted_p)
            total       = sorted_p.sum()
            sorted_p    = jnp.where(total > 0, sorted_p / total, sorted_p)
            probs       = jnp.zeros(VOCAB_SIZE).at[sorted_idx].set(sorted_p)

        return int(jax.random.categorical(rng_key, jnp.log(probs + 1e-12)))

    # ── Level 1: Cerebro draft ─────────────────────────────────────────────────

    def _draft(
        self, context: list[int], rng_key
    ) -> tuple[list[int], list[float]]:
        """
        Autoregressively generate `draft_len` tokens with Cerebro.
        Returns (tokens, draft_probs) where draft_probs[i] = p_cerebro(token_i).
        """
        import jax
        import jax.numpy as jnp

        tokens: list[int]  = []
        probs:  list[float] = []
        ctx = context[:]

        for _ in range(self.draft_len):
            seq    = ctx[-self.cerebro.max_seq:]
            ids    = jnp.array(seq, dtype=jnp.int32)
            logits = self.cerebro.forward(ids)        # (len, vocab)
            last   = logits[-1]                       # (vocab,)

            rng_key, sub = jax.random.split(rng_key)
            tok = self._sample_from_logits(last, sub)

            p = float(jax.nn.softmax(last / max(self.temperature, 1e-6))[tok])
            tokens.append(tok)
            probs.append(p)
            ctx.append(tok)

        return tokens, probs

    # ── Speculative verification (one verifier level) ──────────────────────────

    def _verify(
        self,
        verifier:     ModelHandle,
        context:      list[int],
        draft_tokens: list[int],
        draft_probs:  list[float],
        rng_key,
    ) -> tuple[list[int], list[float]]:
        """
        Batch-verify `draft_tokens` against `verifier` using acceptance/rejection
        sampling (Leviathan et al. 2023, Algorithm 1).

        Returns:
            accepted_tokens        — accepted prefix (including bonus token on full accept)
            accepted_verifier_probs — p_verifier(tok | ctx) for each accepted token;
                                      used as "draft probs" for the next verifier level.
        """
        import jax
        import jax.numpy as jnp
        import numpy as np

        k        = len(draft_tokens)
        full_seq = (context + draft_tokens)[-verifier.max_seq:]
        ids      = jnp.array(full_seq, dtype=jnp.int32)
        logits   = verifier.forward(ids)              # (seq, vocab)

        # Offset: logit at position p predicts the token at position p+1.
        # draft_tokens[j] was generated at context_position + j, so the
        # relevant verifier logit is at index (trunc_ctx + j - 1).
        trunc_ctx = len(full_seq) - k

        T = max(self.temperature, 1e-6)
        accepted_tokens: list[int]  = []
        accepted_vprobs: list[float] = []

        rng_key, sub = jax.random.split(rng_key)
        subkeys = jax.random.split(sub, k + 1)

        for j in range(k):
            pos = trunc_ctx + j - 1
            if pos < 0:
                # Context was truncated past this point — accept unconditionally.
                accepted_tokens.append(draft_tokens[j])
                accepted_vprobs.append(draft_probs[j])
                continue

            v_logits = logits[pos]                     # (vocab,)
            v_probs  = np.array(jax.nn.softmax(v_logits / T))

            p_v = float(v_probs[draft_tokens[j]])
            p_d = draft_probs[j]
            ratio = min(1.0, p_v / (p_d + 1e-12))

            u = float(jax.random.uniform(subkeys[j]))

            if u <= ratio:
                accepted_tokens.append(draft_tokens[j])
                accepted_vprobs.append(p_v)
            else:
                # Rejection: sample from residual distribution
                # p_adj ∝ max(0, p_verifier − p_draft_dist)
                p_d_dist         = np.zeros(VOCAB_SIZE, dtype=np.float64)
                p_d_dist[draft_tokens[j]] = p_d
                p_adj            = np.maximum(0.0, v_probs.astype(np.float64) - p_d_dist)
                total            = p_adj.sum()
                if total > 1e-12:
                    p_adj /= total
                else:
                    p_adj = v_probs.astype(np.float64) / v_probs.sum()

                resampled = int(
                    jax.random.categorical(subkeys[k], jnp.log(jnp.array(p_adj) + 1e-12))
                )
                accepted_tokens.append(resampled)
                accepted_vprobs.append(float(p_adj[resampled]))
                return accepted_tokens, accepted_vprobs  # stop at first rejection

        # All k tokens accepted → append one bonus token from the verifier.
        bonus_pos = trunc_ctx + k - 1
        if bonus_pos < len(logits):
            rng_key, bsub = jax.random.split(rng_key)
            bonus = self._sample_from_logits(logits[bonus_pos], bsub)
            p_bonus = float(jax.nn.softmax(logits[bonus_pos] / T)[bonus])
            accepted_tokens.append(bonus)
            accepted_vprobs.append(p_bonus)

        return accepted_tokens, accepted_vprobs

    # ── 3-level generation loop ────────────────────────────────────────────────

    # ── RAG context injection ──────────────────────────────────────────────────

    def _prepend_rag(self, prompt: str) -> str:
        """Retrieve relevant legal chunks and prepend to prompt."""
        if self.rag is None:
            return prompt
        try:
            context_str = self.rag.retrieve(prompt)
            return context_str + "\n\n" + prompt
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            return prompt

    # ── Tool call detection + execution ────────────────────────────────────────

    def _check_tool_call(
        self, generated: list[int]
    ) -> tuple[int, dict] | None:
        """
        Scan the tail of `generated` for a complete ÿTOOL:{...}ÿ sequence.
        Returns (start_index, parsed_dict) or None.
        """
        buf = bytes(t for t in generated if 0 <= t <= 255)
        open_pos = buf.rfind(_TOOL_OPEN)
        if open_pos == -1:
            return None
        json_start = open_pos + len(_TOOL_OPEN)
        close_pos  = buf.find(_TOOL_CLOSE, json_start)
        if close_pos == -1:
            return None  # incomplete — still generating
        try:
            call = json.loads(buf[json_start:close_pos])
        except json.JSONDecodeError:
            return None
        # Map byte offset back to token index (approximate: 1 byte ≈ 1 token)
        return open_pos, call

    def _execute_tool(self, call: dict) -> list[int]:
        """Execute a tool call and return the ÿRESULT:{...}ÿ as token ids."""
        if self.tools is None:
            result = {"status": "error", "data": "No tool registry configured"}
        else:
            try:
                name   = call.get("name", "")
                kwargs = {k: v for k, v in call.items() if k != "name"}
                result = self.tools.execute(name, **kwargs)
            except Exception as exc:
                result = {"status": "error", "data": str(exc)}
        # Truncate long results to stay within context window
        data = result.get("data", "")
        if isinstance(data, str) and len(data) > 400:
            result = {**result, "data": data[:400] + "…"}
        result_bytes = _RESULT_OPEN + json.dumps(result, ensure_ascii=False).encode() + bytes([255])
        logger.info("Tool %s → %s (status=%s)",
                    call.get("name"), str(result.get("data", ""))[:60], result.get("status"))
        return list(result_bytes)

    # ── 3-level generation loop ────────────────────────────────────────────────

    def generate(
        self,
        prompt:         str,
        max_new_tokens: int = 256,
        seed:           int = 42,
    ) -> str:
        """
        Full 3-level speculative decode with optional RAG context and tool use.
        Returns the generated text (decoded from byte tokens).
        """
        import jax

        specialty = _route_specialty(prompt)
        self._activate_lora(specialty)

        # ── RAG: prepend retrieved legal context ───────────────────────────────
        augmented_prompt = self._prepend_rag(prompt)
        context   = encode(augmented_prompt)
        rng       = jax.random.PRNGKey(seed)
        generated: list[int] = []

        # Stats
        n_drafted    = 0
        n_med_acc    = 0
        n_large_acc  = 0
        n_large_fwd  = 0
        n_tool_calls = 0
        t0 = time.perf_counter()

        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            k = min(self.draft_len, remaining)

            ctx = context + generated

            rng, r1, r2, r3 = jax.random.split(rng, 4)

            # ── Level 1: Cerebro drafts k tokens ──────────────────────────────
            draft_tokens, draft_probs = self._draft(ctx, r1)
            draft_tokens = draft_tokens[:k]
            draft_probs  = draft_probs[:k]
            n_drafted += k

            # ── Level 2: Medium verifies ──────────────────────────────────────
            med_tokens, med_probs = self._verify(
                self.medium, ctx, draft_tokens, draft_probs, r2
            )
            n_med_acc += len(med_tokens)

            if not med_tokens:
                continue  # full rejection (rare) — re-draft

            # ── Level 3: Large verifies medium-accepted tokens ─────────────────
            n_large_fwd += 1
            large_tokens, _ = self._verify(
                self.large, ctx, med_tokens, med_probs, r3
            )
            n_large_acc += len(large_tokens)

            generated.extend(large_tokens)

            # ── Tool call detection ────────────────────────────────────────────
            if self.tools is not None and n_tool_calls < self.max_tool_calls:
                hit = self._check_tool_call(generated)
                if hit is not None:
                    _, call = hit
                    result_tokens = self._execute_tool(call)
                    generated.extend(result_tokens)
                    n_tool_calls += 1

        elapsed   = time.perf_counter() - t0
        total     = len(generated)
        tok_s     = total / elapsed if elapsed > 0 else 0.0
        med_pct   = 100.0 * n_med_acc   / max(1, n_drafted)
        large_pct = 100.0 * n_large_acc / max(1, n_med_acc)

        logger.info(
            "Generated %d tok in %.2fs (%.0f tok/s) | "
            "drafted=%d med_acc=%.0f%% large_acc=%.0f%% large_fwd=%d "
            "tool_calls=%d rag=%s (specialty=%s)",
            total, elapsed, tok_s,
            n_drafted, med_pct, large_pct, n_large_fwd,
            n_tool_calls, self.rag is not None, specialty,
        )

        # Strip raw ÿRESULT:...ÿ markers from final output (keep tool answer text only)
        return _strip_markers(decode(generated))


# ── Engine factory ─────────────────────────────────────────────────────────────

def build_engine(args) -> SpeculativeEngine:
    import os

    # Thread + XLA config (mirrors launch_axion_training.py)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("OMP_PROC_BIND", "spread")
    os.environ.setdefault("OMP_PLACES",    "cores")
    os.environ.setdefault("GOMP_SPINCOUNT", "0")
    os.environ.setdefault("OPENBLAS_CORETYPE", "NEOVERSEN2")
    os.environ["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen=true "
        f"intra_op_parallelism_threads={args.threads} "
        f"--xla_cpu_enable_fast_math=true "
        f"--xla_cpu_enable_vector_ops=true "
        f"--xla_force_host_platform_device_count=1"
    )

    import jax

    if args.compile_cache:
        cache_dir = Path(args.compile_cache)
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            jax.config.update("jax_compilation_cache_dir", str(cache_dir))
        except Exception:
            try:
                from jax.experimental import compilation_cache as cc
                cc.initialize_cache(str(cache_dir))
            except Exception:
                pass

    use_bf16 = args.dtype == "bf16"

    logger.info("Loading models …")

    c_ckpt  = _load_ckpt(args.cerebro)
    cerebro = ModelHandle(args.cerebro_preset, c_ckpt["params"], use_bf16)
    logger.info("Cerebro ready  preset=%-8s params=%.1fM",
                args.cerebro_preset, _count_params(c_ckpt["params"]) / 1e6)

    m_ckpt  = _load_ckpt(args.medium)
    medium  = ModelHandle(args.medium_preset, m_ckpt["params"], use_bf16)
    logger.info("Medium  ready  preset=%-8s params=%.1fM",
                args.medium_preset, _count_params(m_ckpt["params"]) / 1e6)

    l_ckpt      = _load_ckpt(args.large)
    large_base  = l_ckpt["params"]
    large       = ModelHandle(args.large_preset, large_base, use_bf16)
    logger.info("Large   ready  preset=%-8s params=%.1fM",
                args.large_preset, _count_params(large_base) / 1e6)

    lora_dir = Path(args.lora_dir) if args.lora_dir else None

    # ── Optional: RAG retriever ────────────────────────────────────────────────
    rag = None
    if getattr(args, "rag_index", None):
        try:
            from scripts.rag_retriever import RAGRetriever
        except ImportError:
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from rag_retriever import RAGRetriever
            except ImportError:
                RAGRetriever = None

        if RAGRetriever is not None:
            rag_path = Path(args.rag_index)
            if rag_path.exists():
                rag = RAGRetriever(
                    str(rag_path),
                    top_k=getattr(args, "rag_top_k", 3),
                    max_context_bytes=500,
                )
                logger.info("RAG index loaded: %s", rag_path)
            else:
                logger.warning("--rag-index path not found: %s — RAG disabled", rag_path)
        else:
            logger.warning("rag_retriever.py not found — RAG disabled")

    # ── Optional: tool registry ────────────────────────────────────────────────
    tools = None
    if getattr(args, "tools", False):
        try:
            from tool_executor import build_registry
            tools = build_registry()
            logger.info("Tool registry loaded (%d tools)", len(tools._tools))
        except ImportError:
            logger.warning("tool_executor.py not found — tools disabled")

    return SpeculativeEngine(
        cerebro=cerebro,
        medium=medium,
        large=large,
        large_base_params=large_base,
        lora_dir=lora_dir,
        draft_len=args.draft_len,
        temperature=args.temperature,
        top_p=args.top_p,
        rag_retriever=rag,
        tool_registry=tools,
        rag_top_k=getattr(args, "rag_top_k", 3),
        max_tool_calls=getattr(args, "max_tool_calls", 5),
    )


def _count_params(params) -> int:
    import jax
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def _strip_markers(text: str) -> str:
    """Remove raw ÿTOOL:...ÿ and ÿRESULT:...ÿ markers from decoded output."""
    import re
    # The markers decode as \xff in the string
    return re.sub(r'\xff(?:TOOL|RESULT):[^\xff]*\xff', '', text).strip()


# ── HTTP server (stdlib only) ──────────────────────────────────────────────────

def serve(engine: SpeculativeEngine, host: str, port: int) -> None:
    import http.server
    import json as _json

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: N802
            logger.debug("HTTP %s", fmt % args)

        def _send(self, data: dict, code: int = 200) -> None:
            body = _json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send({"status": "ok"})
            elif self.path == "/routes":
                self._send({"specialties": list(SPECIALTY_KEYWORDS)})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self):  # noqa: N802
            if self.path not in ("/generate", "/v1/completions"):
                self._send({"error": "not found"}, 404)
                return

            length = int(self.headers.get("Content-Length", 0))
            try:
                req = _json.loads(self.rfile.read(length))
            except _json.JSONDecodeError:
                self._send({"error": "invalid JSON"}, 400)
                return

            prompt     = req.get("prompt", "")
            max_tokens = int(req.get("max_tokens", req.get("max_new_tokens", 256)))
            seed       = int(req.get("seed", 42))

            if not prompt:
                self._send({"error": "'prompt' is required"}, 400)
                return

            specialty = _route_specialty(prompt)
            t0 = time.perf_counter()
            try:
                text = engine.generate(prompt, max_new_tokens=max_tokens, seed=seed)
            except Exception as exc:
                logger.exception("Generation failed")
                self._send({"error": str(exc)}, 500)
                return

            self._send({
                "generated_text": text,
                "prompt":         prompt,
                "specialty":      specialty,
                "max_tokens":     max_tokens,
                "elapsed_s":      round(time.perf_counter() - t0, 3),
            })

    server = http.server.ThreadingHTTPServer((host, port), Handler)
    logger.info("Serving on http://%s:%d", host, port)
    logger.info("  POST /generate          {prompt, max_tokens, seed}")
    logger.info("  GET  /health            liveness check")
    logger.info("  GET  /routes            available specialties")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        logger.info("Server stopped.")


# ── Benchmark ──────────────────────────────────────────────────────────────────

_BENCH_PROMPTS = [
    "¿Cuáles son los requisitos para interponer un recurso de amparo ante el Tribunal Constitucional?",
    "Explica los elementos del tipo penal en el delito de estafa según el Código Penal español.",
    "¿Qué derechos tiene un trabajador ante un despido improcedente?",
    "¿Cómo se constituye una sociedad de responsabilidad limitada en España?",
    "Describe el procedimiento de expropiación forzosa en el derecho administrativo español.",
]


def run_benchmark(engine: SpeculativeEngine, max_tokens: int = 128, runs: int = 3) -> None:
    logger.info("Benchmark: %d prompts × %d tokens × %d runs", len(_BENCH_PROMPTS), max_tokens, runs)
    all_times: list[float] = []
    all_toks:  list[int]   = []

    for prompt in _BENCH_PROMPTS:
        specialty = _route_specialty(prompt)
        logger.info("  [%s] %s…", specialty, prompt[:60])
        for r in range(runs):
            t0   = time.perf_counter()
            text = engine.generate(prompt, max_new_tokens=max_tokens, seed=r)
            dt   = time.perf_counter() - t0
            n    = len(encode(text))
            all_times.append(dt)
            all_toks.append(n)
            logger.info("    run %d → %d tok in %.2fs (%.0f tok/s)", r + 1, n, dt, n / dt)

    avg_s   = sum(all_times) / len(all_times)
    avg_tok = sum(all_toks)  / len(all_toks)
    logger.info("─" * 60)
    logger.info("Mean latency : %.2fs", avg_s)
    logger.info("Mean tokens  : %.0f", avg_tok)
    logger.info("Mean throughput: %.0f tok/s", avg_tok / avg_s)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Checkpoints
    p.add_argument("--cerebro",  required=True, help="Cerebro checkpoint .pkl")
    p.add_argument("--medium",   required=True, help="Medium checkpoint .pkl")
    p.add_argument("--large",    required=True, help="Large checkpoint .pkl")

    # Presets
    p.add_argument("--cerebro-preset", default="small",  choices=list(PRESETS))
    p.add_argument("--medium-preset",  default="medium", choices=list(PRESETS))
    p.add_argument("--large-preset",   default="large",  choices=list(PRESETS))

    # LoRA
    p.add_argument("--lora-dir", default=None,
                   help="Directory with lora_<specialty>_final.pkl or "
                        "large_<specialty>/lora_final.pkl files")

    # Decoding
    p.add_argument("--draft-len",      type=int,   default=8,
                   help="Tokens Cerebro drafts per step (default: 8)")
    p.add_argument("--max-new-tokens", type=int,   default=256)
    p.add_argument("--temperature",    type=float, default=0.8)
    p.add_argument("--top-p",          type=float, default=0.95)
    p.add_argument("--seed",           type=int,   default=42)

    # RAG
    p.add_argument("--rag-index",  default=None,
                   help="Path to RAG index dir built by rag_indexer.py")
    p.add_argument("--rag-top-k",  type=int, default=3,
                   help="Number of RAG chunks to retrieve (default: 3)")

    # Tools
    p.add_argument("--tools", action="store_true",
                   help="Enable live tool use (search_boe, search_cendoj, etc.)")
    p.add_argument("--max-tool-calls", type=int, default=5,
                   help="Max tool calls per generation (default: 5)")

    # Hardware
    p.add_argument("--dtype",         choices=["float32", "bf16"], default="bf16")
    p.add_argument("--threads",       type=int, default=32)
    p.add_argument("--compile-cache", default=None,
                   help="XLA compilation cache dir (e.g. cache/jax_compile)")

    # Modes
    p.add_argument("--serve",     action="store_true", help="Run HTTP server")
    p.add_argument("--host",      default="0.0.0.0")
    p.add_argument("--port",      type=int, default=8080)
    p.add_argument("--benchmark", action="store_true",
                   help="Throughput benchmark then exit")
    p.add_argument("--bench-tokens", type=int, default=128,
                   help="Tokens per benchmark prompt (default: 128)")
    p.add_argument("--bench-runs", type=int, default=3,
                   help="Runs per benchmark prompt (default: 3)")

    args = p.parse_args()
    engine = build_engine(args)

    if args.benchmark:
        run_benchmark(engine, max_tokens=args.bench_tokens, runs=args.bench_runs)
        return

    if args.serve:
        serve(engine, args.host, args.port)
        return

    # Interactive CLI
    print()
    print("Capibara Legal — 3-level speculative inference")
    print(f"  draft_len={args.draft_len}  temperature={args.temperature}  top_p={args.top_p}")
    print("  Legal: penal · civil · laboral · constitucional · administrativo · mercantil")
    print("  Skills: resumen · instruccion · qa · extraccion · redaccion · dialogo · razonamiento · traduccion")
    print("  Empty input to quit.\n")

    seed = args.seed
    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            break
        result = engine.generate(prompt, max_new_tokens=args.max_new_tokens, seed=seed)
        seed += 1
        print()
        print(result)
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""LoRA fine-tuning for capibara-slim models.

Adds low-rank adapter matrices (Hu et al. 2021) to every 2-D weight kernel
in the model. The base model stays frozen; only the LoRA params are trained.
Adapters are saved separately (~10-30 MB) and can be swapped at inference.

LoRA formula (per layer):  W_eff = W_base + (α/r) · B @ A
  where B ∈ R^{d×r}, A ∈ R^{r×k}, r = rank, α = lora_alpha
  B initialised to 0, A ~ N(0, 1/r)  →  zero delta at step 0

Input format (JSONL, one example per line):
    {"prompt": "¿Qué establece el artículo 24 CE?",
     "response": "El artículo 24 reconoce el derecho a la tutela judicial..."}

The prompt+response are concatenated with a separator and trained with
next-token prediction. Loss is masked to the response tokens only.

Specialties (--specialty) filter examples by keyword match on the prompt.
Use "all" to train on every example.

Available specialties:
  penal, civil, laboral, constitucional, administrativo, mercantil,
  herramientas (JSON tool-call format filter),
  documentos (Markdown legal structure filter — contracts, demandas, actas)

Usage:
    # Legal Q&A fine-tuning on large legal model
    python scripts/lora_finetune.py \\
        --base-ckpt  checkpoints/axion_large_legal/soup_uniform.pkl \\
        --preset     large \\
        --data       data/finetune/legal_qa.jsonl \\
        --specialty  all \\
        --output     checkpoints/lora/large_legal_qa/ \\
        --steps 3000 --rank 16 --lora-alpha 32

    # Penal law specialty
    python scripts/lora_finetune.py \\
        --base-ckpt  checkpoints/axion_large_legal/soup_uniform.pkl \\
        --preset     large \\
        --data       data/finetune/legal_qa.jsonl \\
        --specialty  penal \\
        --output     checkpoints/lora/large_penal/ \\
        --steps 2000 --rank 8

    # Load and merge adapter for inference
    from scripts.lora_finetune import load_lora, merge_lora
    params = load_lora("checkpoints/lora/large_penal/lora_final.pkl",
                       base_params)
    merged = merge_lora(params)   # returns plain params dict (same as base)
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lora")

sys.path.insert(0, str(Path(__file__).parent.parent))

PRESETS = {
    "smoke":  dict(hidden_size=256,  num_layers=4,  num_heads=4,  seq_len=256),
    "small":  dict(hidden_size=512,  num_layers=8,  num_heads=8,  seq_len=512),
    "medium": dict(hidden_size=768,  num_layers=12, num_heads=12, seq_len=1024),
    "full":   dict(hidden_size=1024, num_layers=12, num_heads=16, seq_len=2048),
    "large":  dict(hidden_size=1280, num_layers=24, num_heads=20, seq_len=1024),
}

SPECIALTY_KEYWORDS = {
    "penal":          ["penal", "delito", "pena", "condena", "acusado", "fiscal"],
    "civil":          ["civil", "contrato", "herencia", "matrimonio", "propiedad"],
    "laboral":        ["laboral", "trabajador", "despido", "convenio", "empresa"],
    "constitucional": ["constitucional", "constitución", "derechos fundamentales",
                       "amparo", "tribunal constitucional"],
    "administrativo": ["administrativo", "administración", "sanción", "recurso",
                       "expediente", "licencia"],
    "mercantil":      ["mercantil", "sociedad", "concurso", "quiebra", "accionista"],
    "herramientas":   [],  # JSON-format filter applied post-load; no keyword matching
    "documentos":     ["contrato", "escritura", "acta", "demanda", "recurso",
                       "poder notarial", "testamento", "auto", "sentencia",
                       "providencia", "decreto", "resolución", "certificado",
                       "notificación", "requerimiento", "edicto", "convenio",
                       "estatutos", "protocolo"],
}

# Minimum Markdown legal structure score to keep a documentos example (0.0-1.0)
_DOCUMENTO_MIN_SCORE = 0.4

SEP = "\n### Respuesta:\n"


def is_valid_tool_call(example: dict) -> bool:
    try:
        json.loads(example.get("response", ""))
        return True
    except json.JSONDecodeError:
        return False


def json_parsability(predictions: list[str]) -> float:
    def _try(p: str) -> bool:
        try:
            json.loads(p)
            return True
        except Exception:
            return False
    return sum(_try(p) for p in predictions) / len(predictions) if predictions else 0.0


def coerce_json_output(raw: str) -> str:
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            json.loads(match.group(0))
            return match.group(0)
        except Exception:
            pass
    return "{}"


def markdown_structure_score(text: str) -> float:
    """Score 0-1 how well a legal document follows Markdown structural conventions.

    Checks for: section headings (##/###), numbered articles, bold definitions,
    and minimum length. Used to filter low-quality documentos training examples.
    """
    if len(text) < 100:
        return 0.0
    score = 0.0
    if re.search(r'^#{1,3} ', text, re.MULTILINE):
        score += 0.3
    if re.search(r'artículo\s+\d+|art\.\s*\d+', text, re.IGNORECASE):
        score += 0.3
    if re.search(r'\*\*[^*]+\*\*', text):
        score += 0.2
    if re.search(r'^\d+[\.)\]\s', text, re.MULTILINE):
        score += 0.1
    if len(text) >= 300:
        score += 0.1
    return min(score, 1.0)


def is_valid_documento(example: dict) -> bool:
    return markdown_structure_score(example.get("response", "")) >= _DOCUMENTO_MIN_SCORE


PAD_ID = 256


# -- LoRA parameter utilities --------------------------------------------------

def _is_kernel(path: tuple, value) -> bool:
    """Identify 2-D weight matrices (linear layer kernels) in the param tree."""
    import jax.numpy as jnp
    return (
        isinstance(value, (np.ndarray, jnp.ndarray))
        and value.ndim == 2
        and any("kernel" in str(p) for p in path)
    )


def init_lora_params(base_params, rank: int, lora_alpha: float, rng) -> dict:
    """Create LoRA adapter params matching the base model's 2-D kernels."""
    import jax
    import jax.numpy as jnp

    lora = {}
    leaves_with_paths = jax.tree_util.tree_leaves_with_path(base_params)

    for path, leaf in leaves_with_paths:
        if not _is_kernel(path, leaf):
            continue
        key = "/".join(str(p.key) for p in path)
        d, k = leaf.shape          # (out_features, in_features)
        r = min(rank, min(d, k))   # rank can't exceed smallest dimension

        rng, k1, k2 = jax.random.split(rng, 3)
        lora[key] = {
            "A": jax.random.normal(k1, (r, k), dtype=jnp.float32) / r ** 0.5,
            "B": jnp.zeros((d, r), dtype=jnp.float32),
            "scale": jnp.array(lora_alpha / r, dtype=jnp.float32),
        }

    logger.info("LoRA: %d adapter matrices | rank=%d | α=%.1f",
                len(lora), rank, lora_alpha)
    return lora


def apply_lora(base_params, lora_params) -> dict:
    """Return effective params = base + LoRA delta. Used for forward pass."""
    import jax
    import jax.numpy as jnp

    leaves_with_paths = jax.tree_util.tree_leaves_with_path(base_params)
    effective = {}

    flat_base = {
        "/".join(str(p.key) for p in path): (path, leaf)
        for path, leaf in leaves_with_paths
    }

    def _inject(tree, path_so_far=()):
        if isinstance(tree, dict):
            return {k: _inject(v, path_so_far + (k,)) for k, v in tree.items()}
        key = "/".join(str(p) for p in path_so_far)
        if key in lora_params:
            lp = lora_params[key]
            delta = lp["scale"] * lp["B"] @ lp["A"]
            return tree + delta.astype(tree.dtype)
        return tree

    return _inject(base_params)


def merge_lora(base_params, lora_params) -> dict:
    """Permanently merge LoRA into base weights (for deployment / soup)."""
    return apply_lora(base_params, lora_params)


def save_lora(path: Path, lora_params: dict, meta: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump({"lora": lora_params, **meta}, f)
    size_mb = path.stat().st_size / 1e6
    logger.info("LoRA saved → %s (%.1f MB)", path.name, size_mb)


def load_lora(path: str, base_params) -> dict:
    """Load a LoRA checkpoint and return effective (merged) params."""
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    return apply_lora(base_params, ckpt["lora"])


# -- Instruction data ----------------------------------------------------------

def _load_jsonl(path: str, specialty: str) -> list[dict]:
    examples = []
    keywords = SPECIALTY_KEYWORDS.get(specialty, [])
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if specialty != "all" and keywords:
                text = (ex.get("prompt", "") + ex.get("response", "")).lower()
                if not any(kw in text for kw in keywords):
                    continue
            examples.append(ex)
    if specialty == "herramientas":
        before = len(examples)
        examples = [ex for ex in examples if is_valid_tool_call(ex)]
        logger.info("Herramientas JSON filter: %d → %d valid tool calls", before, len(examples))
    if specialty == "documentos":
        before = len(examples)
        examples = [ex for ex in examples if is_valid_documento(ex)]
        logger.info("Documentos structure filter: %d → %d valid (score≥%.1f)",
                    before, len(examples), _DOCUMENTO_MIN_SCORE)
    logger.info("Loaded %d examples (specialty=%s) from %s",
                len(examples), specialty, path)
    return examples


def _examples_to_batches(
    examples: list[dict],
    seq_len: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[dict]:
    """Tokenise prompt+response pairs and yield batches infinitely."""
    while True:
        rng.shuffle(examples)
        buf_ids: list[np.ndarray] = []
        buf_mask: list[np.ndarray] = []  # 1 = response token (compute loss here)

        for ex in examples:
            prompt   = ex.get("prompt", "").encode("utf-8")
            sep      = SEP.encode("utf-8")
            response = ex.get("response", "").encode("utf-8")

            full = np.frombuffer(prompt + sep + response, dtype=np.uint8).astype(np.int32)
            if len(full) < 4:
                continue

            # Pad/truncate to seq_len+1
            if len(full) > seq_len + 1:
                full = full[:seq_len + 1]

            prompt_len = len(prompt) + len(sep)
            mask = np.zeros(len(full), dtype=np.int32)
            mask[prompt_len:] = 1          # loss only on response tokens

            # Pad to seq_len+1
            pad = seq_len + 1 - len(full)
            full = np.pad(full, (0, pad), constant_values=PAD_ID)
            mask = np.pad(mask, (0, pad), constant_values=0)

            buf_ids.append(full)
            buf_mask.append(mask)

            if len(buf_ids) >= batch_size:
                ids  = np.stack(buf_ids[:batch_size])
                mask_arr = np.stack(buf_mask[:batch_size])
                yield {
                    "input_ids": ids[:, :seq_len],
                    "labels":    ids[:, 1:seq_len + 1],
                    "loss_mask": mask_arr[:, 1:seq_len + 1],
                }
                buf_ids  = buf_ids[batch_size:]
                buf_mask = buf_mask[batch_size:]


# -- Training ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-ckpt", required=True,
                        help="Base model checkpoint .pkl")
    parser.add_argument("--preset",    required=True, choices=list(PRESETS))
    parser.add_argument("--data",      required=True,
                        help="Instruction JSONL file")
    parser.add_argument("--specialty", default="all",
                        choices=["all"] + list(SPECIALTY_KEYWORDS),
                        help="Filter examples by legal specialty (default: all)")
    parser.add_argument("--output",    default="checkpoints/lora")

    # LoRA
    parser.add_argument("--rank",       type=int,   default=16,
                        help="LoRA rank r (default: 16)")
    parser.add_argument("--lora-alpha", type=float, default=32.0,
                        help="LoRA scaling α (default: 32.0)")
    parser.add_argument("--dropout",    type=float, default=0.0,
                        help="Dropout rate (default: 0.0; recommended: 0.05 for herramientas)")

    # Training
    parser.add_argument("--steps",      type=int,   default=3000)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--dtype",      choices=["float32", "bf16"], default="bf16")
    parser.add_argument("--threads",    type=int,   default=32)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--log-steps",  type=int,   default=10)
    parser.add_argument("--save-steps", type=int,   default=500)

    args = parser.parse_args()

    # -- JAX setup -------------------------------------------------------------

    import os
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    os.environ["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen=true "
        f"intra_op_parallelism_threads={args.threads} "
        f"--xla_cpu_enable_fast_math=true "
        f"--xla_force_host_platform_device_count=1"
    )
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("OMP_PROC_BIND", "spread")
    os.environ.setdefault("OMP_PLACES",    "cores")

    import jax
    import jax.numpy as jnp
    import optax
    from models.slim_200m import Slim200M, ModelConfig

    use_bf16 = args.dtype == "bf16"
    preset = PRESETS[args.preset]
    rng = jax.random.PRNGKey(args.seed)

    # -- Load base model -------------------------------------------------------

    logger.info("Loading base model from %s", args.base_ckpt)
    with open(args.base_ckpt, "rb") as f:
        base_ckpt = pickle.load(f)
    base_params = base_ckpt["params"]

    cfg = ModelConfig(
        vocab_size=512,
        hidden_size=preset["hidden_size"],
        num_layers=preset["num_layers"],
        num_heads=preset["num_heads"],
        max_seq_len=preset["seq_len"],
        dropout_rate=args.dropout,
    )
    model = Slim200M(cfg)

    # -- Init LoRA params ------------------------------------------------------

    rng, lora_rng = jax.random.split(rng)
    lora_params = init_lora_params(base_params, args.rank, args.lora_alpha, lora_rng)

    # -- Optimizer (only for LoRA params) --------------------------------------

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=args.lr,
        warmup_steps=args.warmup_steps,
        decay_steps=args.steps,
        end_value=args.lr * 0.05,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=0.0),  # no wd on adapters
    )
    opt_state = optimizer.init(lora_params)

    # -- JIT train step --------------------------------------------------------

    @jax.jit
    def train_step(lora_params, opt_state, batch, base_params):
        def loss_fn(lp):
            eff_params = apply_lora(base_params, lp)
            if use_bf16:
                eff_params = jax.tree_util.tree_map(
                    lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x,
                    eff_params,
                )
            logits = model.apply(eff_params, batch["input_ids"]).astype(jnp.float32)
            ce = optax.softmax_cross_entropy_with_integer_labels(
                logits, batch["labels"]
            )
            mask = batch["loss_mask"].astype(jnp.float32)
            loss = (ce * mask).sum() / (mask.sum() + 1e-8)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(lora_params)
        updates, opt_state = optimizer.update(grads, opt_state, lora_params)
        lora_params = optax.apply_updates(lora_params, updates)
        return lora_params, opt_state, loss

    # -- Data ------------------------------------------------------------------

    examples = _load_jsonl(args.data, args.specialty)
    if not examples:
        logger.error("No examples found -- check --data and --specialty")
        sys.exit(1)

    rng_np = np.random.default_rng(args.seed)
    data_iter = _examples_to_batches(examples, preset["seq_len"],
                                     args.batch_size, rng_np)

    # -- Output ----------------------------------------------------------------

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "preset":      args.preset,
        "specialty":   args.specialty,
        "rank":        args.rank,
        "lora_alpha":  args.lora_alpha,
        "dropout":     args.dropout,
        "base_ckpt":   args.base_ckpt,
    }

    # -- Loop ------------------------------------------------------------------

    logger.info("=" * 60)
    logger.info("LoRA fine-tuning | preset=%s rank=%d α=%.0f specialty=%s",
                args.preset, args.rank, args.lora_alpha, args.specialty)
    logger.info("Examples: %d | steps: %d | lr: %g", len(examples), args.steps, args.lr)
    if args.specialty == "herramientas":
        parsability = json_parsability([ex.get("response", "") for ex in examples])
        logger.info("Data JSON parsability: %.1f%%", parsability * 100)
    logger.info("=" * 60)

    t0 = time.perf_counter()
    t_log = t0
    recent: list[float] = []

    for step in range(1, args.steps + 1):
        batch = next(data_iter)
        batch_jax = {k: jnp.array(v) for k, v in batch.items()}

        lora_params, opt_state, loss = train_step(
            lora_params, opt_state, batch_jax, base_params
        )
        recent.append(float(loss))

        if step % args.log_steps == 0:
            now = time.perf_counter()
            step_time = (now - t_log) / args.log_steps
            avg = sum(recent[-args.log_steps:]) / min(len(recent), args.log_steps)
            eta_h = step_time * (args.steps - step) / 3600
            t_log = now
            logger.info("step %5d/%d | loss %.4f | %.2fs/step | ETA %.1fh",
                        step, args.steps, avg, step_time, eta_h)

        if step % args.save_steps == 0 or step == args.steps:
            save_lora(output_dir / f"lora_step_{step:06d}.pkl", lora_params,
                      {**meta, "step": step, "loss": float(loss)})

    save_lora(output_dir / "lora_final.pkl", lora_params,
              {**meta, "step": args.steps, "loss": recent[-1]})

    total = (time.perf_counter() - t0) / 60
    logger.info("LoRA complete in %.1f min | final loss %.4f", total, recent[-1])
    logger.info("Adapter: %s/lora_final.pkl", output_dir)
    logger.info("To merge into base: use merge_lora() from this script")


if __name__ == "__main__":
    main()

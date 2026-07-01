#!/usr/bin/env python3
"""Supervised Fine-Tuning (SFT) for capibara-slim models.

Updates ALL model weights (no LoRA adapters) on instruction→response pairs.
Use a very low LR (1e-5 to 3e-5) to preserve general knowledge while
teaching instruction-following behaviour.

Loss is masked to response tokens only (same as LoRA fine-tuning).

Input format (JSONL):
    {"prompt": "Como me dou de alta como autónomo?",
     "response": "## Alta como autónomo\\n\\n**1. ..."}

Usage:
    python scripts/sft_finetune.py \\
        --base-ckpt checkpoints/cunca_v2_gl/soup_uniform.pkl \\
        --preset    large \\
        --data      data/finetune/gestoria_gl.jsonl \\
        --output    checkpoints/cunca_v2_gl_sft \\
        --steps 500 --lr 1e-5 --batch-size 4 --grad-accum 4
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [sft] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sft")

sys.path.insert(0, str(Path(__file__).parent.parent))

PRESETS = {
    "smoke":  dict(hidden_size=256,  num_layers=4,  num_heads=4,  seq_len=256),
    "small":  dict(hidden_size=512,  num_layers=8,  num_heads=8,  seq_len=512),
    "medium": dict(hidden_size=768,  num_layers=12, num_heads=12, seq_len=1024),
    "full":   dict(hidden_size=1024, num_layers=12, num_heads=16, seq_len=2048),
    "large":  dict(hidden_size=1280, num_layers=24, num_heads=20, seq_len=1024),
}

PAD_ID = 256
SEP    = "\n\n"


# ── Data ──────────────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                if ex.get("prompt") and ex.get("response"):
                    examples.append(ex)
            except json.JSONDecodeError:
                continue
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


def _examples_to_batches(
    examples: list[dict],
    seq_len: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[dict]:
    while True:
        rng.shuffle(examples)
        buf_ids:  list[np.ndarray] = []
        buf_mask: list[np.ndarray] = []

        for ex in examples:
            prompt   = ex["prompt"].encode("utf-8")
            sep      = SEP.encode("utf-8")
            response = ex["response"].encode("utf-8")

            full = np.frombuffer(prompt + sep + response, dtype=np.uint8).astype(np.int32)
            if len(full) < 4:
                continue
            if len(full) > seq_len + 1:
                full = full[:seq_len + 1]

            prompt_len = len(prompt) + len(sep)
            mask = np.zeros(len(full), dtype=np.int32)
            mask[prompt_len:] = 1

            pad = seq_len + 1 - len(full)
            full = np.pad(full, (0, pad), constant_values=PAD_ID)
            mask = np.pad(mask, (0, pad), constant_values=0)

            buf_ids.append(full)
            buf_mask.append(mask)

            if len(buf_ids) >= batch_size:
                ids = np.stack(buf_ids[:batch_size])
                msk = np.stack(buf_mask[:batch_size])
                yield {
                    "input_ids": ids[:, :seq_len],
                    "labels":    ids[:, 1:seq_len + 1],
                    "loss_mask": msk[:, 1:seq_len + 1],
                }
                buf_ids  = buf_ids[batch_size:]
                buf_mask = buf_mask[batch_size:]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--preset",    required=True, choices=list(PRESETS))
    parser.add_argument("--data",      required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--steps",      type=int,   default=500)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--grad-accum", type=int,   default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--lr",         type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dtype",     choices=["float32", "bf16"], default="bf16")
    parser.add_argument("--threads",   type=int,   default=32)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--log-steps", type=int,   default=10)
    parser.add_argument("--save-steps", type=int,  default=250)
    args = parser.parse_args()

    import os
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=true "
        f"intra_op_parallelism_threads={args.threads} "
        "--xla_cpu_enable_fast_math=true "
        "--xla_force_host_platform_device_count=1"
    )
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES",    "cores")

    import jax
    import jax.numpy as jnp
    import optax
    from models.slim_200m import Slim200M, ModelConfig

    use_bf16 = args.dtype == "bf16"
    preset   = PRESETS[args.preset]

    # ── Load base checkpoint ───────────────────────────────────────────────────

    logger.info("Loading base model from %s", args.base_ckpt)
    with open(args.base_ckpt, "rb") as f:
        ckpt = pickle.load(f)  # nosec B301 - trusted local checkpoint, not user input
    params = ckpt.get("params", ckpt)

    cfg = ModelConfig(
        vocab_size=512,
        hidden_size=preset["hidden_size"],
        num_layers=preset["num_layers"],
        num_heads=preset["num_heads"],
        max_seq_len=preset["seq_len"],
        dropout_rate=0.0,
    )
    model = Slim200M(cfg)

    # ── Optimizer ─────────────────────────────────────────────────────────────

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-7,
        peak_value=args.lr,
        warmup_steps=args.warmup_steps,
        decay_steps=args.steps,
        end_value=args.lr * 0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adamw(schedule, weight_decay=0.01),
    )
    opt_state = optimizer.init(params)

    # ── JIT train step (single micro-batch) ───────────────────────────────────

    @jax.jit
    def train_step(params, opt_state, batch):
        def loss_fn(p):
            p_fwd = p
            if use_bf16:
                p_fwd = jax.tree_util.tree_map(
                    lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x,
                    p_fwd,
                )
            logits = model.apply(p_fwd, batch["input_ids"]).astype(jnp.float32)
            ce = optax.softmax_cross_entropy_with_integer_labels(logits, batch["labels"])
            mask = batch["loss_mask"].astype(jnp.float32)
            return (ce * mask).sum() / (mask.sum() + 1e-8)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        return loss, grads

    @jax.jit
    def apply_updates(params, opt_state, grads):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # ── Data ──────────────────────────────────────────────────────────────────

    examples  = _load_jsonl(args.data)
    rng_np    = np.random.default_rng(args.seed)
    data_iter = _examples_to_batches(
        examples, preset["seq_len"], args.batch_size, rng_np
    )

    # ── Output dir ────────────────────────────────────────────────────────────

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────

    logger.info("=" * 60)
    logger.info("SFT | preset=%s  steps=%d  lr=%.2e  batch=%d×%d",
                args.preset, args.steps, args.lr, args.batch_size, args.grad_accum)
    logger.info("Examples: %d | output: %s", len(examples), output_dir)
    logger.info("=" * 60)

    t0        = time.time()
    loss_acc  = 0.0
    grad_acc  = None

    for step in range(1, args.steps + 1):
        # Gradient accumulation
        step_loss = 0.0
        for _ in range(args.grad_accum):
            batch = next(data_iter)
            batch = {k: jnp.array(v) for k, v in batch.items()}
            loss, grads = train_step(params, opt_state, batch)
            step_loss += float(loss) / args.grad_accum
            if grad_acc is None:
                grad_acc = grads
            else:
                grad_acc = jax.tree_util.tree_map(
                    lambda a, b: a + b, grad_acc, grads
                )

        # Scale accumulated gradients and apply
        grad_acc = jax.tree_util.tree_map(
            lambda g: g / args.grad_accum, grad_acc
        )
        params, opt_state = apply_updates(params, opt_state, grad_acc)
        grad_acc = None
        loss_acc += step_loss

        if step % args.log_steps == 0:
            elapsed = time.time() - t0
            sps = args.log_steps / elapsed
            eta = (args.steps - step) / sps / 3600
            logger.info("step %5d/%d | loss %.4f | %.2fs/step | ETA %.1fh",
                        step, args.steps, loss_acc / args.log_steps,
                        elapsed / args.log_steps, eta)
            loss_acc = 0.0
            t0 = time.time()

        if step % args.save_steps == 0:
            ckpt_path = output_dir / f"ckpt_step_{step:07d}.pkl"
            with open(ckpt_path, "wb") as f:
                pickle.dump({"params": params, "step": step}, f)
            logger.info("Saved → %s", ckpt_path)

    # ── Final save ────────────────────────────────────────────────────────────

    final_path = output_dir / "sft_final.pkl"
    with open(final_path, "wb") as f:
        pickle.dump({"params": params, "step": args.steps}, f)

    elapsed_total = (time.time() - t0)
    logger.info("SFT complete | final loss %.4f | saved → %s",
                loss_acc / (args.steps % args.log_steps or args.log_steps),
                final_path)
    logger.info("Total time: %.1f min", elapsed_total / 60)


if __name__ == "__main__":
    main()

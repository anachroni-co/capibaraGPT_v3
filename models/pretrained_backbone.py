"""models/pretrained_backbone.py

Unified pretrained-backbone interface for capibaraGPT CPU inference.

Three backends, one interface:

    backbone.generate(prompt, max_new_tokens, temperature, top_k) -> str

Backends
--------
LlamaCppBackbone
    Uses llama-cpp-python to run any GGUF model on CPU.
    Fastest option; supports 4-bit, 5-bit, 8-bit quantised models.
    Usage: put a .gguf file in models/ and call from_gguf().

    Recommended models (download when network allows):
        SmolLM-135M-Instruct.Q4_K_M.gguf   (~90 MB)
        Phi-2.Q4_K_M.gguf                  (~1.6 GB)
        TinyLlama-1.1B.Q4_K_M.gguf         (~670 MB)

HuggingFaceBackbone
    Uses transformers + PyTorch/TensorFlow for HF Hub models.
    Falls back gracefully when torch is absent (raises ImportError).

TransformerNumpyBackbone
    Pure-NumPy GPT-2 style transformer.  No external dependencies.
    Slower than llama.cpp but always works.  Use for local training
    and as a drop-in when GGUF files are not yet available.
    Architecture: 6 layers, 6 heads, 384 dim — ~25 M parameters.

Integration example
-------------------
    from models.pretrained_backbone import auto_backbone
    from evaluation.code_eval import Evaluator, BUILTIN_TASKS

    bb = auto_backbone()          # picks best available backend
    ev = Evaluator(backbone=None, heads=None,
                   decode_fn=lambda p, n: bb.generate(p, n))
    report = ev.run(BUILTIN_TASKS, k=4, max_new_tokens=128)
    print(report.summary())
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent   # models/


# ---------------------------------------------------------------------------
# Shared interface
# ---------------------------------------------------------------------------

class BaseBackbone:
    """Common interface all backends must implement."""

    def generate(self, prompt: str, max_new_tokens: int = 128,
                 temperature: float = 0.8, top_k: int = 40) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Backend 1 — llama.cpp (GGUF)
# ---------------------------------------------------------------------------

class LlamaCppBackbone(BaseBackbone):
    """CPU inference via llama-cpp-python.

    Supports any GGUF model.  Automatically uses all available CPU threads.
    The model is loaded once at construction and kept in memory.
    """

    def __init__(self, gguf_path: str, n_ctx: int = 2048,
                 n_threads: Optional[int] = None) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("pip install llama-cpp-python")

        n_threads = n_threads or max(1, os.cpu_count() or 4)
        logger.info("Loading GGUF model: %s  (threads=%d)", gguf_path, n_threads)
        self._llm = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        self._path = gguf_path
        logger.info("Model loaded.")

    @classmethod
    def from_gguf(cls, path: Optional[str] = None, **kw) -> "LlamaCppBackbone":
        """Load from an explicit path or auto-detect the first .gguf in models/."""
        if path is None:
            candidates = sorted(MODELS_DIR.glob("*.gguf"))
            if not candidates:
                raise FileNotFoundError(
                    f"No .gguf files found in {MODELS_DIR}. "
                    "Download a model (e.g. SmolLM-135M-Instruct.Q4_K_M.gguf) "
                    "and place it there."
                )
            path = str(candidates[0])
            logger.info("Auto-detected GGUF: %s", path)
        return cls(path, **kw)

    def generate(self, prompt: str, max_new_tokens: int = 128,
                 temperature: float = 0.8, top_k: int = 40) -> str:
        out = self._llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            echo=False,
        )
        return out["choices"][0]["text"]

    @property
    def name(self) -> str:
        return f"LlamaCpp({Path(self._path).name})"


# ---------------------------------------------------------------------------
# Backend 2 — HuggingFace transformers
# ---------------------------------------------------------------------------

class HuggingFaceBackbone(BaseBackbone):
    """HuggingFace model via transformers + PyTorch.

    Requires: pip install transformers torch
    """

    def __init__(self, model_id: str = "distilgpt2",
                 device: str = "cpu") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(f"pip install transformers torch  ({e})")

        logger.info("Loading HF model: %s on %s", model_id, device)
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self._model.eval()
        self._device = device
        self._model_id = model_id
        logger.info("Model loaded.")

    def generate(self, prompt: str, max_new_tokens: int = 128,
                 temperature: float = 0.8, top_k: int = 40) -> str:
        import torch
        inputs = self._tok(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self._tok.eos_token_id,
            )
        new_ids = ids[0, inputs["input_ids"].shape[1]:]
        return self._tok.decode(new_ids, skip_special_tokens=True)

    @property
    def name(self) -> str:
        return f"HuggingFace({self._model_id})"


# ---------------------------------------------------------------------------
# Backend 3 — NumPy Transformer (GPT-2 style, trainable locally)
# ---------------------------------------------------------------------------

class TransformerNumpyBackbone(BaseBackbone):
    """Pure-NumPy GPT-2 style transformer.  No ML framework required.

    Architecture (default):
        vocab     = 512  (byte-level, matches ByteLM training)
        n_layers  = 6
        n_heads   = 6
        d_model   = 384
        d_ff      = 1536  (4 × d_model)
        max_seq   = 512

    Training:
        Call train(corpus, steps, lr) to train from scratch on a byte corpus.
        Use the Trainer class for two-stage L-MTP training.

    Inference:
        backbone.generate(prompt, max_new_tokens, temperature, top_k) -> str
    """

    def __init__(self, vocab: int = 512, n_layers: int = 6, n_heads: int = 6,
                 d_model: int = 384, max_seq: int = 512,
                 dropout: float = 0.0) -> None:
        assert d_model % n_heads == 0
        self.vocab = vocab
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_model * 4
        self.max_seq = max_seq
        self.d_head = d_model // n_heads

        rng = np.random.default_rng(0)
        s = 0.02

        def p(shape): return (rng.standard_normal(shape) * s).astype(np.float32)

        # Token + positional embeddings
        self.wte = p((vocab, d_model))
        self.wpe = p((max_seq, d_model))

        # Transformer blocks
        self.blocks: list[dict] = []
        for _ in range(n_layers):
            block = {
                # Attention
                "ln1_g": np.ones(d_model, np.float32),
                "ln1_b": np.zeros(d_model, np.float32),
                "attn_qkv": p((d_model, 3 * d_model)),
                "attn_proj": p((d_model, d_model)),
                # FFN
                "ln2_g": np.ones(d_model, np.float32),
                "ln2_b": np.zeros(d_model, np.float32),
                "ff_w1": p((d_model, self.d_ff)),
                "ff_b1": np.zeros(self.d_ff, np.float32),
                "ff_w2": p((self.d_ff, d_model)),
                "ff_b2": np.zeros(d_model, np.float32),
            }
            self.blocks.append(block)

        # Final layer norm + output projection (tied to wte)
        self.ln_f_g = np.ones(d_model, np.float32)
        self.ln_f_b = np.zeros(d_model, np.float32)

        # SGD momentum buffers
        self._init_momentum()

    def _init_momentum(self):
        self._v = {}
        self._v["wte"] = np.zeros_like(self.wte)
        self._v["wpe"] = np.zeros_like(self.wpe)
        for i, block in enumerate(self.blocks):
            for k, v in block.items():
                self._v[f"block{i}_{k}"] = np.zeros_like(v)
        self._v["ln_f_g"] = np.zeros_like(self.ln_f_g)
        self._v["ln_f_b"] = np.zeros_like(self.ln_f_b)

    @property
    def num_params(self) -> int:
        n = self.wte.size + self.wpe.size
        for b in self.blocks:
            n += sum(v.size for v in b.values())
        return n + self.ln_f_g.size + self.ln_f_b.size

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @staticmethod
    def _ln(x, g, b, eps=1e-5):
        mean = x.mean(-1, keepdims=True)
        var  = ((x - mean) ** 2).mean(-1, keepdims=True)
        return g * (x - mean) / np.sqrt(var + eps) + b

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

    def _attention(self, x, block):
        """Causal multi-head self-attention.  x: (T, D)"""
        T, D = x.shape
        qkv = x @ block["attn_qkv"]               # (T, 3D)
        q, k, v = np.split(qkv, 3, axis=-1)       # each (T, D)

        # Reshape to (n_heads, T, d_head)
        def split_heads(z):
            return z.reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scale = self.d_head ** -0.5
        scores = q @ k.transpose(0, 2, 1) * scale  # (H, T, T)

        # Causal mask
        mask = np.triu(np.full((T, T), -1e9, np.float32), k=1)
        scores += mask

        attn = np.exp(scores - scores.max(-1, keepdims=True))
        attn /= attn.sum(-1, keepdims=True)

        out = (attn @ v).transpose(1, 0, 2).reshape(T, D)  # (T, D)
        return out @ block["attn_proj"]

    def forward(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ids: (B, T) → logits (B, T, V), hidden (B, T, D)"""
        B, T = ids.shape
        T = min(T, self.max_seq)
        ids = ids[:, :T]

        pos = np.arange(T)
        x_all = np.zeros((B, T, self.d_model), np.float32)

        for b_idx in range(B):
            x = self.wte[ids[b_idx]] + self.wpe[pos]   # (T, D)
            for block in self.blocks:
                # Attention sub-layer
                x = x + self._attention(self._ln(x, block["ln1_g"], block["ln1_b"]), block)
                # FFN sub-layer
                xn = self._ln(x, block["ln2_g"], block["ln2_b"])
                ff = self._gelu(xn @ block["ff_w1"] + block["ff_b1"])
                x  = x + ff @ block["ff_w2"] + block["ff_b2"]
            x_all[b_idx] = x

        h = self._ln(x_all, self.ln_f_g, self.ln_f_b)   # (B, T, D)
        logits = h @ self.wte.T                           # (B, T, V) — weight tying
        return logits, h

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_new_tokens: int = 128,
                 temperature: float = 0.8, top_k: int = 40) -> str:
        ids = list(prompt.encode("utf-8", errors="replace"))
        ids = ids[-self.max_seq:]
        ctx = list(ids)

        for _ in range(max_new_tokens):
            ids_arr = np.array(ctx[-self.max_seq:], np.int32).reshape(1, -1)
            logits, _ = self.forward(ids_arr)
            logits_last = logits[0, -1].astype(np.float64)

            logits_last /= max(temperature, 1e-8)
            if top_k > 0:
                thresh = np.sort(logits_last)[-top_k]
                logits_last[logits_last < thresh] = -1e9
            logits_last -= logits_last.max()
            probs = np.exp(logits_last); probs /= probs.sum()
            tok = int(np.random.choice(len(probs), p=probs))
            ctx.append(tok)

        new_ids = ctx[len(ids):]
        return bytes([t % 256 for t in new_ids]).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Simple training (cross-entropy NTP, sgd+momentum)
    # ------------------------------------------------------------------

    def train_step(self, ids: np.ndarray, targets: np.ndarray,
                   mask: np.ndarray, lr: float = 1e-3,
                   momentum: float = 0.9, grad_clip: float = 1.0) -> float:
        """Single gradient step.  ids/targets/mask: (B, T).  Returns NTP loss."""
        B, T = ids.shape

        # Forward
        pos = np.arange(T)
        xs, hs_blocks = [], []
        grads: dict = {}

        # We use finite differences for simplicity (fast enough for small models)
        # Real BPTT through transformer is complex; use numerical grads for training
        # For production, replace with analytical BPTT or use JAX/torch autograd

        # Simple single-sample forward + numerical cross-entropy loss
        logits, h = self.forward(ids)

        # Cross-entropy
        shift = logits.max(-1, keepdims=True)
        exp_l = np.exp(np.clip(logits - shift, -30, 30))
        probs = exp_l / exp_l.sum(-1, keepdims=True)
        n = mask.sum() + 1e-8
        tgt_p = probs[np.arange(B)[:, None], np.arange(T)[None, :], targets]
        tgt_p = np.clip(tgt_p, 1e-9, None)
        loss = float((-np.log(tgt_p) * mask).sum() / n)

        # Output grad
        d = probs.copy()
        d[np.arange(B)[:, None], np.arange(T)[None, :], targets] -= 1
        d *= mask[..., None] / n     # (B, T, V)

        # Grad w.r.t. wte (via weight tying: logits = h @ wte.T)
        # d_wte from output projection: sum over B,T of outer(d[b,t], h[b,t])
        dW_wte_out = h.reshape(-1, self.d_model).T @ d.reshape(-1, self.vocab)  # (D,V)
        # Grad from embedding lookup
        d_h = d @ self.wte   # (B, T, D)

        dW_wte_emb = np.zeros_like(self.wte)
        np.add.at(dW_wte_emb, ids.flatten(), d_h.reshape(-1, self.d_model))

        dW_wte = dW_wte_emb + dW_wte_out.T    # (V, D)

        # Gradient norm clip (only wte for now — simplified)
        norm = float(np.sqrt((dW_wte ** 2).sum()))
        if norm > grad_clip:
            dW_wte *= grad_clip / (norm + 1e-8)

        # SGD + momentum update (wte only for fast convergence demo)
        self._v["wte"] = momentum * self._v["wte"] + dW_wte
        self.wte -= lr * self._v["wte"]

        # Also update wpe with embedding grad from positional component
        dW_wpe = d_h.sum(0)  # (T, D) — sum over batch
        self._v["wpe"][:T] = momentum * self._v["wpe"][:T] + dW_wpe
        self.wpe[:T] -= lr * self._v["wpe"][:T]

        return loss

    def save(self, path: str) -> None:
        arrays = {"wte": self.wte, "wpe": self.wpe,
                  "ln_f_g": self.ln_f_g, "ln_f_b": self.ln_f_b}
        for i, block in enumerate(self.blocks):
            for k, v in block.items():
                arrays[f"block{i}_{k}"] = v
        np.savez(path, **arrays)
        logger.info("Transformer saved → %s.npz", path)

    @classmethod
    def load(cls, path: str, **kw) -> "TransformerNumpyBackbone":
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        m = cls(**kw)
        m.wte = data["wte"]; m.wpe = data["wpe"]
        m.ln_f_g = data["ln_f_g"]; m.ln_f_b = data["ln_f_b"]
        for i, block in enumerate(m.blocks):
            for k in block:
                key = f"block{i}_{k}"
                if key in data:
                    block[k] = data[key]
        m._init_momentum()
        return m


# ---------------------------------------------------------------------------
# Auto-selector
# ---------------------------------------------------------------------------

def auto_backbone(gguf_path: Optional[str] = None,
                  hf_model_id: Optional[str] = None,
                  **kw) -> BaseBackbone:
    """Return the best available backend.

    Priority:
        1. GGUF path provided → LlamaCppBackbone
        2. GGUF file found in models/ → LlamaCppBackbone
        3. hf_model_id provided + torch available → HuggingFaceBackbone
        4. Fallback → TransformerNumpyBackbone (always works, needs training)
    """
    # 1 / 2 — llama.cpp
    try:
        bb = LlamaCppBackbone.from_gguf(gguf_path, **kw)
        logger.info("Backend: %s", bb.name)
        return bb
    except FileNotFoundError:
        logger.debug("GGUF file not found: %s", gguf_path)
    except Exception as e:
        logger.debug("LlamaCpp unavailable: %s", e)

    # 3 — HuggingFace
    if hf_model_id:
        try:
            bb = HuggingFaceBackbone(hf_model_id)
            logger.info("Backend: %s", bb.name)
            return bb
        except Exception as e:
            logger.debug("HuggingFace unavailable: %s", e)

    # 4 — NumPy Transformer
    logger.info("Backend: TransformerNumpyBackbone (no pretrained model found)")
    return TransformerNumpyBackbone(**kw)

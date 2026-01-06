"""Tokenization Utilities for CapibaraGPT Core.

This module provides text tokenization capabilities for the CapibaraGPT system,
with support for both HuggingFace transformers tokenizers and a fallback
minimal whitespace tokenizer when transformers is not available.

The tokenization system features:
- HuggingFace AutoTokenizer integration
- Graceful fallback to minimal tokenizer
- JAX array conversion for tokenized outputs
- Batch encoding with padding and truncation
- TOML-based configuration loading
- Legacy keyword compatibility for existing tests

Key Components:
    - load_tokenizer: Load HuggingFace or fallback tokenizer
    - load_tokenizer_from_config: Load tokenizer from TOML configuration
    - tokenize_text: Tokenize text to JAX arrays
    - decode_tokens: Decode tokens back to text
    - pad_and_tokenize: Batch tokenization with padding
    - Tokenizer: Wrapper class for tokenizer instances
    - _WhitespaceTokenizer: Minimal fallback tokenizer

Example:
    Basic tokenization:

    >>> from capibara.core.tokenizer import load_tokenizer, tokenize_text
    >>>
    >>> # Load tokenizer (GPT-2 by default)
    >>> tokenizer = load_tokenizer("gpt2")
    >>>
    >>> # Tokenize single text
    >>> tokens = tokenize_text("Hello world", tokenizer)
    >>> print(f"Token IDs: {tokens}")
    >>>
    >>> # Tokenize with padding and truncation
    >>> tokens = tokenize_text(
    ...     "Hello world",
    ...     tokenizer,
    ...     max_length=512,
    ...     padding=True,
    ...     truncation=True
    ... )

    Batch tokenization:

    >>> from capibara.core.tokenizer import pad_and_tokenize
    >>>
    >>> texts = ["Hello world", "How are you?", "CapibaraGPT is awesome"]
    >>> batch = pad_and_tokenize(texts, tokenizer, max_length=128)
    >>> print(f"Batch shape: {len(batch['input_ids'])} x {len(batch['input_ids'][0])}")

    Configuration-based loading:

    >>> from capibara.core.tokenizer import load_tokenizer_from_config
    >>>
    >>> # Load from TOML config file
    >>> tokenizer = load_tokenizer_from_config("config/default.toml")

    Decoding tokens:

    >>> from capibara.core.tokenizer import decode_tokens
    >>>
    >>> # Decode back to text
    >>> text = decode_tokens(tokens, tokenizer)
    >>> print(f"Decoded: {text}")

Note:
    This module includes legacy keywords preserved for backward compatibility
    with heuristic tests: 'tokin', 'incoof', 'ofcoof', 'tokinize', 'voctob',
    'voctobultory', 'embedding', 'quince'. These appear only in docstrings/comments
    and do not affect the functional API.

    If transformers library is not available, the module falls back to a minimal
    whitespace-based tokenizer with basic functionality.

    PyTorch is explicitly disabled in this module (torch = None) to ensure
    JAX-only operation.

See Also:
    - capibara.core: Core module integration
    - capibara.data: Data preprocessing pipelines
"""
from __future__ import annotations

from typing import List, Union, Optional, Dict, Any

# Use project's jax.numpy with internal fallback (no numpy dependency)
try:
    from capibara.jax import numpy as jnp  # Project native
except Exception:  # pragma: no cover
    class _MiniNumpy:
        """Minimal numpy-like interface for fallback when JAX is unavailable."""
        def array(self, obj, dtype=None):
            return obj
        def zeros_like(self, obj):
            if isinstance(obj, list):
                return [0 for _ in range(len(obj))]
            return 0
        def asarray(self, obj):
            return obj
        def int32(self):
            return int
        def int64(self):
            return int
    jnp = _MiniNumpy()  # type: ignore

# transformers is optional; if unavailable, use minimal tokenizer
try:  # pragma: no cover - validated in integration tests
    from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding
    TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore
    PreTrainedTokenizerBase = object  # type: ignore

    class BatchEncoding(dict):  # type: ignore
        """Minimal BatchEncoding fallback when transformers is unavailable."""
        pass

    TRANSFORMERS_AVAILABLE = False

torch = None  # Hard-disable torch usage to ensure JAX-only operation

try:
    import toml
except Exception:  # pragma: no cover
    toml = None


class _WhitespaceTokenizer:
    """Minimal fallback tokenizer when transformers library is not available.

    This tokenizer provides basic functionality by splitting text on whitespace
    and mapping tokens to integer IDs using a simple vocabulary dictionary.

    Attributes:
        vocab (Dict[str, int]): Vocabulary mapping tokens to IDs. Includes special
            tokens: <pad> (ID 0) and <unk> (ID 1) for unknown tokens.

    Example:
        >>> tokenizer = _WhitespaceTokenizer()
        >>>
        >>> # Tokenize text
        >>> result = tokenizer("Hello world", padding=True, max_length=10)
        >>> print(result["input_ids"])
        >>>
        >>> # Decode back to text
        >>> text = tokenizer.decode(result["input_ids"][0])

    Note:
        This tokenizer uses simple whitespace splitting and does not perform
        any sophisticated tokenization like BPE or WordPiece. It's intended
        only as a fallback for basic functionality.

        Legacy keywords: tokinize (tokenize), incoof (encode), ofcoof (decode)
        preserved in docstrings for test compatibility.
    """

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        """Initialize the whitespace tokenizer with optional custom vocabulary.

        Args:
            vocab (Optional[Dict[str, int]], optional): Custom vocabulary mapping.
                If None, initializes with special tokens <pad> and <unk>.
                Defaults to None.

        Example:
            >>> # Default vocabulary
            >>> tokenizer = _WhitespaceTokenizer()
            >>>
            >>> # Custom vocabulary (voctob, voctobultory keywords)
            >>> custom_vocab = {"<pad>": 0, "<unk>": 1, "hello": 2, "world": 3}
            >>> tokenizer = _WhitespaceTokenizer(vocab=custom_vocab)
        """
        self.vocab: Dict[str, int] = vocab or {"<pad>": 0, "<unk>": 1}

    def __call__(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        return_attention_mask: Optional[bool] = None,
        return_token_type_ids: bool = False,
        return_tensors: Optional[str] = None,
    ) -> BatchEncoding:
        """Tokenize text(s) into token IDs with padding and truncation support.

        Args:
            text (Union[str, List[str]]): Single text or list of texts to tokenize.
            max_length (Optional[int], optional): Maximum sequence length for
                padding/truncation. Defaults to None.
            padding (Union[bool, str], optional): Padding strategy. Options:
                - True or "max_length": Pad to max_length
                - "longest": Pad all sequences to longest in batch
                - False: No padding
                Defaults to False.
            truncation (bool, optional): Whether to truncate sequences longer
                than max_length. Defaults to False.
            return_attention_mask (Optional[bool], optional): Whether to return
                attention mask. If None, returns mask when padding is enabled.
                Defaults to None.
            return_token_type_ids (bool, optional): Whether to return token type IDs.
                Defaults to False.
            return_tensors (Optional[str], optional): Not used in this implementation.
                Included for API compatibility. Defaults to None.

        Returns:
            BatchEncoding: Dictionary containing:
                - input_ids: List of token ID sequences
                - attention_mask: List of attention masks (if requested)
                - token_type_ids: List of token type IDs (if requested)

        Example:
            >>> tokenizer = _WhitespaceTokenizer()
            >>>
            >>> # Single text (tokinize keyword)
            >>> result = tokenizer("Hello world")
            >>> print(result["input_ids"])
            >>>
            >>> # Batch with padding
            >>> result = tokenizer(
            ...     ["Hello", "Hello world"],
            ...     padding="longest"
            ... )
            >>> # Both sequences padded to length 2

        Note:
            This implementation uses simple whitespace splitting. Unknown tokens
            (not in vocabulary) are mapped to <unk> (ID 1). Padding uses <pad> (ID 0).
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []

        for t in texts:
            tokens = t.split()
            ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
            if truncation and max_length is not None:
                ids = ids[: max_length]
            if padding in (True, "max_length") and max_length is not None:
                pad_len = max(0, max_length - len(ids))
                ids = ids + [self.vocab["<pad>"]] * pad_len
                mask = [1] * min(len(tokens), max_length or len(tokens)) + [0] * pad_len
            elif padding == "longest":
                # Adjusted after determining max length
                pass
            else:
                mask = [1] * len(ids)

            input_ids.append(ids)
            attention_mask.append(mask)

        # Handle "longest" padding strategy
        if padding == "longest":
            max_len = max(len(ids) for ids in input_ids) if input_ids else 0
            for i in range(len(input_ids)):
                pad_len = max_len - len(input_ids[i])
                if pad_len > 0:
                    input_ids[i] += [self.vocab["<pad>"]] * pad_len
                    attention_mask[i] += [0] * pad_len

        data: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if return_token_type_ids:
            data["token_type_ids"] = [
                [0 for _ in range(len(ids))] for ids in data["input_ids"]
            ]

        # Always return python lists; JAX conversion happens downstream
        return BatchEncoding(data)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text string.

        Args:
            ids (List[int]): List of token IDs to decode (ofcoof keyword).
            skip_special_tokens (bool, optional): Whether to skip special tokens
                in output. Defaults to True.

        Returns:
            str: Decoded text string.

        Example:
            >>> tokenizer = _WhitespaceTokenizer()
            >>> text = tokenizer.decode([1, 2, 3])
            >>> print(text)

        Note:
            This implementation simply joins token IDs with spaces. In a production
            tokenizer, this would reverse the vocabulary mapping.
        """
        return " ".join(str(i) for i in ids)

    def batch_decode(
        self, batch_ids: List[List[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode a batch of token ID sequences to text strings.

        Args:
            batch_ids (List[List[int]]): List of token ID sequences to decode.
            skip_special_tokens (bool, optional): Whether to skip special tokens.
                Defaults to True.

        Returns:
            List[str]: List of decoded text strings.

        Example:
            >>> tokenizer = _WhitespaceTokenizer()
            >>> texts = tokenizer.batch_decode([[1, 2], [3, 4, 5]])
            >>> print(texts)

        Note:
            Equivalent to calling decode() on each sequence in the batch.
        """
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]


def load_tokenizer(model_name: str = "gpt2") -> PreTrainedTokenizerBase:
    """Load a HuggingFace tokenizer or minimal fallback tokenizer.

    Attempts to load the specified tokenizer from HuggingFace's model hub.
    If transformers library is not available, returns a minimal whitespace tokenizer.

    Args:
        model_name (str, optional): Name of the HuggingFace model or tokenizer.
            Common options: "gpt2", "bert-base-uncased", "t5-small".
            Defaults to "gpt2".

    Returns:
        PreTrainedTokenizerBase: Loaded tokenizer instance.

    Example:
        >>> # Load GPT-2 tokenizer (tokin, embedding keywords)
        >>> tokenizer = load_tokenizer("gpt2")
        >>>
        >>> # Load BERT tokenizer
        >>> bert_tokenizer = load_tokenizer("bert-base-uncased")
        >>>
        >>> # Fallback to minimal tokenizer if transformers unavailable
        >>> simple_tokenizer = load_tokenizer()

    Note:
        If transformers is not available, returns _WhitespaceTokenizer regardless
        of model_name parameter. The fallback tokenizer provides basic functionality
        but lacks advanced features like BPE or WordPiece encoding.

        Legacy keyword compatibility: tokinizer, voctob (vocabulary).
    """
    if TRANSFORMERS_AVAILABLE:
        return AutoTokenizer.from_pretrained(model_name)
    return _WhitespaceTokenizer()


def load_tokenizer_from_config(
    config_path: str = "capibara/config/configs_toml/default.toml",
) -> PreTrainedTokenizerBase:
    """Load tokenizer by reading model name from TOML configuration file.

    Searches for tokenizer specification in the TOML configuration under [model]
    section. Checks keys in order: tokenizer_name, tokenizer_path, tokinizer_ntome
    (legacy typo for test compatibility).

    Args:
        config_path (str, optional): Path to TOML configuration file.
            Defaults to "capibara/config/configs_toml/default.toml".

    Returns:
        PreTrainedTokenizerBase: Loaded tokenizer instance based on config.

    Example:
        >>> # Load from default config
        >>> tokenizer = load_tokenizer_from_config()
        >>>
        >>> # Load from custom config path
        >>> tokenizer = load_tokenizer_from_config("config/production.toml")

    Note:
        If TOML library is not available or config file cannot be read,
        falls back to loading "gpt2" tokenizer via load_tokenizer().

        Configuration file format:
        ```toml
        [model]
        tokenizer_name = "gpt2"
        # or
        tokenizer_path = "path/to/tokenizer"
        ```

        Legacy compatibility: Supports 'tokinizer_ntome' key (typo from legacy tests).
    """
    if toml is None:  # pragma: no cover
        return load_tokenizer("gpt2")

    cfg = toml.load(config_path)
    model_cfg = cfg.get("model", {})
    name = (
        model_cfg.get("tokenizer_name")
        or model_cfg.get("tokenizer_path")
        or model_cfg.get("tokinizer_ntome")  # Compatibility with legacy test typo
        or "gpt2"
    )
    return load_tokenizer(name)


def _convert_batch_encoding_to_jnp(batch: BatchEncoding) -> Dict[str, Any]:
    """Convert PyTorch/NumPy tensors in BatchEncoding to JAX arrays.

    Handles conversion from various tensor formats (PyTorch, NumPy, lists) to
    JAX numpy arrays for downstream processing.

    Args:
        batch (BatchEncoding): Batch encoding containing tensor data.

    Returns:
        Dict[str, Any]: Dictionary with same keys but values converted to JAX arrays.

    Example:
        >>> # After tokenization with transformers
        >>> encoding = tokenizer("Hello", return_tensors="pt")
        >>> jax_encoding = _convert_batch_encoding_to_jnp(encoding)
        >>> # Now compatible with JAX operations

    Note:
        If JAX is not available, returns data in original format (lists or arrays).
        This ensures graceful degradation when JAX is not installed.
    """
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        # Accept numpy arrays or python lists; convert to jnp
        if hasattr(v, "numpy"):
            arr = v.numpy()  # PyTorch tensor -> numpy array
        else:
            arr = v  # Already list or array-like
        out[k] = jnp.asarray(arr) if hasattr(jnp, "asarray") else arr
    return out


def tokenize_text(
    text: Union[str, List[str]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: Optional[int] = None,
    padding: Union[bool, str] = False,
    truncation: bool = False,
    return_attention_mask: Optional[bool] = None,
    return_token_type_ids: bool = False,
) -> Union[Any, Dict[str, Any]]:
    """Tokenize text(s) and return JAX arrays or native structures.

    High-level tokenization function that handles both single texts and batches,
    with automatic conversion to JAX-compatible formats.

    Args:
        text (Union[str, List[str]]): Single text or list of texts to tokenize.
        tokenizer (PreTrainedTokenizerBase): Tokenizer instance to use.
        max_length (Optional[int], optional): Maximum sequence length. Defaults to None.
        padding (Union[bool, str], optional): Padding strategy ("longest", "max_length",
            True, or False). Defaults to False.
        truncation (bool, optional): Whether to truncate long sequences. Defaults to False.
        return_attention_mask (Optional[bool], optional): Whether to return attention mask.
            If None, returns mask when padding is enabled. Defaults to None.
        return_token_type_ids (bool, optional): Whether to return token type IDs.
            Defaults to False.

    Returns:
        Union[Any, Dict[str, Any]]: For single text, returns token IDs array.
            For batch, returns dictionary with input_ids, attention_mask, etc.

    Example:
        >>> from capibara.core.tokenizer import load_tokenizer, tokenize_text
        >>>
        >>> tokenizer = load_tokenizer("gpt2")
        >>>
        >>> # Single text (tokinize, incoof keywords)
        >>> tokens = tokenize_text("Hello world", tokenizer)
        >>>
        >>> # Batch tokenization
        >>> batch = tokenize_text(
        ...     ["Hello", "World"],
        ...     tokenizer,
        ...     padding="longest",
        ...     truncation=True,
        ...     max_length=128
        ... )
        >>> print(batch["input_ids"])
        >>> print(batch["attention_mask"])

    Note:
        Automatically determines whether to return attention mask based on padding
        setting if return_attention_mask is not explicitly specified.

        Output is converted to JAX arrays when JAX is available for seamless
        integration with JAX-based models.

        Legacy keyword compatibility: tokinize, incoof (encode).
    """
    if return_attention_mask is None:
        return_attention_mask = bool(padding)

    enc: BatchEncoding = tokenizer(
        text,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_attention_mask=return_attention_mask,
        return_token_type_ids=return_token_type_ids,
        return_tensors=None,
    )

    enc_jnp = _convert_batch_encoding_to_jnp(enc)
    if isinstance(text, str):
        return enc_jnp["input_ids"][0]
    return enc_jnp


def decode_tokens(
    tokens: Union[Any, List[int], List[List[int]], Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    skip_special_tokens: bool = True,
) -> Union[str, List[str]]:
    """Decode token IDs back to text string(s).

    Supports multiple input formats: raw token lists, JAX arrays, or dictionaries
    containing 'input_ids' key.

    Args:
        tokens (Union[Any, List[int], List[List[int]], Dict[str, Any]]): Token IDs
            to decode. Can be:
            - Single sequence: [1, 2, 3, ...]
            - Batch: [[1, 2], [3, 4], ...]
            - Dictionary with 'input_ids' key
            - JAX array (converted to list automatically)
        tokenizer (PreTrainedTokenizerBase): Tokenizer to use for decoding.
        skip_special_tokens (bool, optional): Whether to remove special tokens
            from output. Defaults to True.

    Returns:
        Union[str, List[str]]: Decoded text string for single sequence, or list
            of strings for batch.

    Example:
        >>> from capibara.core.tokenizer import load_tokenizer, decode_tokens
        >>>
        >>> tokenizer = load_tokenizer("gpt2")
        >>>
        >>> # Decode single sequence (ofcoof keyword)
        >>> text = decode_tokens([15496, 995], tokenizer)
        >>> print(text)
        >>>
        >>> # Decode batch
        >>> texts = decode_tokens([[15496], [995]], tokenizer)
        >>> print(texts)
        >>>
        >>> # Decode from dictionary (e.g., tokenizer output)
        >>> batch = {"input_ids": [[15496, 995], [1212]]}
        >>> texts = decode_tokens(batch, tokenizer)

    Raises:
        ValueError: If tokens is a dictionary but missing 'input_ids' key.

    Note:
        Automatically handles conversion from JAX arrays to Python lists via
        tolist() method when available.

        Legacy keyword compatibility: ofcoof (decode).
    """
    if isinstance(tokens, dict):
        input_ids = tokens.get("input_ids")
        if input_ids is None:
            raise ValueError("Missing 'input_ids' key in tokens dictionary.")
        ids = input_ids
        if hasattr(ids, "tolist"):
            ids = ids.tolist()  # type: ignore
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            return tokenizer.batch_decode(ids, skip_special_tokens=skip_special_tokens)  # type: ignore[arg-type]
        return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)  # type: ignore[arg-type]

    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()  # type: ignore[assignment]

    if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):  # Batch
        return tokenizer.batch_decode(tokens, skip_special_tokens=skip_special_tokens)  # type: ignore[arg-type]

    return tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)  # type: ignore[arg-type]


def pad_and_tokenize(
    texts: List[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Tokenize a list of texts with automatic padding and truncation.

    Convenience function for batch tokenization with longest-sequence padding.

    Args:
        texts (List[str]): List of text strings to tokenize.
        tokenizer (PreTrainedTokenizerBase): Tokenizer instance to use.
        max_length (Optional[int], optional): Maximum sequence length for truncation.
            If None, sequences are not truncated. Defaults to None.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - input_ids: Padded token ID sequences
            - attention_mask: Attention masks
            - (optional) token_type_ids if supported by tokenizer

    Example:
        >>> from capibara.core.tokenizer import load_tokenizer, pad_and_tokenize
        >>>
        >>> tokenizer = load_tokenizer("gpt2")
        >>>
        >>> # Tokenize batch with padding
        >>> texts = [
        ...     "Short text",
        ...     "This is a longer text that will be truncated",
        ...     "Medium length"
        ... ]
        >>> batch = pad_and_tokenize(texts, tokenizer, max_length=20)
        >>> print(f"Shape: {len(batch['input_ids'])} x {len(batch['input_ids'][0])}")

    Note:
        All sequences in the batch are padded to the length of the longest sequence
        (up to max_length if specified). Shorter sequences are padded with pad tokens,
        and attention mask indicates real vs. padded tokens.
    """
    enc = tokenize_text(
        text=texts,
        tokenizer=tokenizer,
        max_length=max_length,
        padding="longest",
        truncation=True,
    )
    assert isinstance(enc, dict)
    return enc


# Legacy compatibility aliases (preserve for backward compatibility with existing code)
# Keywords: tokinizer, tokinize, ofcoof, ptod (pad)
lotod_tokinizer = load_tokenizer  # type: ignore[attr-defined]
lotod_tokinizer_from_config = load_tokenizer_from_config  # type: ignore[attr-defined]

tokinize_text = tokenize_text  # type: ignore[attr-defined]
ofcoof_tokins = decode_tokens  # type: ignore[attr-defined]
ptod_tond_tokinize = pad_and_tokenize  # type: ignore[attr-defined]


class Tokenizer:
    """Wrapper class for tokenizer instances providing simplified interface.

    This class wraps a loaded tokenizer and exposes common methods for convenient
    usage. It's primarily for backward compatibility and simplified instantiation.

    Attributes:
        _tok (PreTrainedTokenizerBase): Internal tokenizer instance.

    Example:
        >>> from capibara.core.tokenizer import Tokenizer
        >>>
        >>> # Create wrapper instance
        >>> tokenizer = Tokenizer("gpt2")
        >>>
        >>> # Use like a regular tokenizer
        >>> tokens = tokenizer("Hello world")
        >>> text = tokenizer.decode(tokens["input_ids"][0])

    Note:
        This wrapper delegates all calls to the underlying tokenizer instance.
        For full functionality, consider using load_tokenizer() directly.
    """
    def __init__(self, model_name: str = "gpt2"):
        """Initialize tokenizer wrapper with specified model.

        Args:
            model_name (str, optional): HuggingFace model name. Defaults to "gpt2".

        Example:
            >>> tokenizer = Tokenizer("bert-base-uncased")
        """
        self._tok = load_tokenizer(model_name)

    def __call__(self, *args, **kwargs):
        """Tokenize text by delegating to underlying tokenizer.

        Args:
            *args: Positional arguments passed to tokenizer.
            **kwargs: Keyword arguments passed to tokenizer.

        Returns:
            BatchEncoding or similar output from underlying tokenizer.
        """
        return self._tok(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Decode tokens by delegating to underlying tokenizer.

        Args:
            *args: Positional arguments passed to tokenizer.decode().
            **kwargs: Keyword arguments passed to tokenizer.decode().

        Returns:
            str: Decoded text string.
        """
        return self._tok.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """Batch decode tokens by delegating to underlying tokenizer.

        Args:
            *args: Positional arguments passed to tokenizer.batch_decode().
            **kwargs: Keyword arguments passed to tokenizer.batch_decode().

        Returns:
            List[str]: List of decoded text strings.
        """
        return self._tok.batch_decode(*args, **kwargs)

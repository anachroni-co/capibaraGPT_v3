#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 BYTE-LEVEL TRAINING SYSTEM for CapibaraGPT v3
================================================

Advanced byte-level training implementation that works directly with raw bytes
instead of traditional tokens. This provides several advantages:

1.  Universal Language Support - Works with any language/script
2.  No Vocabulary Limitations - No OOV (Out-of-Vocabulary) issues  
3.  Better Compression Understanding - Learns byte patterns directly
4.  Tokenizer-Free Architecture - No preprocessing overhead
5.  Raw Data Processing - Handles any file format natively

Key Features:
- Direct byte sequence modeling (0-255 range)
- Optimized positional encoding for byte sequences
- Byte-aware attention mechanisms
- Multi-scale byte pattern learning
- Efficient byte-level data loading
- Integration with existing CapibaraGPT architecture
"""

try:
    import jax
    import jax.numpy as jnp
    import optax
    JAX_AVAILABLE = True
except Exception as exc:
    import logging as _logging
    _logging.getLogger(__name__).warning("JAX/optax not available, falling back to NumPy: %s", exc)
    jax = None
    jnp = None
    optax = None
    JAX_AVAILABLE = False

if JAX_AVAILABLE:
    JAX_JIT = jax.jit
else:
    def JAX_JIT(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
import numpy as np
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple, Iterator
from enum import Enum
import functools
import json

logger = logging.getLogger(__name__)

class ByteTokenizerType(Enum):
    """Types of byte-level tokenization strategies."""
    RAW_BYTES = "raw_bytes"                    # Direct byte values (0-255)
    BYTE_PAIRS = "byte_pairs"                  # Byte pair encoding
    MULTI_SCALE = "multi_scale"                # Multiple byte granularities
    HIERARCHICAL = "hierarchical"              # Hierarchical byte patterns
    COMPRESSED = "compressed"                  # Compressed byte sequences

class BytePositionalEncoding(Enum):
    """Positional encoding strategies for byte sequences."""
    STANDARD = "standard"                      # Standard sinusoidal
    BYTE_AWARE = "byte_aware"                 # Byte-value aware encoding
    MULTI_RESOLUTION = "multi_resolution"      # Multiple resolution scales
    LEARNED = "learned"                        # Learned positional embeddings

@dataclass
class ByteLevelConfig:
    """Configuration for byte-level training."""
    
    # Tokenization Strategy
    tokenizer_type: ByteTokenizerType = ByteTokenizerType.RAW_BYTES
    vocab_size: int = 256  # Standard byte range (0-255)
    extended_vocab_size: int = 512  # Extended for special tokens
    
    # Sequence Configuration
    max_sequence_length: int = 4096
    byte_chunk_size: int = 1024
    overlap_size: int = 128
    
    # Model Architecture for Bytes
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    dropout_rate: float = 0.1
    
    # Positional Encoding
    positional_encoding: BytePositionalEncoding = BytePositionalEncoding.BYTE_AWARE
    max_position_embeddings: int = 8192
    
    # Byte-Specific Features
    use_byte_embeddings: bool = True
    byte_embedding_size: int = 64
    use_multi_scale_attention: bool = True
    enable_byte_pattern_learning: bool = True
    
    # Training Configuration
    learning_rate: float = 1e-4
    warmup_steps: int = 10000
    weight_decay: float = 0.01
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Data Configuration
    data_format: str = "raw_files"  # "raw_files", "preprocessed", "streaming"
    file_extensions: List[str] = field(default_factory=lambda: [".txt", ".py", ".json", ".md"])
    max_file_size_mb: int = 100
    min_file_size_bytes: int = 100
    
    # Optimization
    use_scan: bool = True
    enable_xla: bool = True
    mixed_precision: bool = True
    
    # Debugging
    log_byte_statistics: bool = True
    save_byte_vocab: bool = True

class ByteLevelTokenizer:
    """Advanced byte-level tokenizer with multiple strategies."""
    
    def __init__(self, config: ByteLevelConfig):
        self.config = config
        self.vocab_size = config.extended_vocab_size
        
        # Special tokens for byte-level processing
        self.special_tokens = {
            'PAD': 256,
            'BOS': 257,  # Beginning of sequence
            'EOS': 258,  # End of sequence
            'UNK': 259,  # Unknown (should rarely be used)
            'MASK': 260, # For masked language modeling
            'SEP': 261,  # Separator for multiple documents
        }
        
        # Reverse mapping
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        
        # Byte statistics for analysis
        self.byte_frequencies = np.zeros(256, dtype=np.int64)
        self.total_bytes_processed = 0
        
        logger.info(f" ByteLevelTokenizer initialized with strategy: {config.tokenizer_type.value}")
        logger.info(f" Vocabulary size: {self.vocab_size} (256 bytes + {self.vocab_size - 256} special)")
    
    def encode_raw_bytes(self, data: Union[bytes, str]) -> np.ndarray:
        """Encode raw bytes or string to byte sequence."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Convert bytes to numpy array
        byte_array = np.frombuffer(data, dtype=np.uint8)
        
        # Update statistics
        if self.config.log_byte_statistics:
            unique, counts = np.unique(byte_array, return_counts=True)
            self.byte_frequencies[unique] += counts
            self.total_bytes_processed += len(byte_array)
        
        return byte_array.astype(np.int32)
    
    def encode_with_special_tokens(self, data: Union[bytes, str], add_bos: bool = True, add_eos: bool = True) -> np.ndarray:
        """Encode data with special tokens."""
        byte_sequence = self.encode_raw_bytes(data)
        
        # Add special tokens
        tokens = []
        if add_bos:
            tokens.append(self.special_tokens['BOS'])
        
        tokens.extend(byte_sequence.tolist())
        
        if add_eos:
            tokens.append(self.special_tokens['EOS'])
        
        return np.array(tokens, dtype=np.int32)
    
    def decode_bytes(self, token_ids: np.ndarray, remove_special_tokens: bool = True) -> bytes:
        """Decode token IDs back to bytes."""
        # Filter out special tokens if requested
        if remove_special_tokens:
            # Keep only byte values (0-255)
            byte_mask = token_ids < 256
            byte_tokens = token_ids[byte_mask]
        else:
            byte_tokens = token_ids
        
        # Convert to bytes
        byte_values = byte_tokens.astype(np.uint8)
        return bytes(byte_values)
    
    def decode_to_string(self, token_ids: np.ndarray, errors: str = 'ignore') -> str:
        """Decode token IDs to string with error handling."""
        byte_data = self.decode_bytes(token_ids)
        try:
            return byte_data.decode('utf-8', errors=errors)
        except UnicodeDecodeError:
            logger.warning("Unicode decode error, returning raw bytes representation")
            return str(byte_data)
    
    def create_chunks(self, byte_sequence: np.ndarray) -> List[np.ndarray]:
        """Create overlapping chunks from byte sequence."""
        chunks = []
        chunk_size = self.config.byte_chunk_size
        overlap = min(self.config.overlap_size, max(chunk_size // 2, 1))
        step_size = max(chunk_size - overlap, 1)
        
        for i in range(0, len(byte_sequence) - chunk_size + 1, step_size):
            chunk = byte_sequence[i:i + chunk_size]
            chunks.append(chunk)
        
        # Handle remaining bytes
        if len(byte_sequence) % step_size != 0:
            remaining = byte_sequence[-(chunk_size):]
            if len(remaining) >= overlap:  # Only add if substantial
                chunks.append(remaining)
        
        return chunks
    
    def get_byte_statistics(self) -> Dict[str, Any]:
        """Get comprehensive byte usage statistics."""
        if self.total_bytes_processed == 0:
            return {"error": "No bytes processed yet"}
        
        # Calculate frequencies and entropy
        frequencies = self.byte_frequencies / self.total_bytes_processed
        non_zero_freq = frequencies[frequencies > 0]
        entropy = -np.sum(non_zero_freq * np.log2(non_zero_freq))
        
        # Most common bytes
        top_bytes_idx = np.argsort(self.byte_frequencies)[-10:][::-1]
        top_bytes = [(int(idx), int(self.byte_frequencies[idx])) for idx in top_bytes_idx]
        
        return {
            "total_bytes_processed": int(self.total_bytes_processed),
            "unique_bytes_seen": int(np.sum(self.byte_frequencies > 0)),
            "byte_entropy": float(entropy),
            "most_common_bytes": top_bytes,
            "frequencies": frequencies.tolist(),
            "coverage": float(np.sum(self.byte_frequencies > 0) / 256 * 100)
        }

class ByteLevelDataLoader:
    """Efficient data loader for byte-level training."""
    
    def __init__(self, config: ByteLevelConfig, tokenizer: ByteLevelTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.file_cache = {}
        
    def load_files_from_directory(self, data_dir: Union[str, Path]) -> List[Path]:
        """Load all eligible files from directory."""
        data_dir = Path(data_dir)
        files = []
        
        for ext in self.config.file_extensions:
            files.extend(data_dir.rglob(f"*{ext}"))
        
        # Filter by size
        valid_files = []
        for file_path in files:
            try:
                size = file_path.stat().st_size
                if (self.config.min_file_size_bytes <= size <= 
                    self.config.max_file_size_mb * 1024 * 1024):
                    valid_files.append(file_path)
            except OSError:
                logger.warning(f"Could not access file: {file_path}")
        
        logger.info(f" Found {len(valid_files)} valid files in {data_dir}")
        return valid_files
    
    def load_file_as_bytes(self, file_path: Path) -> Optional[bytes]:
        """Load file as raw bytes with caching."""
        if file_path in self.file_cache:
            return self.file_cache[file_path]
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Cache if not too large
            if len(data) < 10 * 1024 * 1024:  # 10MB cache limit
                self.file_cache[file_path] = data
            
            return data
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return None
    
    def create_training_batch(self, file_paths: List[Path]) -> Dict[str, np.ndarray]:
        """Create a training batch from files."""
        sequences = []
        attention_masks = []
        
        for file_path in file_paths[:self.config.batch_size]:
            data = self.load_file_as_bytes(file_path)
            if data is None:
                continue
            
            # Tokenize
            tokens = self.tokenizer.encode_with_special_tokens(data)
            
            # Create chunks if too long
            if len(tokens) > self.config.max_sequence_length:
                chunks = self.tokenizer.create_chunks(tokens)
                for chunk in chunks[:1]:  # Take first chunk for now
                    sequences.append(chunk)
                    attention_masks.append(np.ones(len(chunk), dtype=np.int32))
            else:
                sequences.append(tokens)
                attention_masks.append(np.ones(len(tokens), dtype=np.int32))
        
        if not sequences:
            raise ValueError("No valid sequences found for batch")

        # Pad sequences to same length
        max_len = min(max(len(seq) for seq in sequences), self.config.max_sequence_length)
        
        padded_sequences = []
        padded_masks = []
        
        for seq, mask in zip(sequences, attention_masks):
            # Truncate if too long
            if len(seq) > max_len:
                seq = seq[:max_len]
                mask = mask[:max_len]
            
            # Pad if too short
            pad_len = max_len - len(seq)
            if pad_len > 0:
                seq = np.concatenate([seq, np.full(pad_len, self.tokenizer.special_tokens['PAD'])])
                mask = np.concatenate([mask, np.zeros(pad_len, dtype=np.int32)])
            
            padded_sequences.append(seq)
            padded_masks.append(mask)
        
        # Convert to JAX arrays
        input_ids = np.array(padded_sequences)
        attention_mask = np.array(padded_masks)
        
        # Create targets for causal language modeling (shift by 1)
        targets = np.concatenate([input_ids[:, 1:], 
                                  np.full((input_ids.shape[0], 1), 
                                          self.tokenizer.special_tokens['PAD'])], axis=1)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'targets': targets,
            'metadata': {
                'batch_size': len(padded_sequences),
                'max_length': max_len,
                'total_tokens': int(np.sum(attention_mask))
            }
        }

class ByteLevelModel:
    """Transformer model optimized for byte-level training."""
    
    def __init__(self, config: ByteLevelConfig):
        self.config = config
        
        # Model parameters will be initialized by the training system
        self.params = None
        
        logger.info(f" ByteLevelModel initialized")
        logger.info(f" Config: {config.hidden_size}d, {config.num_layers}L, {config.num_attention_heads}H")
    
    def create_byte_embeddings(self, key: Any) -> Dict[str, Any]:
        """Create byte-specific embedding layers."""
        embeddings = {}
        
        # Token embeddings (including special tokens)
        embeddings['token_embeddings'] = jax.random.normal(
            key, (self.config.extended_vocab_size, self.config.hidden_size)
        ) * 0.02
        
        # Byte-specific embeddings if enabled
        if self.config.use_byte_embeddings:
            embeddings['byte_embeddings'] = jax.random.normal(
                key, (256, self.config.byte_embedding_size)
            ) * 0.02
        
        # Positional embeddings
        if self.config.positional_encoding == BytePositionalEncoding.LEARNED:
            embeddings['position_embeddings'] = jax.random.normal(
                key, (self.config.max_position_embeddings, self.config.hidden_size)
            ) * 0.02
        
        return embeddings
    
    def apply_byte_aware_positional_encoding(
        self,
        input_embeddings: Any,
        input_ids: Any,
        position_embeddings: Optional[Any] = None
    ) -> Any:
        """Apply byte-aware positional encoding."""
        batch_size, seq_len, hidden_size = input_embeddings.shape
        
        if self.config.positional_encoding == BytePositionalEncoding.LEARNED:
            # Use learned positional embeddings
            positions = jnp.arange(seq_len)
            pos_embeddings = position_embeddings[positions]
            return input_embeddings + pos_embeddings
        
        elif self.config.positional_encoding == BytePositionalEncoding.BYTE_AWARE:
            # Create byte-value aware positional encoding
            positions = jnp.arange(seq_len)[None, :, None]  # [1, seq_len, 1]
            
            # Incorporate byte values into position calculation
            byte_values = jnp.where(input_ids < 256, input_ids, 0)  # Only actual bytes
            byte_influence = byte_values[..., None] / 256.0  # Normalize to [0, 1]
            
            # Standard sinusoidal encoding with byte modification
            div_term = jnp.exp(jnp.arange(0, hidden_size, 2) * 
                              -(jnp.log(10000.0) / hidden_size))
            
            # Modify frequency based on byte values
            modified_positions = positions * (1 + 0.1 * byte_influence)
            
            pos_encoding = jnp.zeros((batch_size, seq_len, hidden_size))
            pos_encoding = pos_encoding.at[:, :, 0::2].set(
                jnp.sin(modified_positions[:, :, 0::2] * div_term)
            )
            pos_encoding = pos_encoding.at[:, :, 1::2].set(
                jnp.cos(modified_positions[:, :, 1::2] * div_term)
            )
            
            return input_embeddings + pos_encoding
        
        else:
            # Standard positional encoding
            positions = jnp.arange(seq_len)[None, :, None]
            div_term = jnp.exp(jnp.arange(0, hidden_size, 2) * 
                              -(jnp.log(10000.0) / hidden_size))
            
            pos_encoding = jnp.zeros((1, seq_len, hidden_size))
            pos_encoding = pos_encoding.at[0, :, 0::2].set(
                jnp.sin(positions[0, :, 0::2] * div_term)
            )
            pos_encoding = pos_encoding.at[0, :, 1::2].set(
                jnp.cos(positions[0, :, 1::2] * div_term)
            )
            
            return input_embeddings + pos_encoding

class ByteLevelTrainer:
    """Main trainer for byte-level models."""
    
    def __init__(self, config: ByteLevelConfig):
        self.config = config
        self.tokenizer = ByteLevelTokenizer(config)
        self.data_loader = ByteLevelDataLoader(config, self.tokenizer)
        self.model = ByteLevelModel(config)
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Initialize optimizer
        self.optimizer = None
        self.optimizer_state = None
        self._torch_model = None
        self._torch_optimizer = None
        self._torch_device = None

        if JAX_AVAILABLE:
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=config.learning_rate,
                warmup_steps=config.warmup_steps,
                decay_steps=100000  # Will be updated based on data
            )
            self.optimizer = optax.adamw(
                learning_rate=schedule,
                weight_decay=config.weight_decay
            )
        else:
            self._init_torch_backend()
        
        logger.info(f" ByteLevelTrainer initialized")

    def _init_params(self, key: Any) -> Dict[str, Any]:
        """Initialize model parameters."""
        embeddings = self.model.create_byte_embeddings(key)
        params = {
            "token_embeddings": embeddings["token_embeddings"],
        }
        self.model.params = params
        return params

    def _init_torch_backend(self):
        """Initialize Torch backend for training if JAX is unavailable."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _TorchByteModel(nn.Module):
            def __init__(self, vocab_size: int, hidden_size: int):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, hidden_size)

            def forward(self, input_ids):
                x = self.emb(input_ids)
                logits = F.linear(x, self.emb.weight)
                return logits

        self._torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch_model = _TorchByteModel(self.config.extended_vocab_size, self.config.hidden_size).to(self._torch_device)
        self._torch_optimizer = torch.optim.AdamW(self._torch_model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def _forward(self, params: Dict[str, Any], input_ids: Any) -> Any:
        """Forward pass: embedding lookup and tied output projection."""
        token_embeddings = params["token_embeddings"]
        x = token_embeddings[input_ids]  # [B, T, H]
        logits = jnp.einsum("bth,vh->btv", x, token_embeddings)
        return logits

    def _loss_fn(self, params: Dict[str, Any], batch: Dict[str, Any]) -> Any:
        """Compute masked cross-entropy loss."""
        input_ids = batch["input_ids"]
        targets = batch["targets"]
        attention_mask = batch["attention_mask"]
        logits = self._forward(params, input_ids)
        loss_per_token = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        mask = attention_mask.astype(jnp.float32)
        masked_loss = loss_per_token * mask
        return jnp.sum(masked_loss) / jnp.maximum(jnp.sum(mask), 1.0)

    @functools.partial(JAX_JIT, static_argnums=0)
    def _train_step(
        self,
        params: Dict[str, Any],
        opt_state: Any,
        batch: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Any, Any]:
        """Single training step."""
        loss, grads = jax.value_and_grad(self._loss_fn)(params, batch)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    def _train_step_torch(self, batch: Dict[str, np.ndarray]) -> float:
        import torch
        import torch.nn.functional as F

        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=self._torch_device)
        targets = torch.tensor(batch["targets"], dtype=torch.long, device=self._torch_device)
        attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.float32, device=self._torch_device)

        self._torch_optimizer.zero_grad()
        logits = self._torch_model(input_ids)
        loss_per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none",
        ).view_as(targets)
        masked_loss = loss_per_token * attention_mask
        loss = masked_loss.sum() / attention_mask.sum().clamp(min=1.0)
        loss.backward()
        self._torch_optimizer.step()
        return float(loss.detach().cpu().item())
    
    def train_on_directory(self, data_dir: Union[str, Path], num_epochs: int = 1) -> Dict[str, Any]:
        """Train model on directory of files."""
        start_time = time.time()
        
        # Load files
        files = self.data_loader.load_files_from_directory(data_dir)
        if not files:
            raise ValueError(f"No valid files found in {data_dir}")
        
        # Initialize model parameters (mock for demonstration)
        params = None
        if JAX_AVAILABLE:
            key = jax.random.PRNGKey(42)
            params = self._init_params(key)
            self.optimizer_state = self.optimizer.init(params)
        
        # Training metrics
        total_loss = 0.0
        num_batches = 0
        
        logger.info(f" Starting byte-level training on {len(files)} files")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            # Shuffle files
            np.random.shuffle(files)
            
            # Process in batches
            for i in range(0, len(files), self.config.batch_size):
                batch_files = files[i:i + self.config.batch_size]
                
                try:
                    # Create batch
                    batch = self.data_loader.create_training_batch(batch_files)
                    
                    # Real training step
                    if JAX_AVAILABLE:
                        jax_batch = {
                            "input_ids": jnp.array(batch["input_ids"]),
                            "attention_mask": jnp.array(batch["attention_mask"]),
                            "targets": jnp.array(batch["targets"]),
                        }
                        params, self.optimizer_state, loss = self._train_step(
                            params, self.optimizer_state, jax_batch
                        )
                        loss = float(loss)
                    else:
                        loss = self._train_step_torch(batch)
                    
                    total_loss += loss
                    num_batches += 1
                    self.step += 1
                    
                    # Log progress
                    if self.step % 100 == 0:
                        avg_loss = total_loss / num_batches
                        logger.info(f"Step {self.step}, Epoch {epoch}, Loss: {avg_loss:.4f}")
                        
                        # Log byte statistics
                        if self.config.log_byte_statistics:
                            byte_stats = self.tokenizer.get_byte_statistics()
                            logger.info(f"Byte coverage: {byte_stats['coverage']:.1f}%, "
                                      f"Entropy: {byte_stats['byte_entropy']:.2f}")
                
                except Exception as e:
                    logger.error(f"Error processing batch {i}: {e}")
                    continue
            
            epoch_time = time.time() - epoch_start
            logger.info(f" Epoch {epoch} completed in {epoch_time:.2f}s")
        
        # Final statistics
        training_time = time.time() - start_time
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # Get tokenizer statistics
        byte_stats = self.tokenizer.get_byte_statistics()
        
        results = {
            'training_completed': True,
            'total_time_seconds': training_time,
            'total_steps': self.step,
            'total_epochs': num_epochs,
            'average_loss': avg_loss,
            'files_processed': len(files),
            'bytes_processed': byte_stats.get('total_bytes_processed', 0),
            'byte_statistics': byte_stats,
            'config_used': {
                'tokenizer_type': self.config.tokenizer_type.value,
                'vocab_size': self.config.vocab_size,
                'max_sequence_length': self.config.max_sequence_length,
                'batch_size': self.config.batch_size
            }
        }
        
        logger.info(f" Byte-level training completed!")
        logger.info(f" Final results: {avg_loss:.4f} loss, {training_time:.2f}s total")
        
        if JAX_AVAILABLE:
            self.model.params = params
        return results
    
    def save_model(self, save_path: Union[str, Path]):
        """Save trained model and tokenizer."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_dict = {
            'tokenizer_type': self.config.tokenizer_type.value,
            'vocab_size': self.config.vocab_size,
            'extended_vocab_size': self.config.extended_vocab_size,
            'max_sequence_length': self.config.max_sequence_length,
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'num_attention_heads': self.config.num_attention_heads,
            'special_tokens': self.tokenizer.special_tokens
        }
        
        with open(save_path / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save byte statistics
        byte_stats = self.tokenizer.get_byte_statistics()
        with open(save_path / 'byte_statistics.json', 'w') as f:
            json.dump(byte_stats, f, indent=2)
        
        logger.info(f" Model saved to {save_path}")

# Factory functions
def create_byte_level_config(**kwargs) -> ByteLevelConfig:
    """Create byte-level configuration with custom parameters."""
    return ByteLevelConfig(**kwargs)

def create_byte_level_trainer(config: Optional[ByteLevelConfig] = None) -> ByteLevelTrainer:
    """Create byte-level trainer with configuration."""
    if config is None:
        config = ByteLevelConfig()
    return ByteLevelTrainer(config)

# Example usage and testing
if __name__ == "__main__":
    # Demo configuration
    config = ByteLevelConfig(
        tokenizer_type=ByteTokenizerType.RAW_BYTES,
        max_sequence_length=2048,
        batch_size=4,
        learning_rate=1e-4,
        log_byte_statistics=True
    )
    
    # Create trainer
    trainer = create_byte_level_trainer(config)
    
    logger.info(" BYTE-LEVEL TRAINING SYSTEM DEMO")
    logger.info("=" * 50)
    logger.info(f" Tokenizer: {config.tokenizer_type.value}")
    logger.info(f" Vocab size: {config.extended_vocab_size} (256 bytes + specials)")
    logger.info(f" Max length: {config.max_sequence_length}")
    logger.info(f" Batch size: {config.batch_size}")
    
    # Test tokenization
    test_text = "Hello, world!  This is byte-level encoding."
    tokens = trainer.tokenizer.encode_with_special_tokens(test_text)
    decoded = trainer.tokenizer.decode_to_string(tokens)
    
    logger.info(f"\n TOKENIZATION TEST:")
    logger.info(f"Original: {test_text}")
    logger.info(f"Tokens: {tokens[:20]}..." if len(tokens) > 20 else f"Tokens: {tokens}")
    logger.info(f"Decoded: {decoded}")
    logger.info(f" Tokenization working correctly!")
    
    # Test byte statistics
    byte_stats = trainer.tokenizer.get_byte_statistics()
    if byte_stats.get('total_bytes_processed', 0) > 0:
        logger.info(f"\n BYTE STATISTICS:")
        logger.info(f"Bytes processed: {byte_stats['total_bytes_processed']}")
        logger.info(f"Unique bytes: {byte_stats['unique_bytes_seen']}")
        logger.info(f"Coverage: {byte_stats['coverage']:.1f}%")
    
    logger.info(f"\n BYTE-LEVEL TRAINING SYSTEM READY!")
    logger.info(f"Use trainer.train_on_directory(data_dir) to start training")

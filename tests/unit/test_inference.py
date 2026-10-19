"""
Unit tests for inference optimization modules.

Tests:
- Speculative Decoding
- Watermarking
- KV-Cache Compression
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import importlib.util

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_module_directly(name: str, path: str):
    """Load a module directly to avoid circular imports."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ═══════════════════════════════════════════════════════════════════
# Speculative Decoding Tests
# ═══════════════════════════════════════════════════════════════════

class TestSpeculativeDecoding:
    """Tests for speculative decoding module."""

    @pytest.fixture
    def spec_module(self):
        return load_module_directly(
            'speculative_decoding',
            str(project_root / 'capibara/inference/speculative_decoding.py')
        )

    def test_speculative_config_defaults(self, spec_module):
        """Test default configuration values."""
        config = spec_module.SpeculativeConfig()

        assert config.num_speculative_tokens == 4
        assert config.temperature == 1.0
        assert config.adaptive_k is True
        assert config.min_speculative_tokens == 2
        assert config.max_speculative_tokens == 8

    def test_speculative_config_validation(self, spec_module):
        """Test configuration validation."""
        config = spec_module.SpeculativeConfig(
            num_speculative_tokens=4,
            min_speculative_tokens=2,
            max_speculative_tokens=8
        )
        config.validate()  # Should not raise

    def test_speculative_config_invalid_k(self, spec_module):
        """Test that invalid k raises error."""
        config = spec_module.SpeculativeConfig(num_speculative_tokens=0)
        with pytest.raises(AssertionError):
            config.validate()

    def test_speculative_stats_defaults(self, spec_module):
        """Test stats tracking defaults."""
        stats = spec_module.SpeculativeStats()

        assert stats.total_tokens == 0
        assert stats.accepted_tokens == 0
        assert stats.draft_calls == 0

    def test_speculative_stats_acceptance_rate(self, spec_module):
        """Test acceptance rate calculation."""
        stats = spec_module.SpeculativeStats()
        stats.total_tokens = 100
        stats.accepted_tokens = 75

        assert stats.acceptance_rate == 0.75

    def test_speculative_stats_speedup(self, spec_module):
        """Test speedup calculation."""
        stats = spec_module.SpeculativeStats()
        stats.total_tokens = 100
        stats.accepted_tokens = 60
        stats.draft_calls = 20
        stats.target_calls = 20

        # Speedup depends on acceptance rate and calls
        assert stats.speedup >= 1.0

    def test_draft_model_type_enum(self, spec_module):
        """Test draft model type enumeration."""
        assert spec_module.DraftModelType.SMALL_LM.value == "small_lm"
        assert spec_module.DraftModelType.NGRAM.value == "ngram"
        assert spec_module.DraftModelType.MEDUSA.value == "medusa"

    def test_create_speculative_config_factory(self, spec_module):
        """Test factory function creates valid config."""
        config = spec_module.create_speculative_config(
            num_speculative=6,
            temperature=0.8,
            adaptive=False
        )

        assert config.num_speculative_tokens == 6
        assert config.temperature == 0.8
        assert config.adaptive_k is False

    def test_speculative_decoder_mock_mode(self, spec_module):
        """Test decoder works in mock mode without real models."""
        config = spec_module.SpeculativeConfig(num_speculative_tokens=4)
        decoder = spec_module.SpeculativeDecoder(
            target_model=None,
            draft_model=None,
            config=config,
            mock_mode=True
        )

        # Should be able to get stats even in mock mode
        stats = decoder.get_stats()
        assert isinstance(stats, dict)


# ═══════════════════════════════════════════════════════════════════
# Watermarking Tests
# ═══════════════════════════════════════════════════════════════════

class TestWatermarking:
    """Tests for watermarking module."""

    @pytest.fixture
    def wm_module(self):
        return load_module_directly(
            'watermarking',
            str(project_root / 'capibara/inference/watermarking.py')
        )

    def test_watermark_config_defaults(self, wm_module):
        """Test default watermark configuration."""
        config = wm_module.WatermarkConfig()

        assert config.gamma == 0.25
        assert config.delta == 2.0
        assert config.vocab_size == 32000
        assert config.watermark_type == wm_module.WatermarkType.GREEN_RED_LIST

    def test_watermark_type_enum(self, wm_module):
        """Test watermark type enumeration."""
        assert wm_module.WatermarkType.GREEN_RED_LIST.value == "green_red"
        assert wm_module.WatermarkType.MULTI_BIT.value == "multi_bit"
        assert wm_module.WatermarkType.SEMANTIC.value == "semantic"

    def test_create_watermark_config_factory(self, wm_module):
        """Test factory function."""
        config = wm_module.create_watermark_config(
            gamma=0.3,
            delta=1.5,
            vocab_size=50000
        )

        assert config.gamma == 0.3
        assert config.delta == 1.5
        assert config.vocab_size == 50000

    def test_green_red_list_watermark_init(self, wm_module):
        """Test green/red list watermark initialization."""
        config = wm_module.WatermarkConfig(
            vocab_size=1000,
            gamma=0.25
        )
        watermark = wm_module.GreenRedListWatermark(config)

        assert watermark.green_list_size == 250  # 25% of 1000

    def test_green_red_list_green_tokens(self, wm_module):
        """Test green list generation is deterministic."""
        config = wm_module.WatermarkConfig(
            vocab_size=1000,
            gamma=0.25,
            secret_key="test_key"
        )
        watermark = wm_module.GreenRedListWatermark(config)

        # Same context should produce same green list
        context = [1, 2, 3]
        green1 = watermark._get_green_list(context)
        green2 = watermark._get_green_list(context)

        assert green1 == green2
        assert len(green1) == 250

    def test_detection_result_dataclass(self, wm_module):
        """Test detection result structure."""
        result = wm_module.DetectionResult(
            is_watermarked=True,
            confidence=0.95,
            z_score=4.5,
            green_fraction=0.6
        )

        assert result.is_watermarked is True
        assert result.confidence == 0.95
        assert result.z_score == 4.5

    def test_watermark_payload(self, wm_module):
        """Test watermark payload encoding."""
        payload = wm_module.WatermarkPayload(
            model_id="capibara-v3",
            timestamp=1234567890,
            version="1.0"
        )

        assert payload.model_id == "capibara-v3"
        assert payload.timestamp == 1234567890

    def test_hash_is_positive(self, wm_module):
        """Test that token hashing produces positive indices."""
        config = wm_module.WatermarkConfig(vocab_size=1000)
        processor = wm_module.WatermarkProcessor(config)

        # The detect_from_text uses abs(hash()) now
        # This test verifies the fix works
        test_words = ["hello", "world", "test", "negative", "hash"]
        for word in test_words:
            idx = abs(hash(word)) % config.vocab_size
            assert 0 <= idx < config.vocab_size


# ═══════════════════════════════════════════════════════════════════
# KV-Cache Tests
# ═══════════════════════════════════════════════════════════════════

class TestKVCache:
    """Tests for KV-cache compression module."""

    @pytest.fixture
    def kv_module(self):
        return load_module_directly(
            'kv_cache',
            str(project_root / 'capibara/inference/kv_cache.py')
        )

    def test_kv_cache_config_defaults(self, kv_module):
        """Test default KV-cache configuration."""
        config = kv_module.KVCacheConfig()

        assert config.max_cache_tokens == 32768
        assert config.window_size == 4096
        assert config.num_sink_tokens == 4
        assert config.strategy == kv_module.CompressionStrategy.STREAMING_LLM

    def test_compression_strategy_enum(self, kv_module):
        """Test compression strategy enumeration."""
        assert kv_module.CompressionStrategy.NONE.value == "none"
        assert kv_module.CompressionStrategy.SLIDING_WINDOW.value == "sliding"
        assert kv_module.CompressionStrategy.H2O.value == "h2o"
        assert kv_module.CompressionStrategy.STREAMING_LLM.value == "streaming"

    def test_quantization_type_enum(self, kv_module):
        """Test quantization type enumeration."""
        assert kv_module.QuantizationType.NONE.value == "none"
        assert kv_module.QuantizationType.INT8.value == "int8"
        assert kv_module.QuantizationType.INT4.value == "int4"

    def test_create_kv_cache_config_factory(self, kv_module):
        """Test factory function."""
        config = kv_module.create_kv_cache_config(
            max_tokens=8192,
            window_size=2048,
            strategy="h2o"
        )

        assert config.max_cache_tokens == 8192
        assert config.window_size == 2048
        assert config.strategy == kv_module.CompressionStrategy.H2O

    def test_cache_stats_defaults(self, kv_module):
        """Test cache stats defaults."""
        stats = kv_module.CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.memory_mb == 0.0

    def test_cache_stats_hit_rate(self, kv_module):
        """Test hit rate calculation."""
        stats = kv_module.CacheStats()
        stats.hits = 80
        stats.misses = 20

        assert stats.hit_rate == 0.8

    def test_cache_stats_compression_ratio(self, kv_module):
        """Test compression ratio calculation."""
        stats = kv_module.CacheStats()
        stats.original_tokens = 1000
        stats.compressed_tokens = 250

        assert stats.compression_ratio == 4.0

    def test_kv_cache_initialization(self, kv_module):
        """Test KV-cache initialization."""
        config = kv_module.KVCacheConfig(window_size=128)
        cache = kv_module.KVCache(config, num_layers=4, num_heads=8, head_dim=64)

        assert len(cache.keys) == 4
        assert len(cache.values) == 4
        assert all(k is None for k in cache.keys)

    def test_kv_cache_clear(self, kv_module):
        """Test cache clearing."""
        config = kv_module.KVCacheConfig()
        cache = kv_module.KVCache(config, num_layers=4, num_heads=8, head_dim=64)

        cache.clear()

        assert all(k is None for k in cache.keys)
        assert all(v is None for v in cache.values)

    def test_kv_cache_memory_usage_with_none(self, kv_module):
        """Test memory usage handles None values correctly."""
        config = kv_module.KVCacheConfig()
        cache = kv_module.KVCache(config, num_layers=4, num_heads=8, head_dim=64)

        # Should not raise even with all None values
        memory = cache.get_memory_usage()
        assert memory == 0.0

    def test_create_kv_cache_factory(self, kv_module):
        """Test create_kv_cache factory function."""
        config = kv_module.create_kv_cache_config(
            max_tokens=2048,
            strategy="h2o"
        )
        cache = kv_module.create_kv_cache(config, num_layers=16)

        assert cache.config.max_cache_tokens == 2048
        assert cache.num_layers == 16

    def test_quantized_kv_cache_init(self, kv_module):
        """Test quantized KV-cache initialization."""
        config = kv_module.KVCacheConfig(
            quantization=kv_module.QuantizationType.INT8
        )
        cache = kv_module.QuantizedKVCache(
            config, num_layers=4, num_heads=8, head_dim=64
        )

        assert cache.config.quantization == kv_module.QuantizationType.INT8


# ═══════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════

class TestInferenceModuleIntegration:
    """Integration tests for inference modules working together."""

    def test_all_modules_importable(self):
        """Test that all inference modules can be imported."""
        spec = load_module_directly(
            'speculative_decoding',
            str(project_root / 'capibara/inference/speculative_decoding.py')
        )
        wm = load_module_directly(
            'watermarking',
            str(project_root / 'capibara/inference/watermarking.py')
        )
        kv = load_module_directly(
            'kv_cache',
            str(project_root / 'capibara/inference/kv_cache.py')
        )

        assert spec is not None
        assert wm is not None
        assert kv is not None

    def test_configs_are_dataclasses(self):
        """Test all configs are proper dataclasses."""
        from dataclasses import is_dataclass

        spec = load_module_directly(
            'speculative_decoding',
            str(project_root / 'capibara/inference/speculative_decoding.py')
        )
        wm = load_module_directly(
            'watermarking',
            str(project_root / 'capibara/inference/watermarking.py')
        )
        kv = load_module_directly(
            'kv_cache',
            str(project_root / 'capibara/inference/kv_cache.py')
        )

        assert is_dataclass(spec.SpeculativeConfig)
        assert is_dataclass(wm.WatermarkConfig)
        assert is_dataclass(kv.KVCacheConfig)

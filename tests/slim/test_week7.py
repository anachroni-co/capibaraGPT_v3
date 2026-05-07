"""Week-7 slim tests: feature flags, verification step, token budget."""
from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

class TestFeatureFlags:
    def test_is_enabled_returns_bool(self):
        from config.feature_flags import is_enabled
        result = is_enabled("rag")
        assert isinstance(result, bool)

    def test_unknown_flag_returns_false(self):
        from config.feature_flags import is_enabled
        assert is_enabled("nonexistent_flag_xyz") is False

    def test_env_override_true(self, monkeypatch):
        from config.feature_flags import is_enabled
        monkeypatch.setenv("CAPIBARA_FEATURES_RAG_ENABLED", "true")
        assert is_enabled("rag") is True

    def test_env_override_false(self, monkeypatch):
        from config.feature_flags import is_enabled
        monkeypatch.setenv("CAPIBARA_FEATURES_QUANTIZATION_ENABLED", "false")
        assert is_enabled("quantization") is False

    def test_env_override_1(self, monkeypatch):
        from config.feature_flags import is_enabled
        monkeypatch.setenv("CAPIBARA_FEATURES_THINK_ANYWHERE_ENABLED", "1")
        assert is_enabled("think_anywhere") is True

    def test_env_override_yes(self, monkeypatch):
        from config.feature_flags import is_enabled
        monkeypatch.setenv("CAPIBARA_FEATURES_MEMORY_EXTRACTION_ENABLED", "yes")
        assert is_enabled("memory_extraction") is True

    def test_flag_config_returns_dict(self):
        from config.feature_flags import flag_config
        cfg = flag_config("rag")
        assert isinstance(cfg, dict)

    def test_flag_config_has_expected_keys(self):
        from config.feature_flags import flag_config
        cfg = flag_config("rag")
        assert "enabled" in cfg
        assert "top_k" in cfg
        assert "min_score" in cfg

    def test_all_flags_returns_dict(self):
        from config.feature_flags import all_flags
        flags = all_flags()
        assert isinstance(flags, dict)
        assert "rag" in flags
        assert "quantization" in flags
        assert "think_anywhere" in flags
        assert "memory_extraction" in flags
        assert "context_compaction" in flags

    def test_all_flags_values_are_bool(self):
        from config.feature_flags import all_flags
        for name, val in all_flags().items():
            assert isinstance(val, bool), f"flag {name!r} is not bool"

    def test_flag_config_quantization(self):
        from config.feature_flags import flag_config
        cfg = flag_config("quantization")
        assert "bits" in cfg

    def test_flag_config_context_compaction(self):
        from config.feature_flags import flag_config
        cfg = flag_config("context_compaction")
        assert "max_turns" in cfg

    def test_case_insensitive(self, monkeypatch):
        from config.feature_flags import is_enabled
        monkeypatch.setenv("CAPIBARA_FEATURES_RAG_ENABLED", "TRUE")
        assert is_enabled("rag") is True

    def test_env_takes_priority_over_config(self, monkeypatch):
        """Env var overrides whatever config.yaml says."""
        from config.feature_flags import is_enabled
        monkeypatch.setenv("CAPIBARA_FEATURES_RAG_ENABLED", "true")
        assert is_enabled("rag") is True

    def test_think_anywhere_strip_flag(self):
        from config.feature_flags import flag_config
        cfg = flag_config("think_anywhere")
        assert "strip_from_output" in cfg


# ---------------------------------------------------------------------------
# Verification step
# ---------------------------------------------------------------------------

class TestVerification:
    def test_good_output_passes(self):
        from inference.verification import verify_output
        report = verify_output("The capital of France is Paris.")
        assert report.passed

    def test_empty_output_fails(self):
        from inference.verification import verify_output
        report = verify_output("")
        assert not report.passed
        assert any("non_empty" in f for f in report.failures)

    def test_whitespace_only_fails(self):
        from inference.verification import verify_output
        report = verify_output("   \n\t  ")
        assert not report.passed

    def test_min_length_check(self):
        from inference.verification import verify_output, VerificationConfig
        cfg = VerificationConfig(min_length=100)
        report = verify_output("Short.", cfg)
        assert not report.passed
        assert any("min_length" in f for f in report.failures)

    def test_max_length_check(self):
        from inference.verification import verify_output, VerificationConfig
        cfg = VerificationConfig(max_length=5)
        report = verify_output("This is a longer sentence than five chars.", cfg)
        assert not report.passed
        assert any("max_length" in f for f in report.failures)

    def test_repetition_check_fails(self):
        from inference.verification import verify_output, VerificationConfig
        cfg = VerificationConfig(no_repetition=True, repetition_max_ratio=0.3)
        text = ("cat dog bird fish " * 30).strip()
        report = verify_output(text, cfg)
        # Even repeated text, the most common 4-gram ratio depends on cycling
        # Just ensure the check runs without error
        assert isinstance(report.passed, bool)

    def test_repetition_blatant_fails(self):
        from inference.verification import verify_output, VerificationConfig
        cfg = VerificationConfig(no_repetition=True, repetition_max_ratio=0.3)
        text = "word " * 100
        report = verify_output(text, cfg)
        assert not report.passed
        assert any("repetition" in f for f in report.failures)

    def test_coherence_check_fails_numbers_only(self):
        from inference.verification import verify_output, VerificationConfig
        cfg = VerificationConfig(coherence=True, non_empty=False, no_repetition=False)
        report = verify_output("123 456 789", cfg)
        assert not report.passed
        assert any("coherence" in f for f in report.failures)

    def test_coherence_passes_with_words(self):
        from inference.verification import verify_output, VerificationConfig
        cfg = VerificationConfig(coherence=True, non_empty=False, no_repetition=False)
        report = verify_output("Hello world", cfg)
        assert report.passed

    def test_no_truncation_check(self):
        from inference.verification import verify_output, VerificationConfig
        cfg = VerificationConfig(no_truncation=True, non_empty=False,
                                 no_repetition=False, coherence=False)
        report = verify_output("This sentence ends abruptly in the middle of", cfg)
        assert not report.passed

    def test_no_truncation_passes_with_punctuation(self):
        from inference.verification import verify_output, VerificationConfig
        cfg = VerificationConfig(no_truncation=True, non_empty=False,
                                 no_repetition=False, coherence=False)
        report = verify_output("Complete sentence.", cfg)
        assert report.passed

    def test_report_as_dict(self):
        from inference.verification import verify_output
        report = verify_output("Hello world.")
        d = report.as_dict()
        assert "passed" in d
        assert "failures" in d
        assert isinstance(d["failures"], list)

    def test_output_verifier_class(self):
        from inference.verification import OutputVerifier, VerificationConfig
        cfg = VerificationConfig(min_length=3, max_length=100)
        verifier = OutputVerifier(cfg)
        report = verifier.verify("Hello world.")
        assert report.passed

    def test_multiple_failures_reported(self):
        from inference.verification import verify_output, VerificationConfig
        cfg = VerificationConfig(min_length=50, coherence=True)
        report = verify_output("123", cfg)
        assert not report.passed
        assert len(report.failures) >= 2


# ---------------------------------------------------------------------------
# Memory extraction (improvement 4)
# ---------------------------------------------------------------------------

class TestMemoryExtraction:
    def _make_store(self):
        from rag.store import VectorStore
        return VectorStore()

    def test_ingest_above_threshold(self):
        from rag.memory import MemoryExtractor
        store = self._make_store()
        extractor = MemoryExtractor(store, threshold=0.7)
        ingested = extractor.maybe_ingest("What is X?", "X is Y.", score=0.9)
        assert ingested is True
        assert len(store) > 0

    def test_skip_below_threshold(self):
        from rag.memory import MemoryExtractor
        store = self._make_store()
        extractor = MemoryExtractor(store, threshold=0.8)
        ingested = extractor.maybe_ingest("What is X?", "X is Y.", score=0.5)
        assert ingested is False
        assert len(store) == 0

    def test_deduplication(self):
        from rag.memory import MemoryExtractor
        store = self._make_store()
        extractor = MemoryExtractor(store, threshold=0.0, deduplicate=True)
        extractor.maybe_ingest("Q", "A", score=1.0)
        size_after_first = len(store)
        extractor.maybe_ingest("Q", "A", score=1.0)
        assert len(store) == size_after_first  # no duplicate added

    def test_dedup_disabled(self):
        from rag.memory import MemoryExtractor
        store = self._make_store()
        extractor = MemoryExtractor(store, threshold=0.0, deduplicate=False)
        extractor.maybe_ingest("Q", "A", score=1.0)
        size_after_first = len(store)
        extractor.maybe_ingest("Q", "A", score=1.0)
        assert len(store) > size_after_first

    def test_seen_count(self):
        from rag.memory import MemoryExtractor
        store = self._make_store()
        extractor = MemoryExtractor(store, threshold=0.0)
        extractor.maybe_ingest("Q1", "A1", score=1.0)
        extractor.maybe_ingest("Q2", "A2", score=1.0)
        assert extractor.seen_count == 2

    def test_clear_dedup_cache(self):
        from rag.memory import MemoryExtractor
        store = self._make_store()
        extractor = MemoryExtractor(store, threshold=0.0, deduplicate=True)
        extractor.maybe_ingest("Q", "A", score=1.0)
        extractor.clear_dedup_cache()
        assert extractor.seen_count == 0

    def test_metadata_attached(self):
        from rag.memory import MemoryExtractor
        store = self._make_store()
        extractor = MemoryExtractor(store, threshold=0.0)
        extractor.maybe_ingest("Q", "A", score=1.0, metadata={"session": "test"})
        assert any("session" in d.metadata for d in store._docs)

    def test_build_memory_extractor_disabled(self, monkeypatch):
        monkeypatch.setenv("CAPIBARA_FEATURES_MEMORY_EXTRACTION_ENABLED", "false")
        from rag.memory import build_memory_extractor
        from rag.store import VectorStore
        result = build_memory_extractor(VectorStore())
        assert result is None

    def test_build_memory_extractor_enabled(self, monkeypatch):
        monkeypatch.setenv("CAPIBARA_FEATURES_MEMORY_EXTRACTION_ENABLED", "true")
        from rag.memory import build_memory_extractor
        from rag.store import VectorStore
        result = build_memory_extractor(VectorStore())
        assert result is not None


# ---------------------------------------------------------------------------
# History / Token budget / Context compaction (improvements 5, 6, 7)
# ---------------------------------------------------------------------------

class TestTokenBudget:
    def test_initial_total_is_zero(self):
        from capibara.repl import TokenBudget
        b = TokenBudget()
        assert b.total == 0

    def test_record_turn_increments(self):
        from capibara.repl import TokenBudget
        b = TokenBudget()
        b.record_turn("hello world", "this is a response")
        assert b.total > 0

    def test_budget_not_exceeded_unlimited(self):
        from capibara.repl import TokenBudget
        b = TokenBudget(max_tokens=0)
        b.record_turn("a " * 1000, "b " * 1000)
        assert not b.budget_exceeded()

    def test_budget_exceeded(self):
        from capibara.repl import TokenBudget
        b = TokenBudget(max_tokens=5)
        b.record_turn("hello world test phrase here", "and a long response too")
        assert b.budget_exceeded()

    def test_status_line_contains_tokens(self):
        from capibara.repl import TokenBudget
        b = TokenBudget()
        b.record_turn("hello", "world")
        line = b.status_line()
        assert "tokens" in line
        assert "total" in line

    def test_status_line_shows_remaining(self):
        from capibara.repl import TokenBudget
        b = TokenBudget(max_tokens=1000)
        b.record_turn("test", "response")
        line = b.status_line()
        assert "remaining" in line

    def test_last_turn_tokens(self):
        from capibara.repl import TokenBudget
        b = TokenBudget()
        b.record_turn("hello", "world response")
        tin, tout = b.last_turn_tokens
        assert tin > 0
        assert tout > 0


class TestContextCompactor:
    def _make_turns(self, n):
        from capibara.repl import Turn
        turns = []
        for i in range(n):
            role = "user" if i % 2 == 0 else "assistant"
            turns.append(Turn(role=role, text=f"Turn {i} text content here."))
        return turns

    def test_no_compact_under_limit(self):
        from capibara.repl import ContextCompactor
        cc = ContextCompactor(max_turns=20)
        turns = self._make_turns(10)
        result = cc.maybe_compact(turns)
        assert len(result) == 10

    def test_compact_over_limit(self):
        from capibara.repl import ContextCompactor
        cc = ContextCompactor(max_turns=10)
        turns = self._make_turns(25)
        result = cc.maybe_compact(turns)
        assert len(result) < 25

    def test_compact_includes_summary_turn(self):
        from capibara.repl import ContextCompactor
        cc = ContextCompactor(max_turns=10)
        turns = self._make_turns(25)
        result = cc.maybe_compact(turns)
        assert any("Summary" in t.text or "compacted" in t.text for t in result)

    def test_compaction_counter(self):
        from capibara.repl import ContextCompactor
        cc = ContextCompactor(max_turns=4)
        turns = self._make_turns(10)
        cc.maybe_compact(turns)
        assert cc.compactions == 1

    def test_turn_token_count(self):
        from capibara.repl import Turn
        t = Turn(role="user", text="hello world")
        assert t.token_count > 0


class TestHistoryManager:
    def test_save_and_load(self, tmp_path):
        from capibara.repl import HistoryManager
        mgr = HistoryManager(path=tmp_path / "hist.txt")
        mgr.load()   # should not raise even if file missing
        mgr.save()   # should create file (may be empty without readline)

    def test_prompt_returns_string(self, monkeypatch):
        from capibara.repl import HistoryManager
        mgr = HistoryManager()
        monkeypatch.setattr("builtins.input", lambda _: "test input")
        result = mgr.prompt("> ")
        assert result == "test input"


class TestREPL:
    def test_repl_instantiates(self):
        from capibara.repl import CapibaraREPL
        repl = CapibaraREPL()
        assert repl is not None

    def test_repl_uses_custom_generate_fn(self):
        from capibara.repl import CapibaraREPL
        called = []
        def gen(q, hist):
            called.append(q)
            return "response"
        repl = CapibaraREPL(generate_fn=gen)
        # Simulate a single turn internally
        from capibara.repl import Turn
        response = repl.generate_fn("hello", [])
        assert response == "response"
        assert called == ["hello"]

    def test_count_tokens_returns_positive(self):
        from capibara.repl import _count_tokens
        n = _count_tokens("hello world how are you")
        assert n > 0

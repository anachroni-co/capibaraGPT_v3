"""Unit tests for core/think_anywhere — no ML dependencies required."""
import pytest
from core.think_anywhere import (
    ThinkAnywhereConfig,
    ThinkAnywhereProcessor,
    ThinkAnywhereReward,
    ThinkAnywhereStreamFilter,
    ParsedResponse,
    RewardResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_RESPONSE = """\
<think>
I need to implement edit distance with DP.
Plan: 2D dp array, fill base cases, iterate.
</think>
def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(<thinkanywhere>base case: fill first column</thinkanywhere>m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[<thinkanywhere>0-based index</thinkanywhere>j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[m][n]
"""

EXPECTED_CLEAN_CODE = """\
def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[m][n]"""


# ---------------------------------------------------------------------------
# ThinkAnywhereConfig
# ---------------------------------------------------------------------------

class TestThinkAnywhereConfig:
    def test_defaults(self):
        cfg = ThinkAnywhereConfig()
        assert cfg.think_open == "<think>"
        assert cfg.ta_open == "<thinkanywhere>"
        assert cfg.structure_reward_weight == pytest.approx(0.1)
        assert cfg.correctness_reward_weight == pytest.approx(0.9)

    def test_open_tag_text_mode(self):
        cfg = ThinkAnywhereConfig(use_special_tokens=False)
        assert cfg.open_tag == "<thinkanywhere>"
        assert cfg.close_tag == "</thinkanywhere>"

    def test_open_tag_special_token_mode(self):
        cfg = ThinkAnywhereConfig(use_special_tokens=True)
        assert cfg.open_tag == "<ta>"
        assert cfg.close_tag == "</ta>"


# ---------------------------------------------------------------------------
# ThinkAnywhereProcessor — format
# ---------------------------------------------------------------------------

class TestThinkAnywhereProcessorFormat:
    def setup_method(self):
        self.proc = ThinkAnywhereProcessor()

    def test_format_prompt_contains_problem(self):
        prompt = self.proc.format_prompt("Write a function to add two numbers.")
        assert "add two numbers" in prompt
        assert "<thinkanywhere>" in prompt  # rules mention the tag

    def test_format_prompt_system_override(self):
        prompt = self.proc.format_prompt("problem", system_override="CUSTOM SYSTEM")
        assert prompt.startswith("CUSTOM SYSTEM")
        assert "problem" in prompt


# ---------------------------------------------------------------------------
# ThinkAnywhereProcessor — parse
# ---------------------------------------------------------------------------

class TestThinkAnywhereProcessorParse:
    def setup_method(self):
        self.proc = ThinkAnywhereProcessor()

    def test_parse_extracts_upfront_thinking(self):
        result = self.proc.parse(VALID_RESPONSE)
        assert "edit distance" in result.upfront_thinking
        assert "DP" in result.upfront_thinking

    def test_parse_extracts_ta_blocks(self):
        result = self.proc.parse(VALID_RESPONSE)
        assert len(result.think_anywhere_blocks) == 2
        assert "base case" in result.think_anywhere_blocks[0]
        assert "0-based" in result.think_anywhere_blocks[1]

    def test_parse_clean_code_has_no_tags(self):
        result = self.proc.parse(VALID_RESPONSE)
        assert "<think>" not in result.clean_code
        assert "<thinkanywhere>" not in result.clean_code
        assert "</thinkanywhere>" not in result.clean_code

    def test_parse_clean_code_is_syntactically_valid(self):
        import ast
        result = self.proc.parse(VALID_RESPONSE)
        ast.parse(result.clean_code)  # must not raise

    def test_parse_is_valid_for_valid_response(self):
        result = self.proc.parse(VALID_RESPONSE)
        assert result.is_valid
        assert result.validation_errors == []

    def test_parse_response_without_think_block_is_invalid(self):
        no_think = "def f(): pass <thinkanywhere>x</thinkanywhere>"
        result = self.proc.parse(no_think)
        assert not result.is_valid
        assert any("thinking block" in e for e in result.validation_errors)

    def test_parse_response_without_ta_block_is_invalid(self):
        no_ta = "<think>thinking</think>\ndef f(): pass"
        result = self.proc.parse(no_ta)
        assert not result.is_valid
        assert any("inline" in e for e in result.validation_errors)

    def test_parse_response_with_unbalanced_tags(self):
        unbalanced = "<think>t</think>\ndef f(<thinkanywhere>x): pass"
        result = self.proc.parse(unbalanced)
        assert not result.is_valid
        assert any("Unbalanced" in e for e in result.validation_errors)


# ---------------------------------------------------------------------------
# ThinkAnywhereProcessor — strip_thinking
# ---------------------------------------------------------------------------

class TestStripThinking:
    def setup_method(self):
        self.proc = ThinkAnywhereProcessor()

    def test_strip_removes_all_blocks(self):
        stripped = self.proc.strip_thinking(VALID_RESPONSE)
        assert "<think>" not in stripped
        assert "</think>" not in stripped
        assert "<thinkanywhere>" not in stripped
        assert "</thinkanywhere>" not in stripped

    def test_strip_preserves_code_structure(self):
        stripped = self.proc.strip_thinking(VALID_RESPONSE)
        assert "def minDistance" in stripped
        assert "dp[i][j]" in stripped
        assert "return dp[m][n]" in stripped

    def test_strip_no_op_on_plain_text(self):
        plain = "def add(a, b):\n    return a + b"
        assert self.proc.strip_thinking(plain) == plain


# ---------------------------------------------------------------------------
# ThinkAnywhereProcessor — special token embedding initialization
# ---------------------------------------------------------------------------

class TestSpecialTokenEmbedding:
    def test_initialize_returns_two_embeddings(self):
        import numpy as np
        proc = ThinkAnywhereProcessor(ThinkAnywhereConfig(use_special_tokens=True))
        vocab_size, hidden = 10, 16
        embeddings = np.random.randn(vocab_size, hidden).astype(np.float32)
        token_ids = {
            "think": 0, "any": 1, "where": 2,
            "<im_start>": 3, "<im_end>": 4,
        }
        e_open, e_close = proc.initialize_special_token_embedding(embeddings, token_ids)
        assert e_open.shape == (hidden,)
        assert e_close.shape == (hidden,)

    def test_initialize_alpha_mixing(self):
        import numpy as np
        cfg = ThinkAnywhereConfig(semantic_mix_alpha=0.5)
        proc = ThinkAnywhereProcessor(cfg)
        vocab_size, hidden = 10, 4
        E = np.eye(vocab_size, hidden, dtype=np.float32)
        token_ids = {
            "think": 0, "any": 1, "where": 2,
            "<im_start>": 3, "<im_end>": 4,
        }
        e_open, _ = proc.initialize_special_token_embedding(E, token_ids)
        # e_open = 0.5 * mean([e0, e1, e2]) + 0.5 * e3
        semantic = np.mean(E[[0, 1, 2]], axis=0)
        expected = 0.5 * semantic + 0.5 * E[3]
        np.testing.assert_allclose(e_open, expected, atol=1e-6)

    def test_initialize_raises_if_seed_tokens_missing(self):
        import numpy as np
        proc = ThinkAnywhereProcessor()
        E = np.ones((5, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="semantic seed tokens"):
            proc.initialize_special_token_embedding(E, {"<im_start>": 0, "<im_end>": 1})


# ---------------------------------------------------------------------------
# ThinkAnywhereReward
# ---------------------------------------------------------------------------

class TestThinkAnywhereReward:
    def setup_method(self):
        self.reward = ThinkAnywhereReward()

    def test_valid_response_correct_code_gets_high_reward(self):
        test_cases = [
            "assert minDistance('horse', 'ros') == 3",
            "assert minDistance('intention', 'execution') == 5",
        ]
        result = self.reward(VALID_RESPONSE, test_cases)
        assert result.structure == pytest.approx(1.0)
        assert result.correctness == pytest.approx(1.0)
        assert result.combined == pytest.approx(1.0)
        assert result.passed_tests == 2
        assert result.total_tests == 2

    def test_invalid_format_gives_zero_structure(self):
        bad = "def f(): return 42"
        result = self.reward(bad, [])
        assert result.structure == pytest.approx(0.0)

    def test_wrong_code_gives_zero_correctness(self):
        # Replace the correct else-branch update with a constant so the
        # function always returns 0 (wrong for any non-trivial input).
        wrong_code = VALID_RESPONSE.replace(
            "dp[i - 1][j - 1]) + 1",
            "0)",
        )
        result = self.reward(wrong_code, ["assert minDistance('ab', 'cd') == 2"])
        assert result.correctness == pytest.approx(0.0)

    def test_no_test_cases_gives_zero_correctness(self):
        result = self.reward(VALID_RESPONSE, [])
        assert result.correctness == pytest.approx(0.0)
        assert result.total_tests == 0

    def test_combined_reward_weighting(self):
        cfg = ThinkAnywhereConfig(structure_reward_weight=0.1)
        reward = ThinkAnywhereReward(cfg)
        result = reward(VALID_RESPONSE, ["assert minDistance('', '') == 0"])
        # structure=1, correctness=1 → combined=1
        assert result.combined == pytest.approx(1.0)

    def test_batch_returns_one_result_per_response(self):
        responses = [VALID_RESPONSE, "<think>x</think>\ndef f(): pass <thinkanywhere>y</thinkanywhere>"]
        results = self.reward.batch(responses)
        assert len(results) == len(responses)

    def test_group_normalized_advantages_zero_mean(self):
        results = self.reward.batch(
            [VALID_RESPONSE] * 4,
            test_cases=["assert minDistance('', '') == 0"],
        )
        advantages = self.reward.group_normalized_advantages(results)
        assert len(advantages) == 4
        # All rewards identical → all advantages ≈ 0
        assert all(abs(a) < 1e-6 for a in advantages)

    def test_timeout_handled_gracefully(self):
        # Replace the first line of the function body with an infinite loop
        # so calling minDistance() hangs.  The test case must call the
        # function to actually trigger the loop.
        infinite_loop = VALID_RESPONSE.replace(
            "m, n = len(word1), len(word2)",
            "while True: pass  # infinite",
        )
        result = self.reward(
            infinite_loop,
            ["assert minDistance('a', 'b') == 1"],
            timeout=0.5,
        )
        assert result.correctness == pytest.approx(0.0)
        assert any("Timeout" in e for e in result.execution_errors)


# ---------------------------------------------------------------------------
# ThinkAnywhereStreamFilter (inference engine streaming helper)
# ---------------------------------------------------------------------------

class TestThinkStreamFilter:
    def test_passthrough_when_no_thinking_blocks(self):
        f = ThinkAnywhereStreamFilter()
        out = "".join(f.feed(c) for c in "hello world")
        out += f.flush()
        assert out == "hello world"

    def test_suppresses_think_block(self):
        f = ThinkAnywhereStreamFilter()
        text = "before<think>hidden</think>after"
        out = "".join(f.feed(c) for c in text) + f.flush()
        assert out == "beforeafter"

    def test_suppresses_thinkanywhere_block(self):
        f = ThinkAnywhereStreamFilter()
        text = "x = <thinkanywhere>reasoning here</thinkanywhere>42"
        out = "".join(f.feed(c) for c in text) + f.flush()
        assert out == "x = 42"

    def test_suppresses_multiple_ta_blocks(self):
        f = ThinkAnywhereStreamFilter()
        text = "a<thinkanywhere>1</thinkanywhere>b<thinkanywhere>2</thinkanywhere>c"
        out = "".join(f.feed(c) for c in text) + f.flush()
        assert out == "abc"

    def test_flush_discards_unclosed_block(self):
        f = ThinkAnywhereStreamFilter()
        out = f.feed("start<thinkanywhere>open but never closed")
        tail = f.flush()
        assert "thinkanywhere" not in out + tail

    def test_handles_chunked_feed(self):
        f = ThinkAnywhereStreamFilter()
        chunks = ["x = ", "<think", "anywhere>", "skip", "</thinkanywhere>", "42"]
        out = "".join(f.feed(c) for c in chunks) + f.flush()
        assert out == "x = 42"

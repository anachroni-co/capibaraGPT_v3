"""Tests for core/special_tokens — one test class per token type."""

import pytest
from core.special_tokens import (
    get_registry,
    SpecialTokenProcessor,
    SpecialTokenStreamFilter,
    ParseResult,
    SearchTokenHandler,
    LangTokenProcessor,
)
from core.special_tokens.base import SpecialTokenConfig
from core.special_tokens.uncertain import UncertainTokenExtractor
from core.special_tokens.lang import LangTokenStreamFilter

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_tokens_registered(self):
        reg = get_registry()
        for name in ("verify", "plan", "uncertain", "search", "lang", "debug"):
            assert name in reg.list_tokens()

    def test_get_config_unknown_raises(self):
        with pytest.raises(KeyError):
            get_registry().get_config("nonexistent")

    def test_strip_all_removes_strip_tokens(self):
        reg = get_registry()
        text = "<verify>ok</verify>hello<plan>step1</plan>world<debug>why</debug>"
        result = reg.strip_all(text)
        assert "hello" in result and "world" in result
        assert "<verify>" not in result
        assert "<plan>" not in result
        assert "<debug>" not in result

    def test_strip_all_keeps_uncertain(self):
        reg = get_registry()
        text = "maybe <uncertain>this part</uncertain> is wrong"
        result = reg.strip_all(result := text)
        assert "<uncertain>" in result   # uncertain has strip_from_output=False


# ---------------------------------------------------------------------------
# Base processor
# ---------------------------------------------------------------------------

class TestSpecialTokenProcessor:
    def _make(self, name, open_tag, close_tag):
        cfg = SpecialTokenConfig(
            name=name, open_tag=open_tag, close_tag=close_tag,
            seed_tokens=["foo"], strip_from_output=True,
        )
        return SpecialTokenProcessor(cfg)

    def test_parse_extracts_blocks(self):
        proc = self._make("x", "<x>", "</x>")
        result = proc.parse("hello <x>block1</x> world <x>block2</x>")
        assert result.clean_text == "hello  world"
        assert result.blocks == ["block1", "block2"]
        assert result.block_count == 2

    def test_parse_no_blocks(self):
        proc = self._make("x", "<x>", "</x>")
        result = proc.parse("plain text")
        assert result.clean_text == "plain text"
        assert result.blocks == []

    def test_strip(self):
        proc = self._make("x", "<x>", "</x>")
        assert proc.strip("pre<x>inner</x>post") == "prepost"

    def test_multiline_block(self):
        proc = self._make("x", "<x>", "</x>")
        result = proc.parse("<x>\nline1\nline2\n</x>after")
        assert "line1" in result.blocks[0]
        assert result.clean_text == "after"


# ---------------------------------------------------------------------------
# Streaming filter
# ---------------------------------------------------------------------------

class TestSpecialTokenStreamFilter:
    def _filter_for(self, name, open_tag, close_tag):
        cfg = SpecialTokenConfig(
            name=name, open_tag=open_tag, close_tag=close_tag,
            seed_tokens=[], strip_from_output=True,
        )
        return SpecialTokenStreamFilter(cfg)

    def _feed_all(self, f, tokens):
        return "".join(f.feed(t) for t in tokens) + f.flush()

    def test_passthrough_no_tags(self):
        f = self._filter_for("v", "<verify>", "</verify>")
        assert self._feed_all(f, ["hello", " world"]) == "hello world"

    def test_strips_complete_block(self):
        f = self._filter_for("v", "<verify>", "</verify>")
        tokens = list("pre<verify>inner</verify>post")
        assert self._feed_all(f, tokens) == "prepost"

    def test_partial_tag_held(self):
        f = self._filter_for("v", "<verify>", "</verify>")
        out = f.feed("<veri")
        assert out == ""  # partial open tag held in buffer

    def test_partial_tag_completes(self):
        f = self._filter_for("v", "<verify>", "</verify>")
        f.feed("<veri")
        f.feed("fy>inner</verify>")
        assert f.flush() == ""  # block stripped

    def test_flush_yields_partial_tag_buffer(self):
        f = self._filter_for("v", "<verify>", "</verify>")
        out = f.feed("<veri")
        assert out == ""        # held back: could be start of <verify>
        assert f.flush() == "<veri"  # not a complete tag, released on flush

    def test_unclosed_block_discarded_on_flush(self):
        f = self._filter_for("v", "<verify>", "</verify>")
        f.feed("<verify>unclosed content")
        assert f.flush() == ""


# ---------------------------------------------------------------------------
# <verify> token
# ---------------------------------------------------------------------------

class TestVerifyToken:
    def setup_method(self):
        self.proc = get_registry().get_processor("verify")

    def test_strips_verify_block(self):
        result = self.proc.parse("code<verify>this looks right</verify>more")
        assert result.clean_text == "codemore"
        assert result.blocks == ["this looks right"]

    def test_config(self):
        cfg = get_registry().get_config("verify")
        assert cfg.open_tag == "<verify>"
        assert cfg.strip_from_output is True


# ---------------------------------------------------------------------------
# <plan> token
# ---------------------------------------------------------------------------

class TestPlanToken:
    def setup_method(self):
        self.proc = get_registry().get_processor("plan")

    def test_strips_plan_block(self):
        result = self.proc.parse("<plan>step1\nstep2</plan>def foo(): ...")
        assert "step1" in result.blocks[0]
        assert "def foo" in result.clean_text

    def test_config(self):
        cfg = get_registry().get_config("plan")
        assert cfg.open_tag == "<plan>"
        assert cfg.strip_from_output is True


# ---------------------------------------------------------------------------
# <uncertain> token
# ---------------------------------------------------------------------------

class TestUncertainToken:
    def test_not_stripped_by_default(self):
        cfg = get_registry().get_config("uncertain")
        assert cfg.strip_from_output is False

    def test_kept_in_strip_all(self):
        text = "I think <uncertain>maybe not</uncertain> correct"
        result = get_registry().strip_all(text)
        assert "<uncertain>" in result

    def test_extractor_detects_uncertainty(self):
        ext = UncertainTokenExtractor()
        assert ext.has_uncertainty("x <uncertain>y</uncertain>")
        assert not ext.has_uncertainty("plain text")

    def test_extractor_strips_markers_keeps_text(self):
        ext = UncertainTokenExtractor()
        out = ext.strip_markers("hello <uncertain>world</uncertain>!")
        assert out == "hello world!"
        assert "<uncertain>" not in out

    def test_extractor_extracts_spans(self):
        ext = UncertainTokenExtractor()
        result = ext.extract("x<uncertain>span1</uncertain>y<uncertain>span2</uncertain>")
        assert result.blocks == ["span1", "span2"]


# ---------------------------------------------------------------------------
# <search> token
# ---------------------------------------------------------------------------

class TestSearchToken:
    def test_strips_when_no_retriever(self):
        handler = SearchTokenHandler()
        text = "before<search>my query</search>after"
        assert handler.process(text) == "beforeafter"

    def test_extracts_queries(self):
        handler = SearchTokenHandler()
        queries = handler.extract_queries("a<search>q1</search>b<search>q2</search>")
        assert queries == ["q1", "q2"]

    def test_count_searches(self):
        handler = SearchTokenHandler()
        assert handler.count_searches("<search>a</search><search>b</search>") == 2

    def test_retriever_called(self):
        class FakeRetriever:
            def retrieve(self, query):
                return [f"result for {query}"]

        handler = SearchTokenHandler(retriever=FakeRetriever())
        out = handler.process("answer: <search>capital of France</search> end")
        assert "result for capital of France" in out
        assert "<search>" not in out

    def test_retriever_exception_strips(self):
        class BrokenRetriever:
            def retrieve(self, query):
                raise RuntimeError("down")

        handler = SearchTokenHandler(retriever=BrokenRetriever())
        out = handler.process("x<search>q</search>y")
        assert "<search>" not in out


# ---------------------------------------------------------------------------
# <lang:XX> token
# ---------------------------------------------------------------------------

class TestLangToken:
    def setup_method(self):
        self.proc = LangTokenProcessor()

    def test_parse_returns_lang_and_block(self):
        text = "Hello <lang:gl>Ola mundo</lang> world"
        clean, blocks = self.proc.parse(text)
        assert clean == "Hello  world"
        assert blocks == [("gl", "Ola mundo")]

    def test_multiple_languages(self):
        text = "<lang:gl>Ola</lang> and <lang:pt>Olá</lang>"
        clean, blocks = self.proc.parse(text)
        assert len(blocks) == 2
        assert blocks[0][0] == "gl"
        assert blocks[1][0] == "pt"

    def test_get_languages_used(self):
        langs = self.proc.get_languages_used("<lang:gl>x</lang><lang:es>y</lang>")
        assert langs == ["gl", "es"]

    def test_has_lang_block(self):
        assert self.proc.has_lang_block("<lang:gl>hola</lang>", "gl")
        assert not self.proc.has_lang_block("<lang:gl>hola</lang>", "pt")

    def test_strip(self):
        out = self.proc.strip("before<lang:gl>Ola</lang>after")
        assert out == "beforeafter"

    def test_stream_filter_strips_lang_block(self):
        f = LangTokenStreamFilter()
        tokens = list("hi<lang:gl>Ola mundo</lang>bye")
        out = "".join(f.feed(t) for t in tokens) + f.flush()
        assert out == "hibye"

    def test_stream_filter_partial_tag_held(self):
        f = LangTokenStreamFilter()
        out = f.feed("<lang")
        assert out == ""


# ---------------------------------------------------------------------------
# <debug> token
# ---------------------------------------------------------------------------

class TestDebugToken:
    def setup_method(self):
        self.proc = get_registry().get_processor("debug")

    def test_strips_debug_block(self):
        text = "result<debug>off by one: i should be i-1</debug>fix"
        result = self.proc.parse(text)
        assert result.clean_text == "resultfix"
        assert "off by one" in result.blocks[0]

    def test_config(self):
        cfg = get_registry().get_config("debug")
        assert cfg.open_tag == "<debug>"
        assert cfg.strip_from_output is True


# ---------------------------------------------------------------------------
# Embedding initialization (requires numpy)
# ---------------------------------------------------------------------------

class TestEmbeddingInit:
    def test_embedding_init_runs(self):
        pytest.importorskip("numpy")
        import numpy as np
        cfg = SpecialTokenConfig(
            name="test_tok",
            open_tag="<test>",
            close_tag="</test>",
            seed_tokens=["check", "assert"],
            alpha=0.5,
            boundary_token="<im_start>",
        )
        proc = SpecialTokenProcessor(cfg)
        vocab_size, hidden = 100, 16
        E = np.random.randn(vocab_size, hidden).astype(np.float32)
        token_ids = {"check": 5, "assert": 10, "<im_start>": 1}
        emb = proc.initialize_embedding(E, token_ids)
        assert emb.shape == (hidden,)

    def test_missing_seed_tokens_fallback(self):
        pytest.importorskip("numpy")
        import numpy as np
        cfg = SpecialTokenConfig(
            name="t2", open_tag="<t2>", close_tag="</t2>",
            seed_tokens=["zzz_unknown"],
        )
        proc = SpecialTokenProcessor(cfg)
        E = np.random.randn(50, 8).astype(np.float32)
        emb = proc.initialize_embedding(E, {})
        assert emb.shape == (8,)

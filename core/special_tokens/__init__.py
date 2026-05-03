"""
Special-token framework for CapibaraGPT.

Generalizes the Think-Anywhere pattern (arXiv:2603.29957) to support
any number of structured meta-token types with:
- semantic-aware embedding initialization
- real-time streaming suppression
- parse / strip / batch processing
- global registry

Token types registered on import:
  verify, plan, uncertain, search, lang, debug
"""

from .base import SpecialTokenConfig, SpecialTokenProcessor, SpecialTokenStreamFilter, ParseResult
from .registry import SpecialTokenRegistry, get_registry, register_token

# Register all built-in token types
from . import verify       # noqa: F401
from . import plan         # noqa: F401
from . import uncertain    # noqa: F401
from . import search       # noqa: F401
from . import lang         # noqa: F401
from . import debug        # noqa: F401
from . import fact_check   # noqa: F401
from . import web_search   # noqa: F401

from .search import SearchTokenHandler
from .lang import LangTokenProcessor
from .fact_check import FactCheckHandler
from .web_search import WebSearchRetriever, WebSearchHandler, WebSearchResult

__all__ = [
    "SpecialTokenConfig",
    "SpecialTokenProcessor",
    "SpecialTokenStreamFilter",
    "ParseResult",
    "SpecialTokenRegistry",
    "get_registry",
    "register_token",
    "SearchTokenHandler",
    "LangTokenProcessor",
    "FactCheckHandler",
    "WebSearchRetriever",
    "WebSearchHandler",
    "WebSearchResult",
]

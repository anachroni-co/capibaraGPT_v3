#!/usr/bin/env python3
"""
Basic web scraping downloader for CapibaraGPT pipeline.

Uses stdlib-only networking and HTML parsing. If BeautifulSoup is available,
it will be used for CSS selectors; otherwise, it falls back to a minimal HTML
parser with tag/id/class selectors.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


_SELECTOR_RE = re.compile(
    r"^(?P<tag>[a-zA-Z0-9_-]+)?(?P<id>#[a-zA-Z0-9_-]+)?(?P<class>\.[a-zA-Z0-9_-]+)?$"
)


@dataclass
class _ParsedSelector:
    tag: Optional[str] = None
    element_id: Optional[str] = None
    class_name: Optional[str] = None

    @classmethod
    def from_selector(cls, selector: str) -> "_ParsedSelector":
        selector = selector.strip()
        if not selector:
            return cls()
        match = _SELECTOR_RE.match(selector)
        if not match:
            return cls(tag=selector)
        tag = match.group("tag")
        element_id = match.group("id")
        class_name = match.group("class")
        return cls(
            tag=tag if tag else None,
            element_id=element_id[1:] if element_id else None,
            class_name=class_name[1:] if class_name else None,
        )


class _SimpleHTMLExtractor(HTMLParser):
    def __init__(self, selectors: Dict[str, _ParsedSelector]):
        super().__init__()
        self._selectors = selectors
        self._active_counts = {key: 0 for key in selectors}
        self._buffers: Dict[str, List[str]] = {key: [] for key in selectors}

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]):
        attr_dict = {k: v for k, v in attrs}
        for name, selector in self._selectors.items():
            if selector.tag and selector.tag != tag:
                continue
            if selector.element_id and attr_dict.get("id") != selector.element_id:
                continue
            if selector.class_name:
                classes = (attr_dict.get("class") or "").split()
                if selector.class_name not in classes:
                    continue
            self._active_counts[name] += 1

    def handle_endtag(self, tag: str):
        for name, selector in self._selectors.items():
            if selector.tag and selector.tag != tag:
                continue
            if self._active_counts[name] > 0:
                self._active_counts[name] -= 1

    def handle_data(self, data: str):
        text = data.strip()
        if not text:
            return
        for name, count in self._active_counts.items():
            if count > 0:
                self._buffers[name].append(text)

    def get_results(self) -> Dict[str, str]:
        return {name: " ".join(parts).strip() for name, parts in self._buffers.items()}


class WebScrapingDownloader:
    """Simple web scraper with optional CSS selector support."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.timeout = float(config.get("timeout", 20))
        self.rate_limit = float(config.get("rate_limit", 0))
        self.user_agent = config.get(
            "user_agent",
            "CapibaraGPT/3.0 (+https://example.com)",
        )
        self._last_request = 0.0

    async def scrape_page(
        self,
        url: str,
        selectors: Optional[Dict[str, str]] = None,
        allow_html: bool = False,
    ) -> Dict[str, Any]:
        """Fetch a page and extract text using selectors."""
        html, status, final_url = await asyncio.to_thread(self._fetch_url, url)
        selector_map = selectors or {"title": "title", "content": "body"}
        extracted = self._extract_fields(html, selector_map)
        text_blob = " ".join(
            value for key, value in extracted.items() if value and key != "title"
        ).strip()
        payload = {
            "url": final_url,
            "status_code": status,
            "title": extracted.get("title", ""),
            "content": extracted.get("content", "") or text_blob,
            "text": text_blob or extracted.get("content", ""),
        }
        if allow_html:
            payload["html"] = html
        return payload

    async def scrape_bulk(
        self,
        urls: Iterable[str],
        selectors: Optional[Dict[str, str]] = None,
        max_concurrent: int = 5,
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _worker(target_url: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.scrape_page(target_url, selectors=selectors)

        tasks = [asyncio.create_task(_worker(url)) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        output: List[Dict[str, Any]] = []
        for item in results:
            if isinstance(item, Exception):
                logger.warning("Web scrape failed: %s", item)
                continue
            output.append(item)
        return output

    def _fetch_url(self, url: str) -> tuple[str, int, str]:
        if self.rate_limit > 0:
            elapsed = time.time() - self._last_request
            if elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed)
        req = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(req, timeout=self.timeout) as response:
            status = getattr(response, "status", 200)
            charset = response.headers.get_content_charset() or "utf-8"
            raw = response.read()
            html = raw.decode(charset, errors="replace")
            final_url = response.geturl()
        self._last_request = time.time()
        return html, int(status), final_url

    def _extract_fields(self, html: str, selectors: Dict[str, str]) -> Dict[str, str]:
        try:
            from bs4 import BeautifulSoup  # type: ignore
            soup = BeautifulSoup(html, "html.parser")
            extracted = {}
            for key, selector in selectors.items():
                nodes = soup.select(selector)
                text = " ".join(n.get_text(" ", strip=True) for n in nodes)
                extracted[key] = text.strip()
            return extracted
        except Exception:
            parsed_selectors = {
                key: _ParsedSelector.from_selector(sel)
                for key, sel in selectors.items()
            }
            parser = _SimpleHTMLExtractor(parsed_selectors)
            parser.feed(html)
            return parser.get_results()

    def resolve_links(self, base_url: str, links: Iterable[str]) -> List[str]:
        return [urljoin(base_url, link) for link in links]

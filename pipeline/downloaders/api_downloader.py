#!/usr/bin/env python3
"""
Basic API downloader for CapibaraGPT pipeline.

Uses stdlib HTTP requests (urllib) and supports pagination helpers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class APIDownloader:
    """API downloader with optional pagination support."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.base_url = config.get("base_url")
        self.timeout = float(config.get("timeout", 20))
        self.headers = config.get("headers", {})
        self.auth = config.get("auth", {})

    async def download(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = self._build_url(endpoint, params)
        return await asyncio.to_thread(self._fetch_json, url)

    async def download_paginated(
        self,
        endpoint: str,
        page_param: str = "page",
        per_page: int = 100,
        max_pages: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        page = 1
        items: List[Any] = []
        params = params.copy() if params else {}
        params.setdefault("per_page", per_page)

        while True:
            params[page_param] = page
            payload = await self.download(endpoint, params=params)
            batch = self._extract_items(payload)
            if not batch:
                break
            items.extend(batch)
            page += 1
            if max_pages and page > max_pages:
                break
        return items

    def _build_url(self, endpoint: str, params: Optional[Dict[str, Any]]) -> str:
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            url = endpoint
        else:
            base = self.base_url or ""
            url = urljoin(base.rstrip("/") + "/", endpoint.lstrip("/"))
        if params:
            parsed = urlparse(url)
            query = urlencode(params, doseq=True)
            url = urlunparse(parsed._replace(query=query))
        return url

    def _fetch_json(self, url: str) -> Any:
        headers = self._build_headers()
        req = Request(url, headers=headers)
        with urlopen(req, timeout=self.timeout) as response:
            raw = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
            text = raw.decode(charset, errors="replace")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("API response is not JSON for %s", url)
            return {"raw": text}

    def _build_headers(self) -> Dict[str, str]:
        headers = {"User-Agent": "CapibaraGPT/3.0 (+https://example.com)"}
        headers.update(self.headers)
        if self.auth:
            auth_type = self.auth.get("type")
            if auth_type == "bearer" and self.auth.get("token"):
                headers["Authorization"] = f"Bearer {self.auth['token']}"
            elif auth_type == "api_key":
                key_name = self.auth.get("header", "X-API-Key")
                headers[key_name] = self.auth.get("token", "")
        return headers

    def _extract_items(self, payload: Any) -> List[Any]:
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return []
        for key in ("items", "data", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return []

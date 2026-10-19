#!/usr/bin/env python3
"""
Direct file downloader for CapibaraGPT pipeline.

Supports HTTP/HTTPS downloads using stdlib urllib.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class DirectDownloader:
    """Download files directly over HTTP/HTTPS."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.timeout = float(config.get("timeout", 60))
        self.chunk_size = int(config.get("chunk_size", 1024 * 1024))
        self.headers = config.get("headers", {})

    async def download(
        self,
        url: str,
        destination: str,
        checksum: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(self._download_sync, url, destination, checksum)

    def _download_sync(
        self,
        url: str,
        destination: str,
        checksum: Optional[str],
    ) -> Dict[str, Any]:
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        req = Request(url, headers=self._build_headers())

        hash_name = None
        hash_obj = None
        if checksum:
            if ":" in checksum:
                hash_name, _ = checksum.split(":", 1)
            else:
                hash_name = "sha256"
            hash_obj = hashlib.new(hash_name)

        total_bytes = 0
        with urlopen(req, timeout=self.timeout) as response, open(dest_path, "wb") as target:
            while True:
                chunk = response.read(self.chunk_size)
                if not chunk:
                    break
                target.write(chunk)
                total_bytes += len(chunk)
                if hash_obj is not None:
                    hash_obj.update(chunk)

        checksum_ok = True
        if checksum and hash_obj is not None:
            _, expected = checksum.split(":", 1) if ":" in checksum else (hash_name, checksum)
            checksum_ok = hash_obj.hexdigest() == expected.lower()
            if not checksum_ok:
                logger.warning("Checksum mismatch for %s", dest_path)

        return {
            "url": url,
            "destination": str(dest_path),
            "bytes_downloaded": total_bytes,
            "checksum_ok": checksum_ok,
        }

    def _build_headers(self) -> Dict[str, str]:
        headers = {"User-Agent": "CapibaraGPT/3.0 (+https://example.com)"}
        headers.update(self.headers)
        return headers

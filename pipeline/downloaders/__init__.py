#!/usr/bin/env python3
"""
Data Downloaders Module.

Includes basic implementations for web scraping, API downloads, direct file
downloads, and the download orchestrator.
"""

from .web_scraper import WebScrapingDownloader
from .api_downloader import APIDownloader
from .direct_downloader import DirectDownloader
from .download_orchestrator import DownloadOrchestrator, DownloadTask, DownloadStats

__all__ = [
    "WebScrapingDownloader",
    "APIDownloader",
    "DirectDownloader",
    "DownloadOrchestrator",
    "DownloadTask",
    "DownloadStats",
]

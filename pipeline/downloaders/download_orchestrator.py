#!/usr/bin/env python3
"""
Download Orchestrator
====================

Coordinates all data download operations including web scraping, API calls, and direct downloads.
Integrates with the complete pipeline from raw data acquisition to training preparation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime, timedelta
import json
import time
from dataclasses import dataclass, asdict

from .web_scraper import WebScrapingDownloader
from .api_downloader import APIDownloader
from .direct_downloader import DirectDownloader

logger = logging.getLogger(__name__)

@dataclass
class DownloadTask:
    """Represents a download task."""
    task_id: str
    source_type: str  # 'web_scraping', 'api', 'direct'
    source_name: str
    target_path: str
    priority: int = 1  # 1=high, 2=medium, 3=low
    estimated_size_gb: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    status: str = "pending"  # pending, running, completed, failed
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class DownloadStats:
    """Download statistics and metrics."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    running_tasks: int = 0
    pending_tasks: int = 0
    total_downloaded_gb: float = 0.0
    download_speed_mbps: float = 0.0
    estimated_completion: Optional[str] = None
    success_rate: float = 0.0

class DownloadOrchestrator:
    """Orchestrates all data download operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.download_queue: List[DownloadTask] = []
        self.active_downloads: Dict[str, DownloadTask] = {}
        self.completed_downloads: List[DownloadTask] = []
        self.failed_downloads: List[DownloadTask] = []

        download_cfg = config.get("download", {})
        self.web_scraper = WebScrapingDownloader(download_cfg.get("web", {}))
        self.api_downloader = APIDownloader(download_cfg.get("api", {}))
        self.direct_downloader = DirectDownloader(download_cfg.get("direct", {}))
        
        # Performance tracking
        self.stats = DownloadStats()
        self.start_time = None
        self.last_stats_update = time.time()
        
        # Storage paths
        self.raw_data_path = Path(config.get("storage", {}).get("raw_data_path", "data/raw"))
        self.cache_path = Path(config.get("storage", {}).get("cache_path", "data/cache"))
        
        # Create directories
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Concurrency limits
        self.max_concurrent_downloads = config.get("processing", {}).get("max_workers", 4)
        self.semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
    
    def add_spanish_news_tasks(self) -> List[str]:
        """Add Spanish news scraping tasks."""
        download_cfg = self.config.get("download", {})
        source_urls = download_cfg.get("spanish_news", {}).get("source_urls", {})
        news_sources = [
            {
                "name": "elpais",
                "estimated_size": 2.0,
                "priority": 1,
                "urls": source_urls.get("elpais", ["https://elpais.com/"]),
            },
            {
                "name": "elmundo", 
                "estimated_size": 1.8,
                "priority": 1,
                "urls": source_urls.get("elmundo", ["https://www.elmundo.es/"]),
            },
            {
                "name": "abc",
                "estimated_size": 1.5,
                "priority": 1,
                "urls": source_urls.get("abc", ["https://www.abc.es/"]),
            },
            {
                "name": "lavanguardia",
                "estimated_size": 1.2,
                "priority": 2,
                "urls": source_urls.get("lavanguardia", ["https://www.lavanguardia.com/"]),
            }
        ]
        
        task_ids = []
        for source in news_sources:
            task = DownloadTask(
                task_id=f"news_{source['name']}_{int(time.time())}",
                source_type="web_scraping",
                source_name=source["name"],
                target_path=str(self.raw_data_path / "spanish_news" / source["name"]),
                priority=source["priority"],
                estimated_size_gb=source["estimated_size"],
                metadata={
                    "seed_urls": source["urls"],
                    "categories": ["politica", "economia", "tecnologia", "cultura", "sociedad"],
                    "max_articles_per_category": 1000,
                    "date_range_days": 30
                }
            )
            self.download_queue.append(task)
            task_ids.append(task.task_id)
        
        logger.info(f" Added {len(task_ids)} Spanish news download tasks")
        return task_ids
    
    def add_academic_tasks(self) -> List[str]:
        """Add Spanish academic repository tasks."""
        download_cfg = self.config.get("download", {})
        source_urls = download_cfg.get("spanish_academic", {}).get("source_urls", {})
        academic_sources = [
            {
                "name": "dialnet",
                "estimated_size": 3.5,
                "priority": 1,
                "urls": source_urls.get("dialnet", ["https://dialnet.unirioja.es/"]),
            },
            {
                "name": "redalyc",
                "estimated_size": 2.8,
                "priority": 1,
                "urls": source_urls.get("redalyc", ["https://www.redalyc.org/"]),
            },
            {
                "name": "scielo_es",
                "estimated_size": 2.2,
                "priority": 2,
                "urls": source_urls.get("scielo_es", ["https://scielo.isciii.es/"]),
            }
        ]
        
        task_ids = []
        for source in academic_sources:
            task = DownloadTask(
                task_id=f"academic_{source['name']}_{int(time.time())}",
                source_type="web_scraping",
                source_name=source["name"],
                target_path=str(self.raw_data_path / "spanish_academic" / source["name"]),
                priority=source["priority"],
                estimated_size_gb=source["estimated_size"],
                metadata={
                    "seed_urls": source["urls"],
                    "fields": ["computer_science", "mathematics", "physics", "engineering"],
                    "max_papers_per_field": 5000,
                    "include_abstracts": True,
                    "include_full_text": False  # Start with abstracts only
                }
            )
            self.download_queue.append(task)
            task_ids.append(task.task_id)
        
        logger.info(f" Added {len(task_ids)} Spanish academic download tasks")
        return task_ids
    
    def add_api_tasks(self) -> List[str]:
        """Add API-based download tasks."""
        download_cfg = self.config.get("download", {})
        api_urls = download_cfg.get("api_data", {}).get("endpoints", {})
        api_sources = [
            {
                "name": "boe_legal",
                "estimated_size": 1.0,
                "priority": 1,
                "endpoint": api_urls.get(
                    "boe_legal",
                    "https://www.boe.es/datosabiertos/api/boe/sumario/20240101",
                ),
            },
            {
                "name": "huggingface_spanish",
                "estimated_size": 5.0,
                "priority": 1,
                "endpoint": api_urls.get(
                    "huggingface_spanish",
                    "https://huggingface.co/api/datasets?limit=50",
                ),
            },
            {
                "name": "wikipedia_es_dumps",
                "estimated_size": 8.0,
                "priority": 2,
                "endpoint": api_urls.get(
                    "wikipedia_es_dumps",
                    "https://dumps.wikimedia.org/eswiki/latest/",
                ),
            }
        ]
        
        task_ids = []
        for source in api_sources:
            task = DownloadTask(
                task_id=f"api_{source['name']}_{int(time.time())}",
                source_type="api",
                source_name=source["name"],
                target_path=str(self.raw_data_path / "api_data" / source["name"]),
                priority=source["priority"],
                estimated_size_gb=source["estimated_size"],
                metadata={
                    "endpoint": source["endpoint"],
                    "update_frequency": "daily" if "boe" in source["name"] else "weekly",
                    "incremental": True
                }
            )
            self.download_queue.append(task)
            task_ids.append(task.task_id)
        
        logger.info(f" Added {len(task_ids)} API download tasks")
        return task_ids
    
    def add_direct_download_tasks(self) -> List[str]:
        """Add direct download tasks."""
        download_cfg = self.config.get("download", {})
        direct_urls = download_cfg.get("direct_downloads", {}).get("urls", {})
        direct_sources = [
            {
                "name": "opensubs_spanish",
                "estimated_size": 12.0,
                "priority": 3,
                "url": direct_urls.get(
                    "opensubs_spanish",
                    "https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/es.txt.zip",
                ),
            },
            {
                "name": "common_crawl_es",
                "estimated_size": 25.0,
                "priority": 3,
                "url": direct_urls.get(
                    "common_crawl_es",
                    "https://data.commoncrawl.org/crawl-data/CC-MAIN-2023-06/warc.paths.gz",
                ),
            }
        ]
        
        task_ids = []
        for source in direct_sources:
            task = DownloadTask(
                task_id=f"direct_{source['name']}_{int(time.time())}",
                source_type="direct",
                source_name=source["name"],
                target_path=str(self.raw_data_path / "direct_downloads" / source["name"]),
                priority=source["priority"],
                estimated_size_gb=source["estimated_size"],
                metadata={
                    "url": source["url"],
                    "compression": "gzip",
                    "chunk_size_mb": 100
                }
            )
            self.download_queue.append(task)
            task_ids.append(task.task_id)
        
        logger.info(f" Added {len(task_ids)} direct download tasks")
        return task_ids
    
    def setup_complete_download_pipeline(self) -> Dict[str, Any]:
        """Set up the complete download pipeline with all sources."""
        logger.info(" Setting up complete download pipeline...")
        
        # Add all task types
        news_tasks = self.add_spanish_news_tasks()
        academic_tasks = self.add_academic_tasks()
        api_tasks = self.add_api_tasks()
        direct_tasks = self.add_direct_download_tasks()
        
        # Sort queue by priority
        self.download_queue.sort(key=lambda x: x.priority)
        
        # Calculate totals
        total_tasks = len(self.download_queue)
        total_estimated_size = sum(task.estimated_size_gb for task in self.download_queue)
        
        pipeline_info = {
            "total_tasks": total_tasks,
            "estimated_total_size_gb": total_estimated_size,
            "task_breakdown": {
                "spanish_news": len(news_tasks),
                "spanish_academic": len(academic_tasks), 
                "api_sources": len(api_tasks),
                "direct_downloads": len(direct_tasks)
            },
            "priority_breakdown": {
                "high_priority": len([t for t in self.download_queue if t.priority == 1]),
                "medium_priority": len([t for t in self.download_queue if t.priority == 2]),
                "low_priority": len([t for t in self.download_queue if t.priority == 3])
            },
            "estimated_duration_hours": total_estimated_size / 2.0,  # Assuming 2GB/hour
            "setup_completed_at": datetime.now().isoformat()
        }
        
        logger.info(f" Pipeline setup complete:")
        logger.info(f"   Total tasks: {total_tasks}")
        logger.info(f"   Estimated size: {total_estimated_size:.1f}GB")
        logger.info(f"   High priority: {pipeline_info['priority_breakdown']['high_priority']} tasks")
        
        return pipeline_info
    
    async def execute_download_task(self, task: DownloadTask) -> bool:
        """Execute a single download task."""
        async with self.semaphore:
            task.status = "running"
            task.started_at = datetime.now().isoformat()
            self.active_downloads[task.task_id] = task
            
            logger.info(f" Starting download: {task.source_name} ({task.source_type})")
            
            try:
                # Execute download based on source type
                if task.source_type == "web_scraping":
                    success = await self._execute_web_scraping(task)
                elif task.source_type == "api":
                    success = await self._execute_api_download(task)
                elif task.source_type == "direct":
                    success = await self._execute_direct_download(task)
                else:
                    raise ValueError(f"Unknown source type: {task.source_type}")
                
                if success:
                    task.status = "completed"
                    task.completed_at = datetime.now().isoformat()
                    self.completed_downloads.append(task)
                    logger.info(f" Completed: {task.source_name}")
                else:
                    raise Exception("Download failed")
                
                return True
                
            except Exception as e:
                task.error_message = str(e)
                task.retry_count += 1
                
                if task.retry_count <= task.max_retries:
                    logger.warning(f"️ Retry {task.retry_count}/{task.max_retries}: {task.source_name}")
                    task.status = "pending"
                    # Add back to queue for retry
                    self.download_queue.insert(0, task)
                else:
                    task.status = "failed"
                    self.failed_downloads.append(task)
                    logger.error(f" Failed: {task.source_name} - {e}")
                
                return False
                
            finally:
                if task.task_id in self.active_downloads:
                    del self.active_downloads[task.task_id]
    
    async def _execute_web_scraping(self, task: DownloadTask) -> bool:
        """Execute web scraping download."""
        urls = task.metadata.get("seed_urls") if task.metadata else None
        if not urls:
            raise ValueError(f"No seed URLs provided for {task.source_name}")

        results = await self.web_scraper.scrape_bulk(
            urls=urls,
            selectors=task.metadata.get("selectors") if task.metadata else None,
            max_concurrent=self.max_concurrent_downloads,
        )

        Path(task.target_path).mkdir(parents=True, exist_ok=True)
        data_file = Path(task.target_path) / f"{task.source_name}_scraped.jsonl"
        with open(data_file, "w", encoding="utf-8") as f:
            for item in results:
                item["source"] = task.source_name
                item["timestamp"] = datetime.now().isoformat()
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f" Web scraping completed: {task.source_name}")
        return True

    
    async def _execute_api_download(self, task: DownloadTask) -> bool:
        """Execute API download."""
        if not task.metadata or not task.metadata.get("endpoint"):
            raise ValueError(f"No API endpoint configured for {task.source_name}")

        endpoint = task.metadata["endpoint"]
        if task.metadata.get("paginated"):
            data = await self.api_downloader.download_paginated(
                endpoint=endpoint,
                page_param=task.metadata.get("page_param", "page"),
                per_page=task.metadata.get("per_page", 100),
                max_pages=task.metadata.get("max_pages"),
            )
        else:
            data = await self.api_downloader.download(endpoint)

        Path(task.target_path).mkdir(parents=True, exist_ok=True)
        data_file = Path(task.target_path) / f"{task.source_name}_api_data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source": task.source_name,
                    "timestamp": datetime.now().isoformat(),
                    "data": data,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(f" API download completed: {task.source_name}")
        return True

    
    async def _execute_direct_download(self, task: DownloadTask) -> bool:
        """Execute direct download."""
        if not task.metadata or not task.metadata.get("url"):
            raise ValueError(f"No download URL configured for {task.source_name}")

        url = task.metadata["url"]
        filename = task.metadata.get("filename")
        if filename:
            destination = Path(task.target_path) / filename
        else:
            destination = Path(task.target_path) / Path(url).name

        await self.direct_downloader.download(
            url=url,
            destination=str(destination),
            checksum=task.metadata.get("checksum"),
        )

        logger.info(f" Direct download completed: {task.source_name}")
        return True

    
    def update_stats(self):
        """Update download statistics."""
        self.stats.total_tasks = len(self.download_queue) + len(self.completed_downloads) + len(self.failed_downloads) + len(self.active_downloads)
        self.stats.completed_tasks = len(self.completed_downloads)
        self.stats.failed_tasks = len(self.failed_downloads)
        self.stats.running_tasks = len(self.active_downloads)
        self.stats.pending_tasks = len(self.download_queue)
        
        if self.stats.total_tasks > 0:
            self.stats.success_rate = self.stats.completed_tasks / (self.stats.completed_tasks + self.stats.failed_tasks) if (self.stats.completed_tasks + self.stats.failed_tasks) > 0 else 0.0
        
        # Calculate downloaded size
        self.stats.total_downloaded_gb = sum(task.estimated_size_gb for task in self.completed_downloads)
        
        # Estimate completion time
        if self.stats.running_tasks > 0 or self.stats.pending_tasks > 0:
            remaining_gb = sum(task.estimated_size_gb for task in self.download_queue) + sum(task.estimated_size_gb for task in self.active_downloads.values())
            estimated_hours = remaining_gb / 2.0  # Assuming 2GB/hour
            estimated_completion = datetime.now() + timedelta(hours=estimated_hours)
            self.stats.estimated_completion = estimated_completion.isoformat()
    
    async def run_download_pipeline(self) -> Dict[str, Any]:
        """Run the complete download pipeline."""
        logger.info(" Starting download pipeline execution...")
        self.start_time = time.time()
        
        # Process queue
        while self.download_queue or self.active_downloads:
            # Start new downloads up to concurrency limit
            while (len(self.active_downloads) < self.max_concurrent_downloads and 
                   self.download_queue):
                task = self.download_queue.pop(0)
                asyncio.create_task(self.execute_download_task(task))
            
            # Update stats periodically
            current_time = time.time()
            if current_time - self.last_stats_update > 5.0:  # Every 5 seconds
                self.update_stats()
                self.last_stats_update = current_time
                
                logger.info(f" Progress: {self.stats.completed_tasks}/{self.stats.total_tasks} completed, "
                           f"{self.stats.running_tasks} running, {self.stats.pending_tasks} pending")
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(1.0)
        
        # Final stats update
        self.update_stats()
        execution_time = time.time() - self.start_time
        
        pipeline_result = {
            "execution_time_seconds": execution_time,
            "stats": asdict(self.stats),
            "completed_tasks": [asdict(task) for task in self.completed_downloads],
            "failed_tasks": [asdict(task) for task in self.failed_downloads],
            "total_data_downloaded_gb": self.stats.total_downloaded_gb,
            "success_rate": self.stats.success_rate,
            "completed_at": datetime.now().isoformat()
        }
        
        logger.info(f" Download pipeline completed!")
        logger.info(f"   Total time: {execution_time:.1f} seconds")
        logger.info(f"   Success rate: {self.stats.success_rate:.1%}")
        logger.info(f"   Data downloaded: {self.stats.total_downloaded_gb:.1f}GB")
        
        return pipeline_result

# Example usage
async def demo_download_pipeline():
    """Demonstrate the complete download pipeline."""
    config = {
        "storage": {
            "raw_data_path": "data/raw",
            "cache_path": "data/cache"
        },
        "processing": {
            "max_workers": 3
        }
    }
    
    orchestrator = DownloadOrchestrator(config)
    
    # Setup pipeline
    pipeline_info = orchestrator.setup_complete_download_pipeline()
    
    # Run pipeline
    result = await orchestrator.run_download_pipeline()
    
    return {"setup": pipeline_info, "execution": result}

if __name__ == "__main__":
    asyncio.run(demo_download_pipeline())

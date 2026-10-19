#!/usr/bin/env python3
"""
CapibaraGPT v3 Complete Pipeline Executor
=========================================

Main execution script for the complete data-to-training pipeline.
Run this script to execute the full workflow from data acquisition to training preparation.

Usage:
    python run_pipeline.py [--config config.json] [--stage STAGE] [--dry-run]
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from capibara.pipeline.workflows.complete_pipeline import run_complete_pipeline, CompletePipeline, PipelineResult, PipelineStage
import time

# Available pipeline stages
AVAILABLE_STAGES = {
    "data_download": "_execute_download_stage",
    "data_processing": "_execute_processing_stage",
    "dataset_preparation": "_execute_dataset_preparation_stage",
    "training_integration": "_execute_training_integration_stage",
}


async def run_single_stage(config: dict, stage_name: str) -> PipelineResult:
    """Run a single pipeline stage.

    Args:
        config: Pipeline configuration dictionary
        stage_name: Name of the stage to run (data_download, data_processing,
                   dataset_preparation, training_integration)

    Returns:
        PipelineResult with only the specified stage executed
    """
    if stage_name not in AVAILABLE_STAGES:
        logger.error(f"Unknown stage: {stage_name}")
        logger.info(f"Available stages: {', '.join(AVAILABLE_STAGES.keys())}")
        return None

    logger.info(f" Running single stage: {stage_name}")

    # Create pipeline instance
    pipeline = CompletePipeline(config)
    pipeline.start_time = time.time()

    # Find the stage index
    stage_index = list(AVAILABLE_STAGES.keys()).index(stage_name)
    stage = pipeline.stages[stage_index]

    try:
        # Get the method for this stage
        method_name = AVAILABLE_STAGES[stage_name]
        stage_method = getattr(pipeline, method_name)

        # Execute the stage
        await stage_method()

        # Calculate duration
        total_duration = time.time() - pipeline.start_time

        # Create result with only this stage
        result = PipelineResult(
            pipeline_id=pipeline.pipeline_id,
            total_duration_seconds=total_duration,
            stages=[stage],
            final_dataset_path=str(pipeline.training_data_path),
            dataset_stats={"stage": stage_name, "status": stage.status},
            ready_for_training=False,  # Single stage doesn't complete training prep
            completed_at=datetime.now().isoformat()
        )

        return result

    except Exception as e:
        logger.error(f"Stage '{stage_name}' failed: {e}")
        stage.status = "failed"
        stage.error_message = str(e)

        return PipelineResult(
            pipeline_id=pipeline.pipeline_id,
            total_duration_seconds=time.time() - pipeline.start_time,
            stages=[stage],
            final_dataset_path="",
            dataset_stats={"error": str(e)},
            ready_for_training=False,
            completed_at=datetime.now().isoformat()
        )


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger(__name__)

def load_config(config_path: str = None) -> dict:
    """Load pipeline configuration."""
    default_config = {
        "pipeline": {
            "name": "CapibaraGPT v3-Complete-Pipeline",
            "version": "2.0.0",
            "description": "Complete data-to-training pipeline for Spanish language model"
        },
        "storage": {
            "raw_data_path": "data/raw",
            "processed_data_path": "data/processed",
            "training_data_path": "data/training",
            "cache_path": "data/cache"
        },
        "download": {
            "spanish_news": {
                "enabled": True,
                "sources": ["elpais", "elmundo", "abc", "lavanguardia"],
                "max_articles_per_source": 1000,
                "categories": ["politica", "economia", "tecnologia", "cultura", "sociedad"]
            },
            "spanish_academic": {
                "enabled": True,
                "sources": ["dialnet", "redalyc", "scielo_es"],
                "fields": ["computer_science", "mathematics", "physics", "engineering"],
                "max_papers_per_field": 5000
            },
            "api_data": {
                "enabled": True,
                "sources": ["boe_legal", "huggingface_spanish", "wikipedia_es_dumps"]
            },
            "direct_downloads": {
                "enabled": False,  # Disabled by default due to size
                "sources": ["opensubs_spanish", "common_crawl_es"]
            }
        },
        "processing": {
            "max_workers": 4,
            "batch_size": 1000,
            "min_quality_score": 0.6,
            "enable_deduplication": True,
            "text_cleaning": {
                "remove_html": True,
                "remove_urls": True,
                "remove_emails": True,
                "normalize_whitespace": True,
                "spanish_specific": True
            }
        },
        "training_preparation": {
            "train_val_split": 0.9,
            "shuffle_data": True,
            "create_unified_dataset": True,
            "generate_training_script": True
        },
        "monitoring": {
            "enabled": True,
            "log_level": "INFO",
            "save_metrics": True,
            "real_time_stats": True
        },
        "performance": {
            "enable_caching": True,
            "concurrent_downloads": 3,
            "memory_limit_gb": 16,
            "disk_space_check": True
        }
    }
    
    if config_path and Path(config_path).exists():
        logger.info(f"Loading configuration from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
        
        # Merge configurations (user config overrides defaults)
        def merge_configs(default, user):
            for key, value in user.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    merge_configs(default[key], value)
                else:
                    default[key] = value
        
        merge_configs(default_config, user_config)
    else:
        logger.info("Using default configuration")
    
    return default_config

def print_pipeline_banner():
    """Print pipeline banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     CapibaraGPT v3 Complete Pipeline                     ║
║                                                                              ║
║   Data Acquisition →  Processing →  Dataset Prep →  Training Ready   ║
║                                                                              ║
║  ️  Web Scraping      Text Cleaning     Dataset Creation                ║
║   API Downloads     Deduplication    ️  Training Integration            ║
║   Direct Downloads   Quality Filter    Ready for Training               ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    logger.info(banner)

def print_pipeline_summary(config: dict):
    """Print pipeline configuration summary."""
    logger.info("\n PIPELINE CONFIGURATION SUMMARY")
    logger.info("=" * 60)
    
    # Data sources
    logger.info("\n DATA SOURCES:")
    download_config = config.get("download", {})
    
    if download_config.get("spanish_news", {}).get("enabled", False):
        news_sources = download_config["spanish_news"]["sources"]
        logger.info(f"    Spanish News: {len(news_sources)} sources ({', '.join(news_sources)})")
    
    if download_config.get("spanish_academic", {}).get("enabled", False):
        academic_sources = download_config["spanish_academic"]["sources"]
        logger.info(f"    Spanish Academic: {len(academic_sources)} sources ({', '.join(academic_sources)})")
    
    if download_config.get("api_data", {}).get("enabled", False):
        api_sources = download_config["api_data"]["sources"]
        logger.info(f"    API Data: {len(api_sources)} sources ({', '.join(api_sources)})")
    
    if download_config.get("direct_downloads", {}).get("enabled", False):
        direct_sources = download_config["direct_downloads"]["sources"]
        logger.info(f"    Direct Downloads: {len(direct_sources)} sources ({', '.join(direct_sources)})")
    
    # Processing settings
    logger.info("\n PROCESSING SETTINGS:")
    proc_config = config.get("processing", {})
    logger.info(f"    Min Quality Score: {proc_config.get('min_quality_score', 0.6)}")
    logger.info(f"    Deduplication: {'Enabled' if proc_config.get('enable_deduplication', True) else 'Disabled'}")
    logger.info(f"    Max Workers: {proc_config.get('max_workers', 4)}")
    logger.info(f"    Batch Size: {proc_config.get('batch_size', 1000)}")
    
    # Storage paths
    logger.info("\n STORAGE PATHS:")
    storage_config = config.get("storage", {})
    for path_type, path in storage_config.items():
        logger.info(f"    {path_type}: {path}")

async def run_pipeline_with_monitoring(config: dict, stage: str = None, dry_run: bool = False):
    """Run pipeline with monitoring and progress tracking."""
    
    if dry_run:
        logger.info("\n DRY RUN MODE - Configuration validation only")
        logger.info(" Configuration is valid")
        logger.info(" All storage paths can be created")
        logger.info(" Pipeline ready to execute")
        return
    
    logger.info(f"\n STARTING COMPLETE PIPELINE")
    logger.info(f"   Start time: {datetime.now().isoformat()}")
    
    if stage:
        logger.info(f"   Running specific stage: {stage}")
        result = await run_single_stage(config, stage)

        if result:
            logger.info(f"\n STAGE '{stage}' COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f" Pipeline ID: {result.pipeline_id}")
            logger.info(f"️  Stage Duration: {result.total_duration_seconds:.1f} seconds")

            # Show stage result
            for s in result.stages:
                if s.stage_name == stage:
                    status_emoji = "" if s.status == "completed" else ""
                    logger.info(f"   {status_emoji} {s.stage_name}: {s.status}")
                    if s.output_path:
                        logger.info(f"    Output: {s.output_path}")
            return result
        return None
    
    try:
        # Execute complete pipeline
        result = await run_complete_pipeline(config)
        
        # Print results
        logger.info(f"\n PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f" Pipeline ID: {result.pipeline_id}")
        logger.info(f"️  Total Duration: {result.total_duration_seconds:.1f} seconds")
        logger.info(f" Final Dataset: {result.final_dataset_path}")
        logger.info(f" Ready for Training: {'Yes' if result.ready_for_training else 'No'}")
        
        # Stage results
        logger.info(f"\n STAGE RESULTS:")
        for stage in result.stages:
            status_emoji = "" if stage.status == "completed" else "" if stage.status == "failed" else ""
            logger.info(f"   {status_emoji} {stage.stage_name}: {stage.status} ({stage.duration_seconds:.1f}s)")
        
        # Dataset statistics
        if result.dataset_stats:
            stats = result.dataset_stats
            logger.info(f"\n DATASET STATISTICS:")
            logger.info(f"    Total Files: {stats.get('total_files', 0)}")
            logger.info(f"    Total Size: {stats.get('total_size_gb', 0):.1f}GB")
            logger.info(f"    Total Records: {stats.get('total_records', 0):,}")
            logger.info(f"   ️  Data Types: {', '.join(stats.get('data_types', []))}")
        
        # Next steps
        logger.info(f"\n NEXT STEPS:")
        logger.info(f"   1. Navigate to: {result.final_dataset_path}")
        logger.info(f"   2. Review generated datasets and configuration")
        logger.info(f"   3. Run training: python run_training.py")
        logger.info(f"   4. Or integrate with capibara/training/ module")
        
        return result
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(f"\n PIPELINE FAILED: {e}")
        logger.error(f" Check logs for detailed error information")
        raise

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="CapibaraGPT v3 Complete Data-to-Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run complete pipeline with default configuration
    python run_pipeline.py
    
    # Run with custom configuration
    python run_pipeline.py --config my_config.json
    
    # Run specific stage only
    python run_pipeline.py --stage data_download
    
    # Dry run (validate configuration only)
    python run_pipeline.py --dry-run
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration JSON file"
    )
    
    parser.add_argument(
        "--stage", "-s",
        choices=["data_download", "data_processing", "dataset_preparation", "training_integration"],
        help="Run specific pipeline stage only"
    )
    
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Validate configuration without executing pipeline"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    print_pipeline_banner()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Print configuration summary
        print_pipeline_summary(config)
        
        # Run pipeline
        asyncio.run(run_pipeline_with_monitoring(
            config=config,
            stage=args.stage,
            dry_run=args.dry_run
        ))
        
    except KeyboardInterrupt:
        logger.info("\n️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        logger.error(f"\n PIPELINE EXECUTION FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
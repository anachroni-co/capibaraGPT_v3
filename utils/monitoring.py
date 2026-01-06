"""
utils monitoring module.

# This module provides functionality for monitoring.
"""

import logging
import time
import psutil
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Memory statistics container"""
    used: float = 0.0
    available: float = 0.0
    percent: float = 0.0
    timestamp: float = field(default_factory=time.time)

class MemoryMonitor:
    """Memory monitoring utility"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.stats_history: List[MemoryStats] = []
        self._monitoring = False
        
    def get_current_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        try:
            memory = psutil.virtual_memory()
            return MemoryStats(
                used=memory.used / (1024**2),  # MB
                available=memory.available / (1024**2),  # MB
                percent=memory.percent,
                timestamp=time.time()
            )
        except Exception as e:
            logger.warning(f"Failed to get memory stats: {e}")
            return MemoryStats()
    
    def start_monitoring(self):
        """Start memory monitoring"""
        self._monitoring = True
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self._monitoring = False
        logger.info("Memory monitoring stopped")
    
    def get_peak_usage(self) -> float:
        """Get peak memory usage from history"""
        if not self.stats_history:
            return 0.0
        return max(stats.used for stats in self.stats_history)
    
    def clear_history(self):
        """Clear monitoring history"""
        self.stats_history.clear()

def main():
    # Main function for this module.
    logger.info("Module monitoring.py starting")
    return True

if __name__ == "__main__":
    main()

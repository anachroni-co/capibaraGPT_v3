"""
System Information
Utilities for obtaining system information.
"""

import os
import sys
import time
import logging
import platform
import psutil #type: ignore
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SystemMonitor:
    """Monitor of system information."""

    def __init__(self):
        """
              Init  .

            TODO: Add detailed description.
            """
        self.system = platform.system()
        self.python_version = sys.version
        self.cpu_count = psutil.cpu_count()
        self.memory_total = psutil.virtual_memory().total

        logger.info(f" System monitor initialized")
        logger.info(f"System: {self.system}")
        logger.info(f"Python: {self.python_version}")
        logger.info(f"CPUs: {self.cpu_count}")
        logger.info(f"Memory: {self.memory_total / (1024**3):.1f} GB")

    def get_system_info(self) -> Dict[str, Any]:
        """
        Obtain system information.

        Returns:
            Dictionary with information
        """
        try:
            cpu_freq = psutil.cpu_freq()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            return {
                "system": {
                    "os": self.system,
                    "python": self.python_version,
                    "cpu_count": self.cpu_count,
                    "cpu_freq": {
                        "current": cpu_freq.current if cpu_freq else 0,
                        "min": cpu_freq.min if cpu_freq else 0,
                        "max": cpu_freq.max if cpu_freq else 0
                    }
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent
                }
            }
        except Exception as e:
            logger.warning(f"️ Error obtaining information: {e}")
            return {}

    def get_process_info(self, pid: Optional[int] = None) -> Dict[str, Any]:
        """
        Obtain process information.

        Args:
            pid: ID of the process or None for the current

        Returns:
            Dictionary with information
        """
        try:
            process = psutil.Process(pid) if pid else psutil.Process()

            with process.oneshot():
                return {
                    "pid": process.pid,
                    "name": process.name(),
                    "status": process.status(),
                    "cpu_percent": process.cpu_percent(),
                    "memory_percent": process.memory_percent(),
                    "memory_info": process.memory_info()._asdict(),
                    "num_threads": process.num_threads(),
                    "io_counters": process.io_counters()._asdict() if process.io_counters() else None
                }
        except Exception as e:
            logger.warning(f"️ Error obtaining process information: {e}")
            return {}

def get_system_info() -> Dict[str, Any]:
    """
    Obtain basic system information.

    Returns:
        Dictionary with information
    """
    monitor = SystemMonitor()
    return monitor.get_system_info()

def check_tpu_availability() -> bool:
    """
    Verify TPU availability.

    Returns:
        True if TPU is available
    """
    try:
        # Try import JAX
        import jax
        devices = jax.devices()
        return any("TPU" in str(device) for device in devices)
    except ImportError:
        return False
    except Exception as e:
        logger.warning(f"️ Error verifying TPU: {e}")
        return False

# Test code moved to capibara/tests/utils/test_utils_comprehensive.py
# To run: python -m capibara.tests.utils.test_utils_comprehensive

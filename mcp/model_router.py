"""
Model Router - Request routing and model execution.

This module provides routing functionality for model requests, handling
version resolution, resource accounting, and request execution within
the Model Control Plane (MCP).

Key Components:
    - ModelRequest: Data class for model request parameters
    - ModelRouter: Main router class for request handling

Author: Skydesk International Dev Team.
"""

from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
import asyncio
import logging
import uuid
from datetime import datetime
from .version_manager import VersionManager
from .resource_manager import ResourceManager

logger = logging.getLogger(__name__)

# ============================================================================
# REQUEST
# ============================================================================

@dataclass
class ModelRequest:
    model_id: str
    input_data: Dict[str, Any]
    version: Optional[str] = None
    priority: int = 1
    timeout: int = 30
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)

# ============================================================================
# ROUTER
# ============================================================================

class ModelRouter:
    def __init__(
        self,
        version_manager: VersionManager,
        resource_manager: ResourceManager,
        model_registry: Dict[str, Any],  # model_id -> callable
    ):
        self.version_manager = version_manager
        self.resource_manager = resource_manager
        self.model_registry = model_registry

        self.active_requests: Dict[str, ModelRequest] = {}
        self.request_history: List[Dict[str, Any]] = []

    async def route_request(self, request: ModelRequest) -> Dict[str, Any]:
        try:
            # Resolve version
            if request.version is None:
                request.version = await self.version_manager.get_active_version(
                    request.model_id
                )
                if request.version is None:
                    raise ValueError(f"No active version for {request.model_id}")

            # Register active request
            self.active_requests[request.request_id] = request

            # Resource accounting
            await self.resource_manager.update_usage(request.model_id)

            # Execute
            result = await asyncio.wait_for(
                self._execute(request),
                timeout=request.timeout
            )

            self._record(request, "success")
            return result

        except asyncio.TimeoutError:
            self._record(request, "timeout")
            raise

        except Exception as e:
            self._record(request, "error", str(e))
            raise

        finally:
            self.active_requests.pop(request.request_id, None)

    async def _execute(self, request: ModelRequest) -> Dict[str, Any]:
        if request.model_id not in self.model_registry:
            raise ValueError(f"Model {request.model_id} not registered")

        model = self.model_registry[request.model_id]

        # Inferencia real (sync o async)
        if asyncio.iscoroutinefunction(model):
            output = await model(request.input_data)
        else:
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(
                None, model, request.input_data
            )

        return {
            "model_id": request.model_id,
            "version": request.version,
            "output": output,
        }

    def _record(self, request: ModelRequest, status: str, error: Optional[str] = None):
        entry = {
            "request_id": request.request_id,
            "model_id": request.model_id,
            "version": request.version,
            "timestamp": datetime.now(),
            "status": status,
        }
        if error:
            entry["error"] = error
        self.request_history.append(entry)

    async def get_request_history(self, model_id: Optional[str] = None):
        if model_id:
            return [h for h in self.request_history if h["model_id"] == model_id]
        return self.request_history

    async def get_active_requests(self):
        return self.active_requests
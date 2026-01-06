"""
BTO Router Module for CapibaraGPT Core Routing System
"""

import logging
from typing import Any, Dict, Optional, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BtoRouterV2(ABC):
    """Base BTO Router V2 class"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.routes = {}
        self.initialized = False
        
    def initialize(self) -> bool:
        """Initialize the router"""
        try:
            self.initialized = True
            logger.info("BtoRouterV2 initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize BtoRouterV2: {e}")
            return False
    
    def add_route(self, path: str, handler: Any) -> bool:
        """Add a route to the router"""
        try:
            self.routes[path] = handler
            logger.debug(f"Added route: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to add route {path}: {e}")
            return False
    
    def route(self, request: Any) -> Any:
        """Route a request"""
        if not self.initialized:
            self.initialize()
            
        # Basic routing logic
        try:
            # Extract path from request if possible
            path = getattr(request, 'path', '/default')
            
            if path in self.routes:
                return self.routes[path](request)
            else:
                return self._default_handler(request)
                
        except Exception as e:
            logger.error(f"Routing error: {e}")
            return self._error_handler(request, e)
    
    def _default_handler(self, request: Any) -> Any:
        """Default request handler"""
        return {"status": "success", "message": "Default BTO router response"}
    
    def _error_handler(self, request: Any, error: Exception) -> Any:
        """Error handler"""
        return {"status": "error", "message": str(error)}
    
    def get_routes(self) -> Dict[str, Any]:
        """Get all registered routes"""
        return self.routes.copy()
    
    def remove_route(self, path: str) -> bool:
        """Remove a route"""
        try:
            if path in self.routes:
                del self.routes[path]
                logger.debug(f"Removed route: {path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove route {path}: {e}")
            return False
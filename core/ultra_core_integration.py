#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra Core Integration Module - CapibaraGPT

Enhanced integration between core and training modules with proper encoding handling,
comprehensive training integration, and optimized performance.
"""

"""Core Integration Module."""

import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Core imports
try:
    from capibara.core.router import EnhancedRouter, RouterConfig, RouterType
    from capibara.core.routing import CoreHttpRouter
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    EnhancedRouter = None
    CoreHttpRouter = None


@dataclass
class IntegrationResult:
    """Result of integration operation."""
    success: bool
    result: Any
    error_message: Optional[str] = None


class CoreIntegration:
    """Core integration between router components."""
    
    def __init__(self):
        self.enhanced_router = None
        self.core_router = None
        
        if CORE_AVAILABLE:
            try:
                self.enhanced_router = EnhancedRouter(RouterConfig(router_type=RouterType.HYBRID))
            except Exception as e:
                logger.warning(f"EnhancedRouter not available: {e}")
            
            try:
                self.core_router = CoreHttpRouter()
            except Exception as e:
                logger.warning(f"CoreHttpRouter not available: {e}")
    
    async def process(
        self, 
        request_data: Any, 
        request_type: str = "general",
        context: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """Process a request."""
        # Try enhanced router
        if self.enhanced_router:
            try:
                result = await self.enhanced_router.route_request(request_data, context)
                if result.success:
                    return IntegrationResult(success=True, result=result)
            except Exception as e:
                logger.debug(f"Enhanced router failed: {e}")
        
        # Fallback to core router
        if self.core_router:
            try:
                result = self.core_router.route(request_type, **(context or {}))
                return IntegrationResult(success=True, result=result)
            except Exception as e:
                logger.debug(f"Core router failed: {e}")
        
        return IntegrationResult(
            success=False, 
            result=None, 
            error_message="No router available"
        )


def create_integration() -> CoreIntegration:
    """Create integration instance."""
    return CoreIntegration()

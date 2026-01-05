"""
Acontext Integration Module for Capibara6

This module provides integration with Acontext for:
- Session management
- Context persistence
- Task observation
- Self-learning capabilities
"""
import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import httpx
import asyncio
from pydantic import BaseModel

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Load Acontext configuration
ACONTEXT_BASE_URL = os.getenv("ACONTEXT_BASE_URL", "http://localhost:8029/api/v1")  # Real Acontext server
ACONTEXT_API_KEY = os.getenv("ACONTEXT_API_KEY", "sk-ac-your-root-api-bearer-token")
ACONTEXT_PROJECT_ID = os.getenv("ACONTEXT_PROJECT_ID", "capibara6-project")

class AcontextSession(BaseModel):
    """Acontext Session representation"""
    id: str
    project_id: str
    space_id: Optional[str] = None
    created_at: datetime

class AcontextIntegration:
    """Integration class for Acontext platform"""

    def __init__(self):
        self.base_url = ACONTEXT_BASE_URL
        self.api_key = ACONTEXT_API_KEY
        self.project_id = ACONTEXT_PROJECT_ID
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0
        )
        self.enabled = bool(self.api_key and self.base_url)
        # Dictionary to keep track of spaces
        self.spaces = {}
        # Dictionary to keep track of sessions
        self.sessions = {}

        if self.enabled:
            logger.info(f"✅ Acontext integration enabled, connecting to: {self.base_url}")
            logger.info(f"📚 Project ID: {self.project_id}")
        else:
            logger.warning("⚠️ Acontext integration disabled - missing API key or base URL")
    
    async def create_session(self, project_id: str, space_id: Optional[str] = None) -> AcontextSession:
        """Create a new Acontext session"""
        if not self.enabled:
            raise HTTPException(status_code=503, detail="Acontext integration not enabled")

        try:
            payload = {}
            if space_id:
                payload["space_id"] = space_id

            response = await self.client.post(
                f"/project/{project_id}/session",
                json=payload
            )

            response.raise_for_status()
            data = response.json()

            session = AcontextSession(
                id=data["id"],
                project_id=project_id,
                space_id=space_id,
                created_at=datetime.now()
            )

            # Store the session in our local dictionary
            self.sessions[data["id"]] = session

            return session
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to create Acontext session: {e}")
            raise HTTPException(status_code=503, detail=f"Acontext session creation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error creating Acontext session: {e}")
            raise HTTPException(status_code=503, detail=f"Acontext session creation error: {str(e)}")
    
    async def send_message_to_session(self, session_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to an Acontext session"""
        if not self.enabled:
            return {"status": "disabled", "message_id": "n/a"}
        
        try:
            payload = {
                "blob": message,
                "format": "openai"
            }
            
            response = await self.client.post(
                f"/session/{session_id}/messages",
                json=payload
            )
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to send message to Acontext session {session_id}: {e}")
            return {"status": "error", "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error sending message to Acontext session {session_id}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def flush_session(self, project_id: str, session_id: str) -> Dict[str, Any]:
        """Flush the session buffer to process tasks"""
        if not self.enabled:
            return {"status": "disabled"}
        
        try:
            response = await self.client.post(
                f"/project/{project_id}/session/{session_id}/flush"
            )
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to flush Acontext session {session_id}: {e}")
            return {"status": "error", "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error flushing Acontext session {session_id}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_session_tasks(self, session_id: str) -> List[Dict[str, Any]]:
        """Get tasks for a session"""
        if not self.enabled:
            return []
        
        try:
            # Note: This endpoint might not exist in the current Acontext API
            # We'll need to check actual API documentation
            response = await self.client.get(
                f"/session/{session_id}/tasks"
            )
            
            response.raise_for_status()
            return response.json().get("items", [])
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to get tasks for Acontext session {session_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting tasks for Acontext session {session_id}: {e}")
            return []
    
    async def create_space(self, project_id: str, name: str) -> Dict[str, Any]:
        """Create a new space for learning"""
        if not self.enabled:
            return {"status": "disabled", "id": "n/a"}

        try:
            payload = {
                "name": name,
                "project_id": project_id
            }

            response = await self.client.post(
                f"/project/{project_id}/space",
                json=payload
            )

            response.raise_for_status()
            result = response.json()

            # Store the space in our local dictionary
            space_id = result["id"]
            self.spaces[space_id] = {
                "id": space_id,
                "name": name,
                "project_id": project_id,
                "created_at": datetime.now().isoformat()
            }

            return result
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to create Acontext space: {e}")
            return {"status": "error", "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error creating Acontext space: {e}")
            return {"status": "error", "error": str(e)}
    
    async def search_space(self, space_id: str, query: str, mode: str = "fast", limit: int = 10) -> Dict[str, Any]:
        """Search a space for learned experiences with enhanced capabilities"""
        if not self.enabled:
            return {"cited_blocks": [], "query": query, "search_metadata": {"total_results": 0, "mode": mode, "limit": limit}}

        try:
            # Use project_id for the search endpoint as required by Acontext API
            response = await self.client.get(
                f"/project/{self.project_id}/space/{space_id}/experience_search",
                params={
                    "query": query,
                    "mode": mode,
                    "limit": limit
                }
            )

            response.raise_for_status()
            result = response.json()

            # Enhance the search results with additional metadata and context
            cited_blocks = result.get("cited_blocks", [])

            # Add search metadata for better context
            enhanced_result = {
                "cited_blocks": cited_blocks,
                "query": query,
                "search_metadata": {
                    "total_results": len(cited_blocks),
                    "mode": mode,
                    "limit": limit,
                    "space_id": space_id,
                    "search_date": datetime.now().isoformat()
                }
            }

            # If we have results, add additional processing to enrich them
            if cited_blocks:
                logger.info(f"🔍 Found {len(cited_blocks)} relevant experiences for query: '{query[:50]}...'")

                # Add relevance scoring if not present
                for block in cited_blocks:
                    if "relevance_score" not in block:
                        block["relevance_score"] = block.get("distance", 0.5)  # Default relevance based on distance

                # Sort by relevance if available
                cited_blocks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            else:
                logger.info(f"🔍 No relevant experiences found for query: '{query[:50]}...'")

            return enhanced_result
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to search Acontext space {space_id}: {e}")
            return {
                "cited_blocks": [],
                "error": str(e),
                "query": query,
                "search_metadata": {"total_results": 0, "mode": mode, "limit": limit, "error": str(e)}
            }
        except Exception as e:
            logger.error(f"Unexpected error searching Acontext space {space_id}: {e}")
            return {
                "cited_blocks": [],
                "error": str(e),
                "query": query,
                "search_metadata": {"total_results": 0, "mode": mode, "limit": limit, "error": str(e)}
            }
    
    async def close(self):
        """Close the HTTP client"""
        if self.client:
            await self.client.aclose()

# Global instance
acontext_client = AcontextIntegration()

async def get_acontext_client():
    """Dependency to get Acontext client"""
    return acontext_client
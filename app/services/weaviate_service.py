"""
Weaviate service for AI Agent Memory Router.

This service provides high-level operations for integrating with Weaviate
vector database for semantic memory storage and retrieval.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from uuid import uuid4

from app.core.weaviate_client import WeaviateClient, get_weaviate_client
from app.core.logging import get_logger
from app.models.memory import MemoryItem, MemoryMetadata
from app.models.agent import Agent

# Setup logger
logger = get_logger(__name__)


class WeaviateService:
    """Service for Weaviate vector database integration."""
    
    def __init__(self, weaviate_client: WeaviateClient = None):
        """Initialize Weaviate service.
        
        Args:
            weaviate_client: Weaviate client instance
        """
        self.weaviate_client = weaviate_client or WeaviateClient()
        self._initialized = False
        
        logger.info("Weaviate service initialized")
    
    async def initialize(self) -> bool:
        """Initialize the Weaviate service.
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize Weaviate client
            success = await self.weaviate_client.initialize()
            
            if success:
                self._initialized = True
                logger.info("Weaviate service initialized successfully")
                return True
            else:
                logger.error("Failed to initialize Weaviate client")
                return False
            
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate service: {e}")
            return False
    
    async def close(self) -> None:
        """Close Weaviate service connections."""
        try:
            await self.weaviate_client.close()
            self._initialized = False
            logger.info("Weaviate service closed")
        except Exception as e:
            logger.error(f"Error closing Weaviate service: {e}")
    
    # Memory Operations
    
    async def store_memory(
        self,
        memory_id: str,
        content: str,
        agent_id: str,
        memory_type: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> bool:
        """Store a memory in Weaviate.
        
        Args:
            memory_id: Unique identifier for the memory
            content: The memory content text
            agent_id: ID of the agent that created this memory
            memory_type: Type of memory
            importance: Importance score (0.0 to 1.0)
            tags: Optional tags
            metadata: Optional metadata
            project_id: Optional project ID
            expires_at: Optional expiration date
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            success = await self.weaviate_client.store_memory(
                memory_id=memory_id,
                content=content,
                agent_id=agent_id,
                memory_type=memory_type,
                importance=importance,
                tags=tags,
                metadata=metadata,
                project_id=project_id,
                expires_at=expires_at
            )
            
            if success:
                logger.info(f"Memory stored successfully in Weaviate: {memory_id}")
            else:
                logger.error(f"Failed to store memory in Weaviate: {memory_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing memory in Weaviate: {e}")
            return False
    
    async def retrieve_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by ID.
        
        Args:
            memory_id: Memory ID to retrieve
            
        Returns:
            Memory data or None if not found
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            memory_data = await self.weaviate_client.get_memory(memory_id)
            
            if memory_data:
                logger.debug(f"Memory retrieved successfully from Weaviate: {memory_id}")
            else:
                logger.debug(f"Memory not found in Weaviate: {memory_id}")
            
            return memory_data
            
        except Exception as e:
            logger.error(f"Error retrieving memory from Weaviate: {e}")
            return None
    
    async def search_memories(
        self,
        query: str,
        agent_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        project_id: Optional[str] = None,
        min_importance: Optional[float] = None,
        limit: int = 10,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search memories using semantic search.
        
        Args:
            query: Search query text
            agent_id: Filter by agent ID
            memory_type: Filter by memory type
            project_id: Filter by project ID
            min_importance: Minimum importance threshold
            limit: Maximum number of results
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of search results
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            results = await self.weaviate_client.search_memories(
                query=query,
                agent_id=agent_id,
                memory_type=memory_type,
                project_id=project_id,
                min_importance=min_importance,
                limit=limit,
                similarity_threshold=similarity_threshold
            )
            
            logger.info(f"Found {len(results)} memories for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching memories in Weaviate: {e}")
            return []
    
    async def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update a memory in Weaviate.
        
        Args:
            memory_id: Memory ID to update
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            success = await self.weaviate_client.update_memory(memory_id, updates)
            
            if success:
                logger.info(f"Memory updated successfully in Weaviate: {memory_id}")
            else:
                logger.error(f"Failed to update memory in Weaviate: {memory_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating memory in Weaviate: {e}")
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from Weaviate.
        
        Args:
            memory_id: Memory ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            success = await self.weaviate_client.delete_memory(memory_id)
            
            if success:
                logger.info(f"Memory deleted successfully from Weaviate: {memory_id}")
            else:
                logger.error(f"Failed to delete memory from Weaviate: {memory_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting memory from Weaviate: {e}")
            return False
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memories in Weaviate.
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            stats = await self.weaviate_client.get_collection_stats()
            
            logger.debug("Retrieved memory statistics from Weaviate")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats from Weaviate: {e}")
            return {"error": str(e)}
    
    async def clear_all_memories(self) -> bool:
        """Clear all memories from Weaviate.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            success = await self.weaviate_client.clear_all_memories()
            
            if success:
                logger.info("All memories cleared from Weaviate")
            else:
                logger.error("Failed to clear memories from Weaviate")
            
            return success
            
        except Exception as e:
            logger.error(f"Error clearing memories from Weaviate: {e}")
            return False
    
    # Advanced Operations
    
    async def search_similar_memories(
        self,
        memory_id: str,
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find memories similar to a given memory.
        
        Args:
            memory_id: Memory ID to find similar memories for
            limit: Maximum number of similar memories to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar memories
        """
        try:
            # First get the original memory
            original_memory = await self.retrieve_memory(memory_id)
            if not original_memory:
                return []
            
            # Search for similar memories using the content
            similar_memories = await self.search_memories(
                query=original_memory.get("content", ""),
                limit=limit + 1,  # +1 to exclude the original
                similarity_threshold=similarity_threshold
            )
            
            # Filter out the original memory
            filtered_memories = [
                memory for memory in similar_memories
                if memory.get("id") != memory_id
            ]
            
            return filtered_memories[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar memories: {e}")
            return []
    
    async def get_memories_by_agent(
        self,
        agent_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get all memories for a specific agent.
        
        Args:
            agent_id: Agent ID to get memories for
            limit: Maximum number of memories to return
            offset: Number of memories to skip
            
        Returns:
            List of memories for the agent
        """
        try:
            # Use a broad search to get all memories for the agent
            memories = await self.search_memories(
                query="",  # Empty query to get all
                agent_id=agent_id,
                limit=limit + offset
            )
            
            # Apply offset
            return memories[offset:offset + limit]
            
        except Exception as e:
            logger.error(f"Error getting memories for agent {agent_id}: {e}")
            return []
    
    async def get_memories_by_project(
        self,
        project_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get all memories for a specific project.
        
        Args:
            project_id: Project ID to get memories for
            limit: Maximum number of memories to return
            offset: Number of memories to skip
            
        Returns:
            List of memories for the project
        """
        try:
            # Use a broad search to get all memories for the project
            memories = await self.search_memories(
                query="",  # Empty query to get all
                project_id=project_id,
                limit=limit + offset
            )
            
            # Apply offset
            return memories[offset:offset + limit]
            
        except Exception as e:
            logger.error(f"Error getting memories for project {project_id}: {e}")
            return []
    
    async def get_memory_timeline(
        self,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get memories in chronological order.
        
        Args:
            agent_id: Filter by agent ID
            project_id: Filter by project ID
            start_date: Start date for timeline
            end_date: End date for timeline
            limit: Maximum number of memories to return
            
        Returns:
            List of memories in chronological order
        """
        try:
            # Get memories (this would need to be enhanced to support date filtering)
            memories = await self.search_memories(
                query="",
                agent_id=agent_id,
                project_id=project_id,
                limit=limit
            )
            
            # Sort by creation date
            sorted_memories = sorted(
                memories,
                key=lambda x: x.get("created_at", ""),
                reverse=True
            )
            
            return sorted_memories
            
        except Exception as e:
            logger.error(f"Error getting memory timeline: {e}")
            return []


# Global service instance
_weaviate_service: Optional[WeaviateService] = None


async def get_weaviate_service() -> WeaviateService:
    """Get or create global Weaviate service instance."""
    global _weaviate_service
    
    if _weaviate_service is None:
        weaviate_client = await get_weaviate_client()
        _weaviate_service = WeaviateService(weaviate_client)
        await _weaviate_service.initialize()
    
    return _weaviate_service


async def close_weaviate_service() -> None:
    """Close global Weaviate service instance."""
    global _weaviate_service
    
    if _weaviate_service:
        await _weaviate_service.close()
        _weaviate_service = None
"""
Cipher service for AI Agent Memory Router.

This service provides high-level operations for integrating with Cipher MCP server
for project-based memory storage and retrieval, with SQLite local storage for
metadata and caching.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from uuid import uuid4
import hashlib

from app.core.cipher_client import CipherAPIClient, CipherMCPError, CipherConnectionError, CipherOperationError
from app.core.sqlite_storage import SQLiteUnifiedStorage
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.memory import MemoryItem, MemoryMetadata
from app.models.agent import Agent

# Get settings
settings = get_settings()

# Setup logger
logger = get_logger(__name__)


class CipherService:
    """Service for Cipher MCP integration with local SQLite storage."""
    
    def __init__(self, sqlite_storage: SQLiteUnifiedStorage, cipher_client: CipherAPIClient = None):
        """Initialize Cipher service.
        
        Args:
            sqlite_storage: SQLite storage instance for local metadata
            cipher_client: Cipher MCP client instance
        """
        self.sqlite_storage = sqlite_storage
        self.cipher_client = cipher_client or CipherAPIClient()
        self._cache_ttl = timedelta(hours=settings.cipher.cache_ttl_hours)
        self._max_retries = settings.cipher.max_retries
        self._retry_delay = settings.cipher.retry_delay
        
        logger.info("Cipher service initialized")
    
    async def initialize(self) -> bool:
        """Initialize the Cipher service.
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize SQLite storage
            await self.sqlite_storage.initialize()
            
            # Test Cipher connection
            await self.cipher_client.connect()
            health = await self.cipher_client.health_check()
            
            logger.info("Cipher service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Cipher service: {e}")
            return False
    
    async def close(self) -> None:
        """Close Cipher service connections."""
        try:
            await self.cipher_client.disconnect()
            await self.sqlite_storage.close()
            logger.info("Cipher service closed")
        except Exception as e:
            logger.error(f"Error closing Cipher service: {e}")
    
    # Memory Operations with Hybrid Storage
    
    async def store_memory(
        self,
        project_id: str,
        agent_id: str,
        memory_content: str,
        memory_type: str = "general",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: int = 1,
        expires_at: Optional[datetime] = None
    ) -> str:
        """Store memory with hybrid storage (Cipher + SQLite).
        
        Args:
            project_id: Project identifier
            agent_id: Agent identifier
            memory_content: Memory content
            memory_type: Type of memory
            tags: Optional tags
            metadata: Optional metadata
            priority: Memory priority (1-10)
            expires_at: Optional expiration time
            
        Returns:
            Memory ID
        """
        memory_id = str(uuid4())
        
        try:
            logger.info(f"Storing memory {memory_id} for project {project_id}")
            
            # Store full content in Cipher
            cipher_response = await self._store_in_cipher(
                project_id=project_id,
                memory_id=memory_id,
                memory_content=memory_content,
                memory_type=memory_type,
                tags=tags,
                metadata=metadata
            )
            
            # Store metadata in SQLite for fast access
            memory_item = MemoryItem(
                id=memory_id,
                agent_id=agent_id,
                content=memory_content,  # Keep a copy for local access
                memory_type=memory_type,
                priority=priority,
                expires_at=expires_at
            )
            
            await self.sqlite_storage.memory.store_memory(memory_item)
            
            # Store additional metadata
            if tags or metadata:
                memory_metadata = MemoryMetadata(
                    id=str(uuid4()),
                    memory_id=memory_id,
                    tags=tags or [],
                    source="cipher",
                    confidence=0.9,
                    embedding_vector=None,  # Will be populated by Cipher
                    vector_dimension=None
                )
                await self.sqlite_storage.metadata.store_metadata(memory_metadata)
            
            # Store routing information
            await self._store_routing_info(
                memory_id=memory_id,
                project_id=project_id,
                agent_id=agent_id,
                cipher_memory_id=cipher_response.get("cipher_memory_id", memory_id)
            )
            
            logger.info(f"Memory stored successfully: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory {memory_id}: {e}")
            # Cleanup on failure
            await self._cleanup_failed_storage(memory_id)
            raise
    
    async def retrieve_memory(
        self,
        project_id: str,
        memory_id: str,
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Retrieve memory with hybrid storage.
        
        Args:
            project_id: Project identifier
            memory_id: Memory identifier
            use_cache: Whether to use local cache
            
        Returns:
            Memory data or None if not found
        """
        try:
            logger.debug(f"Retrieving memory {memory_id} for project {project_id}")
            
            # First try local SQLite for metadata
            local_memory = await self.sqlite_storage.memory.get_memory(memory_id)
            if not local_memory:
                logger.debug(f"Memory not found in local storage: {memory_id}")
                return None
            
            # Check if we should use cached content
            if use_cache and self._is_cache_valid(local_memory):
                logger.debug(f"Using cached content for memory: {memory_id}")
                return self._format_memory_response(local_memory, from_cache=True)
            
            # Retrieve full content from Cipher
            try:
                cipher_memory = await self.cipher_client.retrieve_memory(project_id, memory_id)
                if cipher_memory:
                    # Update local cache with fresh content
                    await self._update_local_cache(memory_id, cipher_memory)
                    return self._format_memory_response(local_memory, cipher_memory)
                else:
                    logger.warning(f"Memory not found in Cipher: {memory_id}")
                    return None
                    
            except CipherConnectionError:
                logger.warning(f"Cipher unavailable, using local cache for memory: {memory_id}")
                return self._format_memory_response(local_memory, from_cache=True)
            
        except Exception as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            raise
    
    async def search_memories(
        self,
        project_id: str,
        query: str,
        agent_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Search memories with hybrid storage.
        
        Args:
            project_id: Project identifier
            query: Search query
            agent_id: Optional agent filter
            memory_type: Optional memory type filter
            tags: Optional tags filter
            limit: Maximum results
            offset: Result offset
            
        Returns:
            Search results
        """
        try:
            logger.info(f"Searching memories for project {project_id} with query: {query}")
            
            # Build search filters
            filters = {}
            if memory_type:
                filters["memory_type"] = memory_type
            if tags:
                filters["tags"] = tags
            
            # Try Cipher search first
            try:
                cipher_results = await self.cipher_client.search_memories(
                    project_id=project_id,
                    query=query,
                    filters=filters,
                    limit=limit,
                    offset=offset
                )
                
                # Enhance results with local metadata
                enhanced_results = await self._enhance_search_results(cipher_results, agent_id)
                
                logger.info(f"Memory search completed: {len(enhanced_results.get('results', []))} results")
                return enhanced_results
                
            except CipherConnectionError:
                logger.warning("Cipher unavailable, falling back to local search")
                return await self._local_search_fallback(
                    project_id=project_id,
                    query=query,
                    agent_id=agent_id,
                    memory_type=memory_type,
                    limit=limit,
                    offset=offset
                )
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            raise
    
    async def update_memory(
        self,
        project_id: str,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update memory in both Cipher and SQLite.
        
        Args:
            project_id: Project identifier
            memory_id: Memory identifier
            updates: Updates to apply
            
        Returns:
            True if update successful
        """
        try:
            logger.info(f"Updating memory {memory_id} for project {project_id}")
            
            # Update in Cipher
            try:
                await self.cipher_client.update_memory(project_id, memory_id, updates)
            except CipherConnectionError:
                logger.warning(f"Cipher unavailable, updating local cache only for memory: {memory_id}")
            
            # Update local SQLite
            await self.sqlite_storage.memory.update_memory(memory_id, updates)
            
            # Update metadata if needed
            if any(key in updates for key in ['tags', 'metadata']):
                metadata_updates = {k: v for k, v in updates.items() if k in ['tags', 'metadata']}
                await self.sqlite_storage.metadata.update_metadata(memory_id, metadata_updates)
            
            logger.info(f"Memory updated successfully: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            raise
    
    async def delete_memory(
        self,
        project_id: str,
        memory_id: str
    ) -> bool:
        """Delete memory from both Cipher and SQLite.
        
        Args:
            project_id: Project identifier
            memory_id: Memory identifier
            
        Returns:
            True if deletion successful
        """
        try:
            logger.info(f"Deleting memory {memory_id} for project {project_id}")
            
            # Delete from Cipher
            try:
                await self.cipher_client.delete_memory(project_id, memory_id)
            except CipherConnectionError:
                logger.warning(f"Cipher unavailable, deleting from local storage only: {memory_id}")
            
            # Delete from local SQLite
            await self.sqlite_storage.memory.delete_memory(memory_id)
            await self.sqlite_storage.metadata.delete_metadata(memory_id)
            
            logger.info(f"Memory deleted successfully: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            raise
    
    # Project Management
    
    async def create_project(
        self,
        project_id: str,
        project_name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new project.
        
        Args:
            project_id: Project identifier
            project_name: Project name
            description: Optional description
            metadata: Optional metadata
            
        Returns:
            True if creation successful
        """
        try:
            logger.info(f"Creating project {project_id}: {project_name}")
            
            # Create in Cipher
            await self.cipher_client.create_project(
                project_id=project_id,
                project_name=project_name,
                description=description,
                metadata=metadata
            )
            
            # Store project info locally for fast access
            await self._store_project_info(project_id, project_name, description, metadata)
            
            logger.info(f"Project created successfully: {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create project {project_id}: {e}")
            raise
    
    async def get_project_info(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project information.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Project information or None if not found
        """
        try:
            # Try local cache first
            local_info = await self._get_local_project_info(project_id)
            if local_info:
                return local_info
            
            # Fallback to Cipher
            return await self.cipher_client.get_project(project_id)
            
        except Exception as e:
            logger.error(f"Failed to get project info {project_id}: {e}")
            return None
    
    # Utility Methods
    
    async def _store_in_cipher(
        self,
        project_id: str,
        memory_id: str,
        memory_content: str,
        memory_type: str,
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Store memory in Cipher with retry logic."""
        for attempt in range(self._max_retries):
            try:
                return await self.cipher_client.store_memory(
                    project_id=project_id,
                    memory_content=memory_content,
                    memory_type=memory_type,
                    tags=tags,
                    metadata=metadata
                )
            except CipherConnectionError as e:
                if attempt < self._max_retries - 1:
                    logger.warning(f"Cipher connection failed (attempt {attempt + 1}), retrying: {e}")
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
                    continue
                else:
                    logger.error(f"Cipher connection failed after {self._max_retries} attempts: {e}")
                    raise
    
    async def _store_routing_info(
        self,
        memory_id: str,
        project_id: str,
        agent_id: str,
        cipher_memory_id: str
    ) -> None:
        """Store routing information in SQLite."""
        # This would integrate with the routing storage
        # For now, we'll store basic routing metadata
        routing_data = {
            "memory_id": memory_id,
            "project_id": project_id,
            "agent_id": agent_id,
            "cipher_memory_id": cipher_memory_id,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Store in a routing table (would need to be added to schema)
        logger.debug(f"Storing routing info for memory {memory_id}")
    
    async def _cleanup_failed_storage(self, memory_id: str) -> None:
        """Clean up failed storage attempts."""
        try:
            await self.sqlite_storage.memory.delete_memory(memory_id)
            await self.sqlite_storage.metadata.delete_metadata(memory_id)
        except Exception as e:
            logger.error(f"Failed to cleanup memory {memory_id}: {e}")
    
    def _is_cache_valid(self, memory_item: MemoryItem) -> bool:
        """Check if local cache is still valid."""
        if not memory_item.updated_at:
            return False
        
        cache_age = datetime.utcnow() - memory_item.updated_at
        return cache_age < self._cache_ttl
    
    def _format_memory_response(
        self, 
        local_memory: MemoryItem, 
        cipher_memory: Optional[Dict[str, Any]] = None,
        from_cache: bool = False
    ) -> Dict[str, Any]:
        """Format memory response."""
        response = {
            "id": local_memory.id,
            "agent_id": local_memory.agent_id,
            "content": cipher_memory.get("content", local_memory.content) if cipher_memory else local_memory.content,
            "memory_type": local_memory.memory_type,
            "priority": local_memory.priority,
            "created_at": local_memory.created_at.isoformat() if local_memory.created_at else None,
            "updated_at": local_memory.updated_at.isoformat() if local_memory.updated_at else None,
            "expires_at": local_memory.expires_at.isoformat() if local_memory.expires_at else None,
            "from_cache": from_cache
        }
        
        if cipher_memory:
            response.update({
                "cipher_metadata": cipher_memory.get("metadata", {}),
                "tags": cipher_memory.get("tags", [])
            })
        
        return response
    
    async def _update_local_cache(self, memory_id: str, cipher_memory: Dict[str, Any]) -> None:
        """Update local cache with fresh Cipher data."""
        try:
            updates = {
                "content": cipher_memory.get("content"),
                "updated_at": datetime.utcnow()
            }
            await self.sqlite_storage.memory.update_memory(memory_id, updates)
        except Exception as e:
            logger.error(f"Failed to update local cache for memory {memory_id}: {e}")
    
    async def _enhance_search_results(
        self, 
        cipher_results: Dict[str, Any], 
        agent_id: Optional[str]
    ) -> Dict[str, Any]:
        """Enhance Cipher search results with local metadata."""
        enhanced_results = cipher_results.copy()
        
        for result in enhanced_results.get("results", []):
            memory_id = result.get("memory_id")
            if memory_id:
                # Get local metadata
                local_memory = await self.sqlite_storage.memory.get_memory(memory_id)
                if local_memory:
                    result.update({
                        "agent_id": local_memory.agent_id,
                        "priority": local_memory.priority,
                        "created_at": local_memory.created_at.isoformat() if local_memory.created_at else None
                    })
        
        return enhanced_results
    
    async def _local_search_fallback(
        self,
        project_id: str,
        query: str,
        agent_id: Optional[str],
        memory_type: Optional[str],
        limit: int,
        offset: int
    ) -> Dict[str, Any]:
        """Fallback to local search when Cipher is unavailable."""
        try:
            # Search local SQLite storage
            local_results = await self.sqlite_storage.memory.search_memories(
                query=query,
                agent_id=agent_id,
                limit=limit,
                offset=offset
            )
            
            # Format results
            results = []
            for memory in local_results:
                if memory_type and memory.memory_type != memory_type:
                    continue
                
                results.append({
                    "memory_id": memory.id,
                    "content": memory.content,
                    "agent_id": memory.agent_id,
                    "memory_type": memory.memory_type,
                    "priority": memory.priority,
                    "created_at": memory.created_at.isoformat() if memory.created_at else None,
                    "from_cache": True
                })
            
            return {
                "results": results,
                "total": len(results),
                "query": query,
                "from_cache": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Local search fallback failed: {e}")
            return {
                "results": [],
                "total": 0,
                "query": query,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _store_project_info(
        self,
        project_id: str,
        project_name: str,
        description: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Store project information locally."""
        # This would store in a projects table
        logger.debug(f"Storing project info locally: {project_id}")
    
    async def _get_local_project_info(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project information from local storage."""
        # This would retrieve from a projects table
        logger.debug(f"Getting local project info: {project_id}")
        return None


# Global service instance
_cipher_service: Optional[CipherService] = None


async def get_cipher_service() -> CipherService:
    """Get or create global Cipher service instance."""
    global _cipher_service
    
    if _cipher_service is None:
        # Initialize SQLite storage
        sqlite_storage = SQLiteUnifiedStorage(
            database_path="/Users/philipclapper/workspace/ai-agent-memory-router/data/ai_agent_memory.db"
        )
        
        _cipher_service = CipherService(sqlite_storage)
        await _cipher_service.initialize()
    
    return _cipher_service


async def close_cipher_service() -> None:
    """Close global Cipher service instance."""
    global _cipher_service
    
    if _cipher_service:
        await _cipher_service.close()
        _cipher_service = None

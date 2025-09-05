"""
Weaviate service for AI Agent Memory Router.

This service provides high-level operations for vector-based memory storage and retrieval
using Weaviate, with SQLite metadata storage for routing decisions.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from uuid import uuid4, UUID

from app.core.logging import get_logger
from app.core.weaviate_client import get_weaviate_client, WeaviateClient
from app.core.sqlite_storage import SQLiteUnifiedStorage
from app.core.config import get_settings
from app.models.memory import MemoryStore, MemoryStoreCreate, MemorySearch, MemorySearchResponse, MemorySearchResult
from app.models.memory import AgentSource, MemoryItem, MemoryMetadata

logger = get_logger(__name__)


class WeaviateMemoryService:
    """Service for Weaviate-based memory operations with SQLite metadata."""
    
    def __init__(self, sqlite_storage: Optional[SQLiteUnifiedStorage] = None):
        """Initialize the Weaviate memory service.
        
        Args:
            sqlite_storage: Optional SQLite storage for metadata and routing
        """
        self.settings = get_settings()
        self.sqlite_storage = sqlite_storage
        self.weaviate_client: Optional[WeaviateClient] = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the Weaviate memory service."""
        try:
            logger.info("Initializing Weaviate memory service")
            
            # Initialize Weaviate client
            self.weaviate_client = await get_weaviate_client()
            
            # Initialize SQLite storage if provided
            if self.sqlite_storage:
                await self.sqlite_storage.initialize()
            
            self._initialized = True
            logger.info("Weaviate memory service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate memory service: {e}")
            return False
    
    async def ensure_agent_exists(self, agent_id: str, agent_name: Optional[str] = None) -> bool:
        """Ensure an agent record exists in SQLite before creating memory records.
        
        Args:
            agent_id: Agent ID to check/create
            agent_name: Optional agent name (will use agent_id if not provided)
            
        Returns:
            True if agent exists or was created successfully, False otherwise
        """
        try:
            if not self.sqlite_storage:
                logger.warning("SQLite storage not available, cannot ensure agent exists")
                return False
            
            # Use the new helper utility to create agent if not exists
            return await self.sqlite_storage.agent.create_agent_if_not_exists(
                agent_id=agent_id,
                agent_name=agent_name,
                agent_type="assistant",
                version="1.0.0"
            )
            
        except Exception as e:
            logger.error(f"Failed to ensure agent {agent_id} exists: {e}")
            return False

    async def store_memory(self, memory_data: MemoryStoreCreate) -> MemoryStore:
        """Store a memory using Weaviate for vectors and SQLite for metadata.
        
        Args:
            memory_data: Memory data to store
            
        Returns:
            Stored memory object
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            memory_id = str(uuid4())
            logger.info(f"Storing memory {memory_id} from agent {memory_data.source.agent_id}")
            
            # Ensure agent exists in SQLite before creating memory records
            agent_name = getattr(memory_data.source, 'agent_name', None)
            agent_exists = await self.ensure_agent_exists(memory_data.source.agent_id, agent_name)
            if not agent_exists:
                logger.warning(f"Could not ensure agent {memory_data.source.agent_id} exists, proceeding with memory storage")
            
            # Convert importance from integer (1-10) to float (0.0-1.0) for Weaviate
            weaviate_importance = memory_data.importance / 10.0
            
            # Store in Weaviate (handles vectorization automatically)
            weaviate_success = await self.weaviate_client.store_memory(
                memory_id=memory_id,
                content=memory_data.content.text,
                agent_id=memory_data.source.agent_id,
                memory_type=memory_data.memory_type,
                importance=weaviate_importance,
                tags=memory_data.tags if hasattr(memory_data, 'tags') else [],
                metadata=memory_data.metadata if hasattr(memory_data, 'metadata') else {},
                project_id=getattr(memory_data.source, 'project_id', 'default'),
                expires_at=memory_data.expiration
            )
            
            if not weaviate_success:
                raise Exception("Failed to store memory in Weaviate")
            
            # Store metadata in SQLite for routing decisions
            if self.sqlite_storage:
                from app.models.memory import MemoryItem
                
                # Convert importance (1-10) to priority (1-4)
                priority = max(1, min(4, int(memory_data.importance / 2.5) + 1))
                
                memory_item = MemoryItem(
                    id=memory_id,
                    agent_id=memory_data.source.agent_id,
                    content=memory_data.content.text,
                    memory_type=memory_data.memory_type,
                    priority=priority,
                    expires_at=memory_data.expiration
                )
                
                await self.sqlite_storage.memory.store_memory(memory_item)
                
                # Store additional metadata
                from app.models.memory import MemoryMetadata
                
                metadata = MemoryMetadata(
                    memory_id=memory_id,
                    tags=memory_data.tags if hasattr(memory_data, 'tags') else [],
                    source="weaviate",
                    confidence=0.8,  # Default confidence for Weaviate-stored memories
                    embedding_vector=None,  # Weaviate handles this
                    vector_dimension=self.settings.weaviate.vector_dimension
                )
                
                await self.sqlite_storage.metadata.store_metadata(metadata)
            
            # Create MemoryStore object
            memory = MemoryStore(
                id=memory_id,
                content=memory_data.content,
                source=memory_data.source,
                memory_type=memory_data.memory_type,
                importance=memory_data.importance,
                expiration=memory_data.expiration,
                access_control={}  # Default empty access control
            )
            
            logger.info(f"Memory stored successfully: {memory_id}")
            return memory
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise
    
    async def get_memory(self, memory_id: Union[str, UUID]) -> Optional[MemoryStore]:
        """Retrieve a memory by ID.
        
        Args:
            memory_id: Memory ID to retrieve (string or UUID)
            
        Returns:
            Memory object or None if not found
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Convert UUID to string if needed
            memory_id_str = str(memory_id)
            
            # Get from Weaviate
            weaviate_data = await self.weaviate_client.get_memory(memory_id_str)
            
            if not weaviate_data:
                logger.warning(f"Memory not found in Weaviate: {memory_id_str}")
                return None
            
            # Get metadata from SQLite if available
            metadata = None
            if self.sqlite_storage:
                metadata = await self.sqlite_storage.metadata.get_metadata(memory_id_str)
            
            # Create MemoryStore object
            from app.models.memory import MemoryContent
            
            content = MemoryContent(
                text=weaviate_data["content"],
                language="en",
                format="text",
                encoding="utf-8"
            )
            
            source = AgentSource(
                agent_id=weaviate_data["agent_id"],
                project_id=weaviate_data.get("project_id", "default")
            )
            
            memory = MemoryStore(
                id=memory_id,
                content=content,
                source=source,
                memory_type=weaviate_data["memory_type"],
                importance=int(weaviate_data["importance"] * 10),  # Convert float (0.0-1.0) back to int (1-10)
                expiration=datetime.fromisoformat(str(weaviate_data["expires_at"])) if weaviate_data.get("expires_at") else None,
                access_control={}  # Default access control
            )
            
            return memory
            
        except Exception as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            return None
    
    async def search_memories(self, search_query: MemorySearch) -> MemorySearchResponse:
        """Search memories using semantic search.
        
        Args:
            search_query: Search query and filters
            
        Returns:
            Search response with results
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Searching memories with query: {search_query.query}")
            
            # Perform semantic search in Weaviate
            weaviate_results = await self.weaviate_client.search_memories(
                query=search_query.query or "",
                agent_id=search_query.agent_id,
                memory_type=search_query.memory_type,
                project_id=None,  # MemorySearch doesn't have project_id field
                min_importance=None,  # MemorySearch doesn't have min_importance field
                limit=search_query.limit,
                similarity_threshold=None  # MemorySearch doesn't have similarity_threshold field
            )
            
            # Convert to MemorySearchResult objects
            results = []
            for weaviate_result in weaviate_results:
                # Create MemoryItem object
                # Convert importance from Weaviate (0.0-1.0) back to priority (1-4)
                weaviate_importance = weaviate_result["importance"]
                if weaviate_importance > 1.0:
                    # If importance is > 1.0, it might be stored as integer (1-10), convert to float first
                    weaviate_importance = weaviate_importance / 10.0
                priority = max(1, min(4, int(weaviate_importance * 4) + 1))
                
                memory_item = MemoryItem(
                    id=UUID(weaviate_result["id"]),
                    agent_id=weaviate_result["agent_id"],
                    content=weaviate_result["content"],
                    memory_type=weaviate_result["memory_type"],
                    priority=priority,
                    created_at=datetime.fromisoformat(str(weaviate_result["created_at"])) if weaviate_result.get("created_at") else datetime.utcnow(),
                    updated_at=datetime.fromisoformat(str(weaviate_result["created_at"])) if weaviate_result.get("created_at") else datetime.utcnow()
                )
                
                # Create MemoryMetadata object if metadata exists
                memory_metadata = None
                if weaviate_result.get("metadata"):
                    memory_metadata = MemoryMetadata(
                        id=uuid4(),
                        memory_id=weaviate_result["id"],
                        tags=weaviate_result.get("metadata", {}).get("tags", []),
                        source=weaviate_result.get("metadata", {}).get("source"),
                        confidence=weaviate_result.get("metadata", {}).get("confidence", 1.0),
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                
                result = MemorySearchResult(
                    memory=memory_item,
                    metadata=memory_metadata,
                    relevance_score=weaviate_result.get("similarity", 0.0),
                    matched_fields=["content"]  # Default to content field matching
                )
                results.append(result)
            
            # Apply additional filtering if needed
            # Note: MemorySearch model has individual filter fields, not a filters dict
            # Additional filtering is handled by the individual fields in the search query
            
            # Apply sorting
            results = self._apply_sorting(results, search_query)
            
            # Apply pagination
            start = search_query.offset
            end = start + search_query.limit
            paginated_results = results[start:end]
            
            response = MemorySearchResponse(
                results=paginated_results,
                total=len(results),
                query=search_query.query or "",
                execution_time=0.0,  # Could be measured if needed
                timestamp=datetime.utcnow()
            )
            
            logger.info(f"Memory search completed: {len(paginated_results)} results found")
            return response
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            raise
    
    async def update_memory(self, memory_id: Union[str, UUID], updates: Dict[str, Any]) -> bool:
        """Update a memory.
        
        Args:
            memory_id: Memory ID to update (string or UUID)
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Convert UUID to string if needed
            memory_id_str = str(memory_id)
            
            # Get the existing memory to find the agent_id
            existing_memory = await self.get_memory(memory_id_str)
            if existing_memory:
                # Ensure agent exists before updating
                await self.ensure_agent_exists(existing_memory.source.agent_id)
            
            # Update in Weaviate
            weaviate_success = await self.weaviate_client.update_memory(memory_id_str, updates)
            
            if not weaviate_success:
                logger.error(f"Failed to update memory in Weaviate: {memory_id_str}")
                return False
            
            # Update metadata in SQLite if available
            if self.sqlite_storage:
                sqlite_updates = {}
                
                if "content" in updates:
                    sqlite_updates["content"] = updates["content"]
                if "memory_type" in updates:
                    sqlite_updates["memory_type"] = updates["memory_type"]
                if "importance" in updates:
                    # Convert importance (1-10) to priority (1-4)
                    priority = max(1, min(4, int(updates["importance"] / 2.5) + 1))
                    sqlite_updates["priority"] = priority
                
                if sqlite_updates:
                    await self.sqlite_storage.memory.update_memory(memory_id_str, sqlite_updates)
            
            logger.info(f"Memory updated successfully: {memory_id_str}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False
    
    async def delete_memory(self, memory_id: Union[str, UUID]) -> bool:
        """Delete a memory.
        
        Args:
            memory_id: Memory ID to delete (string or UUID)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Convert UUID to string if needed
            memory_id_str = str(memory_id)
            
            # Delete from Weaviate
            weaviate_success = await self.weaviate_client.delete_memory(memory_id_str)
            
            if not weaviate_success:
                logger.error(f"Failed to delete memory from Weaviate: {memory_id_str}")
                return False
            
            # Delete from SQLite if available
            if self.sqlite_storage:
                await self.sqlite_storage.memory.delete_memory(memory_id_str)
                await self.sqlite_storage.metadata.delete_metadata(memory_id_str)
            
            logger.info(f"Memory deleted successfully: {memory_id_str}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    async def get_cross_project_memories(
        self,
        query: str,
        source_project_id: str,
        target_project_ids: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[MemorySearchResult]:
        """Get memories from other projects for cross-project knowledge sharing.
        
        Args:
            query: Search query
            source_project_id: Source project ID
            target_project_ids: List of target project IDs (None for all)
            limit: Maximum number of results
            
        Returns:
            List of memories from other projects
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Searching cross-project memories for project {source_project_id}")
            
            # Search in Weaviate with project filters
            weaviate_results = await self.weaviate_client.search_memories(
                query=query,
                project_id=None,  # Search all projects
                limit=limit * 2,  # Get more results for filtering
                similarity_threshold=self.settings.weaviate.similarity_threshold
            )
            
            # Filter out memories from the source project
            cross_project_results = []
            for result in weaviate_results:
                result_project_id = result.get("project_id", "default")
                
                # Skip if it's from the same project
                if result_project_id == source_project_id:
                    continue
                
                # Filter by target projects if specified
                if target_project_ids and result_project_id not in target_project_ids:
                    continue
                
                cross_project_results.append(result)
            
            # Convert to MemorySearchResult objects
            results = []
            for weaviate_result in cross_project_results[:limit]:
                from app.models.memory import MemoryContent
                
                content = MemoryContent(
                    text=weaviate_result["content"],
                    language="en",
                    format="text",
                    encoding="utf-8"
                )
                
                source = AgentSource(
                    agent_id=weaviate_result["agent_id"],
                    project_id=weaviate_result.get("project_id", "default")
                )
                
                result = MemorySearchResult(
                    memory_id=weaviate_result["id"],
                    content=content,
                    source=source,
                    memory_type=weaviate_result["memory_type"],
                    importance=int(weaviate_result["importance"] * 10),  # Convert float (0.0-1.0) back to int (1-10)
                    relevance_score=weaviate_result.get("similarity", 0.0),
                    created_at=datetime.fromisoformat(str(weaviate_result["created_at"])) if weaviate_result.get("created_at") else datetime.utcnow(),
                    metadata=weaviate_result.get("metadata", {})
                )
                results.append(result)
            
            logger.info(f"Found {len(results)} cross-project memories")
            return results
            
        except Exception as e:
            logger.error(f"Failed to get cross-project memories: {e}")
            return []
    
    async def deduplicate_memories(self, agent_id: str, similarity_threshold: float = 0.9) -> Dict[str, Any]:
        """Find and handle duplicate memories for an agent.
        
        Args:
            agent_id: Agent ID to check for duplicates
            similarity_threshold: Similarity threshold for considering memories as duplicates
            
        Returns:
            Dictionary with deduplication results
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Checking for duplicate memories for agent {agent_id}")
            
            # Get all memories for the agent
            agent_memories = await self.weaviate_client.search_memories(
                query="",  # Empty query to get all
                agent_id=agent_id,
                limit=100  # Maximum allowed limit
            )
            
            duplicates = []
            processed = set()
            
            # Compare memories for similarity
            for i, memory1 in enumerate(agent_memories):
                if memory1["id"] in processed:
                    continue
                
                for j, memory2 in enumerate(agent_memories[i+1:], i+1):
                    if memory2["id"] in processed:
                        continue
                    
                    # Simple text similarity check (could be enhanced with actual embedding comparison)
                    content1 = memory1["content"].lower()
                    content2 = memory2["content"].lower()
                    
                    # Calculate simple similarity
                    words1 = set(content1.split())
                    words2 = set(content2.split())
                    
                    if words1 and words2:
                        intersection = words1.intersection(words2)
                        union = words1.union(words2)
                        similarity = len(intersection) / len(union)
                        
                        if similarity >= similarity_threshold:
                            duplicates.append({
                                "memory1": memory1,
                                "memory2": memory2,
                                "similarity": similarity
                            })
                            processed.add(memory1["id"])
                            processed.add(memory2["id"])
            
            result = {
                "agent_id": agent_id,
                "total_memories": len(agent_memories),
                "duplicates_found": len(duplicates),
                "duplicates": duplicates,
                "similarity_threshold": similarity_threshold
            }
            
            logger.info(f"Found {len(duplicates)} duplicate memory pairs for agent {agent_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to deduplicate memories for agent {agent_id}: {e}")
            return {"error": str(e)}
    
    def _apply_additional_filters(self, results: List[MemorySearchResult], filters: Dict[str, Any]) -> List[MemorySearchResult]:
        """Apply additional filters to search results."""
        filtered_results = results
        
        # Filter by importance
        if "min_importance" in filters:
            min_importance = filters["min_importance"]
            filtered_results = [r for r in filtered_results if r.importance >= min_importance]
        
        # Filter by tags
        if "tags" in filters:
            required_tags = set(filters["tags"])
            filtered_results = [
                r for r in filtered_results 
                if required_tags.issubset(set(r.tags if hasattr(r, 'tags') else []))
            ]
        
        return filtered_results
    
    def _apply_sorting(self, results: List[MemorySearchResult], search_query: MemorySearch) -> List[MemorySearchResult]:
        """Apply sorting to search results."""
        reverse = search_query.sort_order == "desc"
        
        if search_query.sort_by == "relevance":
            return sorted(results, key=lambda r: r.relevance_score, reverse=reverse)
        elif search_query.sort_by == "importance":
            return sorted(results, key=lambda r: r.importance, reverse=reverse)
        elif search_query.sort_by == "created_at":
            return sorted(results, key=lambda r: r.created_at, reverse=reverse)
        else:
            # Default: sort by relevance
            return sorted(results, key=lambda r: r.relevance_score, reverse=reverse)
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        try:
            if not self._initialized:
                await self.initialize()
            
            stats = {
                "service_initialized": self._initialized,
                "weaviate_connected": self.weaviate_client is not None,
                "sqlite_available": self.sqlite_storage is not None
            }
            
            if self.weaviate_client:
                collection_stats = await self.weaviate_client.get_collection_stats()
                stats["weaviate_collection"] = collection_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get service stats: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close the Weaviate memory service."""
        try:
            if self.weaviate_client:
                await self.weaviate_client.close()
            
            if self.sqlite_storage:
                await self.sqlite_storage.close()
            
            self._initialized = False
            logger.info("Weaviate memory service closed")
            
        except Exception as e:
            logger.error(f"Error closing Weaviate memory service: {e}")


# Global service instance
_weaviate_service: Optional[WeaviateMemoryService] = None


async def get_weaviate_service(sqlite_storage: Optional[SQLiteUnifiedStorage] = None) -> WeaviateMemoryService:
    """Get the global Weaviate memory service instance."""
    global _weaviate_service
    
    if _weaviate_service is None:
        _weaviate_service = WeaviateMemoryService(sqlite_storage)
        await _weaviate_service.initialize()
    
    return _weaviate_service


async def close_weaviate_service():
    """Close the global Weaviate memory service."""
    global _weaviate_service
    
    if _weaviate_service:
        await _weaviate_service.close()
        _weaviate_service = None

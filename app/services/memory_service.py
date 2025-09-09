"""
Memory service for AI Agent Memory Router.

This service handles business logic for memory operations including
storage, retrieval, search, and routing.
"""

import time
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime

from app.core.logging import get_logger
from app.core.metrics import record_memory_operation, get_metrics_collector
from app.models.memory import (
    MemoryStore, MemoryStoreCreate, MemorySearch, MemorySearchResponse,
    MemorySearchResult, MemoryStats, MemoryRoute, MemoryRouteCreate,
    MemoryItem, MemoryMetadata
)

# Setup logger
logger = get_logger(__name__)


class MemoryService:
    """Service for memory operations."""
    
    def __init__(self):
        """Initialize the memory service."""
        self._memories: Dict[UUID, MemoryStore] = {}
        self._routes: Dict[UUID, MemoryRoute] = {}
        self._stats = {
            "total_memories": 0,
            "memories_by_type": {},
            "memories_by_agent": {},
            "total_routes": 0,
            "routes_by_status": {},
            "storage_size_bytes": 0,
            "last_updated": datetime.utcnow()
        }
    
    async def store_memory(self, memory_data: MemoryStoreCreate) -> MemoryStore:
        """Store a new memory."""
        
        start_time = time.time()
        
        try:
            logger.info(f"Storing memory from agent {memory_data.source.agent_id}")
            
            # Create memory with generated ID
            memory = MemoryStore(
                id=uuid4(),
                content=memory_data.content,
                source=memory_data.source,
                memory_type=memory_data.memory_type,
                importance=memory_data.importance,
                expiration=memory_data.expiration,
                access_control=memory_data.access_control
            )
            
            # Store in memory
            self._memories[memory.id] = memory
            
            # Update statistics
            self._update_stats_after_store(memory)
            
            # Track metrics
            metrics = get_metrics_collector()
            metrics.record_memory_stored(memory.source.agent_id, memory.memory_type)
            
            duration = time.time() - start_time
            record_memory_operation("store", memory.source.agent_id, True)
            
            logger.info(f"Memory stored successfully with ID: {memory.id}")
            return memory
            
        except Exception as e:
            duration = time.time() - start_time
            record_memory_operation("store", "unknown", False)
            logger.error(f"Failed to store memory: {e}")
            raise
    
    async def get_memory(self, memory_id: UUID) -> Optional[MemoryStore]:
        """Retrieve a memory by ID."""
        
        start_time = time.time()
        
        try:
            logger.debug(f"Retrieving memory: {memory_id}")
            
            memory = self._memories.get(memory_id)
            
            if memory:
                duration = time.time() - start_time
                record_memory_operation("retrieve", str(memory_id), True)
                return memory
            else:
                duration = time.time() - start_time
                record_memory_operation("retrieve", str(memory_id), False)
                logger.warning(f"Memory not found: {memory_id}")
                return None
                
        except Exception as e:
            duration = time.time() - start_time
            record_memory_operation("retrieve", str(memory_id), False)
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            raise
    
    async def search_memories(self, search_query: MemorySearch) -> MemorySearchResponse:
        """Search memories based on query and filters."""
        
        start_time = time.time()
        
        try:
            logger.info(f"Searching memories with query: {search_query.query}")
            
            # Apply filters
            filtered_memories = self._apply_search_filters(search_query)
            
            # Apply sorting
            sorted_memories = self._apply_search_sorting(filtered_memories, search_query)
            
            # Apply pagination
            paginated_memories = self._apply_pagination(sorted_memories, search_query)
            
            # Convert to search results
            results = []
            for memory in paginated_memories:
                # Create MemoryItem from MemoryStore
                # Map memory_type from MemoryStore to MemoryItem
                memory_type_mapping = {
                    'conversation': 'context',
                    'knowledge': 'knowledge',
                    'experience': 'experience',
                    'fact': 'fact',
                    'procedure': 'procedure'
                }
                mapped_memory_type = memory_type_mapping.get(memory.memory_type, 'knowledge')
                
                memory_item = MemoryItem(
                    id=memory.id,
                    agent_id=memory.source.agent_id,
                    content=memory.content.text,
                    memory_type=mapped_memory_type,
                    priority=min(4, max(1, memory.importance // 2)),  # Convert 1-10 importance to 1-4 priority
                    expires_at=memory.expiration,
                    created_at=memory.created_at,
                    updated_at=memory.updated_at
                )
                
                # Create MemoryMetadata
                memory_metadata = MemoryMetadata(
                    id=str(uuid4()),
                    memory_id=str(memory.id),
                    tags=[],  # AgentSource doesn't have tags, use empty list
                    source=memory.source.type,  # Use 'type' instead of 'source_type'
                    confidence=1.0,
                    created_at=memory.created_at,
                    updated_at=memory.updated_at
                )
                
                result = MemorySearchResult(
                    memory=memory_item,
                    metadata=memory_metadata,
                    relevance_score=self._calculate_relevance(memory, search_query.query),
                    matched_fields=["content", "memory_type"]
                )
                results.append(result)
            
            # Calculate search time
            search_time = (time.time() - start_time) * 1000
            
            response = MemorySearchResponse(
                results=results,
                total=len(filtered_memories),
                query=search_query.query or "",
                execution_time=search_time / 1000.0,  # Convert to seconds
                timestamp=datetime.utcnow()
            )
            
            duration = time.time() - start_time
            record_memory_operation("search", search_query.query or "unknown", True)
            
            logger.info(f"Memory search completed: {len(results)} results found")
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            record_memory_operation("search", search_query.query or "unknown", False)
            logger.error(f"Failed to search memories: {e}")
            raise
    
    async def create_memory_route(self, route_data: MemoryRouteCreate) -> MemoryRoute:
        """Create a new memory route."""
        
        start_time = time.time()
        
        try:
            logger.info(f"Creating memory route from {route_data.source_agent_id} to {route_data.target_agent_id}")
            
            # Create route with generated ID
            route = MemoryRoute(
                id=uuid4(),
                source_agent_id=route_data.source_agent_id,
                target_agent_id=route_data.target_agent_id,
                memory_id=route_data.memory_id,
                route_type=route_data.route_type,
                priority=route_data.priority,
                metadata=route_data.metadata
            )
            
            # Store route
            self._routes[route.id] = route
            
            # Update statistics
            self._update_stats_after_route(route)
            
            duration = time.time() - start_time
            record_memory_operation("route_create", route_data.source_agent_id, True)
            
            logger.info(f"Memory route created successfully with ID: {route.id}")
            return route
            
        except Exception as e:
            duration = time.time() - start_time
            record_memory_operation("route_create", "unknown", False)
            logger.error(f"Failed to create memory route: {e}")
            raise
    
    async def get_memory_stats(self) -> MemoryStats:
        """Get memory statistics."""
        
        try:
            logger.debug("Retrieving memory statistics")
            
            # Calculate current statistics
            self._calculate_current_stats()
            
            # Convert average importance (1-10) to average priority (1-4)
            avg_importance = self._calculate_average_importance()
            avg_priority = min(4.0, max(1.0, avg_importance / 2.5))  # Convert 1-10 to 1-4
            
            stats = MemoryStats(
                total_memories=self._stats["total_memories"],
                memories_by_type=self._stats["memories_by_type"],
                memories_by_agent=self._stats["memories_by_agent"],
                average_priority=avg_priority,
                total_size_bytes=self._stats["storage_size_bytes"]
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory statistics: {e}")
            raise
    
    async def delete_memory(self, memory_id: UUID) -> bool:
        """Delete a memory by ID."""
        
        start_time = time.time()
        
        try:
            logger.info(f"Deleting memory: {memory_id}")
            
            if memory_id in self._memories:
                memory = self._memories.pop(memory_id)
                
                # Update statistics
                self._update_stats_after_delete(memory)
                
                duration = time.time() - start_time
                record_memory_operation("delete", str(memory_id), True)
                
                logger.info(f"Memory deleted successfully: {memory_id}")
                return True
            else:
                duration = time.time() - start_time
                record_memory_operation("delete", str(memory_id), False)
                logger.warning(f"Memory not found for deletion: {memory_id}")
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            record_memory_operation("delete", str(memory_id), False)
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            raise
    
    def _apply_search_filters(self, search_query: MemorySearch) -> List[MemoryStore]:
        """Apply search filters to memories."""
        
        filtered = list(self._memories.values())
        
        # Filter by query text
        if search_query.query:
            query_lower = search_query.query.lower()
            filtered = [
                m for m in filtered 
                if query_lower in m.content.text.lower() or 
                   any(query_lower in tag.lower() for tag in m.content.tags)
            ]
        
        # Filter by memory type
        if search_query.memory_type:
            filtered = [m for m in filtered if m.memory_type == search_query.memory_type]
        
        # Filter by agent
        if search_query.agent_id:
            filtered = [m for m in filtered if m.source.agent_id == search_query.agent_id]
        
        # Filter by tags
        if search_query.tags:
            required_tags = set(search_query.tags)
            filtered = [
                m for m in filtered 
                if required_tags.issubset(set(m.content.tags if hasattr(m.content, 'tags') else []))
            ]
        
        return filtered
    
    def _apply_search_sorting(self, memories: List[MemoryStore], search_query: MemorySearch) -> List[MemoryStore]:
        """Apply sorting to search results."""
        
        reverse = search_query.sort_order == "desc"
        
        if search_query.sort_by == "relevance":
            # Sort by relevance (placeholder - would use actual relevance scoring)
            return sorted(memories, key=lambda m: m.importance, reverse=reverse)
        elif search_query.sort_by == "created_at":
            return sorted(memories, key=lambda m: m.created_at, reverse=reverse)
        elif search_query.sort_by == "updated_at":
            return sorted(memories, key=lambda m: m.updated_at, reverse=reverse)
        elif search_query.sort_by == "importance":
            return sorted(memories, key=lambda m: m.importance, reverse=reverse)
        elif search_query.sort_by == "confidence":
            return sorted(memories, key=lambda m: m.content.confidence or 0, reverse=reverse)
        else:
            return sorted(memories, key=lambda m: m.created_at, reverse=reverse)
    
    def _apply_pagination(self, memories: List[MemoryStore], search_query: MemorySearch) -> List[MemoryStore]:
        """Apply pagination to search results."""
        
        start = search_query.offset
        end = start + search_query.limit
        
        return memories[start:end]
    
    def _calculate_relevance(self, memory: MemoryStore, query: str) -> float:
        """Calculate relevance score for search result."""
        
        if not query:
            return 1.0
        
        # Simple relevance scoring (placeholder - would use actual semantic search)
        query_lower = query.lower()
        text_lower = memory.content.text.lower()
        
        # Exact match
        if query_lower in text_lower:
            return 0.9
        
        # Partial match
        if any(word in text_lower for word in query_lower.split()):
            return 0.7
        
        # Tag match
        if any(query_lower in tag.lower() for tag in memory.content.tags):
            return 0.6
        
        # Default relevance
        return 0.3
    
    def _update_stats_after_store(self, memory: MemoryStore):
        """Update statistics after storing a memory."""
        
        self._stats["total_memories"] += 1
        
        # Update type statistics
        memory_type = memory.memory_type
        self._stats["memories_by_type"][memory_type] = self._stats["memories_by_type"].get(memory_type, 0) + 1
        
        # Update agent statistics
        agent_id = str(memory.source.agent_id)
        self._stats["memories_by_agent"][agent_id] = self._stats["memories_by_agent"].get(agent_id, 0) + 1
        
        # Update storage size (rough estimate)
        self._stats["storage_size_bytes"] += len(memory.content.text.encode('utf-8'))
        
        self._stats["last_updated"] = datetime.utcnow()
    
    def _update_stats_after_delete(self, memory: MemoryStore):
        """Update statistics after deleting a memory."""
        
        self._stats["total_memories"] -= 1
        
        # Update type statistics
        memory_type = memory.memory_type
        if memory_type in self._stats["memories_by_type"]:
            self._stats["memories_by_type"][memory_type] = max(0, self._stats["memories_by_type"][memory_type] - 1)
        
        # Update agent statistics
        agent_id = str(memory.source.agent_id)
        if agent_id in self._stats["memories_by_agent"]:
            self._stats["memories_by_agent"][agent_id] = max(0, self._stats["memories_by_agent"][agent_id] - 1)
        
        # Update storage size
        self._stats["storage_size_bytes"] = max(0, self._stats["storage_size_bytes"] - len(memory.content.text.encode('utf-8')))
        
        self._stats["last_updated"] = datetime.utcnow()
    
    def _update_stats_after_route(self, route: MemoryRoute):
        """Update statistics after creating a route."""
        
        self._stats["total_routes"] += 1
        
        # Update status statistics
        status = route.status
        self._stats["routes_by_status"][status] = self._stats["routes_by_status"].get(status, 0) + 1
        
        self._stats["last_updated"] = datetime.utcnow()
    
    def _calculate_current_stats(self):
        """Calculate current statistics from actual data."""
        
        # Recalculate from actual memories
        memories = list(self._memories.values())
        routes = list(self._routes.values())
        
        self._stats["total_memories"] = len(memories)
        self._stats["total_routes"] = len(routes)
        
        # Calculate type distribution
        self._stats["memories_by_type"] = {}
        for memory in memories:
            memory_type = memory.memory_type
            self._stats["memories_by_type"][memory_type] = self._stats["memories_by_type"].get(memory_type, 0) + 1
        
        # Calculate agent distribution
        self._stats["memories_by_agent"] = {}
        for memory in memories:
            agent_id = str(memory.source.agent_id)
            self._stats["memories_by_agent"][agent_id] = self._stats["memories_by_agent"].get(agent_id, 0) + 1
        
        # Calculate status distribution
        self._stats["routes_by_status"] = {}
        for route in routes:
            status = route.status
            self._stats["routes_by_status"][status] = self._stats["routes_by_status"].get(status, 0) + 1
        
        # Calculate storage size
        self._stats["storage_size_bytes"] = sum(
            len(memory.content.text.encode('utf-8')) for memory in memories
        )
        
        self._stats["last_updated"] = datetime.utcnow()
    
    def _calculate_average_importance(self) -> float:
        """Calculate average importance of all memories."""
        
        memories = list(self._memories.values())
        if not memories:
            return 0.0
        
        total_importance = sum(memory.importance for memory in memories)
        return total_importance / len(memories)

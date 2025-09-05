"""
Universal Memory Access API endpoints.

This module provides a simplified, project-agnostic interface for any project
to store and retrieve memories without needing to understand the internal
routing complexity of Cipher/Weaviate systems.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from app.core.auth import get_current_project, ProjectCredentials, create_project_auth_response
from app.core.logging import get_logger
from app.core.metrics import record_request, record_request_duration
from app.core.rate_limiting import RateLimiter, get_rate_limiter
from app.models.universal import (
    UniversalMemoryStore, UniversalMemoryResponse, UniversalMemoryRetrieve,
    UniversalMemoryItem, UniversalMemoryListResponse, UniversalMemorySearch,
    UniversalMemorySearchResponse, UniversalMemorySearchResult,
    UniversalMemoryUpdate, UniversalMemoryDelete, UniversalMemoryBatchStore,
    UniversalMemoryBatchResult, UniversalProjectCreate, UniversalProjectResponse,
    UniversalProjectList, UniversalProjectListResponse, UniversalMemoryStats,
    UniversalProjectStats, UniversalAPIError, UniversalAPIStatus,
    UniversalRateLimit, UniversalQuota, MemoryType, Priority, ProjectStatus
)

# Import services
from app.services.cipher_service import get_cipher_service
from app.services.weaviate_service import get_weaviate_service
from app.core.weaviate_client import get_weaviate_client

# Setup
logger = get_logger(__name__)
router = APIRouter(prefix="/universal", tags=["Universal Memory Access"])
security = HTTPBearer()


class UniversalMemoryService:
    """Service for universal memory operations with intelligent backend routing."""
    
    def __init__(self):
        self.cipher_service = None
        self.weaviate_service = None
        self.weaviate_client = None
        self.rate_limiter = get_rate_limiter()
    
    async def initialize(self):
        """Initialize the universal memory service."""
        try:
            self.cipher_service = await get_cipher_service()
            self.weaviate_service = await get_weaviate_service()
            self.weaviate_client = await get_weaviate_client()
            logger.info("Universal Memory Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Universal Memory Service: {e}")
            raise
    
    async def store_memory(
        self,
        project_creds: ProjectCredentials,
        memory_request: UniversalMemoryStore
    ) -> UniversalMemoryResponse:
        """Store memory with intelligent backend selection."""
        start_time = time.time()
        
        try:
            # Generate memory ID
            memory_id = f"uma_{project_creds.project_id}_{int(time.time() * 1000)}"
            
            # Determine storage backend based on project preferences and content
            storage_backend = await self._select_storage_backend(
                project_creds, memory_request
            )
            
            # Store in selected backend
            if storage_backend == "cipher":
                await self._store_in_cipher(memory_id, project_creds, memory_request)
            elif storage_backend == "weaviate":
                await self._store_in_weaviate(memory_id, project_creds, memory_request)
            else:  # hybrid
                await self._store_hybrid(memory_id, project_creds, memory_request)
            
            # Calculate estimated retrieval time
            retrieval_time = await self._estimate_retrieval_time(storage_backend)
            
            return UniversalMemoryResponse(
                memory_id=memory_id,
                project_id=project_creds.project_id,
                status="stored",
                created_at=datetime.utcnow(),
                storage_location=storage_backend,
                estimated_retrieval_time_ms=retrieval_time
            )
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise HTTPException(status_code=500, detail=f"Memory storage failed: {str(e)}")
    
    async def retrieve_memories(
        self,
        project_creds: ProjectCredentials,
        retrieve_request: UniversalMemoryRetrieve
    ) -> UniversalMemoryListResponse:
        """Retrieve memories with intelligent backend selection."""
        start_time = time.time()
        
        try:
            memories = []
            
            # If specific memory ID requested
            if retrieve_request.memory_id:
                memory = await self._get_single_memory(
                    retrieve_request.memory_id, project_creds, retrieve_request
                )
                if memory:
                    memories = [memory]
            else:
                # Retrieve multiple memories
                memories = await self._get_multiple_memories(
                    project_creds, retrieve_request
                )
            
            # Apply filters
            filtered_memories = self._apply_retrieval_filters(memories, retrieve_request)
            
            # Apply pagination
            total = len(filtered_memories)
            offset = retrieve_request.offset
            limit = retrieve_request.limit
            paginated_memories = filtered_memories[offset:offset + limit]
            
            return UniversalMemoryListResponse(
                memories=paginated_memories,
                total=total,
                limit=limit,
                offset=offset,
                has_more=offset + limit < total,
                retrieval_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            raise HTTPException(status_code=500, detail=f"Memory retrieval failed: {str(e)}")
    
    async def search_memories(
        self,
        project_creds: ProjectCredentials,
        search_request: UniversalMemorySearch
    ) -> UniversalMemorySearchResponse:
        """Search memories with intelligent backend selection."""
        start_time = time.time()
        
        try:
            # Determine search strategy
            search_strategy = await self._select_search_strategy(
                project_creds, search_request
            )
            
            results = []
            
            if search_strategy == "semantic":
                # Use Weaviate for semantic search
                results = await self._semantic_search(project_creds, search_request)
            elif search_strategy == "hybrid":
                # Combine both backends
                results = await self._hybrid_search(project_creds, search_request)
            else:
                # Use Cipher for keyword search
                results = await self._keyword_search(project_creds, search_request)
            
            # Apply filters and pagination
            filtered_results = self._apply_search_filters(results, search_request)
            total_results = len(filtered_results)
            paginated_results = filtered_results[
                search_request.offset:search_request.offset + search_request.limit
            ]
            
            return UniversalMemorySearchResponse(
                query=search_request.query,
                results=paginated_results,
                total_results=total_results,
                search_time_ms=(time.time() - start_time) * 1000,
                search_method=search_strategy
            )
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            raise HTTPException(status_code=500, detail=f"Memory search failed: {str(e)}")
    
    async def update_memory(
        self,
        project_creds: ProjectCredentials,
        update_request: UniversalMemoryUpdate
    ) -> Dict[str, Any]:
        """Update memory across all backends."""
        try:
            # Update in all backends where the memory exists
            updated_backends = []
            
            # Try Cipher first
            try:
                if self.cipher_service:
                    success = await self.cipher_service.update_memory(
                        update_request.project_id,
                        update_request.memory_id,
                        update_request.dict(exclude_unset=True, exclude={'project_id', 'memory_id'})
                    )
                    if success:
                        updated_backends.append("cipher")
            except Exception as e:
                logger.warning(f"Failed to update in Cipher: {e}")
            
            # Try Weaviate
            try:
                if self.weaviate_client:
                    updates = update_request.dict(exclude_unset=True, exclude={'project_id', 'memory_id'})
                    success = await self.weaviate_client.update_memory(
                        update_request.memory_id, updates
                    )
                    if success:
                        updated_backends.append("weaviate")
            except Exception as e:
                logger.warning(f"Failed to update in Weaviate: {e}")
            
            if not updated_backends:
                raise HTTPException(status_code=404, detail="Memory not found in any backend")
            
            return {
                "memory_id": update_request.memory_id,
                "project_id": update_request.project_id,
                "status": "updated",
                "updated_backends": updated_backends,
                "timestamp": datetime.utcnow()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
            raise HTTPException(status_code=500, detail=f"Memory update failed: {str(e)}")
    
    async def delete_memory(
        self,
        project_creds: ProjectCredentials,
        delete_request: UniversalMemoryDelete
    ) -> Dict[str, Any]:
        """Delete memory from all backends."""
        try:
            deleted_backends = []
            
            # Delete from Cipher
            try:
                if self.cipher_service:
                    success = await self.cipher_service.delete_memory(
                        delete_request.project_id, delete_request.memory_id
                    )
                    if success:
                        deleted_backends.append("cipher")
            except Exception as e:
                logger.warning(f"Failed to delete from Cipher: {e}")
            
            # Delete from Weaviate
            try:
                if self.weaviate_client:
                    success = await self.weaviate_client.delete_memory(delete_request.memory_id)
                    if success:
                        deleted_backends.append("weaviate")
            except Exception as e:
                logger.warning(f"Failed to delete from Weaviate: {e}")
            
            if not deleted_backends:
                raise HTTPException(status_code=404, detail="Memory not found in any backend")
            
            return {
                "memory_id": delete_request.memory_id,
                "project_id": delete_request.project_id,
                "status": "deleted" if delete_request.permanent else "soft_deleted",
                "deleted_backends": deleted_backends,
                "timestamp": datetime.utcnow()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            raise HTTPException(status_code=500, detail=f"Memory deletion failed: {str(e)}")
    
    # Private helper methods
    
    async def _select_storage_backend(
        self,
        project_creds: ProjectCredentials,
        memory_request: UniversalMemoryStore
    ) -> str:
        """Intelligently select the best storage backend."""
        # Check project preferences
        storage_pref = project_creds.metadata.get("storage_preference", "auto")
        
        if storage_pref != "auto":
            return storage_pref
        
        # Auto-selection logic
        content_length = len(memory_request.content)
        
        # Use Weaviate for semantic content, Cipher for structured data
        if memory_request.memory_type in [MemoryType.KNOWLEDGE, MemoryType.EXPERIENCE]:
            return "weaviate"
        elif content_length > 1000:  # Large content
            return "cipher"
        else:
            return "hybrid"  # Store in both for redundancy
    
    async def _select_search_strategy(
        self,
        project_creds: ProjectCredentials,
        search_request: UniversalMemorySearch
    ) -> str:
        """Select the best search strategy."""
        if search_request.semantic_search:
            return "semantic"
        
        # Check query characteristics
        query_length = len(search_request.query.split())
        if query_length > 3:  # Complex queries benefit from semantic search
            return "hybrid"
        else:
            return "keyword"
    
    async def _store_in_cipher(
        self,
        memory_id: str,
        project_creds: ProjectCredentials,
        memory_request: UniversalMemoryStore
    ):
        """Store memory in Cipher."""
        await self.cipher_service.store_memory(
            project_id=project_creds.project_id,
            agent_id="universal_api",
            memory_content=memory_request.content,
            memory_type=memory_request.memory_type.value,
            tags=memory_request.tags,
            metadata=memory_request.metadata,
            priority=self._convert_priority_to_int(memory_request.priority)
        )
    
    async def _store_in_weaviate(
        self,
        memory_id: str,
        project_creds: ProjectCredentials,
        memory_request: UniversalMemoryStore
    ):
        """Store memory in Weaviate."""
        await self.weaviate_client.store_memory(
            memory_id=memory_id,
            content=memory_request.content,
            agent_id="universal_api",
            memory_type=memory_request.memory_type.value,
            importance=self._convert_priority_to_float(memory_request.priority),
            tags=memory_request.tags,
            metadata=memory_request.metadata,
            project_id=project_creds.project_id,
            expires_at=memory_request.expires_at
        )
    
    async def _store_hybrid(
        self,
        memory_id: str,
        project_creds: ProjectCredentials,
        memory_request: UniversalMemoryStore
    ):
        """Store memory in both backends."""
        await asyncio.gather(
            self._store_in_cipher(memory_id, project_creds, memory_request),
            self._store_in_weaviate(memory_id, project_creds, memory_request),
            return_exceptions=True
        )
    
    async def _get_single_memory(
        self,
        memory_id: str,
        project_creds: ProjectCredentials,
        retrieve_request: UniversalMemoryRetrieve
    ) -> Optional[UniversalMemoryItem]:
        """Get a single memory by ID."""
        # Try Cipher first
        try:
            if self.cipher_service:
                memory_data = await self.cipher_service.retrieve_memory(
                    project_creds.project_id, memory_id, retrieve_request.use_cache
                )
                if memory_data:
                    return self._format_memory_item(memory_data, "cipher")
        except Exception as e:
            logger.warning(f"Failed to retrieve from Cipher: {e}")
        
        # Try Weaviate
        try:
            if self.weaviate_client:
                memory_data = await self.weaviate_client.get_memory(memory_id)
                if memory_data:
                    return self._format_memory_item(memory_data, "weaviate")
        except Exception as e:
            logger.warning(f"Failed to retrieve from Weaviate: {e}")
        
        return None
    
    async def _get_multiple_memories(
        self,
        project_creds: ProjectCredentials,
        retrieve_request: UniversalMemoryRetrieve
    ) -> List[UniversalMemoryItem]:
        """Get multiple memories."""
        memories = []
        
        # Get from Cipher
        try:
            if self.cipher_service:
                cipher_results = await self.cipher_service.search_memories(
                    project_id=project_creds.project_id,
                    query="",  # Get all memories
                    limit=retrieve_request.limit,
                    offset=retrieve_request.offset
                )
                
                for result in cipher_results.get("results", []):
                    memory_item = self._format_memory_item(result, "cipher")
                    if memory_item:
                        memories.append(memory_item)
        except Exception as e:
            logger.warning(f"Failed to retrieve from Cipher: {e}")
        
        return memories
    
    async def _semantic_search(
        self,
        project_creds: ProjectCredentials,
        search_request: UniversalMemorySearch
    ) -> List[UniversalMemorySearchResult]:
        """Perform semantic search using Weaviate."""
        try:
            if not self.weaviate_client:
                return []
            
            results = await self.weaviate_client.search_memories(
                query=search_request.query,
                project_id=project_creds.project_id,
                memory_type=search_request.memory_type.value if search_request.memory_type else None,
                limit=search_request.limit,
                similarity_threshold=0.7
            )
            
            search_results = []
            for result in results:
                memory_item = self._format_memory_item(result, "weaviate")
                if memory_item:
                    search_result = UniversalMemorySearchResult(
                        memory=memory_item,
                        relevance_score=result.get("similarity", 0.0),
                        matched_fields=["content"],
                        search_method="semantic"
                    )
                    search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _keyword_search(
        self,
        project_creds: ProjectCredentials,
        search_request: UniversalMemorySearch
    ) -> List[UniversalMemorySearchResult]:
        """Perform keyword search using Cipher."""
        try:
            if not self.cipher_service:
                return []
            
            results = await self.cipher_service.search_memories(
                project_id=project_creds.project_id,
                query=search_request.query,
                memory_type=search_request.memory_type.value if search_request.memory_type else None,
                tags=search_request.tags,
                limit=search_request.limit,
                offset=search_request.offset
            )
            
            search_results = []
            for result in results.get("results", []):
                memory_item = self._format_memory_item(result, "cipher")
                if memory_item:
                    search_result = UniversalMemorySearchResult(
                        memory=memory_item,
                        relevance_score=0.8,  # Default relevance for keyword search
                        matched_fields=["content"],
                        search_method="keyword"
                    )
                    search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    async def _hybrid_search(
        self,
        project_creds: ProjectCredentials,
        search_request: UniversalMemorySearch
    ) -> List[UniversalMemorySearchResult]:
        """Perform hybrid search using both backends."""
        # Run both searches in parallel
        semantic_results, keyword_results = await asyncio.gather(
            self._semantic_search(project_creds, search_request),
            self._keyword_search(project_creds, search_request),
            return_exceptions=True
        )
        
        # Combine and deduplicate results
        all_results = []
        seen_memory_ids = set()
        
        # Add semantic results first (higher priority)
        for result in semantic_results if isinstance(semantic_results, list) else []:
            if result.memory.memory_id not in seen_memory_ids:
                all_results.append(result)
                seen_memory_ids.add(result.memory.memory_id)
        
        # Add keyword results
        for result in keyword_results if isinstance(keyword_results, list) else []:
            if result.memory.memory_id not in seen_memory_ids:
                all_results.append(result)
                seen_memory_ids.add(result.memory.memory_id)
        
        return all_results
    
    def _format_memory_item(self, memory_data: Dict[str, Any], source: str) -> Optional[UniversalMemoryItem]:
        """Format memory data into UniversalMemoryItem."""
        try:
            return UniversalMemoryItem(
                memory_id=memory_data.get("id") or memory_data.get("memory_id"),
                project_id=memory_data.get("project_id", "unknown"),
                content=memory_data.get("content", ""),
                memory_type=MemoryType(memory_data.get("memory_type", "knowledge")),
                priority=Priority(self._convert_int_to_priority(memory_data.get("priority", 2))),
                tags=memory_data.get("tags", []),
                metadata=memory_data.get("metadata", {}),
                created_at=datetime.fromisoformat(memory_data.get("created_at", datetime.utcnow().isoformat())),
                updated_at=datetime.fromisoformat(memory_data.get("updated_at", datetime.utcnow().isoformat())),
                expires_at=datetime.fromisoformat(memory_data.get("expires_at")) if memory_data.get("expires_at") else None,
                storage_location=source,
                size_bytes=len(memory_data.get("content", "").encode('utf-8'))
            )
        except Exception as e:
            logger.error(f"Failed to format memory item: {e}")
            return None
    
    def _apply_retrieval_filters(
        self,
        memories: List[UniversalMemoryItem],
        retrieve_request: UniversalMemoryRetrieve
    ) -> List[UniversalMemoryItem]:
        """Apply filters to retrieved memories."""
        filtered = memories
        
        if retrieve_request.memory_type:
            filtered = [m for m in filtered if m.memory_type == retrieve_request.memory_type]
        
        if retrieve_request.tags:
            filtered = [m for m in filtered if any(tag in m.tags for tag in retrieve_request.tags)]
        
        return filtered
    
    def _apply_search_filters(
        self,
        results: List[UniversalMemorySearchResult],
        search_request: UniversalMemorySearch
    ) -> List[UniversalMemorySearchResult]:
        """Apply filters to search results."""
        filtered = results
        
        if search_request.memory_type:
            filtered = [r for r in filtered if r.memory.memory_type == search_request.memory_type]
        
        if search_request.tags:
            filtered = [r for r in filtered if any(tag in r.memory.tags for tag in search_request.tags)]
        
        if search_request.priority:
            filtered = [r for r in filtered if r.memory.priority == search_request.priority]
        
        if search_request.created_after:
            filtered = [r for r in filtered if r.memory.created_at >= search_request.created_after]
        
        if search_request.created_before:
            filtered = [r for r in filtered if r.memory.created_at <= search_request.created_before]
        
        return filtered
    
    async def _estimate_retrieval_time(self, storage_backend: str) -> float:
        """Estimate retrieval time for storage backend."""
        # Mock estimation - in real implementation, this would be based on historical data
        estimates = {
            "cipher": 50.0,
            "weaviate": 100.0,
            "hybrid": 75.0
        }
        return estimates.get(storage_backend, 100.0)
    
    def _convert_priority_to_int(self, priority: Priority) -> int:
        """Convert Priority enum to integer."""
        mapping = {
            Priority.LOW: 1,
            Priority.NORMAL: 2,
            Priority.HIGH: 3,
            Priority.CRITICAL: 4
        }
        return mapping.get(priority, 2)
    
    def _convert_priority_to_float(self, priority: Priority) -> float:
        """Convert Priority enum to float for Weaviate importance."""
        mapping = {
            Priority.LOW: 0.25,
            Priority.NORMAL: 0.5,
            Priority.HIGH: 0.75,
            Priority.CRITICAL: 1.0
        }
        return mapping.get(priority, 0.5)
    
    def _convert_int_to_priority(self, priority_int: int) -> str:
        """Convert integer to Priority string."""
        mapping = {
            1: "low",
            2: "normal",
            3: "high",
            4: "critical"
        }
        return mapping.get(priority_int, "normal")


# Global service instance
_universal_service: Optional[UniversalMemoryService] = None


async def get_universal_service() -> UniversalMemoryService:
    """Get or create global universal memory service."""
    global _universal_service
    
    if _universal_service is None:
        _universal_service = UniversalMemoryService()
        await _universal_service.initialize()
    
    return _universal_service


# API Endpoints

@router.post(
    "/memories",
    response_model=UniversalMemoryResponse,
    status_code=201,
    summary="Store memory",
    description="Store a new memory for your project"
)
async def store_memory(
    memory_request: UniversalMemoryStore,
    project: ProjectCredentials = Depends(get_current_project),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> UniversalMemoryResponse:
    """Store a new memory for the authenticated project."""
    
    start_time = time.time()
    
    try:
        logger.info(f"Storing memory for project: {project.project_id}")
        
        # Get universal service
        service = await get_universal_service()
        
        # Store memory
        result = await service.store_memory(project, memory_request)
        
        # Add background task for analytics
        background_tasks.add_task(
            log_memory_operation,
            project.project_id,
            "store",
            result.memory_id
        )
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("POST", "/universal/memories", duration)
        record_request("POST", "/universal/memories", 201)
        
        logger.info(f"Memory stored successfully: {result.memory_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        record_request_duration("POST", "/universal/memories", duration)
        record_request("POST", "/universal/memories", 500)
        
        logger.error(f"Memory storage failed: {e}")
        raise HTTPException(status_code=500, detail="Memory storage failed")


@router.get(
    "/memories",
    response_model=UniversalMemoryListResponse,
    summary="Retrieve memories",
    description="Retrieve memories for your project with filtering and pagination"
)
async def retrieve_memories(
    project_id: str = Query(..., description="Project identifier"),
    memory_id: Optional[str] = Query(None, description="Specific memory ID to retrieve"),
    memory_type: Optional[MemoryType] = Query(None, description="Filter by memory type"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of memories"),
    offset: int = Query(0, ge=0, description="Number of memories to skip"),
    include_metadata: bool = Query(True, description="Include metadata in response"),
    use_cache: bool = Query(True, description="Use cache for faster retrieval"),
    project: ProjectCredentials = Depends(get_current_project)
) -> UniversalMemoryListResponse:
    """Retrieve memories for the authenticated project."""
    
    start_time = time.time()
    
    try:
        logger.info(f"Retrieving memories for project: {project.project_id}")
        
        # Create retrieve request
        retrieve_request = UniversalMemoryRetrieve(
            project_id=project_id,
            memory_id=memory_id,
            memory_type=memory_type,
            tags=tags,
            limit=limit,
            offset=offset,
            include_metadata=include_metadata,
            use_cache=use_cache
        )
        
        # Get universal service
        service = await get_universal_service()
        
        # Retrieve memories
        result = await service.retrieve_memories(project, retrieve_request)
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("GET", "/universal/memories", duration)
        record_request("GET", "/universal/memories", 200)
        
        logger.info(f"Retrieved {len(result.memories)} memories for project {project.project_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        record_request_duration("GET", "/universal/memories", duration)
        record_request("GET", "/universal/memories", 500)
        
        logger.error(f"Memory retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Memory retrieval failed")


@router.post(
    "/memories/search",
    response_model=UniversalMemorySearchResponse,
    summary="Search memories",
    description="Search memories using semantic similarity and filtering"
)
async def search_memories(
    search_request: UniversalMemorySearch,
    project: ProjectCredentials = Depends(get_current_project)
) -> UniversalMemorySearchResponse:
    """Search memories for the authenticated project."""
    
    start_time = time.time()
    
    try:
        logger.info(f"Searching memories for project: {project.project_id}")
        
        # Get universal service
        service = await get_universal_service()
        
        # Search memories
        result = await service.search_memories(project, search_request)
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("POST", "/universal/memories/search", duration)
        record_request("POST", "/universal/memories/search", 200)
        
        logger.info(f"Search completed: {len(result.results)} results for project {project.project_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        record_request_duration("POST", "/universal/memories/search", duration)
        record_request("POST", "/universal/memories/search", 500)
        
        logger.error(f"Memory search failed: {e}")
        raise HTTPException(status_code=500, detail="Memory search failed")


@router.put(
    "/memories/{memory_id}",
    summary="Update memory",
    description="Update an existing memory"
)
async def update_memory(
    memory_id: str,
    update_request: UniversalMemoryUpdate,
    project: ProjectCredentials = Depends(get_current_project)
) -> Dict[str, Any]:
    """Update an existing memory."""
    
    start_time = time.time()
    
    try:
        logger.info(f"Updating memory {memory_id} for project: {project.project_id}")
        
        # Set memory_id from path
        update_request.memory_id = memory_id
        
        # Get universal service
        service = await get_universal_service()
        
        # Update memory
        result = await service.update_memory(project, update_request)
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("PUT", f"/universal/memories/{memory_id}", duration)
        record_request("PUT", f"/universal/memories/{memory_id}", 200)
        
        logger.info(f"Memory updated successfully: {memory_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        record_request_duration("PUT", f"/universal/memories/{memory_id}", duration)
        record_request("PUT", f"/universal/memories/{memory_id}", 500)
        
        logger.error(f"Memory update failed: {e}")
        raise HTTPException(status_code=500, detail="Memory update failed")


@router.delete(
    "/memories/{memory_id}",
    summary="Delete memory",
    description="Delete an existing memory"
)
async def delete_memory(
    memory_id: str,
    project_id: str = Query(..., description="Project identifier"),
    permanent: bool = Query(False, description="Whether to permanently delete"),
    project: ProjectCredentials = Depends(get_current_project)
) -> Dict[str, Any]:
    """Delete an existing memory."""
    
    start_time = time.time()
    
    try:
        logger.info(f"Deleting memory {memory_id} for project: {project.project_id}")
        
        # Create delete request
        delete_request = UniversalMemoryDelete(
            project_id=project_id,
            memory_id=memory_id,
            permanent=permanent
        )
        
        # Get universal service
        service = await get_universal_service()
        
        # Delete memory
        result = await service.delete_memory(project, delete_request)
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("DELETE", f"/universal/memories/{memory_id}", duration)
        record_request("DELETE", f"/universal/memories/{memory_id}", 200)
        
        logger.info(f"Memory deleted successfully: {memory_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        record_request_duration("DELETE", f"/universal/memories/{memory_id}", duration)
        record_request("DELETE", f"/universal/memories/{memory_id}", 500)
        
        logger.error(f"Memory deletion failed: {e}")
        raise HTTPException(status_code=500, detail="Memory deletion failed")


@router.post(
    "/memories/batch",
    response_model=UniversalMemoryBatchResult,
    summary="Batch store memories",
    description="Store multiple memories in a single request"
)
async def batch_store_memories(
    batch_request: UniversalMemoryBatchStore,
    project: ProjectCredentials = Depends(get_current_project)
) -> UniversalMemoryBatchResult:
    """Store multiple memories in a batch operation."""
    
    start_time = time.time()
    
    try:
        logger.info(f"Batch storing {len(batch_request.memories)} memories for project: {project.project_id}")
        
        # Get universal service
        service = await get_universal_service()
        
        # Process memories in batches
        successful = 0
        failed = 0
        errors = []
        memory_ids = []
        
        for i in range(0, len(batch_request.memories), batch_request.batch_size):
            batch = batch_request.memories[i:i + batch_request.batch_size]
            
            for memory_request in batch:
                try:
                    result = await service.store_memory(project, memory_request)
                    successful += 1
                    memory_ids.append(result.memory_id)
                except Exception as e:
                    failed += 1
                    errors.append({
                        "memory_index": i + batch_request.memories.index(memory_request),
                        "error": str(e)
                    })
                    
                    if not batch_request.continue_on_error:
                        break
            
            if not batch_request.continue_on_error and failed > 0:
                break
        
        result = UniversalMemoryBatchResult(
            operation_type="store",
            total_processed=len(batch_request.memories),
            successful=successful,
            failed=failed,
            errors=errors,
            execution_time_ms=(time.time() - start_time) * 1000,
            memory_ids=memory_ids
        )
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("POST", "/universal/memories/batch", duration)
        record_request("POST", "/universal/memories/batch", 200)
        
        logger.info(f"Batch operation completed: {successful} successful, {failed} failed")
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        record_request_duration("POST", "/universal/memories/batch", duration)
        record_request("POST", "/universal/memories/batch", 500)
        
        logger.error(f"Batch memory storage failed: {e}")
        raise HTTPException(status_code=500, detail="Batch memory storage failed")


@router.get(
    "/status",
    response_model=UniversalAPIStatus,
    summary="API status",
    description="Get the current status of the Universal Memory Access API"
)
async def get_api_status() -> UniversalAPIStatus:
    """Get the current status of the Universal Memory Access API."""
    
    try:
        # Check component statuses
        components = {}
        
        # Check Cipher
        try:
            service = await get_universal_service()
            if service.cipher_service:
                components["cipher"] = "healthy"
            else:
                components["cipher"] = "unavailable"
        except Exception:
            components["cipher"] = "unhealthy"
        
        # Check Weaviate
        try:
            service = await get_universal_service()
            if service.weaviate_client:
                components["weaviate"] = "healthy"
            else:
                components["weaviate"] = "unavailable"
        except Exception:
            components["weaviate"] = "unhealthy"
        
        # Check rate limiter
        try:
            rate_limiter = get_rate_limiter()
            components["rate_limiter"] = "healthy"
        except Exception:
            components["rate_limiter"] = "unhealthy"
        
        return UniversalAPIStatus(
            status="healthy" if all(status == "healthy" for status in components.values()) else "degraded",
            version="1.0.0",
            timestamp=datetime.utcnow(),
            components=components,
            uptime_seconds=time.time()  # Mock uptime
        )
        
    except Exception as e:
        logger.error(f"Failed to get API status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get API status")


# Background Tasks

async def log_memory_operation(
    project_id: str,
    operation: str,
    memory_id: str
):
    """Background task to log memory operations."""
    try:
        logger.info(f"Memory operation logged", extra={
            "project_id": project_id,
            "operation": operation,
            "memory_id": memory_id,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Failed to log memory operation: {e}")

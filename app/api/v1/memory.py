"""
Memory management endpoints for AI Agent Memory Router.

This module provides REST API endpoints for memory operations including
routing, storage, retrieval, and search functionality.
"""

import asyncio
import time
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from app.core.database import get_db_session_async
from app.core.metrics import record_request, record_request_duration
from app.core.logging import get_logger

# Setup logger
logger = get_logger(__name__)

# Create router
router = APIRouter()


# Request/Response Models

class MemoryRouteRequest(BaseModel):
    """Request model for memory routing."""
    
    source_agent_id: str = Field(..., description="Source agent identifier")
    target_agent_ids: List[str] = Field(..., description="Target agent identifiers")
    memory_content: str = Field(..., min_length=1, description="Memory content to route")
    priority: str = Field("normal", description="Routing priority")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    
    class Config:
        schema_extra = {
            "example": {
                "source_agent_id": "agent_001",
                "target_agent_ids": ["agent_002", "agent_003"],
                "memory_content": "Important information about project status",
                "priority": "high",
                "context": {
                    "project_id": "proj_123",
                    "urgency": "immediate"
                }
            }
        }


class MemoryRouteResponse(BaseModel):
    """Response model for memory routing."""
    
    route_id: str = Field(..., description="Unique route identifier")
    source_agent_id: str = Field(..., description="Source agent identifier")
    target_agent_ids: List[str] = Field(..., description="Target agent identifiers")
    status: str = Field(..., description="Routing status")
    timestamp: float = Field(..., description="Route creation timestamp")
    priority: str = Field(..., description="Routing priority")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    
    class Config:
        schema_extra = {
            "example": {
                "route_id": "route_1234567890",
                "source_agent_id": "agent_001",
                "target_agent_ids": ["agent_002", "agent_003"],
                "status": "routed",
                "timestamp": 1640995200.0,
                "priority": "high",
                "context": {
                    "project_id": "proj_123",
                    "urgency": "immediate"
                }
            }
        }


class MemoryItem(BaseModel):
    """Model for individual memory items."""
    
    memory_id: str = Field(..., description="Unique memory identifier")
    agent_id: str = Field(..., description="Agent identifier")
    content: str = Field(..., description="Memory content")
    memory_type: str = Field(..., description="Type of memory")
    tags: List[str] = Field(default_factory=list, description="Memory tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: float = Field(..., description="Creation timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "memory_id": "mem_1234567890",
                "agent_id": "agent_001",
                "content": "Project milestone achieved",
                "memory_type": "achievement",
                "tags": ["milestone", "project", "success"],
                "metadata": {"project_id": "proj_123"},
                "created_at": 1640995200.0
            }
        }


class PaginatedMemoryResponse(BaseModel):
    """Response model for paginated memory results."""
    
    items: List[MemoryItem] = Field(..., description="Memory items")
    total: int = Field(..., description="Total number of memories")
    limit: int = Field(..., description="Number of items per page")
    offset: int = Field(..., description="Number of items skipped")
    has_more: bool = Field(..., description="Whether there are more items")


class MemorySearchRequest(BaseModel):
    """Request model for memory search."""
    
    query: str = Field(..., description="Search query")
    agent_id: Optional[str] = Field(None, description="Filter by agent ID")
    memory_type: Optional[str] = Field(None, description="Filter by memory type")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Number of results to skip")


class MemorySearchResult(BaseModel):
    """Model for memory search results."""
    
    memory_id: str = Field(..., description="Memory identifier")
    agent_id: str = Field(..., description="Agent identifier")
    content: str = Field(..., description="Memory content")
    relevance_score: float = Field(..., description="Search relevance score")
    memory_type: str = Field(..., description="Memory type")
    tags: List[str] = Field(default_factory=list, description="Memory tags")
    created_at: float = Field(..., description="Creation timestamp")


class MemorySearchResponse(BaseModel):
    """Response model for memory search."""
    
    query: str = Field(..., description="Search query")
    results: List[MemorySearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_time_ms: float = Field(..., description="Search execution time")


# Cipher-specific models

class CipherMemoryStoreRequest(BaseModel):
    """Request model for storing memory in Cipher."""
    
    project_id: str = Field(..., description="Project identifier")
    agent_id: str = Field(..., description="Agent identifier")
    memory_content: str = Field(..., min_length=1, description="Memory content to store")
    memory_type: str = Field("general", description="Type of memory")
    tags: Optional[List[str]] = Field(None, description="Memory tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    priority: int = Field(1, ge=1, le=10, description="Memory priority (1-10)")


class CipherMemoryStoreResponse(BaseModel):
    """Response model for storing memory in Cipher."""
    
    memory_id: str = Field(..., description="Unique memory identifier")
    project_id: str = Field(..., description="Project identifier")
    agent_id: str = Field(..., description="Agent identifier")
    memory_type: str = Field(..., description="Type of memory")
    status: str = Field(..., description="Storage status")
    timestamp: float = Field(..., description="Storage timestamp")


class CipherMemoryRetrieveResponse(BaseModel):
    """Response model for retrieving memory from Cipher."""
    
    memory_id: str = Field(..., description="Memory identifier")
    project_id: str = Field(..., description="Project identifier")
    memory_data: Optional[Dict[str, Any]] = Field(None, description="Memory data")
    status: str = Field(..., description="Retrieval status")
    from_cache: bool = Field(False, description="Whether data came from cache")
    timestamp: float = Field(..., description="Retrieval timestamp")


class CipherMemorySearchRequest(BaseModel):
    """Request model for searching memories in Cipher."""
    
    project_id: str = Field(..., description="Project identifier")
    query: str = Field(..., min_length=1, description="Search query")
    agent_id: Optional[str] = Field(None, description="Filter by agent ID")
    memory_type: Optional[str] = Field(None, description="Filter by memory type")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Number of results to skip")


class CipherMemorySearchResponse(BaseModel):
    """Response model for searching memories in Cipher."""
    
    project_id: str = Field(..., description="Project identifier")
    query: str = Field(..., description="Search query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total: int = Field(..., description="Total number of results")
    from_cache: bool = Field(False, description="Whether results came from cache")
    timestamp: float = Field(..., description="Search timestamp")


class CipherProjectCreateRequest(BaseModel):
    """Request model for creating a project in Cipher."""
    
    project_id: str = Field(..., description="Project identifier")
    project_name: str = Field(..., min_length=1, description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Project metadata")


class CipherProjectCreateResponse(BaseModel):
    """Response model for creating a project in Cipher."""
    
    project_id: str = Field(..., description="Project identifier")
    project_name: str = Field(..., description="Project name")
    status: str = Field(..., description="Creation status")
    timestamp: float = Field(..., description="Creation timestamp")


# Endpoints

@router.post(
    "/route",
    response_model=MemoryRouteResponse,
    status_code=201,
    summary="Route memory between AI agents",
    description="Intelligently route memory content between specified AI agents"
)
async def route_memory(
    request: MemoryRouteRequest,
    background_tasks: BackgroundTasks
) -> MemoryRouteResponse:
    """Route memory between AI agents with intelligent decision making."""
    
    start_time = time.time()
    
    try:
        logger.info(f"Routing memory from {request.source_agent_id} to {request.target_agent_ids}")
        
        # Add background task for logging and analytics
        background_tasks.add_task(
            log_memory_routing_attempt,
            request.source_agent_id,
            request.target_agent_ids,
            request.priority
        )
        
        # Execute routing logic (mock implementation for now)
        route_id = f"route_{int(time.time() * 1000)}"
        result = MemoryRouteResponse(
            route_id=route_id,
            source_agent_id=request.source_agent_id,
            target_agent_ids=request.target_agent_ids,
            status="routed",
            timestamp=time.time(),
            priority=request.priority,
            context=request.context
        )
        
        # Add background task for notification
        background_tasks.add_task(
            notify_target_agents,
            request.target_agent_ids,
            result
        )
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("POST", "/memory/route", duration)
        record_request("POST", "/memory/route", 201)
        
        logger.info(f"Memory routing completed: {route_id}")
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration("POST", "/memory/route", duration)
        record_request("POST", "/memory/route", 500)
        
        logger.error(f"Memory routing failed: {e}")
        raise HTTPException(status_code=500, detail="Memory routing failed")


@router.get(
    "/{agent_id}",
    response_model=PaginatedMemoryResponse,
    summary="Retrieve agent memories",
    description="Get paginated list of memories for a specific agent"
)
async def get_agent_memories(
    agent_id: str,
    limit: int = Query(10, ge=1, le=100, description="Number of memories to return"),
    offset: int = Query(0, ge=0, description="Number of memories to skip"),
    memory_type: Optional[str] = Query(None, description="Filter by memory type"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags")
) -> PaginatedMemoryResponse:
    """Retrieve memories for a specific agent with pagination and filtering."""
    
    start_time = time.time()
    
    try:
        logger.info(f"Retrieving memories for agent: {agent_id}")
        
        # Mock implementation for now
        mock_memories = [
            MemoryItem(
                memory_id=f"mem_{i}",
                agent_id=agent_id,
                content=f"Sample memory content {i}",
                memory_type=memory_type or "general",
                tags=tags or ["sample"],
                created_at=time.time() - (i * 3600)
            )
            for i in range(1, min(limit + 1, 6))
        ]
        
        total_count = 25  # Mock total count
        
        result = PaginatedMemoryResponse(
            items=mock_memories,
            total=total_count,
            limit=limit,
            offset=offset,
            has_more=offset + limit < total_count
        )
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("GET", f"/memory/{agent_id}", 200)
        record_request("GET", f"/memory/{agent_id}", 200)
        
        logger.info(f"Retrieved {len(mock_memories)} memories for agent {agent_id}")
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration("GET", f"/memory/{agent_id}", 500)
        record_request("GET", f"/memory/{agent_id}", 500)
        
        logger.error(f"Failed to retrieve memories for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve memories")


@router.post(
    "/search",
    response_model=MemorySearchResponse,
    summary="Search agent memories",
    description="Search memories using semantic similarity and filtering"
)
async def search_memories(request: MemorySearchRequest) -> MemorySearchResponse:
    """Search memories using semantic similarity and filtering."""
    
    start_time = time.time()
    
    try:
        logger.info(f"Searching memories with query: {request.query}")
        
        # Mock implementation for now
        mock_results = [
            MemorySearchResult(
                memory_id=f"mem_{i}",
                agent_id=request.agent_id or f"agent_{i}",
                content=f"Sample search result {i} matching '{request.query}'",
                relevance_score=0.9 - (i * 0.1),
                memory_type=request.memory_type or "general",
                tags=request.tags or ["search_result"],
                created_at=time.time() - (i * 3600)
            )
            for i in range(1, min(request.limit + 1, 6))
        ]
        
        result = MemorySearchResponse(
            query=request.query,
            results=mock_results,
            total_results=15,  # Mock total
            search_time_ms=(time.time() - start_time) * 1000
        )
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("POST", "/memory/search", 200)
        record_request("POST", "/memory/search", 200)
        
        logger.info(f"Memory search completed: {len(mock_results)} results")
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration("POST", "/memory/search", 500)
        record_request("POST", "/memory/search", 500)
        
        logger.error(f"Memory search failed: {e}")
        raise HTTPException(status_code=500, detail="Memory search failed")


@router.get(
    "/route/{route_id}",
    response_model=MemoryRouteResponse,
    summary="Get memory route details",
    description="Retrieve details of a specific memory route"
)
async def get_memory_route(route_id: str) -> MemoryRouteResponse:
    """Get details of a specific memory route."""
    
    start_time = time.time()
    
    try:
        logger.info(f"Retrieving memory route: {route_id}")
        
        # Mock implementation for now
        result = MemoryRouteResponse(
            route_id=route_id,
            source_agent_id="agent_001",
            target_agent_ids=["agent_002", "agent_003"],
            status="completed",
            timestamp=time.time() - 3600,
            priority="normal",
            context={"route_type": "information_sharing"}
        )
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("GET", f"/memory/route/{route_id}", 200)
        record_request("GET", f"/memory/route/{route_id}", 200)
        
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration("GET", f"/memory/route/{route_id}", 500)
        record_request("GET", f"/memory/route/{route_id}", 500)
        
        logger.error(f"Failed to retrieve memory route {route_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve memory route")


@router.get(
    "/stats/overview",
    summary="Get memory statistics overview",
    description="Retrieve overview statistics for memory operations"
)
async def get_memory_stats_overview():
    """Get overview statistics for memory operations."""
    
    start_time = time.time()
    
    try:
        logger.info("Retrieving memory statistics overview")
        
        # Mock implementation for now
        stats = {
            "total_memories": 1250,
            "total_routes": 890,
            "successful_routes": 845,
            "failed_routes": 45,
            "success_rate": 0.949,
            "average_routing_time_ms": 45.2,
            "top_agents": [
                {"agent_id": "agent_1", "memory_count": 150},
                {"agent_id": "agent_2", "memory_count": 120},
                {"agent_id": "agent_3", "memory_count": 95}
            ],
            "timestamp": time.time()
        }
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("GET", "/memory/stats/overview", 200)
        record_request("GET", "/memory/stats/overview", 200)
        
        return stats
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration("GET", "/memory/stats/overview", 500)
        record_request("GET", "/memory/stats/overview", 500)
        
        logger.error(f"Failed to retrieve memory statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve memory statistics")


# Background Tasks

async def log_memory_routing_attempt(
    source_agent_id: str,
    target_agent_ids: List[str],
    priority: str
):
    """Background task to log memory routing attempt."""
    try:
        logger.info(f"Memory routing attempt logged", extra={
            "source_agent_id": source_agent_id,
            "target_agent_ids": target_agent_ids,
            "priority": priority,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Failed to log memory routing attempt: {e}")


async def notify_target_agents(
    target_agent_ids: List[str],
    route_result: MemoryRouteResponse
):
    """Background task to notify target agents."""
    try:
        logger.info(f"Notifying target agents: {target_agent_ids}")
        
        # Mock notification logic
        for agent_id in target_agent_ids:
            logger.info(f"Notification sent to agent: {agent_id}")
            
        logger.info(f"Target agent notifications completed")
        
    except Exception as e:
        logger.error(f"Failed to notify target agents: {e}")


# Cipher Integration Endpoints

@router.post(
    "/cipher/store",
    response_model=CipherMemoryStoreResponse,
    status_code=201,
    summary="Store memory in Cipher",
    description="Store memory content in Cipher with hybrid SQLite storage"
)
async def cipher_store_memory(request: CipherMemoryStoreRequest) -> CipherMemoryStoreResponse:
    """Store memory in Cipher with hybrid storage."""
    
    start_time = time.time()
    
    try:
        logger.info(f"Storing memory in Cipher for project {request.project_id}, agent {request.agent_id}")
        
        from app.services.cipher_service import get_cipher_service
        
        cipher_service = await get_cipher_service()
        
        memory_id = await cipher_service.store_memory(
            project_id=request.project_id,
            agent_id=request.agent_id,
            memory_content=request.memory_content,
            memory_type=request.memory_type,
            tags=request.tags,
            metadata=request.metadata,
            priority=request.priority
        )
        
        result = CipherMemoryStoreResponse(
            memory_id=memory_id,
            project_id=request.project_id,
            agent_id=request.agent_id,
            memory_type=request.memory_type,
            status="stored",
            timestamp=time.time()
        )
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("POST", "/memory/cipher/store", 201)
        record_request("POST", "/memory/cipher/store", 201)
        
        logger.info(f"Cipher memory storage completed: {memory_id}")
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration("POST", "/memory/cipher/store", 500)
        record_request("POST", "/memory/cipher/store", 500)
        
        logger.error(f"Cipher memory storage failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cipher memory storage failed: {str(e)}")


@router.get(
    "/cipher/retrieve/{project_id}/{memory_id}",
    response_model=CipherMemoryRetrieveResponse,
    summary="Retrieve memory from Cipher",
    description="Retrieve memory content from Cipher with cache support"
)
async def cipher_retrieve_memory(
    project_id: str,
    memory_id: str,
    use_cache: bool = Query(True, description="Whether to use local cache")
) -> CipherMemoryRetrieveResponse:
    """Retrieve memory from Cipher with cache support."""
    
    start_time = time.time()
    
    try:
        logger.info(f"Retrieving memory {memory_id} from Cipher for project {project_id}")
        
        from app.services.cipher_service import get_cipher_service
        
        cipher_service = await get_cipher_service()
        
        memory_data = await cipher_service.retrieve_memory(
            project_id=project_id,
            memory_id=memory_id,
            use_cache=use_cache
        )
        
        if memory_data:
            result = CipherMemoryRetrieveResponse(
                memory_id=memory_id,
                project_id=project_id,
                memory_data=memory_data,
                status="retrieved",
                from_cache=memory_data.get("from_cache", False),
                timestamp=time.time()
            )
        else:
            result = CipherMemoryRetrieveResponse(
                memory_id=memory_id,
                project_id=project_id,
                memory_data=None,
                status="not_found",
                from_cache=False,
                timestamp=time.time()
            )
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("GET", f"/memory/cipher/retrieve/{project_id}/{memory_id}", 200)
        record_request("GET", f"/memory/cipher/retrieve/{project_id}/{memory_id}", 200)
        
        logger.info(f"Cipher memory retrieval completed: {memory_id}")
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration("GET", f"/memory/cipher/retrieve/{project_id}/{memory_id}", 500)
        record_request("GET", f"/memory/cipher/retrieve/{project_id}/{memory_id}", 500)
        
        logger.error(f"Cipher memory retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cipher memory retrieval failed: {str(e)}")


@router.post(
    "/cipher/search",
    response_model=CipherMemorySearchResponse,
    summary="Search memories in Cipher",
    description="Search memories in Cipher with semantic similarity and filtering"
)
async def cipher_search_memories(request: CipherMemorySearchRequest) -> CipherMemorySearchResponse:
    """Search memories in Cipher with semantic similarity and filtering."""
    
    start_time = time.time()
    
    try:
        logger.info(f"Searching memories in Cipher for project {request.project_id} with query: {request.query}")
        
        from app.services.cipher_service import get_cipher_service
        
        cipher_service = await get_cipher_service()
        
        search_results = await cipher_service.search_memories(
            project_id=request.project_id,
            query=request.query,
            agent_id=request.agent_id,
            memory_type=request.memory_type,
            tags=request.tags,
            limit=request.limit,
            offset=request.offset
        )
        
        result = CipherMemorySearchResponse(
            project_id=request.project_id,
            query=request.query,
            results=search_results.get("results", []),
            total=search_results.get("total", 0),
            from_cache=search_results.get("from_cache", False),
            timestamp=time.time()
        )
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("POST", "/memory/cipher/search", 200)
        record_request("POST", "/memory/cipher/search", 200)
        
        logger.info(f"Cipher memory search completed: {len(result.results)} results")
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration("POST", "/memory/cipher/search", 500)
        record_request("POST", "/memory/cipher/search", 500)
        
        logger.error(f"Cipher memory search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cipher memory search failed: {str(e)}")


@router.put(
    "/cipher/update/{project_id}/{memory_id}",
    summary="Update memory in Cipher",
    description="Update memory content in Cipher"
)
async def cipher_update_memory(
    project_id: str,
    memory_id: str,
    updates: Dict[str, Any]
):
    """Update memory in Cipher."""
    
    start_time = time.time()
    
    try:
        logger.info(f"Updating memory {memory_id} in Cipher for project {project_id}")
        
        from app.services.cipher_service import get_cipher_service
        
        cipher_service = await get_cipher_service()
        
        success = await cipher_service.update_memory(
            project_id=project_id,
            memory_id=memory_id,
            updates=updates
        )
        
        result = {
            "memory_id": memory_id,
            "project_id": project_id,
            "updates": updates,
            "status": "updated" if success else "failed",
            "timestamp": time.time()
        }
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("PUT", f"/memory/cipher/update/{project_id}/{memory_id}", 200)
        record_request("PUT", f"/memory/cipher/update/{project_id}/{memory_id}", 200)
        
        logger.info(f"Cipher memory update completed: {memory_id}")
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration("PUT", f"/memory/cipher/update/{project_id}/{memory_id}", 500)
        record_request("PUT", f"/memory/cipher/update/{project_id}/{memory_id}", 500)
        
        logger.error(f"Cipher memory update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cipher memory update failed: {str(e)}")


@router.delete(
    "/cipher/delete/{project_id}/{memory_id}",
    summary="Delete memory from Cipher",
    description="Delete memory content from Cipher"
)
async def cipher_delete_memory(
    project_id: str,
    memory_id: str
):
    """Delete memory from Cipher."""
    
    start_time = time.time()
    
    try:
        logger.info(f"Deleting memory {memory_id} from Cipher for project {project_id}")
        
        from app.services.cipher_service import get_cipher_service
        
        cipher_service = await get_cipher_service()
        
        success = await cipher_service.delete_memory(
            project_id=project_id,
            memory_id=memory_id
        )
        
        result = {
            "memory_id": memory_id,
            "project_id": project_id,
            "status": "deleted" if success else "failed",
            "timestamp": time.time()
        }
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("DELETE", f"/memory/cipher/delete/{project_id}/{memory_id}", 200)
        record_request("DELETE", f"/memory/cipher/delete/{project_id}/{memory_id}", 200)
        
        logger.info(f"Cipher memory deletion completed: {memory_id}")
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration("DELETE", f"/memory/cipher/delete/{project_id}/{memory_id}", 500)
        record_request("DELETE", f"/memory/cipher/delete/{project_id}/{memory_id}", 500)
        
        logger.error(f"Cipher memory deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cipher memory deletion failed: {str(e)}")


@router.post(
    "/cipher/projects",
    response_model=CipherProjectCreateResponse,
    status_code=201,
    summary="Create project in Cipher",
    description="Create a new project in Cipher for memory organization"
)
async def cipher_create_project(request: CipherProjectCreateRequest) -> CipherProjectCreateResponse:
    """Create a new project in Cipher."""
    
    start_time = time.time()
    
    try:
        logger.info(f"Creating project {request.project_id}: {request.project_name} in Cipher")
        
        from app.services.cipher_service import get_cipher_service
        
        cipher_service = await get_cipher_service()
        
        success = await cipher_service.create_project(
            project_id=request.project_id,
            project_name=request.project_name,
            description=request.description,
            metadata=request.metadata
        )
        
        result = CipherProjectCreateResponse(
            project_id=request.project_id,
            project_name=request.project_name,
            status="created" if success else "failed",
            timestamp=time.time()
        )
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("POST", "/memory/cipher/projects", 201)
        record_request("POST", "/memory/cipher/projects", 201)
        
        logger.info(f"Cipher project creation completed: {request.project_id}")
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration("POST", "/memory/cipher/projects", 500)
        record_request("POST", "/memory/cipher/projects", 500)
        
        logger.error(f"Cipher project creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cipher project creation failed: {str(e)}")


@router.get(
    "/cipher/projects/{project_id}",
    summary="Get project information from Cipher",
    description="Retrieve project information from Cipher"
)
async def cipher_get_project(project_id: str):
    """Get project information from Cipher."""
    
    start_time = time.time()
    
    try:
        logger.info(f"Getting project {project_id} from Cipher")
        
        from app.services.cipher_service import get_cipher_service
        
        cipher_service = await get_cipher_service()
        
        project_info = await cipher_service.get_project_info(project_id)
        
        if project_info:
            result = {
                "project_id": project_id,
                "project_info": project_info,
                "status": "found",
                "timestamp": time.time()
            }
        else:
            result = {
                "project_id": project_id,
                "status": "not_found",
                "timestamp": time.time()
            }
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("GET", f"/memory/cipher/projects/{project_id}", 200)
        record_request("GET", f"/memory/cipher/projects/{project_id}", 200)
        
        logger.info(f"Cipher project retrieval completed: {project_id}")
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration("GET", f"/memory/cipher/projects/{project_id}", 500)
        record_request("GET", f"/memory/cipher/projects/{project_id}", 500)
        
        logger.error(f"Cipher project retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cipher project retrieval failed: {str(e)}")


@router.get(
    "/cipher/projects",
    summary="List projects in Cipher",
    description="List all projects in Cipher"
)
async def cipher_list_projects(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of projects to return"),
    offset: int = Query(0, ge=0, description="Number of projects to skip")
):
    """List all projects in Cipher."""
    
    start_time = time.time()
    
    try:
        logger.info("Listing projects in Cipher")
        
        from app.core.cipher_client import get_cipher_client
        
        cipher_client = await get_cipher_client()
        
        projects_data = await cipher_client.list_projects(limit=limit, offset=offset)
        
        result = {
            "projects": projects_data.get("projects", []),
            "total": projects_data.get("total", 0),
            "limit": limit,
            "offset": offset,
            "timestamp": time.time()
        }
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("GET", "/memory/cipher/projects", 200)
        record_request("GET", "/memory/cipher/projects", 200)
        
        logger.info(f"Cipher project listing completed: {len(result['projects'])} projects")
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration("GET", "/memory/cipher/projects", 500)
        record_request("GET", "/memory/cipher/projects", 500)
        
        logger.error(f"Cipher project listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cipher project listing failed: {str(e)}")


@router.get(
    "/cipher/health",
    summary="Check Cipher MCP server health",
    description="Check the health status of Cipher MCP server"
)
async def cipher_health_check():
    """Check Cipher MCP server health."""
    
    start_time = time.time()
    
    try:
        logger.info("Checking Cipher MCP server health")
        
        from app.core.cipher_client import get_cipher_client
        
        cipher_client = await get_cipher_client()
        
        health_data = await cipher_client.health_check()
        
        result = {
            "status": "healthy" if health_data.get("status") == "ok" else "unhealthy",
            "health_data": health_data,
            "timestamp": time.time()
        }
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration("GET", "/memory/cipher/health", 200)
        record_request("GET", "/memory/cipher/health", 200)
        
        logger.info(f"Cipher health check completed: {result['status']}")
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration("GET", "/memory/cipher/health", 500)
        record_request("GET", "/memory/cipher/health", 500)
        
        logger.error(f"Cipher health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

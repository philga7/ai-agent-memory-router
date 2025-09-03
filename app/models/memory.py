"""
Memory models for AI Agent Memory Router.

This module defines Pydantic models for memory-related operations
including routing, storage, and search functionality.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from uuid import UUID

from .common import TimestampMixin, IDMixin, PaginationParams, PaginatedResponse


class MemoryContent(BaseModel):
    """Memory content model."""
    
    text: str = Field(..., description="Memory text content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    tags: Optional[List[str]] = Field(default_factory=list, description="Memory tags")
    confidence: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v


class MemorySource(BaseModel):
    """Memory source model."""
    
    agent_id: UUID = Field(..., description="Source agent identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Source timestamp")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Source context")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemoryRoute(BaseModel, TimestampMixin, IDMixin):
    """Memory route model."""
    
    source_agent_id: UUID = Field(..., description="Source agent identifier")
    target_agent_id: UUID = Field(..., description="Target agent identifier")
    memory_id: UUID = Field(..., description="Memory identifier")
    route_type: str = Field(..., description="Route type (direct, broadcast, selective)")
    priority: int = Field(default=1, ge=1, le=10, description="Route priority (1-10)")
    status: str = Field(default="pending", description="Route status")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Route metadata")
    
    @validator('route_type')
    def validate_route_type(cls, v):
        valid_types = ['direct', 'broadcast', 'selective', 'conditional']
        if v not in valid_types:
            raise ValueError(f'Route type must be one of: {", ".join(valid_types)}')
        return v
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['pending', 'active', 'completed', 'failed', 'cancelled']
        if v not in valid_statuses:
            raise ValueError(f'Status must be one of: {", ".join(valid_statuses)}')
        return v


class MemoryRouteCreate(BaseModel):
    """Model for creating a memory route."""
    
    source_agent_id: UUID = Field(..., description="Source agent identifier")
    target_agent_id: UUID = Field(..., description="Target agent identifier")
    memory_id: UUID = Field(..., description="Memory identifier")
    route_type: str = Field(default="direct", description="Route type")
    priority: int = Field(default=1, ge=1, le=10, description="Route priority")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Route metadata")
    
    @validator('route_type')
    def validate_route_type(cls, v):
        valid_types = ['direct', 'broadcast', 'selective', 'conditional']
        if v not in valid_types:
            raise ValueError(f'Route type must be one of: {", ".join(valid_types)}')
        return v


class MemoryRouteResponse(BaseModel):
    """Response model for memory route operations."""
    
    route: MemoryRoute = Field(..., description="Memory route")
    success: bool = Field(..., description="Operation success status")
    message: Optional[str] = Field(None, description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemoryStore(BaseModel, TimestampMixin, IDMixin):
    """Memory storage model."""
    
    content: MemoryContent = Field(..., description="Memory content")
    source: MemorySource = Field(..., description="Memory source")
    memory_type: str = Field(..., description="Memory type (conversation, knowledge, experience)")
    importance: int = Field(default=5, ge=1, le=10, description="Memory importance (1-10)")
    expiration: Optional[datetime] = Field(None, description="Memory expiration timestamp")
    access_control: Optional[Dict[str, List[UUID]]] = Field(default_factory=dict, description="Access control rules")
    
    @validator('memory_type')
    def validate_memory_type(cls, v):
        valid_types = ['conversation', 'knowledge', 'experience', 'fact', 'procedure']
        if v not in valid_types:
            raise ValueError(f'Memory type must be one of: {", ".join(valid_types)}')
        return v
    
    @validator('importance')
    def validate_importance(cls, v):
        if v < 1 or v > 10:
            raise ValueError('Importance must be between 1 and 10')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemoryStoreCreate(BaseModel):
    """Model for creating a memory store."""
    
    content: MemoryContent = Field(..., description="Memory content")
    source: MemorySource = Field(..., description="Memory source")
    memory_type: str = Field(..., description="Memory type")
    importance: int = Field(default=5, ge=1, le=10, description="Memory importance")
    expiration: Optional[datetime] = Field(None, description="Memory expiration timestamp")
    access_control: Optional[Dict[str, List[UUID]]] = Field(default_factory=dict, description="Access control rules")
    
    @validator('memory_type')
    def validate_memory_type(cls, v):
        valid_types = ['conversation', 'knowledge', 'experience', 'fact', 'procedure']
        if v not in valid_types:
            raise ValueError(f'Memory type must be one of: {", ".join(valid_types)}')
        return v


class MemoryStoreResponse(BaseModel):
    """Response model for memory store operations."""
    
    memory: MemoryStore = Field(..., description="Stored memory")
    success: bool = Field(..., description="Operation success status")
    message: Optional[str] = Field(None, description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemorySearch(BaseModel):
    """Memory search model."""
    
    query: str = Field(..., description="Search query")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Search filters")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Result offset")
    sort_by: str = Field(default="relevance", description="Sort field")
    sort_order: str = Field(default="desc", regex="^(asc|desc)$", description="Sort order")
    include_metadata: bool = Field(default=True, description="Include metadata in results")
    
    @validator('limit')
    def validate_limit(cls, v):
        if v < 1 or v > 100:
            raise ValueError('Limit must be between 1 and 100')
        return v
    
    @validator('sort_by')
    def validate_sort_by(cls, v):
        valid_fields = ['relevance', 'created_at', 'updated_at', 'importance', 'confidence']
        if v not in valid_fields:
            raise ValueError(f'Sort field must be one of: {", ".join(valid_fields)}')
        return v


class MemorySearchResult(BaseModel):
    """Individual memory search result."""
    
    memory_id: UUID = Field(..., description="Memory identifier")
    content: MemoryContent = Field(..., description="Memory content")
    source: MemorySource = Field(..., description="Memory source")
    memory_type: str = Field(..., description="Memory type")
    importance: int = Field(..., description="Memory importance")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Search relevance score")
    created_at: datetime = Field(..., description="Creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemorySearchResponse(BaseModel):
    """Response model for memory search operations."""
    
    results: List[MemorySearchResult] = Field(..., description="Search results")
    total: int = Field(..., description="Total matching results")
    query: str = Field(..., description="Original search query")
    filters: Dict[str, Any] = Field(..., description="Applied filters")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Search timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemoryStats(BaseModel):
    """Memory statistics model."""
    
    total_memories: int = Field(..., description="Total number of memories")
    memories_by_type: Dict[str, int] = Field(..., description="Memories grouped by type")
    memories_by_agent: Dict[UUID, int] = Field(..., description="Memories grouped by agent")
    average_importance: float = Field(..., description="Average memory importance")
    total_routes: int = Field(..., description="Total number of memory routes")
    routes_by_status: Dict[str, int] = Field(..., description="Routes grouped by status")
    storage_size_bytes: int = Field(..., description="Total storage size in bytes")
    last_updated: datetime = Field(..., description="Last statistics update")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemoryRouteStats(BaseModel):
    """Memory route statistics model."""
    
    total_routes: int = Field(..., description="Total number of routes")
    routes_by_type: Dict[str, int] = Field(..., description="Routes grouped by type")
    routes_by_status: Dict[str, int] = Field(..., description="Routes grouped by status")
    routes_by_priority: Dict[int, int] = Field(..., description="Routes grouped by priority")
    average_route_time_ms: float = Field(..., description="Average route processing time")
    successful_routes: int = Field(..., description="Number of successful routes")
    failed_routes: int = Field(..., description="Number of failed routes")
    last_updated: datetime = Field(..., description="Last statistics update")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemoryAccessControl(BaseModel):
    """Memory access control model."""
    
    memory_id: UUID = Field(..., description="Memory identifier")
    agent_permissions: Dict[UUID, List[str]] = Field(..., description="Agent permissions")
    group_permissions: Optional[Dict[str, List[str]]] = Field(default_factory=dict, description="Group permissions")
    public_read: bool = Field(default=False, description="Public read access")
    public_write: bool = Field(default=False, description="Public write access")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MemoryBatchOperation(BaseModel):
    """Batch memory operation model."""
    
    operation: str = Field(..., description="Operation type (store, update, delete)")
    memories: List[Union[MemoryStoreCreate, Dict[str, Any]]] = Field(..., description="Memories to operate on")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Operation options")
    
    @validator('operation')
    def validate_operation(cls, v):
        valid_operations = ['store', 'update', 'delete', 'route']
        if v not in valid_operations:
            raise ValueError(f'Operation must be one of: {", ".join(valid_operations)}')
        return v


class MemoryBatchResult(BaseModel):
    """Batch memory operation result model."""
    
    operation: str = Field(..., description="Operation type")
    total_memories: int = Field(..., description="Total memories processed")
    successful: int = Field(..., description="Successfully processed memories")
    failed: int = Field(..., description="Failed memories")
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Operation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

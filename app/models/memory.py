"""
Memory-related models for AI Agent Memory Router.

This module defines the data models for memory storage, routing,
and management operations.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from uuid import UUID
from pydantic import BaseModel, Field, validator

from .common import TimestampMixin, IDMixin


class MemoryItem(BaseModel, TimestampMixin, IDMixin):
    """Memory item model for storage."""
    
    agent_id: str = Field(..., description="Agent identifier")
    content: str = Field(..., description="Memory content")
    memory_type: str = Field(..., description="Memory type (knowledge, experience, context)")
    priority: int = Field(default=1, ge=1, le=4, description="Memory priority (1=low, 2=normal, 3=high, 4=critical)")
    expires_at: Optional[datetime] = Field(None, description="Memory expiration timestamp")
    
    @validator('memory_type')
    def validate_memory_type(cls, v):
        valid_types = ['knowledge', 'experience', 'context', 'fact', 'procedure']
        if v not in valid_types:
            raise ValueError(f'Memory type must be one of: {", ".join(valid_types)}')
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        if v < 1 or v > 4:
            raise ValueError('Priority must be between 1 and 4')
        return v


class MemoryMetadata(BaseModel, TimestampMixin, IDMixin):
    """Memory metadata model."""
    
    memory_id: str = Field(..., description="Memory identifier")
    tags: List[str] = Field(default_factory=list, description="Memory tags")
    source: Optional[str] = Field(None, description="Memory source")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    embedding_vector: Optional[str] = Field(None, description="Base64 encoded vector for semantic search")
    vector_dimension: Optional[int] = Field(None, description="Dimension of the embedding vector")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v


class MemoryContent(BaseModel):
    """Memory content model."""
    
    text: str = Field(..., description="Memory text content")
    language: str = Field(default="en", description="Content language code")
    format: str = Field(default="text", description="Content format (text, markdown, json)")
    encoding: str = Field(default="utf-8", description="Content encoding")
    
    @validator('format')
    def validate_format(cls, v):
        valid_formats = ['text', 'markdown', 'json', 'html', 'xml']
        if v not in valid_formats:
            raise ValueError(f'Format must be one of: {", ".join(valid_formats)}')
        return v


class MemorySource(BaseModel):
    """Memory source model."""
    
    type: str = Field(..., description="Source type (conversation, document, api, etc.)")
    identifier: str = Field(..., description="Source identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Source timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Source metadata")
    
    @validator('type')
    def validate_source_type(cls, v):
        valid_types = ['conversation', 'document', 'api', 'user_input', 'system', 'external']
        if v not in valid_types:
            raise ValueError(f'Source type must be one of: {", ".join(valid_types)}')
        return v


class MemoryRoute(BaseModel, TimestampMixin, IDMixin):
    """Memory route model."""
    
    source_agent_id: str = Field(..., description="Source agent identifier")
    target_agent_id: str = Field(..., description="Target agent identifier")
    memory_id: str = Field(..., description="Memory identifier")
    route_type: str = Field(..., description="Route type (direct, broadcast, selective)")
    priority: int = Field(default=1, ge=1, le=10, description="Route priority (1-10)")
    status: str = Field(default="pending", description="Route status")
    routing_reason: Optional[str] = Field(None, description="Reason for routing decision")
    delivered_at: Optional[datetime] = Field(None, description="Delivery timestamp")
    acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment timestamp")
    
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
    
    source_agent_id: str = Field(..., description="Source agent identifier")
    target_agent_id: str = Field(..., description="Target agent identifier")
    memory_id: str = Field(..., description="Memory identifier")
    route_type: str = Field(default="direct", description="Route type")
    priority: int = Field(default=1, ge=1, le=10, description="Route priority")
    routing_reason: Optional[str] = Field(None, description="Reason for routing decision")


class MemoryRouteResponse(BaseModel):
    """Response model for memory route operations."""
    
    route: MemoryRoute = Field(..., description="Memory route")
    success: bool = Field(..., description="Operation success status")
    message: Optional[str] = Field(None, description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class MemoryStore(BaseModel, TimestampMixin, IDMixin):
    """Memory storage model."""
    
    content: MemoryContent = Field(..., description="Memory content")
    source: MemorySource = Field(..., description="Memory source")
    memory_type: str = Field(..., description="Memory type (conversation, knowledge, experience)")
    importance: int = Field(default=5, ge=1, le=10, description="Memory importance (1-10)")
    expiration: Optional[datetime] = Field(None, description="Memory expiration timestamp")
    access_control: Optional[Dict[str, List[str]]] = Field(default_factory=dict, description="Access control rules")
    
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


class MemoryStoreCreate(BaseModel):
    """Model for creating a memory store."""
    
    content: MemoryContent = Field(..., description="Memory content")
    source: MemorySource = Field(..., description="Memory source")
    memory_type: str = Field(..., description="Memory type")
    importance: int = Field(default=5, ge=1, le=10, description="Memory importance")
    expiration: Optional[datetime] = Field(None, description="Memory expiration timestamp")


class MemoryStoreResponse(BaseModel):
    """Response model for memory store operations."""
    
    memory: MemoryStore = Field(..., description="Memory store")
    success: bool = Field(..., description="Operation success status")
    message: Optional[str] = Field(None, description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class MemorySearch(BaseModel):
    """Memory search parameters."""
    
    query: str = Field(..., description="Search query")
    agent_id: Optional[str] = Field(None, description="Filter by agent ID")
    memory_type: Optional[str] = Field(None, description="Filter by memory type")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: int = Field(default=10, ge=1, le=100, description="Result limit")
    offset: int = Field(default=0, ge=0, description="Result offset")
    sort_by: str = Field(default="relevance", description="Sort field")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$", description="Sort order")
    include_metadata: bool = Field(default=True, description="Include metadata in results")
    
    @validator('limit')
    def validate_limit(cls, v):
        if v < 1 or v > 100:
            raise ValueError('Limit must be between 1 and 100')
        return v
    
    @validator('offset')
    def validate_offset(cls, v):
        if v < 0:
            raise ValueError('Offset must be non-negative')
        return v


class MemorySearchResult(BaseModel):
    """Memory search result."""
    
    memory: MemoryItem = Field(..., description="Memory item")
    metadata: Optional[MemoryMetadata] = Field(None, description="Memory metadata")
    relevance_score: float = Field(..., description="Search relevance score")
    matched_fields: List[str] = Field(default_factory=list, description="Fields that matched the query")
    
    @validator('relevance_score')
    def validate_relevance_score(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Relevance score must be between 0.0 and 1.0')
        return v


class MemorySearchResponse(BaseModel):
    """Response model for memory search operations."""
    
    results: List[MemorySearchResult] = Field(..., description="Search results")
    total: int = Field(..., description="Total number of results")
    query: str = Field(..., description="Search query")
    execution_time: float = Field(..., description="Search execution time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Search timestamp")


class MemoryStats(BaseModel):
    """Memory statistics model."""
    
    total_memories: int = Field(..., description="Total number of memories")
    memories_by_type: Dict[str, int] = Field(default_factory=dict, description="Memory count by type")
    memories_by_agent: Dict[str, int] = Field(default_factory=dict, description="Memory count by agent")
    average_priority: float = Field(..., description="Average memory priority")
    oldest_memory: Optional[datetime] = Field(None, description="Oldest memory timestamp")
    newest_memory: Optional[datetime] = Field(None, description="Newest memory timestamp")
    total_size_bytes: int = Field(..., description="Total memory size in bytes")
    
    @validator('average_priority')
    def validate_average_priority(cls, v):
        if v < 1.0 or v > 4.0:
            raise ValueError('Average priority must be between 1.0 and 4.0')
        return v


class MemoryRouteStats(BaseModel):
    """Memory route statistics model."""
    
    total_routes: int = Field(..., description="Total number of routes")
    routes_by_status: Dict[str, int] = Field(default_factory=dict, description="Route count by status")
    routes_by_type: Dict[str, int] = Field(default_factory=dict, description="Route count by type")
    average_delivery_time: Optional[float] = Field(None, description="Average delivery time in seconds")
    success_rate: float = Field(..., description="Route success rate (0.0-1.0)")
    total_bandwidth_bytes: int = Field(..., description="Total bandwidth used in bytes")
    
    @validator('success_rate')
    def validate_success_rate(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Success rate must be between 0.0 and 1.0')
        return v


class MemoryAccessControl(BaseModel):
    """Memory access control model."""
    
    memory_id: str = Field(..., description="Memory identifier")
    agent_id: str = Field(..., description="Agent identifier")
    permission: str = Field(..., description="Permission level (read, write, delete, admin)")
    granted_at: datetime = Field(default_factory=datetime.utcnow, description="Permission granted timestamp")
    expires_at: Optional[datetime] = Field(None, description="Permission expiration timestamp")
    granted_by: Optional[str] = Field(None, description="Agent that granted the permission")
    
    @validator('permission')
    def validate_permission(cls, v):
        valid_permissions = ['read', 'write', 'delete', 'admin']
        if v not in valid_permissions:
            raise ValueError(f'Permission must be one of: {", ".join(valid_permissions)}')
        return v


class MemoryBatchOperation(BaseModel):
    """Memory batch operation model."""
    
    operation_type: str = Field(..., description="Operation type (create, update, delete)")
    memories: List[Union[MemoryItem, Dict[str, Any]]] = Field(..., description="List of memories or memory data")
    batch_size: int = Field(default=100, ge=1, le=1000, description="Batch size")
    continue_on_error: bool = Field(default=False, description="Continue processing on error")
    
    @validator('operation_type')
    def validate_operation_type(cls, v):
        valid_types = ['create', 'update', 'delete', 'search', 'export']
        if v not in valid_types:
            raise ValueError(f'Operation type must be one of: {", ".join(valid_types)}')
        return v


class MemoryBatchResult(BaseModel):
    """Memory batch operation result model."""
    
    operation_type: str = Field(..., description="Operation type")
    total_processed: int = Field(..., description="Total items processed")
    successful: int = Field(..., description="Successfully processed items")
    failed: int = Field(..., description="Failed items")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Error details")
    execution_time: float = Field(..., description="Total execution time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Operation timestamp")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_processed == 0:
            return 0.0
        return self.successful / self.total_processed

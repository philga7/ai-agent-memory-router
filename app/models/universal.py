"""
Universal Memory Access API models.

This module defines the simplified models for the universal memory access API
that any project can use to store and retrieve memories without needing to
understand the internal routing complexity.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, validator, HttpUrl
from enum import Enum

from .common import TimestampMixin, IDMixin, PaginationParams, PaginatedResponse


class MemoryType(str, Enum):
    """Memory type enumeration."""
    KNOWLEDGE = "knowledge"
    EXPERIENCE = "experience"
    CONTEXT = "context"
    FACT = "fact"
    PROCEDURE = "procedure"
    CONVERSATION = "conversation"


class Priority(str, Enum):
    """Priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ProjectStatus(str, Enum):
    """Project status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"


# Universal Memory Models

class UniversalMemoryStore(BaseModel):
    """Universal memory storage model - simplified interface for any project."""
    
    project_id: str = Field(..., min_length=1, max_length=100, description="Project identifier")
    content: str = Field(..., min_length=1, description="Memory content to store")
    memory_type: MemoryType = Field(default=MemoryType.KNOWLEDGE, description="Type of memory")
    priority: Priority = Field(default=Priority.NORMAL, description="Memory priority")
    tags: List[str] = Field(default_factory=list, description="Memory tags for organization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    expires_at: Optional[datetime] = Field(None, description="Memory expiration timestamp")
    
    @validator('project_id')
    def validate_project_id(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Project ID must contain only alphanumeric characters, hyphens, and underscores')
        return v.lower()
    
    @validator('tags')
    def validate_tags(cls, v):
        if len(v) > 20:
            raise ValueError('Maximum 20 tags allowed')
        for tag in v:
            if len(tag) > 50:
                raise ValueError('Tag length cannot exceed 50 characters')
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        if len(v) > 50:
            raise ValueError('Maximum 50 metadata keys allowed')
        return v


class UniversalMemoryResponse(BaseModel):
    """Universal memory storage response."""
    
    memory_id: str = Field(..., description="Unique memory identifier")
    project_id: str = Field(..., description="Project identifier")
    status: str = Field(..., description="Storage status")
    created_at: datetime = Field(..., description="Creation timestamp")
    storage_location: str = Field(..., description="Storage location (cipher, weaviate, local)")
    estimated_retrieval_time_ms: float = Field(..., description="Estimated retrieval time in milliseconds")


class UniversalMemoryRetrieve(BaseModel):
    """Universal memory retrieval request."""
    
    project_id: str = Field(..., description="Project identifier")
    memory_id: Optional[str] = Field(None, description="Specific memory ID to retrieve")
    memory_type: Optional[MemoryType] = Field(None, description="Filter by memory type")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of memories to retrieve")
    offset: int = Field(default=0, ge=0, description="Number of memories to skip")
    include_metadata: bool = Field(default=True, description="Include metadata in response")
    use_cache: bool = Field(default=True, description="Use cache for faster retrieval")


class UniversalMemoryItem(BaseModel):
    """Universal memory item response."""
    
    memory_id: str = Field(..., description="Memory identifier")
    project_id: str = Field(..., description="Project identifier")
    content: str = Field(..., description="Memory content")
    memory_type: MemoryType = Field(..., description="Memory type")
    priority: Priority = Field(..., description="Memory priority")
    tags: List[str] = Field(..., description="Memory tags")
    metadata: Dict[str, Any] = Field(..., description="Memory metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    storage_location: str = Field(..., description="Storage location")
    size_bytes: int = Field(..., description="Memory size in bytes")


class UniversalMemoryListResponse(BaseModel):
    """Universal memory list response."""
    
    memories: List[UniversalMemoryItem] = Field(..., description="List of memories")
    total: int = Field(..., description="Total number of memories")
    limit: int = Field(..., description="Number of memories returned")
    offset: int = Field(..., description="Number of memories skipped")
    has_more: bool = Field(..., description="Whether there are more memories")
    retrieval_time_ms: float = Field(..., description="Retrieval time in milliseconds")


class UniversalMemorySearch(BaseModel):
    """Universal memory search request."""
    
    project_id: str = Field(..., description="Project identifier")
    query: str = Field(..., min_length=1, description="Search query")
    memory_type: Optional[MemoryType] = Field(None, description="Filter by memory type")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    priority: Optional[Priority] = Field(None, description="Filter by priority")
    created_after: Optional[datetime] = Field(None, description="Filter memories created after this date")
    created_before: Optional[datetime] = Field(None, description="Filter memories created before this date")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")
    include_metadata: bool = Field(default=True, description="Include metadata in results")
    semantic_search: bool = Field(default=True, description="Use semantic similarity search")


class UniversalMemorySearchResult(BaseModel):
    """Universal memory search result."""
    
    memory: UniversalMemoryItem = Field(..., description="Memory item")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Search relevance score")
    matched_fields: List[str] = Field(default_factory=list, description="Fields that matched the query")
    search_method: str = Field(..., description="Search method used (semantic, keyword, hybrid)")


class UniversalMemorySearchResponse(BaseModel):
    """Universal memory search response."""
    
    query: str = Field(..., description="Search query")
    results: List[UniversalMemorySearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")
    search_method: str = Field(..., description="Primary search method used")


class UniversalMemoryUpdate(BaseModel):
    """Universal memory update request."""
    
    project_id: str = Field(..., description="Project identifier")
    memory_id: str = Field(..., description="Memory identifier")
    content: Optional[str] = Field(None, description="Updated memory content")
    memory_type: Optional[MemoryType] = Field(None, description="Updated memory type")
    priority: Optional[Priority] = Field(None, description="Updated priority")
    tags: Optional[List[str]] = Field(None, description="Updated tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
    expires_at: Optional[datetime] = Field(None, description="Updated expiration timestamp")


class UniversalMemoryDelete(BaseModel):
    """Universal memory deletion request."""
    
    project_id: str = Field(..., description="Project identifier")
    memory_id: str = Field(..., description="Memory identifier")
    permanent: bool = Field(default=False, description="Whether to permanently delete (vs soft delete)")


class UniversalMemoryBatchStore(BaseModel):
    """Universal memory batch storage request."""
    
    project_id: str = Field(..., description="Project identifier")
    memories: List[UniversalMemoryStore] = Field(..., min_items=1, max_items=100, description="List of memories to store")
    batch_size: int = Field(default=10, ge=1, le=50, description="Batch processing size")
    continue_on_error: bool = Field(default=False, description="Continue processing if individual memories fail")


class UniversalMemoryBatchResult(BaseModel):
    """Universal memory batch operation result."""
    
    operation_type: str = Field(..., description="Operation type")
    total_processed: int = Field(..., description="Total items processed")
    successful: int = Field(..., description="Successfully processed items")
    failed: int = Field(..., description="Failed items")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Error details")
    execution_time_ms: float = Field(..., description="Total execution time in milliseconds")
    memory_ids: List[str] = Field(default_factory=list, description="IDs of successfully processed memories")


# Project Management Models

class UniversalProjectCreate(BaseModel):
    """Universal project creation request."""
    
    project_id: str = Field(..., min_length=1, max_length=100, description="Project identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Project name")
    description: Optional[str] = Field(None, max_length=1000, description="Project description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Project metadata")
    storage_preference: str = Field(default="auto", description="Storage preference (auto, cipher, weaviate, local)")
    retention_days: Optional[int] = Field(None, ge=1, le=3650, description="Memory retention period in days")
    
    @validator('project_id')
    def validate_project_id(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Project ID must contain only alphanumeric characters, hyphens, and underscores')
        return v.lower()
    
    @validator('storage_preference')
    def validate_storage_preference(cls, v):
        valid_preferences = ['auto', 'cipher', 'weaviate', 'local']
        if v not in valid_preferences:
            raise ValueError(f'Storage preference must be one of: {", ".join(valid_preferences)}')
        return v


class UniversalProjectResponse(BaseModel):
    """Universal project response."""
    
    project_id: str = Field(..., description="Project identifier")
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    status: ProjectStatus = Field(..., description="Project status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    metadata: Dict[str, Any] = Field(..., description="Project metadata")
    storage_preference: str = Field(..., description="Storage preference")
    retention_days: Optional[int] = Field(None, description="Memory retention period")
    memory_count: int = Field(..., description="Total number of memories")
    total_size_bytes: int = Field(..., description="Total memory size in bytes")


class UniversalProjectList(BaseModel):
    """Universal project list request."""
    
    status: Optional[ProjectStatus] = Field(None, description="Filter by project status")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of projects")
    offset: int = Field(default=0, ge=0, description="Number of projects to skip")
    include_stats: bool = Field(default=True, description="Include project statistics")


class UniversalProjectListResponse(BaseModel):
    """Universal project list response."""
    
    projects: List[UniversalProjectResponse] = Field(..., description="List of projects")
    total: int = Field(..., description="Total number of projects")
    limit: int = Field(..., description="Number of projects returned")
    offset: int = Field(..., description="Number of projects skipped")
    has_more: bool = Field(..., description="Whether there are more projects")


# Statistics and Analytics Models

class UniversalMemoryStats(BaseModel):
    """Universal memory statistics."""
    
    project_id: str = Field(..., description="Project identifier")
    total_memories: int = Field(..., description="Total number of memories")
    memories_by_type: Dict[str, int] = Field(..., description="Memory count by type")
    memories_by_priority: Dict[str, int] = Field(..., description="Memory count by priority")
    total_size_bytes: int = Field(..., description="Total memory size in bytes")
    average_memory_size_bytes: float = Field(..., description="Average memory size")
    oldest_memory: Optional[datetime] = Field(None, description="Oldest memory timestamp")
    newest_memory: Optional[datetime] = Field(None, description="Newest memory timestamp")
    storage_distribution: Dict[str, int] = Field(..., description="Memory distribution by storage location")


class UniversalProjectStats(BaseModel):
    """Universal project statistics."""
    
    total_projects: int = Field(..., description="Total number of projects")
    active_projects: int = Field(..., description="Number of active projects")
    total_memories: int = Field(..., description="Total number of memories across all projects")
    total_size_bytes: int = Field(..., description="Total memory size across all projects")
    average_memories_per_project: float = Field(..., description="Average memories per project")
    top_projects_by_memory: List[Dict[str, Any]] = Field(..., description="Top projects by memory count")


# Error and Status Models

class UniversalAPIError(BaseModel):
    """Universal API error response."""
    
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")


class UniversalAPIStatus(BaseModel):
    """Universal API status response."""
    
    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Status timestamp")
    components: Dict[str, str] = Field(..., description="Component statuses")
    uptime_seconds: float = Field(..., description="API uptime in seconds")


# Rate Limiting Models

class UniversalRateLimit(BaseModel):
    """Universal rate limit information."""
    
    project_id: str = Field(..., description="Project identifier")
    requests_per_minute: int = Field(..., description="Allowed requests per minute")
    requests_per_hour: int = Field(..., description="Allowed requests per hour")
    requests_per_day: int = Field(..., description="Allowed requests per day")
    current_usage: Dict[str, int] = Field(..., description="Current usage statistics")
    reset_times: Dict[str, datetime] = Field(..., description="Rate limit reset times")


class UniversalQuota(BaseModel):
    """Universal quota information."""
    
    project_id: str = Field(..., description="Project identifier")
    max_memories: Optional[int] = Field(None, description="Maximum number of memories")
    max_storage_bytes: Optional[int] = Field(None, description="Maximum storage in bytes")
    current_memories: int = Field(..., description="Current number of memories")
    current_storage_bytes: int = Field(..., description="Current storage usage in bytes")
    usage_percentage: float = Field(..., description="Usage percentage (0-100)")

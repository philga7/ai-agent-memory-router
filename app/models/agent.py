"""
Agent models for AI Agent Memory Router.

This module defines Pydantic models for AI agent management
including registration, capabilities, and status tracking.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator, HttpUrl
from uuid import UUID

from .common import TimestampMixin, IDMixin, PaginationParams, PaginatedResponse


class AgentCapabilities(BaseModel):
    """AI agent capabilities model."""
    
    memory_read: bool = Field(default=True, description="Can read memories")
    memory_write: bool = Field(default=True, description="Can write memories")
    memory_search: bool = Field(default=True, description="Can search memories")
    context_access: bool = Field(default=True, description="Can access conversation context")
    routing: bool = Field(default=False, description="Can create memory routes")
    admin: bool = Field(default=False, description="Has administrative privileges")
    custom_tools: Optional[List[str]] = Field(default_factory=list, description="Custom tool capabilities")
    
    @validator('custom_tools')
    def validate_custom_tools(cls, v):
        if v is not None:
            # Ensure unique tool names
            unique_tools = list(set(v))
            if len(unique_tools) != len(v):
                raise ValueError('Custom tools must have unique names')
        return v


class AgentStatus(BaseModel):
    """AI agent status model."""
    
    status: str = Field(..., description="Agent status")
    last_heartbeat: datetime = Field(..., description="Last heartbeat timestamp")
    uptime: float = Field(..., description="Agent uptime in seconds")
    memory_count: int = Field(default=0, description="Number of memories owned")
    route_count: int = Field(default=0, description="Number of active routes")
    error_count: int = Field(default=0, description="Number of errors encountered")
    performance_metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Performance metrics")
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['online', 'offline', 'busy', 'error', 'maintenance']
        if v not in valid_statuses:
            raise ValueError(f'Status must be one of: {", ".join(valid_statuses)}')
        return v
    
    @validator('uptime')
    def validate_uptime(cls, v):
        if v < 0:
            raise ValueError('Uptime must be non-negative')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Agent(BaseModel, TimestampMixin, IDMixin):
    """AI agent model."""
    
    name: str = Field(..., description="Agent name")
    description: Optional[str] = Field(None, description="Agent description")
    agent_type: str = Field(..., description="Agent type (assistant, tool, router, monitor)")
    version: str = Field(..., description="Agent version")
    endpoint_url: Optional[HttpUrl] = Field(None, description="Agent endpoint URL")
    capabilities: AgentCapabilities = Field(..., description="Agent capabilities")
    status: AgentStatus = Field(..., description="Agent status")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    tags: Optional[List[str]] = Field(default_factory=list, description="Agent tags")
    access_token: Optional[str] = Field(None, description="Access token for authentication")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Name cannot be empty')
        if len(v) > 100:
            raise ValueError('Name cannot exceed 100 characters')
        return v.strip()
    
    @validator('agent_type')
    def validate_agent_type(cls, v):
        valid_types = ['assistant', 'tool', 'router', 'monitor', 'gateway', 'specialist']
        if v not in valid_types:
            raise ValueError(f'Agent type must be one of: {", ".join(valid_types)}')
        return v
    
    @validator('version')
    def validate_version(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Version cannot be empty')
        return v.strip()
    
    @validator('tags')
    def validate_tags(cls, v):
        if v is not None:
            # Ensure unique tags
            unique_tags = list(set(v))
            if len(unique_tags) != len(v):
                raise ValueError('Tags must be unique')
            # Validate tag length
            for tag in v:
                if len(tag) > 50:
                    raise ValueError('Tag length cannot exceed 50 characters')
        return v


class AgentContext(BaseModel):
    """Agent context model for storing conversation and session state."""
    
    agent_id: str = Field(..., description="Agent identifier")
    context_type: str = Field(..., description="Context type (session, project, user)")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Context data")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    project_id: Optional[str] = Field(None, description="Project identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Context creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Context last update timestamp")
    expires_at: Optional[datetime] = Field(None, description="Context expiration timestamp")
    
    @validator('context_type')
    def validate_context_type(cls, v):
        valid_types = ['session', 'project', 'user', 'conversation', 'task']
        if v not in valid_types:
            raise ValueError(f'Context type must be one of: {", ".join(valid_types)}')
        return v


class AgentCreate(BaseModel):
    """Model for creating an AI agent."""
    
    name: str = Field(..., description="Agent name")
    description: Optional[str] = Field(None, description="Agent description")
    agent_type: str = Field(..., description="Agent type")
    version: str = Field(..., description="Agent version")
    endpoint_url: Optional[HttpUrl] = Field(None, description="Agent endpoint URL")
    capabilities: AgentCapabilities = Field(..., description="Agent capabilities")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    tags: Optional[List[str]] = Field(default_factory=list, description="Agent tags")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Name cannot be empty')
        if len(v) > 100:
            raise ValueError('Name cannot exceed 100 characters')
        return v.strip()
    
    @validator('agent_type')
    def validate_agent_type(cls, v):
        valid_types = ['assistant', 'tool', 'router', 'monitor', 'gateway', 'specialist']
        if v not in valid_types:
            raise ValueError(f'Agent type must be one of: {", ".join(valid_types)}')
        return v


class AgentUpdate(BaseModel):
    """Model for updating an AI agent."""
    
    name: Optional[str] = Field(None, description="Agent name")
    description: Optional[str] = Field(None, description="Agent description")
    agent_type: Optional[str] = Field(None, description="Agent type")
    version: Optional[str] = Field(None, description="Agent version")
    endpoint_url: Optional[HttpUrl] = Field(None, description="Agent endpoint URL")
    capabilities: Optional[AgentCapabilities] = Field(None, description="Agent capabilities")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    tags: Optional[List[str]] = Field(None, description="Agent tags")
    
    @validator('name')
    def validate_name(cls, v):
        if v is not None:
            if len(v.strip()) == 0:
                raise ValueError('Name cannot be empty')
            if len(v) > 100:
                raise ValueError('Name cannot exceed 100 characters')
            return v.strip()
        return v
    
    @validator('agent_type')
    def validate_agent_type(cls, v):
        if v is not None:
            valid_types = ['assistant', 'tool', 'router', 'monitor', 'gateway', 'specialist']
            if v not in valid_types:
                raise ValueError(f'Agent type must be one of: {", ".join(valid_types)}')
        return v


class AgentResponse(BaseModel):
    """Response model for agent operations."""
    
    agent: Agent = Field(..., description="Agent information")
    success: bool = Field(..., description="Operation success status")
    message: Optional[str] = Field(None, description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentHeartbeat(BaseModel):
    """Agent heartbeat model."""
    
    agent_id: UUID = Field(..., description="Agent identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Heartbeat timestamp")
    status: str = Field(..., description="Current agent status")
    uptime: float = Field(..., description="Agent uptime in seconds")
    memory_count: int = Field(default=0, description="Current memory count")
    route_count: int = Field(default=0, description="Current route count")
    error_count: int = Field(default=0, description="Current error count")
    performance_metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Performance metrics")
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['online', 'offline', 'busy', 'error', 'maintenance']
        if v not in valid_statuses:
            raise ValueError(f'Status must be one of: {", ".join(valid_statuses)}')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentDeregister(BaseModel):
    """Agent deregistration model."""
    
    agent_id: UUID = Field(..., description="Agent identifier")
    reason: Optional[str] = Field(None, description="Deregistration reason")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Deregistration timestamp")
    cleanup_options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Cleanup options")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentSearch(BaseModel):
    """Agent search model."""
    
    query: Optional[str] = Field(None, description="Search query")
    agent_type: Optional[str] = Field(None, description="Filter by agent type")
    status: Optional[str] = Field(None, description="Filter by status")
    capabilities: Optional[List[str]] = Field(default_factory=list, description="Filter by capabilities")
    tags: Optional[List[str]] = Field(default_factory=list, description="Filter by tags")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Result offset")
    sort_by: str = Field(default="name", description="Sort field")
    sort_order: str = Field(default="asc", pattern="^(asc|desc)$", description="Sort order")
    
    @validator('agent_type')
    def validate_agent_type(cls, v):
        if v is not None:
            valid_types = ['assistant', 'tool', 'router', 'monitor', 'gateway', 'specialist']
            if v not in valid_types:
                raise ValueError(f'Agent type must be one of: {", ".join(valid_types)}')
        return v
    
    @validator('status')
    def validate_status(cls, v):
        if v is not None:
            valid_statuses = ['online', 'offline', 'busy', 'error', 'maintenance']
            if v not in valid_statuses:
                raise ValueError(f'Status must be one of: {", ".join(valid_statuses)}')
        return v
    
    @validator('sort_by')
    def validate_sort_by(cls, v):
        valid_fields = ['name', 'agent_type', 'status', 'created_at', 'last_heartbeat']
        if v not in valid_fields:
            raise ValueError(f'Sort field must be one of: {", ".join(valid_fields)}')
        return v


class AgentSearchResult(BaseModel):
    """Individual agent search result."""
    
    agent_id: UUID = Field(..., description="Agent identifier")
    name: str = Field(..., description="Agent name")
    description: Optional[str] = Field(None, description="Agent description")
    agent_type: str = Field(..., description="Agent type")
    version: str = Field(..., description="Agent version")
    status: AgentStatus = Field(..., description="Agent status")
    capabilities: AgentCapabilities = Field(..., description="Agent capabilities")
    tags: List[str] = Field(..., description="Agent tags")
    created_at: datetime = Field(..., description="Creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentSearchResponse(BaseModel):
    """Response model for agent search operations."""
    
    results: List[AgentSearchResult] = Field(..., description="Search results")
    total: int = Field(..., description="Total matching results")
    query: Optional[str] = Field(None, description="Original search query")
    filters: Dict[str, Any] = Field(..., description="Applied filters")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Search timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentStats(BaseModel):
    """Agent statistics model."""
    
    total_agents: int = Field(..., description="Total number of agents")
    agents_by_type: Dict[str, int] = Field(..., description="Agents grouped by type")
    agents_by_status: Dict[str, int] = Field(..., description="Agents grouped by status")
    online_agents: int = Field(..., description="Number of online agents")
    offline_agents: int = Field(..., description="Number of offline agents")
    total_memories: int = Field(..., description="Total memories across all agents")
    total_routes: int = Field(..., description="Total routes across all agents")
    average_uptime: float = Field(..., description="Average agent uptime")
    last_updated: datetime = Field(..., description="Last statistics update")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentGroup(BaseModel):
    """Agent group model."""
    
    name: str = Field(..., description="Group name")
    description: Optional[str] = Field(None, description="Group description")
    agent_ids: List[UUID] = Field(default_factory=list, description="Agent identifiers in group")
    capabilities: AgentCapabilities = Field(..., description="Group capabilities")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Name cannot be empty')
        if len(v) > 100:
            raise ValueError('Name cannot exceed 100 characters')
        return v.strip()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentBatchOperation(BaseModel):
    """Batch agent operation model."""
    
    operation: str = Field(..., description="Operation type (update, deregister, status_update)")
    agents: List[Union[AgentUpdate, AgentDeregister, AgentHeartbeat]] = Field(..., description="Agents to operate on")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Operation options")
    
    @validator('operation')
    def validate_operation(cls, v):
        valid_operations = ['update', 'deregister', 'status_update', 'capability_update']
        if v not in valid_operations:
            raise ValueError(f'Operation must be one of: {", ".join(valid_operations)}')
        return v


class AgentBatchResult(BaseModel):
    """Batch agent operation result model."""
    
    operation: str = Field(..., description="Operation type")
    total_agents: int = Field(..., description="Total agents processed")
    successful: int = Field(..., description="Successfully processed agents")
    failed: int = Field(..., description="Failed agents")
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Operation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

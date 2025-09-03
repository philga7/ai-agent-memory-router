"""
Context models for AI Agent Memory Router.

This module defines Pydantic models for conversation context management
including context updates, retrieval, and search functionality.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from uuid import UUID

from .common import TimestampMixin, IDMixin, PaginationParams, PaginatedResponse


class ContextParticipant(BaseModel):
    """Conversation participant model."""
    
    agent_id: UUID = Field(..., description="Participant agent identifier")
    role: str = Field(..., description="Participant role (user, assistant, system, observer)")
    name: Optional[str] = Field(None, description="Participant name")
    capabilities: Optional[List[str]] = Field(default_factory=list, description="Participant capabilities")
    joined_at: datetime = Field(default_factory=datetime.utcnow, description="Join timestamp")
    last_activity: datetime = Field(default_factory=datetime.utcnow, description="Last activity timestamp")
    
    @validator('role')
    def validate_role(cls, v):
        valid_roles = ['user', 'assistant', 'system', 'observer', 'moderator']
        if v not in valid_roles:
            raise ValueError(f'Role must be one of: {", ".join(valid_roles)}')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ContextMessage(BaseModel):
    """Context message model."""
    
    message_id: UUID = Field(..., description="Message identifier")
    sender_id: UUID = Field(..., description="Sender agent identifier")
    content: str = Field(..., description="Message content")
    message_type: str = Field(default="text", description="Message type")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Message metadata")
    parent_message_id: Optional[UUID] = Field(None, description="Parent message identifier")
    reactions: Optional[Dict[str, List[UUID]]] = Field(default_factory=dict, description="Message reactions")
    
    @validator('message_type')
    def validate_message_type(cls, v):
        valid_types = ['text', 'image', 'file', 'command', 'system', 'error']
        if v not in valid_types:
            raise ValueError(f'Message type must be one of: {", ".join(valid_types)}')
        return v
    
    @validator('content')
    def validate_content(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Message content cannot be empty')
        if len(v) > 10000:
            raise ValueError('Message content cannot exceed 10000 characters')
        return v.strip()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationContext(BaseModel, TimestampMixin, IDMixin):
    """Conversation context model."""
    
    title: Optional[str] = Field(None, description="Conversation title")
    description: Optional[str] = Field(None, description="Conversation description")
    participants: List[ContextParticipant] = Field(default_factory=list, description="Conversation participants")
    messages: List[ContextMessage] = Field(default_factory=list, description="Conversation messages")
    context_type: str = Field(default="conversation", description="Context type")
    status: str = Field(default="active", description="Context status")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    tags: Optional[List[str]] = Field(default_factory=list, description="Context tags")
    max_participants: int = Field(default=10, ge=2, le=100, description="Maximum participants")
    max_messages: int = Field(default=1000, ge=1, le=10000, description="Maximum messages")
    
    @validator('context_type')
    def validate_context_type(cls, v):
        valid_types = ['conversation', 'task', 'project', 'meeting', 'support', 'training']
        if v not in valid_types:
            raise ValueError(f'Context type must be one of: {", ".join(valid_types)}')
        return v
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['active', 'paused', 'completed', 'archived', 'deleted']
        if v not in valid_statuses:
            raise ValueError(f'Status must be one of: {", ".join(valid_statuses)}')
        return v
    
    @validator('participants')
    def validate_participants(cls, v):
        if len(v) > 100:
            raise ValueError('Cannot exceed 100 participants')
        # Ensure unique agent IDs
        agent_ids = [p.agent_id for p in v]
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError('Duplicate participant agent IDs not allowed')
        return v
    
    @validator('messages')
    def validate_messages(cls, v):
        if len(v) > 10000:
            raise ValueError('Cannot exceed 10000 messages')
        return v
    
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


class ContextUpdate(BaseModel):
    """Context update model."""
    
    context_id: UUID = Field(..., description="Context identifier")
    update_type: str = Field(..., description="Update type")
    data: Dict[str, Any] = Field(..., description="Update data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Update timestamp")
    agent_id: Optional[UUID] = Field(None, description="Updating agent identifier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('update_type')
    def validate_update_type(cls, v):
        valid_types = ['message_add', 'participant_add', 'participant_remove', 'status_change', 'metadata_update']
        if v not in valid_types:
            raise ValueError(f'Update type must be one of: {", ".join(valid_types)}')
        return v
    
    @validator('data')
    def validate_data(cls, v):
        if not v:
            raise ValueError('Update data cannot be empty')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ContextResponse(BaseModel):
    """Response model for context operations."""
    
    context: ConversationContext = Field(..., description="Conversation context")
    success: bool = Field(..., description="Operation success status")
    message: Optional[str] = Field(None, description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ContextSearch(BaseModel):
    """Context search model."""
    
    query: Optional[str] = Field(None, description="Search query")
    context_type: Optional[str] = Field(None, description="Filter by context type")
    status: Optional[str] = Field(None, description="Filter by status")
    participant_ids: Optional[List[UUID]] = Field(default_factory=list, description="Filter by participant IDs")
    tags: Optional[List[str]] = Field(default_factory=list, description="Filter by tags")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="Date range filter")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Result offset")
    sort_by: str = Field(default="updated_at", description="Sort field")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$", description="Sort order")
    
    @validator('context_type')
    def validate_context_type(cls, v):
        if v is not None:
            valid_types = ['conversation', 'task', 'project', 'meeting', 'support', 'training']
            if v not in valid_types:
                raise ValueError(f'Context type must be one of: {", ".join(valid_types)}')
        return v
    
    @validator('status')
    def validate_status(cls, v):
        if v is not None:
            valid_statuses = ['active', 'paused', 'completed', 'archived', 'deleted']
            if v not in valid_statuses:
                raise ValueError(f'Status must be one of: {", ".join(valid_statuses)}')
        return v
    
    @validator('sort_by')
    def validate_sort_by(cls, v):
        valid_fields = ['created_at', 'updated_at', 'title', 'participant_count', 'message_count']
        if v not in valid_fields:
            raise ValueError(f'Sort field must be one of: {", ".join(valid_fields)}')
        return v
    
    @validator('date_range')
    def validate_date_range(cls, v):
        if v is not None:
            if 'start' not in v or 'end' not in v:
                raise ValueError('Date range must include start and end dates')
            if v['start'] >= v['end']:
                raise ValueError('Start date must be before end date')
        return v


class ContextSearchResult(BaseModel):
    """Individual context search result."""
    
    context_id: UUID = Field(..., description="Context identifier")
    title: Optional[str] = Field(None, description="Context title")
    description: Optional[str] = Field(None, description="Context description")
    context_type: str = Field(..., description="Context type")
    status: str = Field(..., description="Context status")
    participant_count: int = Field(..., description="Number of participants")
    message_count: int = Field(..., description="Number of messages")
    tags: List[str] = Field(..., description="Context tags")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Search relevance score")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ContextSearchResponse(BaseModel):
    """Response model for context search operations."""
    
    results: List[ContextSearchResult] = Field(..., description="Search results")
    total: int = Field(..., description="Total matching results")
    query: Optional[str] = Field(None, description="Original search query")
    filters: Dict[str, Any] = Field(..., description="Applied filters")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Search timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ContextStats(BaseModel):
    """Context statistics model."""
    
    total_contexts: int = Field(..., description="Total number of contexts")
    contexts_by_type: Dict[str, int] = Field(..., description="Contexts grouped by type")
    contexts_by_status: Dict[str, int] = Field(..., description="Contexts grouped by status")
    active_contexts: int = Field(..., description="Number of active contexts")
    total_participants: int = Field(..., description="Total participants across all contexts")
    total_messages: int = Field(..., description="Total messages across all contexts")
    average_participants: float = Field(..., description="Average participants per context")
    average_messages: float = Field(..., description="Average messages per context")
    contexts_by_date: Dict[str, int] = Field(..., description="Contexts created by date")
    last_updated: datetime = Field(..., description="Last statistics update")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ContextParticipantStats(BaseModel):
    """Context participant statistics model."""
    
    context_id: UUID = Field(..., description="Context identifier")
    total_participants: int = Field(..., description="Total participants")
    participants_by_role: Dict[str, int] = Field(..., description="Participants grouped by role")
    active_participants: int = Field(..., description="Active participants")
    participant_activity: Dict[UUID, Dict[str, Any]] = Field(..., description="Participant activity data")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ContextMessageStats(BaseModel):
    """Context message statistics model."""
    
    context_id: UUID = Field(..., description="Context identifier")
    total_messages: int = Field(..., description="Total messages")
    messages_by_type: Dict[str, int] = Field(..., description="Messages grouped by type")
    messages_by_sender: Dict[UUID, int] = Field(..., description="Messages grouped by sender")
    message_timeline: Dict[str, int] = Field(..., description="Messages by time period")
    average_message_length: float = Field(..., description="Average message length")
    last_message: datetime = Field(..., description="Last message timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ContextAccessControl(BaseModel):
    """Context access control model."""
    
    context_id: UUID = Field(..., description="Context identifier")
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


class ContextBatchOperation(BaseModel):
    """Batch context operation model."""
    
    operation: str = Field(..., description="Operation type (update, archive, delete)")
    contexts: List[Union[ContextUpdate, Dict[str, Any]]] = Field(..., description="Contexts to operate on")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Operation options")
    
    @validator('operation')
    def validate_operation(cls, v):
        valid_operations = ['update', 'archive', 'delete', 'status_change', 'participant_update']
        if v not in valid_operations:
            raise ValueError(f'Operation must be one of: {", ".join(valid_operations)}')
        return v


class ContextBatchResult(BaseModel):
    """Batch context operation result model."""
    
    operation: str = Field(..., description="Operation type")
    total_contexts: int = Field(..., description="Total contexts processed")
    successful: int = Field(..., description="Successfully processed contexts")
    failed: int = Field(..., description="Failed contexts")
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Operation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SimpleContext(BaseModel, TimestampMixin, IDMixin):
    """Simple context model for storing basic conversation context data."""
    
    conversation_id: str = Field(..., description="Conversation identifier")
    agent_id: str = Field(..., description="Agent identifier")
    context_data: Dict[str, Any] = Field(..., description="Context data as JSON")
    context_type: str = Field(default="conversation", description="Context type")
    expires_at: Optional[datetime] = Field(None, description="When context expires")
    
    @validator('context_type')
    def validate_context_type(cls, v):
        valid_types = ['conversation', 'task', 'project', 'meeting', 'support', 'training']
        if v not in valid_types:
            raise ValueError(f'Context type must be one of: {", ".join(valid_types)}')
        return v

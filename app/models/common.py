"""
Common models and base classes for AI Agent Memory Router.

This module provides base classes and common models used across
all other model modules.
"""

from datetime import datetime
from typing import Generic, TypeVar, Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4


# Generic type for pagination
T = TypeVar('T')


class TimestampMixin(BaseModel):
    """Mixin to add timestamp fields to models."""
    
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class IDMixin(BaseModel):
    """Mixin to add ID field to models."""
    
    id: UUID = Field(default_factory=uuid4, description="Unique identifier")


class PaginationParams(BaseModel):
    """Parameters for paginated requests."""
    
    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    size: int = Field(default=20, ge=1, le=100, description="Page size (max 100)")
    sort_by: Optional[str] = Field(default="created_at", description="Sort field")
    sort_order: str = Field(default="desc", regex="^(asc|desc)$", description="Sort order")
    
    @validator('page')
    def validate_page(cls, v):
        if v < 1:
            raise ValueError('Page must be at least 1')
        return v
    
    @validator('size')
    def validate_size(cls, v):
        if v < 1:
            raise ValueError('Size must be at least 1')
        if v > 100:
            raise ValueError('Size cannot exceed 100')
        return v


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response model."""
    
    items: List[T] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Page size")
    pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")
    
    @validator('pages')
    def calculate_pages(cls, v, values):
        if 'total' in values and 'size' in values:
            return (values['total'] + values['size'] - 1) // values['size']
        return v
    
    @validator('has_next')
    def calculate_has_next(cls, v, values):
        if 'page' in values and 'pages' in values:
            return values['page'] < values['pages']
        return v
    
    @validator('has_prev')
    def calculate_has_prev(cls, v, values):
        if 'page' in values:
            return values['page'] > 1
        return v


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SuccessResponse(BaseModel):
    """Standard success response model."""
    
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthCheck(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    version: str = Field(..., description="Application version")
    uptime: float = Field(..., description="Application uptime in seconds")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ComponentStatus(BaseModel):
    """Component status model."""
    
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Component status")
    last_check: datetime = Field(default_factory=datetime.utcnow, description="Last check timestamp")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SystemInfo(BaseModel):
    """System information model."""
    
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment (dev, staging, prod)")
    uptime: float = Field(..., description="System uptime in seconds")
    memory_usage: Optional[str] = Field(None, description="Memory usage")
    cpu_usage: Optional[str] = Field(None, description="CPU usage")
    active_connections: Optional[int] = Field(None, description="Active connections")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MetricsData(BaseModel):
    """Metrics data model."""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")
    metrics: Dict[str, Any] = Field(..., description="Metrics data")
    performance: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SearchQuery(BaseModel):
    """Base search query model."""
    
    query: str = Field(..., description="Search query string")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Search filters")
    limit: Optional[int] = Field(default=20, ge=1, le=100, description="Maximum results")
    offset: Optional[int] = Field(default=0, ge=0, description="Result offset")
    
    @validator('limit')
    def validate_limit(cls, v):
        if v is not None and (v < 1 or v > 100):
            raise ValueError('Limit must be between 1 and 100')
        return v
    
    @validator('offset')
    def validate_offset(cls, v):
        if v is not None and v < 0:
            raise ValueError('Offset must be non-negative')
        return v


class FilterCondition(BaseModel):
    """Filter condition model."""
    
    field: str = Field(..., description="Field to filter on")
    operator: str = Field(..., description="Filter operator (eq, ne, gt, lt, gte, lte, in, nin, contains)")
    value: Any = Field(..., description="Filter value")
    
    @validator('operator')
    def validate_operator(cls, v):
        valid_operators = ['eq', 'ne', 'gt', 'lt', 'gte', 'lte', 'in', 'nin', 'contains']
        if v not in valid_operators:
            raise ValueError(f'Operator must be one of: {", ".join(valid_operators)}')
        return v


class SortOption(BaseModel):
    """Sort option model."""
    
    field: str = Field(..., description="Field to sort by")
    order: str = Field(default="asc", regex="^(asc|desc)$", description="Sort order")
    
    @validator('order')
    def validate_order(cls, v):
        if v not in ['asc', 'desc']:
            raise ValueError('Order must be either "asc" or "desc"')
        return v


class BulkOperation(BaseModel):
    """Bulk operation model."""
    
    operation: str = Field(..., description="Operation type (create, update, delete)")
    items: List[Dict[str, Any]] = Field(..., description="Items to operate on")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Operation options")
    
    @validator('operation')
    def validate_operation(cls, v):
        valid_operations = ['create', 'update', 'delete']
        if v not in valid_operations:
            raise ValueError(f'Operation must be one of: {", ".join(valid_operations)}')
        return v


class BulkResult(BaseModel):
    """Bulk operation result model."""
    
    operation: str = Field(..., description="Operation type")
    total_items: int = Field(..., description="Total items processed")
    successful: int = Field(..., description="Successfully processed items")
    failed: int = Field(..., description="Failed items")
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Operation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

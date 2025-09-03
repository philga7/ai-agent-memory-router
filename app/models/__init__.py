"""
Models package for AI Agent Memory Router.

This package contains all Pydantic models used for data validation,
API requests/responses, and internal data structures.
"""

from .memory import (
    MemoryRoute,
    MemoryRouteCreate,
    MemoryRouteResponse,
    MemorySearch,
    MemorySearchResponse,
    MemoryStore,
    MemoryStoreResponse,
    MemoryStats
)

from .agent import (
    Agent,
    AgentCreate,
    AgentUpdate,
    AgentResponse,
    AgentCapabilities,
    AgentStatus,
    AgentStats
)

from .context import (
    ConversationContext,
    ContextUpdate,
    ContextResponse,
    ContextSearch,
    ContextSearchResponse,
    ContextStats
)

from .common import (
    BaseModel,
    TimestampMixin,
    IDMixin,
    PaginationParams,
    PaginatedResponse,
    ErrorResponse,
    SuccessResponse
)

__all__ = [
    # Memory models
    "MemoryRoute",
    "MemoryRouteCreate", 
    "MemoryRouteResponse",
    "MemorySearch",
    "MemorySearchResponse",
    "MemoryStore",
    "MemoryStoreResponse",
    "MemoryStats",
    
    # Agent models
    "Agent",
    "AgentCreate",
    "AgentUpdate", 
    "AgentResponse",
    "AgentCapabilities",
    "AgentStatus",
    "AgentStats",
    
    # Context models
    "ConversationContext",
    "ContextUpdate",
    "ContextResponse", 
    "ContextSearch",
    "ContextSearchResponse",
    "ContextStats",
    
    # Common models
    "BaseModel",
    "TimestampMixin",
    "IDMixin",
    "PaginationParams",
    "PaginatedResponse",
    "ErrorResponse",
    "SuccessResponse"
]

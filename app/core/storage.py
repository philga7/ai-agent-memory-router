"""
Storage abstraction layer for AI Agent Memory Router.

This module provides abstract interfaces that different storage backends
(SQLite, PostgreSQL, etc.) must implement, enabling easy migration
between database systems.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from contextlib import asynccontextmanager

from app.models.memory import MemoryItem, MemoryRoute, MemoryMetadata
from app.models.agent import Agent, AgentContext
from app.models.context import ConversationContext, SimpleContext


class StorageError(Exception):
    """Base exception for storage operations."""
    pass


class ConnectionError(StorageError):
    """Database connection error."""
    pass


class TransactionError(StorageError):
    """Database transaction error."""
    pass


class MemoryStorage(ABC):
    """Abstract interface for memory storage operations."""
    
    @abstractmethod
    async def store_memory(self, memory: MemoryItem) -> str:
        """Store a memory item and return its ID."""
        pass
    
    @abstractmethod
    async def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory item by ID."""
        pass
    
    @abstractmethod
    async def search_memories(
        self, 
        query: str, 
        agent_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[MemoryItem]:
        """Search memories by content or metadata."""
        pass
    
    @abstractmethod
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory item."""
        pass
    
    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory item."""
        pass
    
    @abstractmethod
    async def get_agent_memories(
        self, 
        agent_id: str, 
        limit: int = 10, 
        offset: int = 0
    ) -> List[MemoryItem]:
        """Get all memories for a specific agent."""
        pass


class MetadataStorage(ABC):
    """Abstract interface for metadata storage operations."""
    
    @abstractmethod
    async def store_metadata(self, metadata: MemoryMetadata) -> str:
        """Store memory metadata."""
        pass
    
    @abstractmethod
    async def get_metadata(self, memory_id: str) -> Optional[MemoryMetadata]:
        """Retrieve metadata for a memory."""
        pass
    
    @abstractmethod
    async def update_metadata(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update memory metadata."""
        pass
    
    @abstractmethod
    async def delete_metadata(self, memory_id: str) -> bool:
        """Delete memory metadata."""
        pass


class RoutingStorage(ABC):
    """Abstract interface for routing decision storage."""
    
    @abstractmethod
    async def store_route(self, route: MemoryRoute) -> str:
        """Store a memory routing decision."""
        pass
    
    @abstractmethod
    async def get_route(self, route_id: str) -> Optional[MemoryRoute]:
        """Retrieve a routing decision."""
        pass
    
    @abstractmethod
    async def get_agent_routes(
        self, 
        agent_id: str, 
        limit: int = 10, 
        offset: int = 0
    ) -> List[MemoryRoute]:
        """Get routing decisions for an agent."""
        pass
    
    @abstractmethod
    async def update_route(self, route_id: str, updates: Dict[str, Any]) -> bool:
        """Update a routing decision."""
        pass
    
    @abstractmethod
    async def delete_route(self, route_id: str) -> bool:
        """Delete a routing decision."""
        pass


class AgentStorage(ABC):
    """Abstract interface for agent storage operations."""
    
    @abstractmethod
    async def store_agent(self, agent: Agent) -> str:
        """Store an agent."""
        pass
    
    @abstractmethod
    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Retrieve an agent by ID."""
        pass
    
    @abstractmethod
    async def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> bool:
        """Update an agent."""
        pass
    
    @abstractmethod
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent."""
        pass
    
    @abstractmethod
    async def list_agents(self, limit: int = 100, offset: int = 0) -> List[Agent]:
        """List all agents."""
        pass


class ContextStorage(ABC):
    """Abstract interface for conversation context storage."""
    
    @abstractmethod
    async def store_context(self, context: SimpleContext) -> str:
        """Store conversation context."""
        pass
    
    @abstractmethod
    async def get_context(self, context_id: str) -> Optional[SimpleContext]:
        """Retrieve conversation context."""
        pass
    
    @abstractmethod
    async def update_context(self, context_id: str, updates: Dict[str, Any]) -> bool:
        """Update conversation context."""
        pass
    
    @abstractmethod
    async def delete_context(self, context_id: str) -> bool:
        """Delete conversation context."""
        pass


class StorageManager(ABC):
    """Abstract interface for overall storage management."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the storage system."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close storage connections."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check storage system health."""
        pass
    
    @abstractmethod
    async def backup(self, backup_path: str) -> bool:
        """Create a backup of the storage."""
        pass
    
    @abstractmethod
    async def restore(self, backup_path: str) -> bool:
        """Restore from a backup."""
        pass
    
    @abstractmethod
    async def migrate(self, target_version: str) -> bool:
        """Migrate to a new storage version."""
        pass


class TransactionManager(ABC):
    """Abstract interface for transaction management."""
    
    @abstractmethod
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        pass
    
    @abstractmethod
    async def commit(self) -> None:
        """Commit the current transaction."""
        pass
    
    @abstractmethod
    async def rollback(self) -> None:
        """Rollback the current transaction."""
        pass


# Composite storage interface that combines all storage types
class UnifiedStorage(ABC):
    """Unified interface combining all storage types."""
    
    @property
    @abstractmethod
    def memory(self) -> MemoryStorage:
        """Access to memory storage."""
        pass
    
    @property
    @abstractmethod
    def metadata(self) -> MetadataStorage:
        """Access to metadata storage."""
        pass
    
    @property
    @abstractmethod
    def routing(self) -> RoutingStorage:
        """Access to routing storage."""
        pass
    
    @property
    @abstractmethod
    def agent(self) -> AgentStorage:
        """Access to agent storage."""
        pass
    
    @property
    @abstractmethod
    def context(self) -> ContextStorage:
        """Access to context storage."""
        pass
    
    @property
    @abstractmethod
    def manager(self) -> StorageManager:
        """Access to storage management."""
        pass
    
    @property
    @abstractmethod
    def transaction(self) -> TransactionManager:
        """Access to transaction management."""
        pass

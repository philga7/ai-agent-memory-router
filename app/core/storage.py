"""
Storage abstraction layer for AI Agent Memory Router.
This layer provides a unified interface for different database backends.
Currently supports SQLite with easy migration path to PostgreSQL.
"""

import os
import sqlite3
import json
import logging
from typing import Dict, List, Optional, Any, Union
from contextlib import contextmanager
from datetime import datetime
import asyncio
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Storage exceptions
class StorageError(Exception):
    """Base exception for storage operations."""
    pass

class ConnectionError(StorageError):
    """Exception raised for connection issues."""
    pass

class TransactionError(StorageError):
    """Exception raised for transaction issues."""
    pass

class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the storage backend."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close connections and cleanup."""
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        pass
    
    @abstractmethod
    async def execute_transaction(self, queries: List[tuple]) -> bool:
        """Execute multiple queries in a transaction."""
        pass

# Storage interface abstractions
class MemoryStorage(ABC):
    """Abstract interface for memory storage operations."""
    
    @abstractmethod
    async def store_memory(self, memory_data: Dict[str, Any]) -> str:
        """Store a memory and return its ID."""
        pass
    
    @abstractmethod
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory by ID."""
        pass
    
    @abstractmethod
    async def search_memories(self, query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search memories based on query and filters."""
        pass
    
    @abstractmethod
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory."""
        pass
    
    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        pass

class MetadataStorage(ABC):
    """Abstract interface for metadata storage operations."""
    
    @abstractmethod
    async def store_metadata(self, memory_id: str, metadata: Dict[str, Any]) -> bool:
        """Store metadata for a memory."""
        pass
    
    @abstractmethod
    async def get_metadata(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a memory."""
        pass
    
    @abstractmethod
    async def update_metadata(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update metadata for a memory."""
        pass
    
    @abstractmethod
    async def delete_metadata(self, memory_id: str) -> bool:
        """Delete metadata for a memory."""
        pass

class RoutingStorage(ABC):
    """Abstract interface for routing storage operations."""
    
    @abstractmethod
    async def create_route(self, route_data: Dict[str, Any]) -> str:
        """Create a memory route and return its ID."""
        pass
    
    @abstractmethod
    async def get_route(self, route_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a route by ID."""
        pass
    
    @abstractmethod
    async def update_route_status(self, route_id: str, status: str) -> bool:
        """Update route status."""
        pass
    
    @abstractmethod
    async def get_routes_by_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all routes for a specific agent."""
        pass

class AgentStorage(ABC):
    """Abstract interface for agent storage operations."""
    
    @abstractmethod
    async def store_agent(self, agent_data: Dict[str, Any]) -> str:
        """Store an agent and return its ID."""
        pass
    
    @abstractmethod
    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
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

class ContextStorage(ABC):
    """Abstract interface for context storage operations."""
    
    @abstractmethod
    async def store_context(self, context_data: Dict[str, Any]) -> str:
        """Store a conversation context and return its ID."""
        pass
    
    @abstractmethod
    async def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a context by ID."""
        pass
    
    @abstractmethod
    async def update_context(self, context_id: str, updates: Dict[str, Any]) -> bool:
        """Update a context."""
        pass
    
    @abstractmethod
    async def delete_context(self, context_id: str) -> bool:
        """Delete a context."""
        pass

class TransactionManager(ABC):
    """Abstract interface for transaction management."""
    
    @abstractmethod
    async def begin_transaction(self):
        """Begin a new transaction."""
        pass
    
    @abstractmethod
    async def commit_transaction(self):
        """Commit the current transaction."""
        pass
    
    @abstractmethod
    async def rollback_transaction(self):
        """Rollback the current transaction."""
        pass
    
    @abstractmethod
    async def __aenter__(self):
        """Async context manager entry."""
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass

class UnifiedStorage(ABC):
    """Abstract interface for unified storage operations."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the unified storage system."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close all storage connections."""
        pass
    
    @property
    @abstractmethod
    def memory_storage(self) -> MemoryStorage:
        """Get memory storage interface."""
        pass
    
    @property
    @abstractmethod
    def metadata_storage(self) -> MetadataStorage:
        """Get metadata storage interface."""
        pass
    
    @property
    @abstractmethod
    def routing_storage(self) -> RoutingStorage:
        """Get routing storage interface."""
        pass
    
    @property
    @abstractmethod
    def agent_storage(self) -> AgentStorage:
        """Get agent storage interface."""
        pass
    
    @property
    @abstractmethod
    def context_storage(self) -> ContextStorage:
        """Get context storage interface."""
        pass

class SQLiteBackend(StorageBackend):
    """SQLite storage backend implementation."""
    
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.connection = None
        self._ensure_data_directory()
    
    def _ensure_data_directory(self):
        """Ensure the data directory exists."""
        data_dir = Path(self.database_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> bool:
        """Initialize SQLite database and create tables."""
        try:
            # Run initialization in thread pool since SQLite is synchronous
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._initialize_sync)
            logger.info(f"SQLite database initialized at {self.database_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            return False
    
    def _initialize_sync(self):
        """Synchronous initialization of SQLite database."""
        with sqlite3.connect(self.database_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Read and execute initialization script
            init_script_path = Path(__file__).parent.parent.parent / "docker" / "sqlite" / "init.sql"
            if init_script_path.exists():
                with open(init_script_path, 'r') as f:
                    init_script = f.read()
                    conn.executescript(init_script)
                    conn.commit()
            else:
                logger.warning("SQLite initialization script not found, using basic schema")
                self._create_basic_schema(conn)
    
    def _create_basic_schema(self, conn: sqlite3.Connection):
        """Create basic schema if initialization script is not available."""
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                capabilities TEXT,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                tags TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
            );
        """)
        conn.commit()
    
    async def close(self) -> None:
        """Close SQLite connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    async def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute_query_sync, query, params)
    
    def _execute_query_sync(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Synchronous query execution."""
        with sqlite3.connect(self.database_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            if query.strip().upper().startswith(('SELECT', 'PRAGMA')):
                return [dict(row) for row in cursor.fetchall()]
            else:
                conn.commit()
                return [{"affected_rows": cursor.rowcount}]
    
    async def execute_transaction(self, queries: List[tuple]) -> bool:
        """Execute multiple queries in a transaction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute_transaction_sync, queries)
    
    def _execute_transaction_sync(self, queries: List[tuple]) -> bool:
        """Synchronous transaction execution."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.row_factory = sqlite3.Row
                for query, params in queries:
                    conn.execute(query, params)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            return False

class StorageManager:
    """Main storage manager that handles backend selection and operations."""
    
    def __init__(self):
        self.backend: Optional[StorageBackend] = None
        self.database_type: str = "sqlite"
    
    async def initialize(self, database_url: str = None) -> bool:
        """Initialize storage with the appropriate backend."""
        if not database_url:
            # Default to SQLite
            data_dir = Path(__file__).parent.parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            database_url = f"sqlite:///{data_dir}/ai_agent_memory.db"
        
        if database_url.startswith("sqlite://"):
            self.database_type = "sqlite"
            db_path = database_url.replace("sqlite:///", "")
            self.backend = SQLiteBackend(db_path)
        elif database_url.startswith("postgresql"):
            # Future PostgreSQL support
            self.database_type = "postgresql"
            raise NotImplementedError("PostgreSQL backend not yet implemented")
        else:
            raise ValueError(f"Unsupported database URL: {database_url}")
        
        return await self.backend.initialize()
    
    async def close(self) -> None:
        """Close storage connections."""
        if self.backend:
            await self.backend.close()
    
    async def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a query using the current backend."""
        if not self.backend:
            raise RuntimeError("Storage not initialized")
        return await self.backend.execute_query(query, params)
    
    async def execute_transaction(self, queries: List[tuple]) -> bool:
        """Execute a transaction using the current backend."""
        if not self.backend:
            raise RuntimeError("Storage not initialized")
        return await self.backend.execute_transaction(queries)
    
    def get_database_type(self) -> str:
        """Get the current database type."""
        return self.database_type

# Global storage manager instance
storage_manager = StorageManager()

# Convenience functions for easy access
async def get_storage() -> StorageManager:
    """Get the global storage manager instance."""
    return storage_manager

async def execute_query(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Execute a query using the global storage manager."""
    return await storage_manager.execute_query(query, params)

async def execute_transaction(queries: List[tuple]) -> bool:
    """Execute a transaction using the global storage manager."""
    return await storage_manager.execute_transaction(queries)

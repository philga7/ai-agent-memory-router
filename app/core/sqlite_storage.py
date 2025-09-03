"""
SQLite storage implementation for AI Agent Memory Router.

This module implements the storage abstraction interfaces using SQLite
as the backend database. It provides a complete storage solution that
can be easily migrated to PostgreSQL in the future.
"""

import json
import sqlite3
import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
import aiosqlite
import uuid


def safe_json_dumps(obj):
    """Safely serialize objects to JSON, handling datetime objects."""
    if obj is None:
        return None
    
    def datetime_handler(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    return json.dumps(obj, default=datetime_handler)

from app.core.storage import (
    MemoryStorage, MetadataStorage, RoutingStorage, AgentStorage,
    ContextStorage, StorageManager, TransactionManager, UnifiedStorage,
    StorageError, ConnectionError, TransactionError
)
from app.models.memory import MemoryItem, MemoryRoute, MemoryMetadata
from app.models.agent import Agent, AgentContext
from app.models.context import ConversationContext, SimpleContext


logger = logging.getLogger(__name__)


class SQLiteConnectionPool:
    """Manages SQLite database connections with async support."""
    
    def __init__(self, database_path: str, max_connections: int = 10):
        self.database_path = database_path
        self.max_connections = max_connections
        self._connections = asyncio.Queue(maxsize=max_connections)
        self._initialized = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return await self.get_connection()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass
    
    async def initialize(self):
        """Initialize the connection pool."""
        if self._initialized:
            return
        
        # Create database directory if it doesn't exist
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        await self._init_schema()
        
        # Pre-populate connection pool
        for _ in range(self.max_connections):
            conn = await aiosqlite.connect(self.database_path)
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.execute("PRAGMA journal_mode = WAL")
            self._connections.put_nowait(conn)
        
        self._initialized = True
        logger.info(f"SQLite connection pool initialized with {self.max_connections} connections")
    
    async def _init_schema(self):
        """Initialize database schema from SQL file."""
        schema_path = Path(__file__).parent / "database_schema.sql"
        if not schema_path.exists():
            # Try relative to project root
            schema_path = Path(__file__).parent.parent.parent / "app" / "core" / "database_schema.sql"
            if not schema_path.exists():
                logger.warning("Database schema file not found, using basic schema")
                return
        
        logger.info(f"Initializing schema from: {schema_path}")
        
        async with aiosqlite.connect(self.database_path) as conn:
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Execute the entire schema as one statement
            try:
                logger.debug("Executing complete schema...")
                await conn.executescript(schema_sql)
                await conn.commit()
                logger.info("Database schema initialized successfully")
            except Exception as e:
                logger.error(f"Failed to execute schema: {e}")
                raise
    
    async def get_connection(self) -> aiosqlite.Connection:
        """Get a database connection from the pool."""
        if not self._initialized:
            await self.initialize()
        
        try:
            conn = await asyncio.wait_for(self._connections.get(), timeout=5.0)
            return conn
        except asyncio.TimeoutError:
            raise ConnectionError("Timeout waiting for database connection")
    
    async def return_connection(self, conn: aiosqlite.Connection):
        """Return a connection to the pool."""
        try:
            self._connections.put_nowait(conn)
        except asyncio.QueueFull:
            await conn.close()
    
    async def close(self):
        """Close all connections in the pool."""
        while not self._connections.empty():
            conn = await self._connections.get()
            await conn.close()
        self._initialized = False


class SQLiteTransactionManager(TransactionManager):
    """SQLite transaction management implementation."""
    
    def __init__(self, connection_pool: SQLiteConnectionPool):
        self.connection_pool = connection_pool
        self._current_connection: Optional[aiosqlite.Connection] = None
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        conn = await self.connection_pool.get_connection()
        self._current_connection = conn
        
        try:
            await conn.execute("BEGIN TRANSACTION")
            yield self
            await conn.commit()
        except Exception as e:
            await conn.rollback()
            raise TransactionError(f"Transaction failed: {e}")
        finally:
            await self.connection_pool.return_connection(conn)
            self._current_connection = None
    
    async def commit(self):
        """Commit the current transaction."""
        if self._current_connection:
            await self._current_connection.commit()
    
    async def rollback(self):
        """Rollback the current transaction."""
        if self._current_connection:
            await self._current_connection.rollback()


class SQLiteMemoryStorage(MemoryStorage):
    """SQLite implementation of memory storage."""
    
    def __init__(self, connection_pool: SQLiteConnectionPool):
        self.connection_pool = connection_pool
    
    async def _execute_with_connection(self, operation):
        """Execute a database operation with proper connection management."""
        conn = await self.connection_pool.get_connection()
        try:
            return await operation(conn)
        finally:
            await self.connection_pool.return_connection(conn)
    
    async def _get_connection(self):
        """Get a database connection."""
        return await self.connection_pool.get_connection()
    
    async def _return_connection(self, conn):
        """Return a database connection."""
        await self.connection_pool.return_connection(conn)
    
    async def store_memory(self, memory: MemoryItem) -> str:
        """Store a memory item and return its ID."""
        memory_id = str(uuid.uuid4())
        
        async def _store(conn):
            await conn.execute("""
                INSERT INTO memory_items (id, agent_id, content, memory_type, priority, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                memory_id,
                memory.agent_id,
                memory.content,
                memory.memory_type,
                memory.priority,
                memory.expires_at
            ))
            await conn.commit()
            return memory_id
        
        result = await self._execute_with_connection(_store)
        logger.info(f"Stored memory {memory_id} for agent {memory.agent_id}")
        return result
    
    async def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory item by ID."""
        conn = await self.connection_pool.get_connection()
        try:
            async with conn.execute("""
                SELECT id, agent_id, content, memory_type, priority, created_at, updated_at, expires_at
                FROM memory_items WHERE id = ?
            """, (memory_id,)) as cursor:
                row = await cursor.fetchone()
        finally:
            await self.connection_pool.return_connection(conn)
        
        if not row:
            return None
        
        return MemoryItem(
            id=row[0],
            agent_id=row[1],
            content=row[2],
            memory_type=row[3],
            priority=row[4],
            created_at=datetime.fromisoformat(row[5]) if row[5] else None,
            updated_at=datetime.fromisoformat(row[6]) if row[6] else None,
            expires_at=datetime.fromisoformat(row[7]) if row[7] else None
        )
    
    async def search_memories(
        self, 
        query: str, 
        agent_id: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[MemoryItem]:
        """Search memories by content or metadata."""
        sql = """
            SELECT DISTINCT mi.id, mi.agent_id, mi.content, mi.memory_type, 
                   mi.priority, mi.created_at, mi.updated_at, mi.expires_at
            FROM memory_items mi
            LEFT JOIN memory_metadata mm ON mi.id = mm.memory_id
            WHERE (mi.content LIKE ? OR mm.tags LIKE ?)
        """
        params = [f"%{query}%", f"%{query}%"]
        
        if agent_id:
            sql += " AND mi.agent_id = ?"
            params.append(agent_id)
        
        sql += " ORDER BY mi.created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        conn = await self.connection_pool.get_connection()
        try:
            async with conn.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
        finally:
            await self.connection_pool.return_connection(conn)
        
        memories = []
        for row in rows:
            memory = MemoryItem(
                id=row[0],
                agent_id=row[1],
                content=row[2],
                memory_type=row[3],
                priority=row[4],
                created_at=datetime.fromisoformat(row[5]) if row[5] else None,
                updated_at=datetime.fromisoformat(row[6]) if row[6] else None,
                expires_at=datetime.fromisoformat(row[7]) if row[7] else None
            )
            memories.append(memory)
        
        return memories
    
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory item."""
        if not updates:
            return True
        
        set_clauses = []
        params = []
        
        for key, value in updates.items():
            if key in ['content', 'memory_type', 'priority', 'expires_at']:
                set_clauses.append(f"{key} = ?")
                params.append(value)
        
        if not set_clauses:
            return True
        
        sql = f"UPDATE memory_items SET {', '.join(set_clauses)} WHERE id = ?"
        params.append(memory_id)
        
        conn = await self.connection_pool.get_connection()
        try:
            await conn.execute(sql, params)
            await conn.commit()
        finally:
            await self.connection_pool.return_connection(conn)
        
        logger.info(f"Updated memory {memory_id}")
        return True
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory item."""
        conn = await self.connection_pool.get_connection()
        try:
            await conn.execute("DELETE FROM memory_items WHERE id = ?", (memory_id,))
            await conn.commit()
        finally:
            await self.connection_pool.return_connection(conn)
        
        logger.info(f"Deleted memory {memory_id}")
        return True
    
    async def get_agent_memories(
        self, 
        agent_id: str, 
        limit: int = 10, 
        offset: int = 0
    ) -> List[MemoryItem]:
        """Get all memories for a specific agent."""
        conn = await self.connection_pool.get_connection()
        try:
            async with conn.execute("""
                SELECT id, agent_id, content, memory_type, priority, created_at, updated_at, expires_at
                FROM memory_items 
                WHERE agent_id = ? 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            """, (agent_id, limit, offset)) as cursor:
                rows = await cursor.fetchall()
        finally:
            await self.connection_pool.return_connection(conn)
        
        memories = []
        for row in rows:
            memory = MemoryItem(
                id=row[0],
                agent_id=row[1],
                content=row[2],
                memory_type=row[3],
                priority=row[4],
                created_at=datetime.fromisoformat(row[5]) if row[5] else None,
                updated_at=datetime.fromisoformat(row[6]) if row[6] else None,
                expires_at=datetime.fromisoformat(row[7]) if row[7] else None
            )
            memories.append(memory)
        
        return memories


class SQLiteMetadataStorage(MetadataStorage):
    """SQLite implementation of metadata storage."""
    
    def __init__(self, connection_pool: SQLiteConnectionPool):
        self.connection_pool = connection_pool
    
    async def store_metadata(self, metadata: MemoryMetadata) -> str:
        """Store memory metadata."""
        metadata_id = str(uuid.uuid4())
        
        conn = await self.connection_pool.get_connection()
        try:
            await conn.execute("""
                INSERT INTO memory_metadata (id, memory_id, tags, source, confidence, 
                                          embedding_vector, vector_dimension)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata_id,
                metadata.memory_id,
                json.dumps(metadata.tags) if metadata.tags else None,
                metadata.source,
                metadata.confidence,
                metadata.embedding_vector,
                metadata.vector_dimension
            ))
            await conn.commit()
        finally:
            await self.connection_pool.return_connection(conn)
        
        return metadata_id
    
    async def get_metadata(self, memory_id: str) -> Optional[MemoryMetadata]:
        """Retrieve metadata for a memory."""
        conn = await self.connection_pool.get_connection()
        try:
            async with conn.execute("""
                SELECT id, memory_id, tags, source, confidence, embedding_vector, vector_dimension,
                       created_at, updated_at
                FROM memory_metadata WHERE memory_id = ?
            """, (memory_id,)) as cursor:
                row = await cursor.fetchone()
        finally:
            await self.connection_pool.return_connection(conn)
        
        if not row:
            return None
        
        return MemoryMetadata(
            id=row[0],
            memory_id=row[1],
            tags=json.loads(row[2]) if row[2] else [],
            source=row[3],
            confidence=row[4],
            embedding_vector=row[5],
            vector_dimension=row[6],
            created_at=datetime.fromisoformat(row[7]) if row[7] else None,
            updated_at=datetime.fromisoformat(row[8]) if row[8] else None
        )
    
    async def update_metadata(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update memory metadata."""
        if not updates:
            return True
        
        set_clauses = []
        params = []
        
        for key, value in updates.items():
            if key in ['tags', 'source', 'confidence', 'embedding_vector', 'vector_dimension']:
                if key == 'tags':
                    value = json.dumps(value) if value else None
                set_clauses.append(f"{key} = ?")
                params.append(value)
        
        if not set_clauses:
            return True
        
        sql = f"UPDATE memory_metadata SET {', '.join(set_clauses)} WHERE memory_id = ?"
        params.append(memory_id)
        
        conn = await self.connection_pool.get_connection()
        try:
            await conn.execute(sql, params)
            await conn.commit()
        finally:
            await self.connection_pool.return_connection(conn)
        
        return True
    
    async def delete_metadata(self, memory_id: str) -> bool:
        """Delete memory metadata."""
        conn = await self.connection_pool.get_connection()
        try:
            await conn.execute("DELETE FROM memory_metadata WHERE memory_id = ?", (memory_id,))
            await conn.commit()
        finally:
            await self.connection_pool.return_connection(conn)
        
        return True


class SQLiteRoutingStorage(RoutingStorage):
    """SQLite implementation of routing storage."""
    
    def __init__(self, connection_pool: SQLiteConnectionPool):
        self.connection_pool = connection_pool
    
    async def store_route(self, route: MemoryRoute) -> str:
        """Store a memory routing decision."""
        route_id = str(uuid.uuid4())
        
        conn = await self.connection_pool.get_connection()
        try:
            await conn.execute("""
                INSERT INTO memory_routes (id, source_agent_id, target_agent_id, memory_id,
                                         route_type, priority, status, routing_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                route_id,
                route.source_agent_id,
                route.target_agent_id,
                route.memory_id,
                route.route_type,
                route.priority,
                route.status,
                route.routing_reason
            ))
            await conn.commit()
        finally:
            await self.connection_pool.return_connection(conn)
        
        return route_id
    
    async def get_route(self, route_id: str) -> Optional[MemoryRoute]:
        """Retrieve a routing decision."""
        conn = await self.connection_pool.get_connection()
        try:
            async with conn.execute("""
                SELECT id, source_agent_id, target_agent_id, memory_id, route_type,
                       priority, status, routing_reason, created_at, delivered_at, acknowledged_at
                FROM memory_routes WHERE id = ?
            """, (route_id,)) as cursor:
                row = await cursor.fetchone()
        finally:
            await self.connection_pool.return_connection(conn)
        
        if not row:
            return None
        
        return MemoryRoute(
            id=row[0],
            source_agent_id=row[1],
            target_agent_id=row[2],
            memory_id=row[3],
            route_type=row[4],
            priority=row[5],
            status=row[6],
            routing_reason=row[7],
            created_at=datetime.fromisoformat(row[8]) if row[8] else None,
            delivered_at=datetime.fromisoformat(row[9]) if row[9] else None,
            acknowledged_at=datetime.fromisoformat(row[10]) if row[10] else None
        )
    
    async def get_agent_routes(
        self, 
        agent_id: str, 
        limit: int = 10, 
        offset: int = 0
    ) -> List[MemoryRoute]:
        """Get routing decisions for an agent."""
        conn = await self.connection_pool.get_connection()
        try:
            async with conn.execute("""
                SELECT id, source_agent_id, target_agent_id, memory_id, route_type,
                       priority, status, routing_reason, created_at, delivered_at, acknowledged_at
                FROM memory_routes 
                WHERE source_agent_id = ? OR target_agent_id = ?
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            """, (agent_id, agent_id, limit, offset)) as cursor:
                rows = await cursor.fetchall()
        finally:
            await self.connection_pool.return_connection(conn)
        
        routes = []
        for row in rows:
            route = MemoryRoute(
                id=row[0],
                source_agent_id=row[1],
                target_agent_id=row[2],
                memory_id=row[3],
                route_type=row[4],
                priority=row[5],
                status=row[6],
                routing_reason=row[7],
                created_at=datetime.fromisoformat(row[8]) if row[8] else None,
                delivered_at=datetime.fromisoformat(row[9]) if row[9] else None,
                acknowledged_at=datetime.fromisoformat(row[10]) if row[10] else None
            )
            routes.append(route)
        
        return routes
    
    async def update_route(self, route_id: str, updates: Dict[str, Any]) -> bool:
        """Update a routing decision."""
        if not updates:
            return True
        
        set_clauses = []
        params = []
        
        for key, value in updates.items():
            if key in ['status', 'routing_reason', 'delivered_at', 'acknowledged_at']:
                if key in ['delivered_at', 'acknowledged_at'] and value:
                    value = value.isoformat()
                set_clauses.append(f"{key} = ?")
                params.append(value)
        
        if not set_clauses:
            return True
        
        sql = f"UPDATE memory_routes SET {', '.join(set_clauses)} WHERE id = ?"
        params.append(route_id)
        
        conn = await self.connection_pool.get_connection()
        try:
            await conn.execute(sql, params)
            await conn.commit()
        finally:
            await self.connection_pool.return_connection(conn)
        
        return True
    
    async def delete_route(self, route_id: str) -> bool:
        """Delete a routing decision."""
        conn = await self.connection_pool.get_connection()
        try:
            await conn.execute("DELETE FROM memory_routes WHERE id = ?", (route_id,))
            await conn.commit()
        finally:
            await self.connection_pool.return_connection(conn)
        
        return True


class SQLiteAgentStorage(AgentStorage):
    """SQLite implementation of agent storage."""
    
    def __init__(self, connection_pool: SQLiteConnectionPool):
        self.connection_pool = connection_pool
    
    async def store_agent(self, agent: Agent) -> str:
        """Store an agent."""
        agent_id = str(uuid.uuid4())
        
        conn = await self.connection_pool.get_connection()
        try:
            await conn.execute("""
                INSERT INTO agents (id, name, description, agent_type, version, capabilities, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_id,
                agent.name,
                agent.description,
                agent.agent_type,
                agent.version,
                safe_json_dumps(agent.capabilities.model_dump()) if agent.capabilities else None,
                safe_json_dumps(agent.status.model_dump()) if agent.status else None,
                safe_json_dumps(agent.metadata) if agent.metadata else None
            ))
            await conn.commit()
        finally:
            await self.connection_pool.return_connection(conn)
        
        return agent_id
    
    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Retrieve an agent by ID."""
        conn = await self.connection_pool.get_connection()
        try:
            async with conn.execute("""
                SELECT id, name, description, agent_type, version, capabilities, status, metadata, created_at, updated_at
                FROM agents WHERE id = ?
            """, (agent_id,)) as cursor:
                row = await cursor.fetchone()
        finally:
            await self.connection_pool.return_connection(conn)
        
        if not row:
            return None
        
        # Parse capabilities and status from JSON
        capabilities_data = json.loads(row[5]) if row[5] else {}
        status_data = json.loads(row[6]) if row[6] else {}
        
        # Reconstruct the AgentCapabilities and AgentStatus objects
        from app.models.agent import AgentCapabilities, AgentStatus
        
        capabilities = AgentCapabilities(**capabilities_data) if capabilities_data else None
        status = AgentStatus(**status_data) if status_data else None
        
        return Agent(
            id=row[0],
            name=row[1],
            description=row[2],
            agent_type=row[3],
            version=row[4],
            capabilities=capabilities,
            status=status,
            metadata=json.loads(row[7]) if row[7] else {},
            created_at=datetime.fromisoformat(row[8]) if row[8] else None,
            updated_at=datetime.fromisoformat(row[9]) if row[9] else None
        )
    
    async def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> bool:
        """Update an agent."""
        if not updates:
            return True
        
        set_clauses = []
        params = []
        
        for key, value in updates.items():
            if key in ['name', 'description', 'capabilities', 'status', 'metadata']:
                if key in ['capabilities', 'metadata']:
                    value = json.dumps(value) if value else None
                set_clauses.append(f"{key} = ?")
                params.append(value)
        
        if not set_clauses:
            return True
        
        sql = f"UPDATE agents SET {', '.join(set_clauses)} WHERE id = ?"
        params.append(agent_id)
        
        conn = await self.connection_pool.get_connection()
        try:
            await conn.execute(sql, params)
            await conn.commit()
        finally:
            await self.connection_pool.return_connection(conn)
        
        return True
    
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent."""
        conn = await self.connection_pool.get_connection()
        try:
            await conn.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
            await conn.commit()
        finally:
            await self.connection_pool.return_connection(conn)
        
        return True
    
    async def list_agents(self, limit: int = 100, offset: int = 0) -> List[Agent]:
        """List all agents."""
        conn = await self.connection_pool.get_connection()
        try:
            async with conn.execute("""
                SELECT id, name, description, agent_type, version, capabilities, status, metadata, created_at, updated_at
                FROM agents 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            """, (limit, offset)) as cursor:
                rows = await cursor.fetchall()
        finally:
            await self.connection_pool.return_connection(conn)
        
        agents = []
        for row in rows:
            # Parse capabilities and status from JSON
            capabilities_data = json.loads(row[5]) if row[5] else {}
            status_data = json.loads(row[6]) if row[6] else {}
            
            # Reconstruct the AgentCapabilities and AgentStatus objects
            from app.models.agent import AgentCapabilities, AgentStatus
            
            capabilities = AgentCapabilities(**capabilities_data) if capabilities_data else None
            status = AgentStatus(**status_data) if status_data else None
            
            agent = Agent(
                id=row[0],
                name=row[1],
                description=row[2],
                agent_type=row[3],
                version=row[4],
                capabilities=capabilities,
                status=status,
                metadata=json.loads(row[7]) if row[7] else {},
                created_at=datetime.fromisoformat(row[8]) if row[8] else None,
                updated_at=datetime.fromisoformat(row[9]) if row[9] else None
            )
            agents.append(agent)
        
        return agents


class SQLiteContextStorage(ContextStorage):
    """SQLite implementation of context storage."""
    
    def __init__(self, connection_pool: SQLiteConnectionPool):
        self.connection_pool = connection_pool
    
    async def store_context(self, context: SimpleContext) -> str:
        """Store conversation context."""
        context_id = str(uuid.uuid4())
        
        conn = await self.connection_pool.get_connection()
        try:
            await conn.execute("""
                INSERT INTO conversation_contexts (id, conversation_id, agent_id, context_data, 
                                                context_type, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                context_id,
                context.conversation_id,
                context.agent_id,
                json.dumps(context.context_data),
                context.context_type,
                context.expires_at.isoformat() if context.expires_at else None
            ))
            await conn.commit()
        finally:
            await self.connection_pool.return_connection(conn)
        
        return context_id
    
    async def get_context(self, context_id: str) -> Optional[SimpleContext]:
        """Retrieve conversation context."""
        conn = await self.connection_pool.get_connection()
        try:
            async with conn.execute("""
                SELECT id, conversation_id, agent_id, context_data, context_type, 
                       created_at, updated_at, expires_at
                FROM conversation_contexts WHERE id = ?
            """, (context_id,)) as cursor:
                row = await cursor.fetchone()
        finally:
            await self.connection_pool.return_connection(conn)
        
        if not row:
            return None
        
        return SimpleContext(
            id=row[0],
            conversation_id=row[1],
            agent_id=row[2],
            context_data=json.loads(row[3]),
            context_type=row[4],
            expires_at=datetime.fromisoformat(row[7]) if row[7] else None
        )
    
    async def update_context(self, context_id: str, updates: Dict[str, Any]) -> bool:
        """Update conversation context."""
        if not updates:
            return True
        
        set_clauses = []
        params = []
        
        for key, value in updates.items():
            if key in ['context_data', 'context_type', 'expires_at']:
                if key == 'context_data':
                    value = json.dumps(value)
                elif key == 'expires_at' and value:
                    value = value.isoformat()
                set_clauses.append(f"{key} = ?")
                params.append(value)
        
        if not set_clauses:
            return True
        
        sql = f"UPDATE conversation_contexts SET {', '.join(set_clauses)} WHERE id = ?"
        params.append(context_id)
        
        conn = await self.connection_pool.get_connection()
        try:
            await conn.execute(sql, params)
            await conn.commit()
        finally:
            await self.connection_pool.return_connection(conn)
        
        return True
    
    async def delete_context(self, context_id: str) -> bool:
        """Delete conversation context."""
        conn = await self.connection_pool.get_connection()
        try:
            await conn.execute("DELETE FROM conversation_contexts WHERE id = ?", (context_id,))
            await conn.commit()
        finally:
            await self.connection_pool.return_connection(conn)
        
        return True


class SQLiteStorageManager(StorageManager):
    """SQLite implementation of storage management."""
    
    def __init__(self, connection_pool: SQLiteConnectionPool):
        self.connection_pool = connection_pool
    
    async def initialize(self) -> bool:
        """Initialize the storage system."""
        try:
            await self.connection_pool.initialize()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
            return False
    
    async def close(self) -> None:
        """Close storage connections."""
        await self.connection_pool.close()
    
    async def health_check(self) -> bool:
        """Check storage system health."""
        try:
            conn = await self.connection_pool.get_connection()
            try:
                await conn.execute("SELECT 1")
            finally:
                await self.connection_pool.return_connection(conn)
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def backup(self, backup_path: str) -> bool:
        """Create a backup of the storage."""
        try:
            import shutil
            shutil.copy2(self.connection_pool.database_path, backup_path)
            logger.info(f"Backup created at {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    async def restore(self, backup_path: str) -> bool:
        """Restore from a backup."""
        try:
            import shutil
            shutil.copy2(backup_path, self.connection_pool.database_path)
            logger.info(f"Restore completed from {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    async def migrate(self, target_version: str) -> bool:
        """Migrate to a new storage version."""
        # This will be implemented when we add migration support
        logger.info(f"Migration to version {target_version} not yet implemented")
        return True


class SQLiteUnifiedStorage(UnifiedStorage):
    """SQLite implementation of unified storage interface."""
    
    def __init__(self, database_path: str, max_connections: int = 10):
        self.connection_pool = SQLiteConnectionPool(database_path, max_connections)
        self._memory = SQLiteMemoryStorage(self.connection_pool)
        self._metadata = SQLiteMetadataStorage(self.connection_pool)
        self._routing = SQLiteRoutingStorage(self.connection_pool)
        self._agent = SQLiteAgentStorage(self.connection_pool)
        self._context = SQLiteContextStorage(self.connection_pool)
        self._manager = SQLiteStorageManager(self.connection_pool)
        self._transaction = SQLiteTransactionManager(self.connection_pool)
    
    @property
    def memory(self) -> MemoryStorage:
        return self._memory
    
    @property
    def metadata(self) -> MetadataStorage:
        return self._metadata
    
    @property
    def routing(self) -> RoutingStorage:
        return self._routing
    
    @property
    def agent(self) -> AgentStorage:
        return self._agent
    
    @property
    def context(self) -> ContextStorage:
        return self._context
    
    @property
    def manager(self) -> StorageManager:
        return self._manager
    
    @property
    def transaction(self) -> TransactionManager:
        return self._transaction

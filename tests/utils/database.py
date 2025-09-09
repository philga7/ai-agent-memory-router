"""
Database utilities for testing
"""

import asyncio
from typing import AsyncGenerator, List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.memory import MemoryStore
from app.models.agent import Agent
from app.models.context import ConversationContext


class TestDatabaseManager:
    """Manager for test database operations."""
    
    def __init__(self, engine: AsyncEngine):
        self.engine = engine
        self.session_factory = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def create_session(self) -> AsyncSession:
        """Create a new database session."""
        return self.session_factory()
    
    async def cleanup_database(self) -> None:
        """Clean up all test data from database."""
        async with self.create_session() as session:
            # Delete all test data
            await session.execute(text("DELETE FROM memory_stores WHERE metadata->>'environment' = 'test'"))
            await session.execute(text("DELETE FROM agents WHERE metadata->>'environment' = 'test'"))
            await session.execute(text("DELETE FROM contexts WHERE metadata->>'environment' = 'test'"))
            await session.commit()
    
    async def seed_test_data(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        """Seed database with test data."""
        async with self.create_session() as session:
            # Seed memory stores
            if 'memory_stores' in data:
                for memory_data in data['memory_stores']:
                    memory = MemoryStore(**memory_data)
                    session.add(memory)
            
            # Seed agents
            if 'agents' in data:
                for agent_data in data['agents']:
                    agent = Agent(**agent_data)
                    session.add(agent)
            
            # Seed contexts
            if 'contexts' in data:
                for context_data in data['contexts']:
                    context = ConversationContext(**context_data)
                    session.add(context)
            
            await session.commit()
    
    async def get_table_counts(self) -> Dict[str, int]:
        """Get row counts for all tables."""
        async with self.create_session() as session:
            counts = {}
            
            # Count memory stores
            result = await session.execute(text("SELECT COUNT(*) FROM memory_stores"))
            counts['memory_stores'] = result.scalar()
            
            # Count agents
            result = await session.execute(text("SELECT COUNT(*) FROM agents"))
            counts['agents'] = result.scalar()
            
            # Count contexts
            result = await session.execute(text("SELECT COUNT(*) FROM contexts"))
            counts['contexts'] = result.scalar()
            
            return counts
    
    async def verify_data_integrity(self) -> Dict[str, bool]:
        """Verify data integrity in test database."""
        async with self.create_session() as session:
            integrity_checks = {}
            
            # Check for orphaned records
            result = await session.execute(text("""
                SELECT COUNT(*) FROM memory_stores m 
                LEFT JOIN agents a ON m.metadata->>'agent_id' = a.id 
                WHERE a.id IS NULL AND m.metadata->>'agent_id' IS NOT NULL
            """))
            integrity_checks['orphaned_memory_stores'] = result.scalar() == 0
            
            # Check for invalid timestamps
            result = await session.execute(text("""
                SELECT COUNT(*) FROM memory_stores 
                WHERE created_at > updated_at
            """))
            integrity_checks['invalid_timestamps'] = result.scalar() == 0
            
            return integrity_checks


class DatabaseTestHelper:
    """Helper class for database testing operations."""
    
    @staticmethod
    async def create_test_memory_store(
        session: AsyncSession,
        content: str = "Test memory content",
        agent_id: str = "test-agent-123",
        **kwargs
    ) -> MemoryStore:
        """Create a test memory store."""
        memory_data = {
            "content": content,
            "metadata": {
                "agent_id": agent_id,
                "environment": "test",
                "memory_type": "knowledge",
                "tags": ["testing"],
                "importance": 5,
                **kwargs.get("metadata", {})
            },
            "priority": kwargs.get("priority", "normal"),
            "access_control": kwargs.get("access_control", {"read": [agent_id], "write": [agent_id]})
        }
        
        memory = MemoryStore(**memory_data)
        session.add(memory)
        await session.commit()
        await session.refresh(memory)
        return memory
    
    @staticmethod
    async def create_test_agent(
        session: AsyncSession,
        name: str = "Test Agent",
        agent_type: str = "memory_router",
        **kwargs
    ) -> Agent:
        """Create a test agent."""
        agent_data = {
            "name": name,
            "description": kwargs.get("description", "A test agent"),
            "agent_type": agent_type,
            "status": kwargs.get("status", "active"),
            "capabilities": kwargs.get("capabilities", ["memory_storage", "memory_retrieval"]),
            "metadata": {
                "environment": "test",
                "version": "1.0.0",
                **kwargs.get("metadata", {})
            }
        }
        
        agent = Agent(**agent_data)
        session.add(agent)
        await session.commit()
        await session.refresh(agent)
        return agent
    
    @staticmethod
    async def create_test_context(
        session: AsyncSession,
        title: str = "Test Conversation",
        **kwargs
    ) -> ConversationContext:
        """Create a test context."""
        from app.models.context import ContextParticipant, ContextMessage
        from uuid import uuid4
        from datetime import datetime, timezone
        
        context_data = {
            "title": title,
            "description": kwargs.get("description", "A test conversation"),
            "context_type": kwargs.get("context_type", "conversation"),
            "status": kwargs.get("status", "active"),
            "participants": kwargs.get("participants", [
                ContextParticipant(
                    agent_id=uuid4(),
                    role="user",
                    name="Test User",
                    capabilities=["chat"]
                ),
                ContextParticipant(
                    agent_id=uuid4(),
                    role="assistant",
                    name="Test Agent",
                    capabilities=["memory_storage", "memory_retrieval"]
                )
            ]),
            "messages": kwargs.get("messages", [
                ContextMessage(
                    message_id=uuid4(),
                    sender_id=uuid4(),
                    content="Hello, how are you?",
                    message_type="text",
                    timestamp=datetime.now(timezone.utc)
                )
            ]),
            "tags": kwargs.get("tags", ["test", "conversation"]),
            "metadata": {
                "environment": "test",
                "session_id": "test-session-123",
                **kwargs.get("metadata", {})
            }
        }
        
        context = ConversationContext(**context_data)
        session.add(context)
        await session.commit()
        await session.refresh(context)
        return context
    
    @staticmethod
    async def cleanup_test_data(session: AsyncSession) -> None:
        """Clean up all test data from session."""
        # Delete test memory stores
        await session.execute(text("DELETE FROM memory_stores WHERE metadata->>'environment' = 'test'"))
        
        # Delete test agents
        await session.execute(text("DELETE FROM agents WHERE metadata->>'environment' = 'test'"))
        
        # Delete test contexts
        await session.execute(text("DELETE FROM contexts WHERE metadata->>'environment' = 'test'"))
        
        await session.commit()
    
    @staticmethod
    async def get_test_data_counts(session: AsyncSession) -> Dict[str, int]:
        """Get counts of test data in session."""
        counts = {}
        
        # Count test memory stores
        result = await session.execute(text("SELECT COUNT(*) FROM memory_stores WHERE metadata->>'environment' = 'test'"))
        counts['memory_stores'] = result.scalar()
        
        # Count test agents
        result = await session.execute(text("SELECT COUNT(*) FROM agents WHERE metadata->>'environment' = 'test'"))
        counts['agents'] = result.scalar()
        
        # Count test contexts
        result = await session.execute(text("SELECT COUNT(*) FROM contexts WHERE metadata->>'environment' = 'test'"))
        counts['contexts'] = result.scalar()
        
        return counts


class DatabaseTransactionHelper:
    """Helper for database transaction testing."""
    
    @staticmethod
    async def test_transaction_rollback(session: AsyncSession, operation) -> None:
        """Test that a transaction can be rolled back."""
        try:
            await operation(session)
            await session.commit()
        except Exception:
            await session.rollback()
            raise
    
    @staticmethod
    async def test_concurrent_operations(
        session1: AsyncSession,
        session2: AsyncSession,
        operation1,
        operation2
    ) -> None:
        """Test concurrent database operations."""
        # Start both operations concurrently
        task1 = asyncio.create_task(operation1(session1))
        task2 = asyncio.create_task(operation2(session2))
        
        # Wait for both to complete
        await asyncio.gather(task1, task2)
    
    @staticmethod
    async def test_isolation_levels(session: AsyncSession, operation) -> None:
        """Test different isolation levels."""
        # Test with default isolation level
        await operation(session)
        
        # Test with read committed
        await session.execute(text("SET TRANSACTION ISOLATION LEVEL READ COMMITTED"))
        await operation(session)
        
        # Test with serializable
        await session.execute(text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE"))
        await operation(session)


class DatabasePerformanceHelper:
    """Helper for database performance testing."""
    
    @staticmethod
    async def measure_query_performance(session: AsyncSession, query: str) -> float:
        """Measure query execution time."""
        import time
        
        start_time = time.time()
        await session.execute(text(query))
        end_time = time.time()
        
        return end_time - start_time
    
    @staticmethod
    async def benchmark_bulk_operations(
        session: AsyncSession,
        create_operation,
        count: int = 1000
    ) -> Dict[str, float]:
        """Benchmark bulk database operations."""
        import time
        
        # Benchmark bulk insert
        start_time = time.time()
        for i in range(count):
            await create_operation(session, i)
        await session.commit()
        insert_time = time.time() - start_time
        
        # Benchmark bulk select
        start_time = time.time()
        result = await session.execute(text("SELECT COUNT(*) FROM memory_stores WHERE metadata->>'environment' = 'test'"))
        select_time = time.time() - start_time
        
        return {
            "bulk_insert_time": insert_time,
            "bulk_select_time": select_time,
            "insert_rate": count / insert_time if insert_time > 0 else 0,
            "select_rate": 1 / select_time if select_time > 0 else 0
        }

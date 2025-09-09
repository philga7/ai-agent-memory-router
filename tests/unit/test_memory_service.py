"""
Unit tests for memory service functionality.

Tests the core memory management operations including storage, retrieval,
and routing decisions.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from typing import Dict, Any, List
from uuid import uuid4

from app.services.memory_service import MemoryService
from app.models.memory import MemoryStoreCreate, MemoryContent, AgentSource, MemorySearch
from app.core.config import get_settings


class TestMemoryService:
    """Test memory service functionality."""
    
    @pytest.fixture
    def memory_service(self):
        """Create memory service instance for testing."""
        return MemoryService()
    
    @pytest.fixture
    def sample_memory_data(self):
        """Create sample memory data for testing."""
        return MemoryStoreCreate(
            content=MemoryContent(
                text="This is a test memory content",
                format="text"
            ),
            source=AgentSource(
                agent_id="test-agent-456",
                agent_type="assistant",
                session_id="test-session-123"
            ),
            memory_type="conversation",
            importance=5
        )
    
    def test_memory_service_initialization(self, memory_service):
        """Test that memory service initializes correctly."""
        assert memory_service is not None
        assert hasattr(memory_service, 'store_memory')
        assert hasattr(memory_service, 'get_memory')
        assert hasattr(memory_service, 'search_memories')
        assert hasattr(memory_service, 'delete_memory')
        assert hasattr(memory_service, 'create_memory_route')
        assert hasattr(memory_service, 'get_memory_stats')
    
    @pytest.mark.asyncio
    async def test_store_memory_success(self, memory_service, sample_memory_data):
        """Test successful memory storage."""
        result = await memory_service.store_memory(sample_memory_data)
        
        assert result is not None
        assert result.content.text == "This is a test memory content"
        assert result.source.agent_id == "test-agent-456"
        assert result.memory_type == "conversation"
        assert result.importance == 5
    
    @pytest.mark.asyncio
    async def test_get_memory_success(self, memory_service, sample_memory_data):
        """Test successful memory retrieval."""
        # First store a memory
        stored_memory = await memory_service.store_memory(sample_memory_data)
        memory_id = stored_memory.id
        
        # Then retrieve it
        result = await memory_service.get_memory(memory_id)
        
        assert result is not None
        assert result.id == memory_id
        assert result.content.text == "This is a test memory content"
    
    @pytest.mark.asyncio
    async def test_get_memory_not_found(self, memory_service):
        """Test memory retrieval when memory doesn't exist."""
        non_existent_id = uuid4()
        result = await memory_service.get_memory(non_existent_id)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_search_memories_success(self, memory_service, sample_memory_data):
        """Test successful memory search."""
        # First store a memory
        await memory_service.store_memory(sample_memory_data)
        
        # Then search for it
        search_query = MemorySearch(
            query="test memory",
            limit=10,
            offset=0
        )
        
        result = await memory_service.search_memories(search_query)
        
        assert result is not None
        assert hasattr(result, 'results')
        assert hasattr(result, 'total')
        assert result.total >= 0
    
    @pytest.mark.asyncio
    async def test_search_memories_empty_results(self, memory_service):
        """Test memory search with no results."""
        search_query = MemorySearch(
            query="non-existent content",
            limit=10,
            offset=0
        )
        
        result = await memory_service.search_memories(search_query)
        
        assert result is not None
        assert result.total == 0
        assert len(result.results) == 0
    
    @pytest.mark.asyncio
    async def test_delete_memory_success(self, memory_service, sample_memory_data):
        """Test successful memory deletion."""
        # First store a memory
        stored_memory = await memory_service.store_memory(sample_memory_data)
        memory_id = stored_memory.id
        
        # Then delete it
        result = await memory_service.delete_memory(memory_id)
        
        assert result is True
        
        # Verify it's deleted
        retrieved = await memory_service.get_memory(memory_id)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, memory_service):
        """Test memory deletion when memory doesn't exist."""
        non_existent_id = uuid4()
        result = await memory_service.delete_memory(non_existent_id)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_memory_stats(self, memory_service, sample_memory_data):
        """Test getting memory statistics."""
        # Store some memories
        await memory_service.store_memory(sample_memory_data)
        
        # Get stats
        stats = await memory_service.get_memory_stats()
        
        assert stats is not None
        assert hasattr(stats, 'total_memories')
        assert hasattr(stats, 'memories_by_type')
        assert hasattr(stats, 'memories_by_agent')
        assert stats.total_memories >= 1
    
    @pytest.mark.asyncio
    async def test_memory_service_concurrent_operations(self, memory_service, sample_memory_data):
        """Test memory service handles concurrent operations."""
        import asyncio
        
        # Create multiple concurrent storage operations
        tasks = [
            memory_service.store_memory(sample_memory_data)
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        assert all(result is not None for result in results)
        
        # Verify all memories were stored
        stats = await memory_service.get_memory_stats()
        assert stats.total_memories >= 5
    
    @pytest.mark.asyncio
    async def test_memory_service_error_handling(self, memory_service):
        """Test memory service error handling."""
        # Test that service handles None inputs gracefully
        with pytest.raises(AttributeError):
            # This should fail because None doesn't have a 'source' attribute
            await memory_service.store_memory(None)
    
    @pytest.mark.asyncio
    async def test_memory_service_with_different_types(self, memory_service):
        """Test memory service with different memory types."""
        # Test with different memory types
        memory_types = ["conversation", "knowledge", "experience", "fact", "procedure"]
        
        for memory_type in memory_types:
            memory_data = MemoryStoreCreate(
                content=MemoryContent(
                    text=f"Test {memory_type} memory",
                    format="text"
                ),
                source=AgentSource(
                    agent_id="test-agent-456",
                    agent_type="assistant",
                    session_id="test-session-123"
                ),
                memory_type=memory_type,
                importance=5
            )
            
            result = await memory_service.store_memory(memory_data)
            assert result is not None
            assert result.memory_type == memory_type
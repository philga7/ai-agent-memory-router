"""
Test data fixtures for AI Agent Memory Router
"""

import pytest
from typing import Dict, Any, List
from datetime import datetime, timezone
from uuid import uuid4

from app.models.memory import (
    MemoryStoreCreate, MemoryStore, MemorySearch, MemorySearchResponse,
    MemoryStats, MemoryItem, MemoryMetadata, MemoryContent, MemorySource, AgentSource,
    MemorySearchResult
)
from app.models.agent import Agent, AgentCreate, AgentUpdate, AgentSearch, AgentCapabilities, AgentStatus
from app.models.context import ContextUpdate, ContextSearch, ContextResponse, ConversationContext, ContextParticipant, ContextMessage


@pytest.fixture
def sample_memory_store_create():
    """Sample memory store create data."""
    return MemoryStoreCreate(
        content=MemoryContent(
            text="This is a test memory for unit testing",
            language="en",
            format="text",
            encoding="utf-8"
        ),
        source=MemorySource(
            type="agent",
            identifier="test-agent",
            metadata={"project_id": "test-project"}
        ),
        memory_type="knowledge",
        importance=5,
        access_control={"read": ["test-agent"], "write": ["test-agent"]}
    )


@pytest.fixture
def sample_memory_store():
    """Sample memory store data."""
    return MemoryStore(
        id=uuid4(),
        content=MemoryContent(
            text="This is a test memory for unit testing",
            language="en",
            format="text",
            encoding="utf-8"
        ),
        source=MemorySource(
            type="agent",
            identifier="test-agent",
            metadata={"project_id": "test-project"}
        ),
        memory_type="knowledge",
        importance=5,
        access_control={"read": ["test-agent"], "write": ["test-agent"]},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_memory_search():
    """Sample memory search data."""
    return MemorySearch(
        query="test memory",
        filters={
            "agent_id": "test-agent",
            "memory_type": "knowledge"
        },
        limit=10,
        offset=0
    )


@pytest.fixture
def sample_memory_search_response():
    """Sample memory search response data."""
    return MemorySearchResponse(
        results=[
            MemorySearchResult(
                memory=MemoryItem(
                    id=uuid4(),
                    agent_id="test-agent",
                    content="This is a test memory for unit testing",
                    memory_type="knowledge",
                    priority=2,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                ),
                metadata=MemoryMetadata(
                    id=uuid4(),
                    memory_id=str(uuid4()),
                    tags=["testing", "unit-test"],
                    source="test-agent",
                    confidence=0.95
                ),
                relevance_score=0.95,
                matched_fields=["content", "memory_type"]
            )
        ],
        total=1,
        query="test memory",
        execution_time=0.1
    )


@pytest.fixture
def sample_memory_stats():
    """Sample memory stats data."""
    return MemoryStats(
        total_memories=100,
        memories_by_type={"knowledge": 50, "experience": 30, "context": 20},
        memories_by_agent={"agent-1": 40, "agent-2": 35, "agent-3": 25},
        average_priority=2.5,
        oldest_memory=datetime.now(timezone.utc),
        newest_memory=datetime.now(timezone.utc),
        total_size_bytes=1024000
    )


@pytest.fixture
def sample_agent():
    """Sample agent data."""
    return Agent(
        id=uuid4(),
        name="Test Agent",
        description="A test agent for unit testing",
        agent_type="router",
        version="1.0.0",
        capabilities=AgentCapabilities(
            memory_read=True,
            memory_write=True,
            memory_search=True,
            context_access=True,
            routing=True,
            admin=False,
            custom_tools=["memory_storage", "memory_retrieval", "routing"]
        ),
        status=AgentStatus(
            status="online",
            last_heartbeat=datetime.now(timezone.utc),
            uptime=3600.0,
            memory_count=10,
            route_count=5,
            error_count=0
        ),
        metadata={
            "environment": "test",
            "created_by": "test-user"
        },
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_agent_create():
    """Sample agent create data."""
    return AgentCreate(
        name="Test Agent",
        description="A test agent for unit testing",
        agent_type="router",
        version="1.0.0",
        capabilities=AgentCapabilities(
            memory_read=True,
            memory_write=True,
            memory_search=True,
            context_access=True,
            routing=True,
            admin=False,
            custom_tools=["memory_storage", "memory_retrieval", "routing"]
        ),
        metadata={
            "environment": "test",
            "created_by": "test-user"
        }
    )


@pytest.fixture
def sample_agent_update():
    """Sample agent update data."""
    return AgentUpdate(
        name="Updated Test Agent",
        description="An updated test agent for unit testing",
        status="inactive",
        metadata={
            "version": "1.1.0",
            "environment": "test",
            "updated_by": "test-user"
        }
    )


@pytest.fixture
def sample_agent_search():
    """Sample agent search data."""
    return AgentSearch(
        query="test agent",
        filters={
            "agent_type": "memory_router",
            "status": "active"
        },
        limit=10,
        offset=0
    )


@pytest.fixture
def sample_context():
    """Sample context data."""
    return ConversationContext(
        id=uuid4(),
        title="Test Conversation",
        description="A test conversation for unit testing",
        context_type="conversation",
        status="active",
        participants=[
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
        ],
        messages=[
            ContextMessage(
                message_id=uuid4(),
                sender_id=uuid4(),
                content="Hello, how are you?",
                message_type="text",
                timestamp=datetime.now(timezone.utc)
            ),
            ContextMessage(
                message_id=uuid4(),
                sender_id=uuid4(),
                content="I'm doing well, thank you!",
                message_type="text",
                timestamp=datetime.now(timezone.utc)
            )
        ],
        tags=["test", "conversation"],
        metadata={
            "session_id": "session-123",
            "environment": "test"
        },
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_context_create():
    """Sample context create data."""
    return ConversationContext(
        title="Test Conversation",
        description="A test conversation for unit testing",
        context_type="conversation",
        participants=[
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
        ],
        messages=[
            ContextMessage(
                message_id=uuid4(),
                sender_id=uuid4(),
                content="Hello, how are you?",
                message_type="text",
                timestamp=datetime.now(timezone.utc)
            )
        ],
        tags=["test", "conversation"],
        metadata={
            "session_id": "session-123",
            "environment": "test"
        }
    )


@pytest.fixture
def sample_context_update():
    """Sample context update data."""
    return ContextUpdate(
        status="inactive",
        messages=[
            {
                "role": "user",
                "content": "Hello, how are you?",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "role": "agent",
                "content": "I'm doing well, thank you!",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "role": "user",
                "content": "That's great to hear!",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ],
        metadata={
            "session_id": "session-123",
            "environment": "test",
            "updated_by": "test-user"
        }
    )


@pytest.fixture
def sample_context_search():
    """Sample context search data."""
    return ContextSearch(
        query="test conversation",
        filters={
            "agent_id": "agent-id-123",
            "context_type": "conversation",
            "status": "active"
        },
        limit=10,
        offset=0
    )


@pytest.fixture
def sample_memory_list():
    """Sample list of memory data."""
    return [
        {
            "id": f"memory-id-{i}",
            "content": f"Test memory content {i}",
            "metadata": {
                "project_id": "test-project",
                "agent_id": f"test-agent-{i}",
                "memory_type": "knowledge",
                "tags": ["testing", "unit-test"],
                "importance": i % 5 + 1
            },
            "priority": "normal",
            "access_control": {"read": [f"test-agent-{i}"], "write": [f"test-agent-{i}"]},
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        for i in range(1, 6)
    ]


@pytest.fixture
def sample_agent_list():
    """Sample list of agent data."""
    return [
        {
            "id": f"agent-id-{i}",
            "name": f"Test Agent {i}",
            "description": f"A test agent {i} for unit testing",
            "agent_type": "memory_router",
            "status": "active",
            "capabilities": ["memory_storage", "memory_retrieval", "routing"],
            "metadata": {
                "version": "1.0.0",
                "environment": "test",
                "created_by": "test-user"
            },
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        for i in range(1, 6)
    ]


@pytest.fixture
def sample_context_list():
    """Sample list of context data."""
    return [
        {
            "id": f"context-id-{i}",
            "conversation_id": f"conversation-id-{i}",
            "agent_id": f"agent-id-{i}",
            "context_type": "conversation",
            "status": "active",
            "participants": [f"agent-id-{i}", f"user-id-{i}"],
            "messages": [
                {
                    "role": "user",
                    "content": f"Hello, how are you? (Message {i})",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ],
            "tags": ["test", "conversation"],
            "metadata": {
                "session_id": f"session-{i}",
                "environment": "test"
            },
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        for i in range(1, 6)
    ]


@pytest.fixture
def sample_error_response():
    """Sample error response data."""
    return {
        "error": "Test error message",
        "code": "TEST_ERROR",
        "details": {
            "field": "test_field",
            "message": "Test validation error"
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@pytest.fixture
def sample_success_response():
    """Sample success response data."""
    return {
        "success": True,
        "message": "Operation completed successfully",
        "data": {
            "id": "test-id-123",
            "status": "completed"
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

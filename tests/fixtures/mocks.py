"""
Mock fixtures for testing
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from app.services.memory_service import MemoryService
from app.services.routing_service import IntelligentRoutingService
from app.services.cipher_service import CipherService
from app.services.weaviate_service import WeaviateService


@pytest.fixture
def mock_cipher_service():
    """Mock Cipher service for testing."""
    mock = AsyncMock(spec=CipherService)
    
    # Configure default return values
    mock.store_memory.return_value = "cipher-memory-id-123"
    mock.retrieve_memory.return_value = {
        "id": "cipher-memory-id-123",
        "content": "Test memory content",
        "metadata": {"project_id": "test-project"},
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z"
    }
    mock.search_memories.return_value = [
        {
            "id": "cipher-memory-id-123",
            "content": "Test memory content",
            "metadata": {"project_id": "test-project"},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }
    ]
    mock.delete_memory.return_value = True
    mock.update_memory.return_value = True
    mock.create_project.return_value = "project-id-123"
    mock.get_project_info.return_value = {
        "project_id": "project-id-123",
        "name": "Test Project",
        "created_at": "2024-01-01T00:00:00Z"
    }
    
    return mock


@pytest.fixture
def mock_weaviate_service():
    """Mock Weaviate service for testing."""
    mock = AsyncMock(spec=WeaviateService)
    
    # Configure default return values
    mock.store_memory.return_value = "weaviate-memory-id-456"
    mock.retrieve_memory.return_value = {
        "id": "weaviate-memory-id-456",
        "content": "Test memory content",
        "metadata": {"agent_id": "test-agent"},
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z"
    }
    mock.search_memories.return_value = [
        {
            "id": "weaviate-memory-id-456",
            "content": "Test memory content",
            "metadata": {"agent_id": "test-agent"},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }
    ]
    mock.delete_memory.return_value = True
    mock.get_memory_stats.return_value = {
        "total_memories": 1,
        "total_size_bytes": 1024,
        "average_priority": 2.5,
        "last_updated": "2024-01-01T00:00:00Z"
    }
    
    return mock


@pytest.fixture
def mock_routing_service():
    """Mock routing service for testing."""
    mock = AsyncMock(spec=IntelligentRoutingService)
    
    # Configure default return values
    mock.route_memory.return_value = {
        "target_service": "cipher",
        "confidence": 0.95,
        "reasoning": "Project-specific memory",
        "route_id": "route-123"
    }
    mock.get_routing_statistics.return_value = {
        "total_routes": 10,
        "successful_routes": 9,
        "failed_routes": 1,
        "average_confidence": 0.92
    }
    mock.health_check.return_value = {
        "status": "healthy",
        "last_check": "2024-01-01T00:00:00Z"
    }
    
    return mock


@pytest.fixture
def mock_memory_service():
    """Mock memory service for testing."""
    mock = AsyncMock(spec=MemoryService)
    
    # Configure default return values
    mock.store_memory.return_value = "memory-id-789"
    mock.get_memory.return_value = {
        "id": "memory-id-789",
        "content": "Test memory content",
        "metadata": {"agent_id": "test-agent"},
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z"
    }
    mock.search_memories.return_value = {
        "results": [
            {
                "id": "memory-id-789",
                "content": "Test memory content",
                "metadata": {"agent_id": "test-agent"},
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z"
            }
        ],
        "total": 1,
        "page": 1,
        "size": 10
    }
    mock.delete_memory.return_value = True
    mock.get_memory_stats.return_value = {
        "total_memories": 1,
        "total_size_bytes": 1024,
        "average_priority": 2.5,
        "last_updated": "2024-01-01T00:00:00Z"
    }
    
    return mock


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for testing."""
    mock = MagicMock()
    
    # Configure default responses
    mock.get.return_value.status_code = 200
    mock.get.return_value.json.return_value = {"status": "ok"}
    mock.post.return_value.status_code = 201
    mock.post.return_value.json.return_value = {"id": "created-id"}
    mock.put.return_value.status_code = 200
    mock.put.return_value.json.return_value = {"status": "updated"}
    mock.delete.return_value.status_code = 204
    
    return mock


@pytest.fixture
def mock_async_http_client():
    """Mock async HTTP client for testing."""
    mock = AsyncMock()
    
    # Configure default responses
    mock.get.return_value.status_code = 200
    mock.get.return_value.json.return_value = {"status": "ok"}
    mock.post.return_value.status_code = 201
    mock.post.return_value.json.return_value = {"id": "created-id"}
    mock.put.return_value.status_code = 200
    mock.put.return_value.json.return_value = {"status": "updated"}
    mock.delete.return_value.status_code = 204
    
    return mock


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    mock = MagicMock()
    mock.info = MagicMock()
    mock.debug = MagicMock()
    mock.warning = MagicMock()
    mock.error = MagicMock()
    mock.critical = MagicMock()
    return mock


@pytest.fixture
def mock_uuid():
    """Mock UUID generation for testing."""
    with patch('uuid.uuid4') as mock_uuid4:
        mock_uuid4.return_value.hex = "test-uuid-123456789"
        yield mock_uuid4


@pytest.fixture
def mock_datetime():
    """Mock datetime for testing."""
    with patch('datetime.datetime') as mock_dt:
        mock_dt.utcnow.return_value.isoformat.return_value = "2024-01-01T00:00:00Z"
        mock_dt.now.return_value.isoformat.return_value = "2024-01-01T00:00:00Z"
        yield mock_dt


@pytest.fixture
def mock_time():
    """Mock time for testing."""
    with patch('time.time') as mock_time:
        mock_time.return_value = 1704067200.0  # 2024-01-01T00:00:00Z
        yield mock_time


@pytest.fixture
def mock_os_environ():
    """Mock environment variables for testing."""
    with patch.dict('os.environ', {
        'TESTING': 'true',
        'DATABASE_URL': 'sqlite+aiosqlite:///:memory:',
        'CIPHER_API_URL': 'http://localhost:3001',
        'WEAVIATE_URL': 'http://localhost:8080',
        'LOG_LEVEL': 'DEBUG',
        'SECRET_KEY': 'test-secret-key-for-testing-only',
        'ALGORITHM': 'HS256',
        'ACCESS_TOKEN_EXPIRE_MINUTES': '30',
    }) as mock_env:
        yield mock_env


@pytest.fixture
def mock_file_system():
    """Mock file system operations for testing."""
    with patch('builtins.open', create=True) as mock_open:
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_file.read.return_value = "test file content"
        mock_file.write.return_value = None
        mock_open.return_value = mock_file
        yield mock_open


@pytest.fixture
def mock_async_sleep():
    """Mock async sleep for testing."""
    with patch('asyncio.sleep') as mock_sleep:
        mock_sleep.return_value = None
        yield mock_sleep


@pytest.fixture
def mock_retry():
    """Mock retry mechanism for testing."""
    with patch('tenacity.retry') as mock_retry:
        def retry_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        mock_retry.side_effect = retry_decorator
        yield mock_retry

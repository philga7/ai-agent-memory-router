"""
Pytest configuration and fixtures for AI Agent Memory Router
Adapted from test-automation-harness patterns for Python
"""

import asyncio
import os
import tempfile
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Import application components
from app.main import app
from app.core.database import get_db_session, Base
from app.core.config import get_settings
from app.services.memory_service import MemoryService
from app.services.routing_service import IntelligentRoutingService
from app.services.cipher_service import CipherService
from app.services.weaviate_service import WeaviateService

# Import test utilities
from tests.fixtures.database import test_engine, test_db_session, test_db_url
from tests.fixtures.mocks import (
    mock_cipher_service, mock_weaviate_service, mock_routing_service,
    mock_memory_service, mock_http_client, mock_async_http_client
)
from tests.fixtures.data import (
    sample_memory_store_create, sample_memory_store, sample_memory_search,
    sample_memory_search_response, sample_memory_stats, sample_agent,
    sample_agent_create, sample_agent_update, sample_context, sample_context_create
)
from tests.utils.config import test_config, test_settings, test_environment, environment_manager
from tests.utils.helpers import (
    TestHelper, AsyncTestHelper, DatabaseTestHelper, APITestHelper,
    MockHelper, PerformanceTestHelper, ValidationTestHelper, db_test_helper
)


# Event loop fixture is now imported from tests.fixtures.database
# Test settings fixture is now imported from tests.utils.config
# Test engine and database session fixtures are now imported from tests.fixtures.database


@pytest.fixture
def test_client(test_db_session) -> TestClient:
    """Create test client with database session override."""
    def override_get_db():
        return test_db_session
    
    app.dependency_overrides[get_db_session] = override_get_db
    
    client = TestClient(app)
    yield client
    
    app.dependency_overrides.clear()


# Mock service fixtures are now imported from tests.fixtures.mocks
# Sample data fixtures are now imported from tests.fixtures.data
# Test environment setup is now imported from tests.utils.config


@pytest.fixture
def memory_service_with_mocks(mock_cipher_service, mock_weaviate_service, mock_routing_service):
    """Create memory service with mocked dependencies."""
    service = MemoryService()
    service.cipher_service = mock_cipher_service
    service.weaviate_service = mock_weaviate_service
    service.routing_service = mock_routing_service
    return service


@pytest.fixture
def routing_service_with_mocks(mock_cipher_service, mock_weaviate_service):
    """Create routing service with mocked dependencies."""
    service = IntelligentRoutingService()
    service.cipher_service = mock_cipher_service
    service.weaviate_service = mock_weaviate_service
    return service


@pytest.fixture
def test_client_with_auth(test_client):
    """Create test client with authentication headers."""
    def add_auth_headers(headers: dict = None):
        auth_headers = {"Authorization": "Bearer test-token"}
        if headers:
            auth_headers.update(headers)
        return auth_headers
    
    test_client.add_auth_headers = add_auth_headers
    return test_client


@pytest.fixture
async def async_test_client():
    """Create async test client for testing."""
    from httpx import AsyncClient
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def test_database_cleanup(test_db_session):
    """Clean up test database after each test."""
    yield test_db_session
    
    # Clean up any test data
    try:
        # This would be implemented based on your specific database cleanup needs
        pass
    except Exception:
        pass


@pytest.fixture
def performance_monitor():
    """Monitor performance during tests."""
    import time
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        def get_duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    monitor = PerformanceMonitor()
    monitor.start()
    yield monitor
    monitor.stop()


@pytest.fixture
def test_logger():
    """Create test logger for testing."""
    import logging
    
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    
    # Create a test handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


@pytest.fixture
def mock_external_services():
    """Mock all external services for testing."""
    with pytest.MonkeyPatch() as mp:
        # Mock external HTTP calls
        mp.setattr("httpx.AsyncClient.get", AsyncMock(return_value=MagicMock(status_code=200, json=lambda: {})))
        mp.setattr("httpx.AsyncClient.post", AsyncMock(return_value=MagicMock(status_code=201, json=lambda: {})))
        mp.setattr("httpx.AsyncClient.put", AsyncMock(return_value=MagicMock(status_code=200, json=lambda: {})))
        mp.setattr("httpx.AsyncClient.delete", AsyncMock(return_value=MagicMock(status_code=204)))
        
        yield mp


@pytest.fixture
def test_data_cleanup():
    """Clean up test data after tests."""
    cleanup_items = []
    
    def add_cleanup(item):
        cleanup_items.append(item)
    
    yield add_cleanup
    
    # Perform cleanup
    for item in cleanup_items:
        try:
            if hasattr(item, 'cleanup'):
                item.cleanup()
            elif hasattr(item, 'delete'):
                item.delete()
        except Exception:
            pass


# Helper fixtures
@pytest.fixture
def test_helper():
    """Create test helper instance."""
    return TestHelper()


@pytest.fixture
def async_test_helper():
    """Create async test helper instance."""
    return AsyncTestHelper()


@pytest.fixture
def database_test_helper():
    """Create database test helper instance."""
    return DatabaseTestHelper()


@pytest.fixture
def api_test_helper():
    """Create API test helper instance."""
    return APITestHelper()


@pytest.fixture
def mock_helper():
    """Create mock helper instance."""
    return MockHelper()


@pytest.fixture
def performance_test_helper():
    """Create performance test helper instance."""
    return PerformanceTestHelper()


@pytest.fixture
def validation_test_helper():
    """Create validation test helper instance."""
    return ValidationTestHelper()


# Test markers for categorization
pytestmark = [
    pytest.mark.asyncio,
]

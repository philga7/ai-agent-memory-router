"""
Test configuration utilities for AI Agent Memory Router
"""

import os
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path

import pytest


class TestConfig:
    """Test configuration class."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent.parent
        self.project_root = self.test_dir.parent
        self.temp_dir = None
    
    def setup_temp_directory(self) -> str:
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp(prefix="ai_agent_memory_router_test_")
        return self.temp_dir
    
    def cleanup_temp_directory(self) -> None:
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def get_test_database_url(self) -> str:
        """Get test database URL."""
        return "sqlite+aiosqlite:///:memory:"
    
    def get_test_file_database_url(self) -> str:
        """Get test file database URL."""
        if not self.temp_dir:
            self.setup_temp_directory()
        return f"sqlite+aiosqlite:///{self.temp_dir}/test.db"
    
    def get_test_settings(self) -> Dict[str, Any]:
        """Get test settings."""
        return {
            "TESTING": "true",
            "DATABASE_URL": self.get_test_database_url(),
            "CIPHER_API_URL": "http://localhost:3001",
            "WEAVIATE_URL": "http://localhost:8080",
            "LOG_LEVEL": "DEBUG",
            "SECRET_KEY": "test-secret-key-for-testing-only",
            "ALGORITHM": "HS256",
            "ACCESS_TOKEN_EXPIRE_MINUTES": "30",
            "REDIS_URL": "redis://localhost:6379/0",
            "CHROMA_URL": "http://localhost:8000",
            "ENVIRONMENT": "test"
        }
    
    def get_integration_test_settings(self) -> Dict[str, Any]:
        """Get integration test settings."""
        return {
            **self.get_test_settings(),
            "DATABASE_URL": self.get_test_file_database_url(),
            "ENABLE_EXTERNAL_SERVICES": "true"
        }
    
    def get_e2e_test_settings(self) -> Dict[str, Any]:
        """Get end-to-end test settings."""
        return {
            **self.get_test_settings(),
            "DATABASE_URL": self.get_test_file_database_url(),
            "ENABLE_EXTERNAL_SERVICES": "true",
            "ENABLE_REAL_SERVICES": "true"
        }


@pytest.fixture(scope="session")
def test_config():
    """Provide TestConfig instance."""
    config = TestConfig()
    yield config
    config.cleanup_temp_directory()


@pytest.fixture
def test_settings(test_config):
    """Provide test settings."""
    return test_config.get_test_settings()


@pytest.fixture
def integration_test_settings(test_config):
    """Provide integration test settings."""
    return test_config.get_integration_test_settings()


@pytest.fixture
def e2e_test_settings(test_config):
    """Provide end-to-end test settings."""
    return test_config.get_e2e_test_settings()


@pytest.fixture
def test_database_url(test_config):
    """Provide test database URL."""
    return test_config.get_test_database_url()


@pytest.fixture
def test_file_database_url(test_config):
    """Provide test file database URL."""
    return test_config.get_test_file_database_url()


@pytest.fixture
def temp_test_directory(test_config):
    """Provide temporary test directory."""
    temp_dir = test_config.setup_temp_directory()
    yield temp_dir
    test_config.cleanup_temp_directory()


class EnvironmentManager:
    """Manage environment variables for tests."""
    
    def __init__(self):
        self.original_env = {}
        self.test_env = {}
    
    def set_test_environment(self, env_vars: Dict[str, str]) -> None:
        """Set test environment variables."""
        self.test_env = env_vars.copy()
        for key, value in env_vars.items():
            self.original_env[key] = os.environ.get(key)
            os.environ[key] = value
    
    def restore_environment(self) -> None:
        """Restore original environment variables."""
        for key in self.test_env.keys():
            if key in self.original_env:
                original_value = self.original_env[key]
                if original_value is None:
                    # Remove the environment variable if it was None originally
                    if key in os.environ:
                        del os.environ[key]
                else:
                    os.environ[key] = original_value
            elif key in os.environ:
                del os.environ[key]
        self.original_env.clear()
        self.test_env.clear()


@pytest.fixture
def environment_manager():
    """Provide EnvironmentManager instance."""
    manager = EnvironmentManager()
    yield manager
    manager.restore_environment()


@pytest.fixture
def test_environment(test_config, environment_manager):
    """Set up test environment."""
    env_vars = test_config.get_test_settings()
    environment_manager.set_test_environment(env_vars)
    yield env_vars
    environment_manager.restore_environment()


@pytest.fixture
def integration_test_environment(test_config, environment_manager):
    """Set up integration test environment."""
    env_vars = test_config.get_integration_test_settings()
    environment_manager.set_test_environment(env_vars)
    yield env_vars
    environment_manager.restore_environment()


@pytest.fixture
def e2e_test_environment(test_config, environment_manager):
    """Set up end-to-end test environment."""
    env_vars = test_config.get_e2e_test_settings()
    environment_manager.set_test_environment(env_vars)
    yield env_vars
    environment_manager.restore_environment()


class TestDataManager:
    """Manage test data for different test scenarios."""
    
    def __init__(self):
        self.test_data = {}
    
    def add_test_data(self, key: str, data: Any) -> None:
        """Add test data."""
        self.test_data[key] = data
    
    def get_test_data(self, key: str) -> Any:
        """Get test data."""
        return self.test_data.get(key)
    
    def clear_test_data(self) -> None:
        """Clear all test data."""
        self.test_data.clear()
    
    def get_memory_test_data(self) -> Dict[str, Any]:
        """Get memory test data."""
        return {
            "sample_memory": {
                "content": "This is a test memory",
                "metadata": {
                    "project_id": "test-project",
                    "agent_id": "test-agent",
                    "memory_type": "knowledge",
                    "tags": ["testing", "unit-test"],
                    "importance": 5
                },
                "priority": "normal"
            },
            "sample_memory_list": [
                {
                    "content": f"Test memory {i}",
                    "metadata": {
                        "project_id": "test-project",
                        "agent_id": f"test-agent-{i}",
                        "memory_type": "knowledge",
                        "tags": ["testing", "unit-test"],
                        "importance": i % 5 + 1
                    },
                    "priority": "normal"
                }
                for i in range(1, 6)
            ]
        }
    
    def get_agent_test_data(self) -> Dict[str, Any]:
        """Get agent test data."""
        return {
            "sample_agent": {
                "name": "Test Agent",
                "description": "A test agent for unit testing",
                "agent_type": "memory_router",
                "capabilities": ["memory_storage", "memory_retrieval", "routing"],
                "metadata": {
                    "version": "1.0.0",
                    "environment": "test"
                }
            },
            "sample_agent_list": [
                {
                    "name": f"Test Agent {i}",
                    "description": f"A test agent {i} for unit testing",
                    "agent_type": "memory_router",
                    "capabilities": ["memory_storage", "memory_retrieval", "routing"],
                    "metadata": {
                        "version": "1.0.0",
                        "environment": "test"
                    }
                }
                for i in range(1, 6)
            ]
        }
    
    def get_context_test_data(self) -> Dict[str, Any]:
        """Get context test data."""
        return {
            "sample_context": {
                "conversation_id": "test-conversation-123",
                "agent_id": "test-agent-123",
                "context_type": "conversation",
                "participants": ["test-agent-123", "test-user-123"],
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, how are you?",
                        "timestamp": "2024-01-01T00:00:00Z"
                    }
                ],
                "tags": ["test", "conversation"],
                "metadata": {
                    "session_id": "session-123",
                    "environment": "test"
                }
            },
            "sample_context_list": [
                {
                    "conversation_id": f"test-conversation-{i}",
                    "agent_id": f"test-agent-{i}",
                    "context_type": "conversation",
                    "participants": [f"test-agent-{i}", f"test-user-{i}"],
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Hello, how are you? (Message {i})",
                            "timestamp": "2024-01-01T00:00:00Z"
                        }
                    ],
                    "tags": ["test", "conversation"],
                    "metadata": {
                        "session_id": f"session-{i}",
                        "environment": "test"
                    }
                }
                for i in range(1, 6)
            ]
        }


@pytest.fixture
def test_data_manager():
    """Provide TestDataManager instance."""
    manager = TestDataManager()
    yield manager
    manager.clear_test_data()


@pytest.fixture
def memory_test_data(test_data_manager):
    """Provide memory test data."""
    return test_data_manager.get_memory_test_data()


@pytest.fixture
def agent_test_data(test_data_manager):
    """Provide agent test data."""
    return test_data_manager.get_agent_test_data()


@pytest.fixture
def context_test_data(test_data_manager):
    """Provide context test data."""
    return test_data_manager.get_context_test_data()

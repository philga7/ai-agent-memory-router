"""
Unit tests for test fixtures and utilities
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from tests.fixtures.database import test_engine, test_db_session
from tests.fixtures.mocks import (
    mock_cipher_service, mock_weaviate_service, mock_routing_service,
    mock_memory_service, mock_http_client, mock_async_http_client
)
from tests.fixtures.data import (
    sample_memory_store_create, sample_memory_store, sample_memory_search,
    sample_memory_search_response, sample_memory_stats, sample_agent,
    sample_agent_create, sample_agent_update, sample_context, sample_context_create
)
from tests.utils.config import test_config, test_settings, test_environment
from tests.utils.helpers import (
    TestHelper, AsyncTestHelper, DatabaseTestHelper, APITestHelper,
    MockHelper, PerformanceTestHelper, ValidationTestHelper
)
from tests.utils.database import TestDatabaseManager, DatabaseTestHelper as DBHelper


class TestDatabaseFixtures:
    """Test database fixtures."""
    
    @pytest.mark.asyncio
    async def test_test_engine_fixture(self, test_engine):
        """Test that test engine fixture works."""
        # test_engine is an async generator, we need to get the actual engine
        async for engine in test_engine:
            assert engine is not None
            assert hasattr(engine, 'dispose')
            break
    
    @pytest.mark.asyncio
    async def test_test_db_session_fixture(self, test_db_session):
        """Test that test database session fixture works."""
        # test_db_session is an async generator, we need to get the actual session
        async for session in test_db_session:
            assert session is not None
            assert hasattr(session, 'commit')
            assert hasattr(session, 'rollback')
            break
    
    @pytest.mark.asyncio
    async def test_database_manager(self, test_engine):
        """Test database manager functionality."""
        # test_engine is an async generator, we need to get the actual engine
        async for engine in test_engine:
            manager = TestDatabaseManager(engine)

            # Test session creation
            session = await manager.create_session()
            assert session is not None

            # Test table counts
            counts = await manager.get_table_counts()
            assert isinstance(counts, dict)
            assert 'memory_stores' in counts
            assert 'agents' in counts
            assert 'contexts' in counts

            # Test data integrity
            integrity = await manager.verify_data_integrity()
            assert isinstance(integrity, dict)
            assert 'orphaned_memory_stores' in integrity
            assert 'invalid_timestamps' in integrity
            break


class TestMockFixtures:
    """Test mock fixtures."""
    
    def test_mock_cipher_service_fixture(self, mock_cipher_service):
        """Test that mock cipher service fixture works."""
        assert mock_cipher_service is not None
        assert hasattr(mock_cipher_service, 'store_memory')
        assert hasattr(mock_cipher_service, 'retrieve_memory')
        assert hasattr(mock_cipher_service, 'search_memories')
        assert hasattr(mock_cipher_service, 'delete_memory')
        assert hasattr(mock_cipher_service, 'update_memory')
        assert hasattr(mock_cipher_service, 'create_project')
        assert hasattr(mock_cipher_service, 'get_project_info')
    
    def test_mock_weaviate_service_fixture(self, mock_weaviate_service):
        """Test that mock weaviate service fixture works."""
        assert mock_weaviate_service is not None
        assert hasattr(mock_weaviate_service, 'store_memory')
        assert hasattr(mock_weaviate_service, 'retrieve_memory')
        assert hasattr(mock_weaviate_service, 'search_memories')
        assert hasattr(mock_weaviate_service, 'delete_memory')
        assert hasattr(mock_weaviate_service, 'get_memory_stats')
    
    def test_mock_routing_service_fixture(self, mock_routing_service):
        """Test that mock routing service fixture works."""
        assert mock_routing_service is not None
        assert hasattr(mock_routing_service, 'route_memory')
        assert hasattr(mock_routing_service, 'get_routing_statistics')
        assert hasattr(mock_routing_service, 'health_check')
    
    def test_mock_memory_service_fixture(self, mock_memory_service):
        """Test that mock memory service fixture works."""
        assert mock_memory_service is not None
        assert hasattr(mock_memory_service, 'store_memory')
        assert hasattr(mock_memory_service, 'get_memory')
        assert hasattr(mock_memory_service, 'search_memories')
        assert hasattr(mock_memory_service, 'delete_memory')
        assert hasattr(mock_memory_service, 'get_memory_stats')
    
    def test_mock_http_client_fixture(self, mock_http_client):
        """Test that mock HTTP client fixture works."""
        assert mock_http_client is not None
        assert hasattr(mock_http_client, 'get')
        assert hasattr(mock_http_client, 'post')
        assert hasattr(mock_http_client, 'put')
        assert hasattr(mock_http_client, 'delete')
    
    def test_mock_async_http_client_fixture(self, mock_async_http_client):
        """Test that mock async HTTP client fixture works."""
        assert mock_async_http_client is not None
        assert hasattr(mock_async_http_client, 'get')
        assert hasattr(mock_async_http_client, 'post')
        assert hasattr(mock_async_http_client, 'put')
        assert hasattr(mock_async_http_client, 'delete')


class TestDataFixtures:
    """Test data fixtures."""
    
    def test_sample_memory_store_create_fixture(self, sample_memory_store_create):
        """Test that sample memory store create fixture works."""
        assert sample_memory_store_create is not None
        assert hasattr(sample_memory_store_create, 'content')
        assert hasattr(sample_memory_store_create, 'source')
        assert hasattr(sample_memory_store_create, 'memory_type')
        assert hasattr(sample_memory_store_create, 'importance')
        assert hasattr(sample_memory_store_create, 'access_control')
    
    def test_sample_memory_store_fixture(self, sample_memory_store):
        """Test that sample memory store fixture works."""
        assert sample_memory_store is not None
        assert hasattr(sample_memory_store, 'id')
        assert hasattr(sample_memory_store, 'content')
        assert hasattr(sample_memory_store, 'source')
        assert hasattr(sample_memory_store, 'memory_type')
        assert hasattr(sample_memory_store, 'importance')
        assert hasattr(sample_memory_store, 'access_control')
        assert hasattr(sample_memory_store, 'created_at')
        assert hasattr(sample_memory_store, 'updated_at')
    
    def test_sample_memory_search_fixture(self, sample_memory_search):
        """Test that sample memory search fixture works."""
        assert sample_memory_search is not None
        assert hasattr(sample_memory_search, 'query')
        assert hasattr(sample_memory_search, 'agent_id')
        assert hasattr(sample_memory_search, 'memory_type')
        assert hasattr(sample_memory_search, 'limit')
        assert hasattr(sample_memory_search, 'offset')
    
    def test_sample_memory_search_response_fixture(self, sample_memory_search_response):
        """Test that sample memory search response fixture works."""
        assert sample_memory_search_response is not None
        assert hasattr(sample_memory_search_response, 'results')
        assert hasattr(sample_memory_search_response, 'total')
        assert hasattr(sample_memory_search_response, 'query')
        assert hasattr(sample_memory_search_response, 'execution_time')
    
    def test_sample_memory_stats_fixture(self, sample_memory_stats):
        """Test that sample memory stats fixture works."""
        assert sample_memory_stats is not None
        assert hasattr(sample_memory_stats, 'total_memories')
        assert hasattr(sample_memory_stats, 'total_size_bytes')
        assert hasattr(sample_memory_stats, 'average_priority')
        assert hasattr(sample_memory_stats, 'memories_by_type')
        assert hasattr(sample_memory_stats, 'memories_by_agent')
    
    def test_sample_agent_fixture(self, sample_agent):
        """Test that sample agent fixture works."""
        assert sample_agent is not None
        assert hasattr(sample_agent, 'id')
        assert hasattr(sample_agent, 'name')
        assert hasattr(sample_agent, 'description')
        assert hasattr(sample_agent, 'agent_type')
        assert hasattr(sample_agent, 'status')
        assert hasattr(sample_agent, 'capabilities')
        assert hasattr(sample_agent, 'metadata')
        assert hasattr(sample_agent, 'created_at')
        assert hasattr(sample_agent, 'updated_at')
    
    def test_sample_agent_create_fixture(self, sample_agent_create):
        """Test that sample agent create fixture works."""
        assert sample_agent_create is not None
        assert hasattr(sample_agent_create, 'name')
        assert hasattr(sample_agent_create, 'description')
        assert hasattr(sample_agent_create, 'agent_type')
        assert hasattr(sample_agent_create, 'capabilities')
        assert hasattr(sample_agent_create, 'metadata')
    
    def test_sample_agent_update_fixture(self, sample_agent_update):
        """Test that sample agent update fixture works."""
        assert sample_agent_update is not None
        assert hasattr(sample_agent_update, 'name')
        assert hasattr(sample_agent_update, 'description')
        assert hasattr(sample_agent_update, 'agent_type')
        assert hasattr(sample_agent_update, 'metadata')
    
    def test_sample_context_fixture(self, sample_context):
        """Test that sample context fixture works."""
        assert sample_context is not None
        assert hasattr(sample_context, 'id')
        assert hasattr(sample_context, 'title')
        assert hasattr(sample_context, 'description')
        assert hasattr(sample_context, 'context_type')
        assert hasattr(sample_context, 'status')
        assert hasattr(sample_context, 'participants')
        assert hasattr(sample_context, 'messages')
        assert hasattr(sample_context, 'tags')
        assert hasattr(sample_context, 'metadata')
        assert hasattr(sample_context, 'created_at')
        assert hasattr(sample_context, 'updated_at')
    
    def test_sample_context_create_fixture(self, sample_context_create):
        """Test that sample context create fixture works."""
        assert sample_context_create is not None
        assert hasattr(sample_context_create, 'title')
        assert hasattr(sample_context_create, 'description')
        assert hasattr(sample_context_create, 'context_type')
        assert hasattr(sample_context_create, 'participants')
        assert hasattr(sample_context_create, 'messages')
        assert hasattr(sample_context_create, 'tags')
        assert hasattr(sample_context_create, 'metadata')


class TestConfigFixtures:
    """Test configuration fixtures."""
    
    def test_test_config_fixture(self, test_config):
        """Test that test config fixture works."""
        assert test_config is not None
        assert hasattr(test_config, 'get_test_settings')
        assert hasattr(test_config, 'get_test_database_url')
        assert hasattr(test_config, 'setup_temp_directory')
        assert hasattr(test_config, 'cleanup_temp_directory')
    
    def test_test_settings_fixture(self, test_settings):
        """Test that test settings fixture works."""
        assert test_settings is not None
        assert isinstance(test_settings, dict)
        assert 'TESTING' in test_settings
        assert 'DATABASE_URL' in test_settings
        assert 'CIPHER_API_URL' in test_settings
        assert 'WEAVIATE_URL' in test_settings
        assert 'LOG_LEVEL' in test_settings
        assert 'SECRET_KEY' in test_settings
    
    def test_test_environment_fixture(self, test_environment):
        """Test that test environment fixture works."""
        assert test_environment is not None
        assert isinstance(test_environment, dict)
        assert 'TESTING' in test_environment
        assert 'DATABASE_URL' in test_environment


class TestHelperFixtures:
    """Test helper fixtures."""
    
    def test_test_helper_fixture(self, test_helper):
        """Test that test helper fixture works."""
        assert test_helper is not None
        assert hasattr(test_helper, 'generate_test_id')
        assert hasattr(test_helper, 'generate_test_timestamp')
        assert hasattr(test_helper, 'create_test_metadata')
        assert hasattr(test_helper, 'assert_response_structure')
        assert hasattr(test_helper, 'assert_error_response')
        assert hasattr(test_helper, 'assert_success_response')
    
    def test_async_test_helper_fixture(self, async_test_helper):
        """Test that async test helper fixture works."""
        assert async_test_helper is not None
        assert hasattr(async_test_helper, 'wait_for_condition')
        assert hasattr(async_test_helper, 'retry_async_operation')
    
    def test_database_test_helper_fixture(self, db_test_helper):
        """Test that database test helper fixture works."""
        assert db_test_helper is not None
        assert hasattr(db_test_helper, 'assert_database_record')
        assert hasattr(db_test_helper, 'assert_timestamps')
        assert hasattr(db_test_helper, 'assert_id_field')
    
    def test_api_test_helper_fixture(self, api_test_helper):
        """Test that API test helper fixture works."""
        assert api_test_helper is not None
        assert hasattr(api_test_helper, 'assert_http_status')
        assert hasattr(api_test_helper, 'assert_json_response')
        assert hasattr(api_test_helper, 'assert_pagination_response')
        assert hasattr(api_test_helper, 'create_auth_headers')
        assert hasattr(api_test_helper, 'create_content_type_headers')
    
    def test_mock_helper_fixture(self, mock_helper):
        """Test that mock helper fixture works."""
        assert mock_helper is not None
        assert hasattr(mock_helper, 'create_mock_response')
        assert hasattr(mock_helper, 'create_mock_error_response')
        assert hasattr(mock_helper, 'create_mock_success_response')
    
    def test_performance_test_helper_fixture(self, performance_test_helper):
        """Test that performance test helper fixture works."""
        assert performance_test_helper is not None
        assert hasattr(performance_test_helper, 'measure_execution_time')
        assert hasattr(performance_test_helper, 'measure_async_execution_time')
        assert hasattr(performance_test_helper, 'assert_performance_threshold')
    
    def test_validation_test_helper_fixture(self, validation_test_helper):
        """Test that validation test helper fixture works."""
        assert validation_test_helper is not None
        assert hasattr(validation_test_helper, 'assert_required_fields')
        assert hasattr(validation_test_helper, 'assert_field_types')
        assert hasattr(validation_test_helper, 'assert_field_values')
        assert hasattr(validation_test_helper, 'assert_field_ranges')


class TestHelperFunctionality:
    """Test helper functionality."""
    
    def test_test_helper_generate_test_id(self, test_helper):
        """Test test helper ID generation."""
        test_id = test_helper.generate_test_id()
        assert isinstance(test_id, str)
        assert test_id.startswith("test-")
        
        custom_id = test_helper.generate_test_id("custom")
        assert isinstance(custom_id, str)
        assert custom_id.startswith("custom-")
    
    def test_test_helper_generate_test_timestamp(self, test_helper):
        """Test test helper timestamp generation."""
        timestamp = test_helper.generate_test_timestamp()
        assert timestamp is not None
        assert hasattr(timestamp, 'isoformat')
    
    def test_test_helper_create_test_metadata(self, test_helper):
        """Test test helper metadata creation."""
        metadata = test_helper.create_test_metadata()
        assert isinstance(metadata, dict)
        assert 'test_id' in metadata
        assert 'created_at' in metadata
        assert 'environment' in metadata
        
        custom_metadata = test_helper.create_test_metadata(custom_field="custom_value")
        assert custom_metadata['custom_field'] == "custom_value"
    
    def test_test_helper_assert_response_structure(self, test_helper):
        """Test test helper response structure assertion."""
        response_data = {
            "success": True,
            "data": {"id": "123"},
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        # Should not raise
        test_helper.assert_response_structure(response_data, ["success", "data", "timestamp"])
        
        # Should raise
        with pytest.raises(AssertionError):
            test_helper.assert_response_structure(response_data, ["missing_field"])
    
    def test_test_helper_assert_error_response(self, test_helper):
        """Test test helper error response assertion."""
        error_response = {
            "error": "Test error",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        # Should not raise
        test_helper.assert_error_response(error_response)
        
        # Should raise
        with pytest.raises(AssertionError):
            test_helper.assert_error_response({"success": True})
    
    def test_test_helper_assert_success_response(self, test_helper):
        """Test test helper success response assertion."""
        success_response = {
            "success": True,
            "data": {"id": "123"},
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        # Should not raise
        test_helper.assert_success_response(success_response)
        
        # Should raise
        with pytest.raises(AssertionError):
            test_helper.assert_success_response({"error": "Test error"})
    
    @pytest.mark.asyncio
    async def test_async_test_helper_wait_for_condition(self, async_test_helper):
        """Test async test helper wait for condition."""
        # Test successful condition
        condition_met = False
        
        async def condition():
            return condition_met
        
        # Start the condition check
        task = asyncio.create_task(async_test_helper.wait_for_condition(condition, timeout=1.0))
        
        # Set condition to True after a short delay
        await asyncio.sleep(0.1)
        condition_met = True
        
        # Should complete without error
        await task
    
    @pytest.mark.asyncio
    async def test_async_test_helper_retry_operation(self, async_test_helper):
        """Test async test helper retry operation."""
        call_count = 0
        
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Test error")
            return "success"
        
        result = await async_test_helper.retry_async_operation(failing_operation, max_retries=3)
        assert result == "success"
        assert call_count == 3
    
    def test_mock_helper_create_mock_response(self, mock_helper):
        """Test mock helper response creation."""
        response = mock_helper.create_mock_response(
            status_code=201,
            data={"id": "123"},
            headers={"Content-Type": "application/json"}
        )
        
        assert response["status_code"] == 201
        assert response["data"]["id"] == "123"
        assert response["headers"]["Content-Type"] == "application/json"
    
    def test_mock_helper_create_mock_error_response(self, mock_helper):
        """Test mock helper error response creation."""
        error_response = mock_helper.create_mock_error_response(
            error_code="TEST_ERROR",
            error_message="Test error message"
        )
        
        assert error_response["error"] == "Test error message"
        assert error_response["code"] == "TEST_ERROR"
        assert "timestamp" in error_response
    
    def test_mock_helper_create_mock_success_response(self, mock_helper):
        """Test mock helper success response creation."""
        success_response = mock_helper.create_mock_success_response(
            data={"id": "123"},
            message="Operation completed"
        )
        
        assert success_response["success"] is True
        assert success_response["message"] == "Operation completed"
        assert success_response["data"]["id"] == "123"
        assert "timestamp" in success_response

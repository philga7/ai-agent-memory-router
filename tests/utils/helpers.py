"""
Test helper utilities for AI Agent Memory Router
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


class TestHelper:
    """Helper class for test utilities."""
    
    @staticmethod
    def generate_test_id(prefix: str = "test") -> str:
        """Generate a test ID with prefix."""
        return f"{prefix}-{uuid4().hex[:8]}"
    
    @staticmethod
    def generate_test_timestamp() -> datetime:
        """Generate a test timestamp."""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def create_test_metadata(**kwargs) -> Dict[str, Any]:
        """Create test metadata with default values."""
        default_metadata = {
            "test_id": TestHelper.generate_test_id(),
            "created_at": TestHelper.generate_test_timestamp().isoformat(),
            "environment": "test"
        }
        default_metadata.update(kwargs)
        return default_metadata
    
    @staticmethod
    def assert_response_structure(response_data: Dict[str, Any], expected_fields: List[str]) -> None:
        """Assert that response has expected structure."""
        for field in expected_fields:
            assert field in response_data, f"Missing field: {field}"
    
    @staticmethod
    def assert_error_response(response_data: Dict[str, Any]) -> None:
        """Assert that response is an error response."""
        assert "error" in response_data or "detail" in response_data
        assert "timestamp" in response_data or "created_at" in response_data
    
    @staticmethod
    def assert_success_response(response_data: Dict[str, Any]) -> None:
        """Assert that response is a success response."""
        assert "success" in response_data or "data" in response_data
        assert "timestamp" in response_data or "created_at" in response_data


class AsyncTestHelper:
    """Helper class for async test utilities."""
    
    @staticmethod
    async def wait_for_condition(
        condition_func,
        timeout: float = 5.0,
        interval: float = 0.1,
        error_message: str = "Condition not met within timeout"
    ) -> None:
        """Wait for a condition to be true."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
                return
            await asyncio.sleep(interval)
        raise TimeoutError(error_message)
    
    @staticmethod
    async def retry_async_operation(
        operation,
        max_retries: int = 3,
        delay: float = 0.1,
        backoff_factor: float = 2.0
    ) -> Any:
        """Retry an async operation with exponential backoff."""
        last_exception = None
        current_delay = delay
        
        for attempt in range(max_retries + 1):
            try:
                return await operation()
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor
                else:
                    raise last_exception


class DatabaseTestHelper:
    """Helper class for database test utilities."""
    
    @staticmethod
    def assert_database_record(record: Any, expected_fields: List[str]) -> None:
        """Assert that database record has expected fields."""
        for field in expected_fields:
            assert hasattr(record, field), f"Missing field: {field}"
    
    @staticmethod
    def assert_timestamps(record: Any) -> None:
        """Assert that record has proper timestamps."""
        assert hasattr(record, 'created_at') or hasattr(record, 'created_at')
        assert hasattr(record, 'updated_at') or hasattr(record, 'updated_at')
    
    @staticmethod
    def assert_id_field(record: Any) -> None:
        """Assert that record has an ID field."""
        assert hasattr(record, 'id') or hasattr(record, 'id')
        assert getattr(record, 'id', None) is not None


class APITestHelper:
    """Helper class for API test utilities."""
    
    @staticmethod
    def assert_http_status(response, expected_status: int) -> None:
        """Assert HTTP response status."""
        assert response.status_code == expected_status, f"Expected {expected_status}, got {response.status_code}"
    
    @staticmethod
    def assert_json_response(response) -> Dict[str, Any]:
        """Assert response is JSON and return parsed data."""
        assert response.headers.get("content-type", "").startswith("application/json")
        return response.json()
    
    @staticmethod
    def assert_pagination_response(response_data: Dict[str, Any]) -> None:
        """Assert response has pagination structure."""
        required_fields = ["results", "total", "page", "size"]
        for field in required_fields:
            assert field in response_data, f"Missing pagination field: {field}"
    
    @staticmethod
    def create_auth_headers(token: str) -> Dict[str, str]:
        """Create authorization headers."""
        return {"Authorization": f"Bearer {token}"}
    
    @staticmethod
    def create_content_type_headers(content_type: str = "application/json") -> Dict[str, str]:
        """Create content type headers."""
        return {"Content-Type": content_type}


class MockHelper:
    """Helper class for mock utilities."""
    
    @staticmethod
    def create_mock_response(
        status_code: int = 200,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Create a mock HTTP response."""
        return {
            "status_code": status_code,
            "data": data or {},
            "headers": headers or {}
        }
    
    @staticmethod
    def create_mock_error_response(
        error_code: str = "TEST_ERROR",
        error_message: str = "Test error message"
    ) -> Dict[str, Any]:
        """Create a mock error response."""
        return {
            "error": error_message,
            "code": error_code,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    @staticmethod
    def create_mock_success_response(
        data: Optional[Dict[str, Any]] = None,
        message: str = "Operation completed successfully"
    ) -> Dict[str, Any]:
        """Create a mock success response."""
        return {
            "success": True,
            "message": message,
            "data": data or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class PerformanceTestHelper:
    """Helper class for performance test utilities."""
    
    @staticmethod
    def measure_execution_time(func):
        """Decorator to measure function execution time."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time:.4f} seconds")
            return result
        return wrapper
    
    @staticmethod
    async def measure_async_execution_time(func):
        """Decorator to measure async function execution time."""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Async execution time: {execution_time:.4f} seconds")
            return result
        return wrapper
    
    @staticmethod
    def assert_performance_threshold(execution_time: float, threshold: float) -> None:
        """Assert that execution time is within threshold."""
        assert execution_time <= threshold, f"Execution time {execution_time:.4f}s exceeds threshold {threshold}s"


class ValidationTestHelper:
    """Helper class for validation test utilities."""
    
    @staticmethod
    def assert_required_fields(data: Dict[str, Any], required_fields: List[str]) -> None:
        """Assert that data has all required fields."""
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
    
    @staticmethod
    def assert_field_types(data: Dict[str, Any], field_types: Dict[str, type]) -> None:
        """Assert that fields have correct types."""
        for field, expected_type in field_types.items():
            assert field in data, f"Missing field: {field}"
            assert isinstance(data[field], expected_type), f"Field {field} should be {expected_type}, got {type(data[field])}"
    
    @staticmethod
    def assert_field_values(data: Dict[str, Any], field_values: Dict[str, Any]) -> None:
        """Assert that fields have expected values."""
        for field, expected_value in field_values.items():
            assert field in data, f"Missing field: {field}"
            assert data[field] == expected_value, f"Field {field} should be {expected_value}, got {data[field]}"
    
    @staticmethod
    def assert_field_ranges(data: Dict[str, Any], field_ranges: Dict[str, tuple]) -> None:
        """Assert that numeric fields are within expected ranges."""
        for field, (min_val, max_val) in field_ranges.items():
            assert field in data, f"Missing field: {field}"
            assert min_val <= data[field] <= max_val, f"Field {field} should be between {min_val} and {max_val}, got {data[field]}"


# Pytest fixtures for common test scenarios
@pytest.fixture
def test_helper():
    """Provide TestHelper instance."""
    return TestHelper


@pytest.fixture
def async_test_helper():
    """Provide AsyncTestHelper instance."""
    return AsyncTestHelper


@pytest.fixture
def db_test_helper():
    """Provide DatabaseTestHelper instance."""
    return DatabaseTestHelper


@pytest.fixture
def api_test_helper():
    """Provide APITestHelper instance."""
    return APITestHelper


@pytest.fixture
def mock_helper():
    """Provide MockHelper instance."""
    return MockHelper


@pytest.fixture
def performance_test_helper():
    """Provide PerformanceTestHelper instance."""
    return PerformanceTestHelper


@pytest.fixture
def validation_test_helper():
    """Provide ValidationTestHelper instance."""
    return ValidationTestHelper

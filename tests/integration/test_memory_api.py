"""
Integration tests for Memory API endpoints.

This module tests the complete memory management API including routing,
storage, retrieval, and search functionality with database operations.
"""

import pytest
import time
from typing import Dict, Any, List
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from app.core.database import get_db_session
from app.models.memory import MemoryStore, MemoryRoute
from app.services.memory_service import MemoryService
from app.services.cipher_service import CipherService


@pytest.mark.integration
class TestMemoryRoutingAPI:
    """Test memory routing API endpoints."""
    
    @pytest.mark.asyncio
    async def test_route_memory_success(self, test_client: TestClient):
        """Test successful memory routing between agents."""
        # Arrange
        route_request = {
            "source_agent_id": "agent_001",
            "target_agent_ids": ["agent_002", "agent_003"],
            "memory_content": "Important project update: milestone achieved",
            "priority": "high",
            "context": {
                "project_id": "proj_123",
                "urgency": "immediate"
            }
        }
        
        # Act
        response = test_client.post("/api/v1/memory/route", json=route_request)
        
        # Assert
        assert response.status_code == 201
        data = response.json()
        
        assert "route_id" in data
        assert data["source_agent_id"] == "agent_001"
        assert data["target_agent_ids"] == ["agent_002", "agent_003"]
        assert data["status"] == "routed"
        assert data["priority"] == "high"
        assert data["context"]["project_id"] == "proj_123"
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_route_memory_validation_error(self, test_client: TestClient):
        """Test memory routing with invalid input data."""
        # Arrange
        invalid_request = {
            "source_agent_id": "",  # Empty agent ID
            "target_agent_ids": [],  # Empty target list
            "memory_content": "",  # Empty content
            "priority": "invalid_priority"
        }
        
        # Act
        response = test_client.post("/api/v1/memory/route", json=invalid_request)
        
        # Assert
        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert "Validation error" in data["error"]
    
    @pytest.mark.asyncio
    async def test_route_memory_large_content(self, test_client: TestClient):
        """Test memory routing with large content."""
        # Arrange
        large_content = "x" * 10000  # 10KB content
        route_request = {
            "source_agent_id": "agent_001",
            "target_agent_ids": ["agent_002"],
            "memory_content": large_content,
            "priority": "normal"
        }
        
        # Act
        response = test_client.post("/api/v1/memory/route", json=route_request)
        
        # Assert
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "routed"
    
    @pytest.mark.asyncio
    async def test_route_memory_multiple_targets(self, test_client: TestClient):
        """Test memory routing to multiple target agents."""
        # Arrange
        route_request = {
            "source_agent_id": "agent_001",
            "target_agent_ids": ["agent_002", "agent_003", "agent_004", "agent_005"],
            "memory_content": "Broadcast message to all team members",
            "priority": "normal"
        }
        
        # Act
        response = test_client.post("/api/v1/memory/route", json=route_request)
        
        # Assert
        assert response.status_code == 201
        data = response.json()
        assert len(data["target_agent_ids"]) == 4
        assert "agent_002" in data["target_agent_ids"]
        assert "agent_003" in data["target_agent_ids"]
        assert "agent_004" in data["target_agent_ids"]
        assert "agent_005" in data["target_agent_ids"]


@pytest.mark.integration
class TestMemoryRetrievalAPI:
    """Test memory retrieval API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_agent_memories_success(self, test_client: TestClient):
        """Test successful retrieval of agent memories."""
        # Arrange
        agent_id = "agent_001"
        limit = 10
        offset = 0
        
        # Act
        response = test_client.get(
            f"/api/v1/memory/{agent_id}",
            params={"limit": limit, "offset": offset}
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert "items" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert "has_more" in data
        
        assert data["limit"] == limit
        assert data["offset"] == offset
        assert isinstance(data["items"], list)
        assert len(data["items"]) <= limit
    
    @pytest.mark.asyncio
    async def test_get_agent_memories_with_filters(self, test_client: TestClient):
        """Test retrieval of agent memories with filtering."""
        # Arrange
        agent_id = "agent_001"
        memory_type = "achievement"
        tags = ["milestone", "success"]
        
        # Act
        response = test_client.get(
            f"/api/v1/memory/{agent_id}",
            params={
                "memory_type": memory_type,
                "tags": tags,
                "limit": 5
            }
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Verify all returned memories match the filters
        for memory in data["items"]:
            assert memory["agent_id"] == agent_id
            assert memory["memory_type"] == memory_type
            # Check if any of the requested tags are present
            assert any(tag in memory["tags"] for tag in tags)
    
    @pytest.mark.asyncio
    async def test_get_agent_memories_pagination(self, test_client: TestClient):
        """Test memory retrieval pagination."""
        # Arrange
        agent_id = "agent_001"
        limit = 3
        
        # Act - Get first page
        response1 = test_client.get(
            f"/api/v1/memory/{agent_id}",
            params={"limit": limit, "offset": 0}
        )
        
        # Act - Get second page
        response2 = test_client.get(
            f"/api/v1/memory/{agent_id}",
            params={"limit": limit, "offset": limit}
        )
        
        # Assert
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Verify pagination
        assert data1["offset"] == 0
        assert data2["offset"] == limit
        assert data1["limit"] == limit
        assert data2["limit"] == limit
        
        # Verify no overlap between pages
        memory_ids_1 = {mem["memory_id"] for mem in data1["items"]}
        memory_ids_2 = {mem["memory_id"] for mem in data2["items"]}
        assert len(memory_ids_1.intersection(memory_ids_2)) == 0
    
    @pytest.mark.asyncio
    async def test_get_agent_memories_invalid_agent(self, test_client: TestClient):
        """Test retrieval with invalid agent ID."""
        # Arrange
        invalid_agent_id = "nonexistent_agent"
        
        # Act
        response = test_client.get(f"/api/v1/memory/{invalid_agent_id}")
        
        # Assert
        assert response.status_code == 200  # API returns empty list for invalid agents
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0


@pytest.mark.integration
class TestMemorySearchAPI:
    """Test memory search API endpoints."""
    
    @pytest.mark.asyncio
    async def test_search_memories_success(self, test_client: TestClient):
        """Test successful memory search."""
        # Arrange
        search_request = {
            "query": "project milestone",
            "agent_id": "agent_001",
            "limit": 10,
            "offset": 0
        }
        
        # Act
        response = test_client.post("/api/v1/memory/search", json=search_request)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert "query" in data
        assert "results" in data
        assert "total_results" in data
        assert "search_time_ms" in data
        
        assert data["query"] == "project milestone"
        assert isinstance(data["results"], list)
        assert data["total_results"] >= 0
        assert data["search_time_ms"] >= 0
    
    @pytest.mark.asyncio
    async def test_search_memories_with_filters(self, test_client: TestClient):
        """Test memory search with various filters."""
        # Arrange
        search_request = {
            "query": "important update",
            "agent_id": "agent_001",
            "memory_type": "notification",
            "tags": ["urgent", "update"],
            "limit": 5
        }
        
        # Act
        response = test_client.post("/api/v1/memory/search", json=search_request)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Verify search results match filters
        for result in data["results"]:
            assert result["agent_id"] == "agent_001"
            assert result["memory_type"] == "notification"
            # Check if any of the requested tags are present
            assert any(tag in result["tags"] for tag in ["urgent", "update"])
    
    @pytest.mark.asyncio
    async def test_search_memories_empty_query(self, test_client: TestClient):
        """Test memory search with empty query."""
        # Arrange
        search_request = {
            "query": "",
            "limit": 10
        }
        
        # Act
        response = test_client.post("/api/v1/memory/search", json=search_request)
        
        # Assert
        assert response.status_code == 422  # Validation error for empty query
    
    @pytest.mark.asyncio
    async def test_search_memories_large_limit(self, test_client: TestClient):
        """Test memory search with large limit."""
        # Arrange
        search_request = {
            "query": "test query",
            "limit": 1000  # Exceeds maximum allowed limit
        }
        
        # Act
        response = test_client.post("/api/v1/memory/search", json=search_request)
        
        # Assert
        assert response.status_code == 422  # Validation error for limit too large


@pytest.mark.integration
class TestMemoryRouteDetailsAPI:
    """Test memory route details API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_memory_route_success(self, test_client: TestClient):
        """Test successful retrieval of memory route details."""
        # Arrange
        route_id = "route_1234567890"
        
        # Act
        response = test_client.get(f"/api/v1/memory/route/{route_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["route_id"] == route_id
        assert "source_agent_id" in data
        assert "target_agent_ids" in data
        assert "status" in data
        assert "timestamp" in data
        assert "priority" in data
    
    @pytest.mark.asyncio
    async def test_get_memory_route_invalid_id(self, test_client: TestClient):
        """Test retrieval with invalid route ID."""
        # Arrange
        invalid_route_id = "invalid_route_id"
        
        # Act
        response = test_client.get(f"/api/v1/memory/route/{invalid_route_id}")
        
        # Assert
        assert response.status_code == 200  # API returns mock data for any route ID
        data = response.json()
        assert data["route_id"] == invalid_route_id


@pytest.mark.integration
class TestMemoryStatsAPI:
    """Test memory statistics API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_memory_stats_overview(self, test_client: TestClient):
        """Test retrieval of memory statistics overview."""
        # Act
        response = test_client.get("/api/v1/memory/stats/overview")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert "total_memories" in data
        assert "total_routes" in data
        assert "successful_routes" in data
        assert "failed_routes" in data
        assert "success_rate" in data
        assert "average_routing_time_ms" in data
        assert "top_agents" in data
        assert "timestamp" in data
        
        # Verify data types and ranges
        assert isinstance(data["total_memories"], int)
        assert isinstance(data["total_routes"], int)
        assert isinstance(data["success_rate"], float)
        assert 0 <= data["success_rate"] <= 1
        assert isinstance(data["top_agents"], list)


@pytest.mark.integration
class TestCipherIntegrationAPI:
    """Test Cipher integration API endpoints."""
    
    @pytest.mark.asyncio
    async def test_cipher_store_memory_success(self, test_client: TestClient, mock_cipher_service):
        """Test successful memory storage in Cipher."""
        # Arrange
        store_request = {
            "project_id": "test_project",
            "agent_id": "agent_001",
            "memory_content": "Test memory content for Cipher storage",
            "memory_type": "test",
            "tags": ["test", "cipher"],
            "metadata": {"test": True},
            "priority": 5
        }
        
        # Mock the cipher service response
        mock_cipher_service.store_memory.return_value = "cipher_memory_123"
        
        # Act
        response = test_client.post("/api/v1/memory/cipher/store", json=store_request)
        
        # Assert
        assert response.status_code == 201
        data = response.json()
        
        assert data["memory_id"] == "cipher_memory_123"
        assert data["project_id"] == "test_project"
        assert data["agent_id"] == "agent_001"
        assert data["memory_type"] == "test"
        assert data["status"] == "stored"
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_cipher_retrieve_memory_success(self, test_client: TestClient, mock_cipher_service):
        """Test successful memory retrieval from Cipher."""
        # Arrange
        project_id = "test_project"
        memory_id = "cipher_memory_123"
        
        # Mock the cipher service response
        mock_memory_data = {
            "content": "Retrieved memory content",
            "metadata": {"retrieved": True},
            "from_cache": False
        }
        mock_cipher_service.retrieve_memory.return_value = mock_memory_data
        
        # Act
        response = test_client.get(f"/api/v1/memory/cipher/retrieve/{project_id}/{memory_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["memory_id"] == memory_id
        assert data["project_id"] == project_id
        assert data["memory_data"] == mock_memory_data
        assert data["status"] == "retrieved"
        assert data["from_cache"] is False
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_cipher_retrieve_memory_not_found(self, test_client: TestClient, mock_cipher_service):
        """Test memory retrieval when memory is not found."""
        # Arrange
        project_id = "test_project"
        memory_id = "nonexistent_memory"
        
        # Mock the cipher service to return None (not found)
        mock_cipher_service.retrieve_memory.return_value = None
        
        # Act
        response = test_client.get(f"/api/v1/memory/cipher/retrieve/{project_id}/{memory_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["memory_id"] == memory_id
        assert data["project_id"] == project_id
        assert data["memory_data"] is None
        assert data["status"] == "not_found"
        assert data["from_cache"] is False
    
    @pytest.mark.asyncio
    async def test_cipher_search_memories_success(self, test_client: TestClient, mock_cipher_service):
        """Test successful memory search in Cipher."""
        # Arrange
        search_request = {
            "project_id": "test_project",
            "query": "test search query",
            "agent_id": "agent_001",
            "limit": 10,
            "offset": 0
        }
        
        # Mock the cipher service response
        mock_search_results = {
            "results": [
                {
                    "memory_id": "mem_1",
                    "content": "Test result 1",
                    "relevance_score": 0.9
                },
                {
                    "memory_id": "mem_2",
                    "content": "Test result 2",
                    "relevance_score": 0.8
                }
            ],
            "total": 2,
            "from_cache": False
        }
        mock_cipher_service.search_memories.return_value = mock_search_results
        
        # Act
        response = test_client.post("/api/v1/memory/cipher/search", json=search_request)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["project_id"] == "test_project"
        assert data["query"] == "test search query"
        assert len(data["results"]) == 2
        assert data["total"] == 2
        assert data["from_cache"] is False
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_cipher_update_memory_success(self, test_client: TestClient, mock_cipher_service):
        """Test successful memory update in Cipher."""
        # Arrange
        project_id = "test_project"
        memory_id = "cipher_memory_123"
        updates = {
            "content": "Updated memory content",
            "tags": ["updated", "test"]
        }
        
        # Mock the cipher service response
        mock_cipher_service.update_memory.return_value = True
        
        # Act
        response = test_client.put(
            f"/api/v1/memory/cipher/update/{project_id}/{memory_id}",
            json=updates
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["memory_id"] == memory_id
        assert data["project_id"] == project_id
        assert data["updates"] == updates
        assert data["status"] == "updated"
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_cipher_delete_memory_success(self, test_client: TestClient, mock_cipher_service):
        """Test successful memory deletion from Cipher."""
        # Arrange
        project_id = "test_project"
        memory_id = "cipher_memory_123"
        
        # Mock the cipher service response
        mock_cipher_service.delete_memory.return_value = True
        
        # Act
        response = test_client.delete(f"/api/v1/memory/cipher/delete/{project_id}/{memory_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["memory_id"] == memory_id
        assert data["project_id"] == project_id
        assert data["status"] == "deleted"
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_cipher_create_project_success(self, test_client: TestClient, mock_cipher_service):
        """Test successful project creation in Cipher."""
        # Arrange
        create_request = {
            "project_id": "new_test_project",
            "project_name": "New Test Project",
            "description": "A test project for Cipher integration",
            "metadata": {"test": True}
        }
        
        # Mock the cipher service response
        mock_cipher_service.create_project.return_value = True
        
        # Act
        response = test_client.post("/api/v1/memory/cipher/projects", json=create_request)
        
        # Assert
        assert response.status_code == 201
        data = response.json()
        
        assert data["project_id"] == "new_test_project"
        assert data["project_name"] == "New Test Project"
        assert data["status"] == "created"
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_cipher_get_project_success(self, test_client: TestClient, mock_cipher_service):
        """Test successful project retrieval from Cipher."""
        # Arrange
        project_id = "test_project"
        
        # Mock the cipher service response
        mock_project_info = {
            "name": "Test Project",
            "description": "A test project",
            "created_at": "2024-01-01T00:00:00Z",
            "metadata": {"test": True}
        }
        mock_cipher_service.get_project_info.return_value = mock_project_info
        
        # Act
        response = test_client.get(f"/api/v1/memory/cipher/projects/{project_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["project_id"] == project_id
        assert data["project_info"] == mock_project_info
        assert data["status"] == "found"
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_cipher_list_projects_success(self, test_client: TestClient, mock_cipher_service):
        """Test successful project listing from Cipher."""
        # Arrange
        limit = 10
        offset = 0
        
        # Mock the cipher service response
        mock_projects_data = {
            "projects": [
                {"project_id": "proj_1", "name": "Project 1"},
                {"project_id": "proj_2", "name": "Project 2"}
            ],
            "total": 2
        }
        mock_cipher_service.list_projects.return_value = mock_projects_data
        
        # Act
        response = test_client.get(
            "/api/v1/memory/cipher/projects",
            params={"limit": limit, "offset": offset}
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["projects"]) == 2
        assert data["total"] == 2
        assert data["limit"] == limit
        assert data["offset"] == offset
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_cipher_health_check_success(self, test_client: TestClient, mock_cipher_service):
        """Test successful Cipher health check."""
        # Arrange
        # Mock the cipher service response
        mock_health_data = {
            "status": "ok",
            "version": "1.0.0",
            "uptime": 3600
        }
        mock_cipher_service.health_check.return_value = mock_health_data
        
        # Act
        response = test_client.get("/api/v1/memory/cipher/health")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["health_data"] == mock_health_data
        assert "timestamp" in data


@pytest.mark.integration
class TestMemoryAPIPerformance:
    """Test memory API performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_memory_routing_performance(self, test_client: TestClient, performance_monitor):
        """Test memory routing performance under normal load."""
        # Arrange
        route_request = {
            "source_agent_id": "agent_001",
            "target_agent_ids": ["agent_002"],
            "memory_content": "Performance test memory content",
            "priority": "normal"
        }
        
        # Act
        performance_monitor.start()
        response = test_client.post("/api/v1/memory/route", json=route_request)
        performance_monitor.stop()
        
        # Assert
        assert response.status_code == 201
        duration = performance_monitor.get_duration()
        assert duration is not None
        assert duration < 1.0  # Should complete within 1 second
    
    @pytest.mark.asyncio
    async def test_memory_search_performance(self, test_client: TestClient, performance_monitor):
        """Test memory search performance."""
        # Arrange
        search_request = {
            "query": "performance test query",
            "limit": 10
        }
        
        # Act
        performance_monitor.start()
        response = test_client.post("/api/v1/memory/search", json=search_request)
        performance_monitor.stop()
        
        # Assert
        assert response.status_code == 200
        duration = performance_monitor.get_duration()
        assert duration is not None
        assert duration < 2.0  # Search should complete within 2 seconds
        
        # Verify search time is reported in response
        data = response.json()
        assert data["search_time_ms"] >= 0
        assert data["search_time_ms"] < 2000  # Less than 2 seconds in milliseconds


@pytest.mark.integration
class TestMemoryAPISecurity:
    """Test memory API security features."""
    
    @pytest.mark.asyncio
    async def test_memory_routing_sql_injection_protection(self, test_client: TestClient):
        """Test protection against SQL injection in memory routing."""
        # Arrange
        malicious_request = {
            "source_agent_id": "agent_001'; DROP TABLE memories; --",
            "target_agent_ids": ["agent_002"],
            "memory_content": "Test content",
            "priority": "normal"
        }
        
        # Act
        response = test_client.post("/api/v1/memory/route", json=malicious_request)
        
        # Assert
        assert response.status_code == 201  # Should handle gracefully
        data = response.json()
        assert data["source_agent_id"] == malicious_request["source_agent_id"]  # Should be treated as literal string
    
    @pytest.mark.asyncio
    async def test_memory_search_xss_protection(self, test_client: TestClient):
        """Test protection against XSS in memory search."""
        # Arrange
        malicious_request = {
            "query": "<script>alert('xss')</script>",
            "limit": 10
        }
        
        # Act
        response = test_client.post("/api/v1/memory/search", json=malicious_request)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        # The malicious script should be treated as literal text, not executed
        assert "<script>" in data["query"]
    
    @pytest.mark.asyncio
    async def test_memory_content_size_limits(self, test_client: TestClient):
        """Test memory content size limits."""
        # Arrange - Create very large content (1MB)
        large_content = "x" * (1024 * 1024)
        route_request = {
            "source_agent_id": "agent_001",
            "target_agent_ids": ["agent_002"],
            "memory_content": large_content,
            "priority": "normal"
        }
        
        # Act
        response = test_client.post("/api/v1/memory/route", json=route_request)
        
        # Assert
        # Should either succeed or fail gracefully with appropriate error
        assert response.status_code in [201, 413, 422]  # Created, Payload Too Large, or Validation Error

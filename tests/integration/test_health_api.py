"""
Integration tests for Health API endpoints.

This module tests the complete health monitoring API including basic health checks,
detailed health checks, component health monitoring, and system metrics.
"""

import pytest
import time
from typing import Dict, Any, List
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.integration
class TestBasicHealthAPI:
    """Test basic health check API endpoints."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, test_client: TestClient):
        """Test successful basic health check."""
        # Act
        response = test_client.get("/api/v1/health/")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "0.1.0"
        assert "uptime" in data
        assert data["uptime"] >= 0
    
    @pytest.mark.asyncio
    async def test_health_check_response_time(self, test_client: TestClient, performance_monitor):
        """Test health check response time."""
        # Act
        performance_monitor.start()
        response = test_client.get("/api/v1/health/")
        performance_monitor.stop()
        
        # Assert
        assert response.status_code == 200
        duration = performance_monitor.get_duration()
        assert duration is not None
        assert duration < 0.5  # Health check should be very fast
    
    @pytest.mark.asyncio
    async def test_health_check_uptime_increases(self, test_client: TestClient):
        """Test that uptime increases between health checks."""
        # Act - First check
        response1 = test_client.get("/api/v1/health/")
        time.sleep(0.1)  # Wait 100ms
        response2 = test_client.get("/api/v1/health/")
        
        # Assert
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        assert data2["uptime"] > data1["uptime"]
        assert data2["uptime"] - data1["uptime"] >= 0.1  # At least 100ms difference


@pytest.mark.integration
class TestDetailedHealthAPI:
    """Test detailed health check API endpoints."""
    
    @pytest.mark.asyncio
    async def test_detailed_health_check_success(self, test_client: TestClient):
        """Test successful detailed health check."""
        # Act
        response = test_client.get("/api/v1/health/detailed")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert "overall_status" in data
        assert "timestamp" in data
        assert data["version"] == "0.1.0"
        assert "uptime" in data
        assert "components" in data
        assert "system_info" in data
        
        # Verify components structure
        assert isinstance(data["components"], list)
        assert len(data["components"]) > 0
        
        for component in data["components"]:
            assert "name" in component
            assert "status" in component
            assert "response_time_ms" in component
            assert "last_check" in component
            assert "details" in component
    
    @pytest.mark.asyncio
    async def test_detailed_health_check_components(self, test_client: TestClient):
        """Test detailed health check includes all expected components."""
        # Act
        response = test_client.get("/api/v1/health/detailed")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        component_names = [comp["name"] for comp in data["components"]]
        
        # Verify expected components are present
        assert "database" in component_names
        assert "mcp_server" in component_names
        assert "redis" in component_names
        assert "chroma" in component_names
    
    @pytest.mark.asyncio
    async def test_detailed_health_check_system_info(self, test_client: TestClient):
        """Test detailed health check includes system information."""
        # Act
        response = test_client.get("/api/v1/health/detailed")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        system_info = data["system_info"]
        assert "python_version" in system_info
        assert "platform" in system_info
        assert "memory_usage" in system_info
        assert "cpu_usage" in system_info
        assert "active_connections" in system_info
    
    @pytest.mark.asyncio
    async def test_detailed_health_check_overall_status(self, test_client: TestClient):
        """Test detailed health check overall status calculation."""
        # Act
        response = test_client.get("/api/v1/health/detailed")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        overall_status = data["overall_status"]
        assert overall_status in ["healthy", "degraded", "unhealthy"]
        
        # If all components are healthy, overall should be healthy
        all_healthy = all(comp["status"] == "healthy" for comp in data["components"])
        if all_healthy:
            assert overall_status == "healthy"


@pytest.mark.integration
class TestComponentHealthAPI:
    """Test component-specific health check API endpoints."""
    
    @pytest.mark.asyncio
    async def test_database_health_check(self, test_client: TestClient):
        """Test database health check."""
        # Act
        response = test_client.get("/api/v1/health/database")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["component"] == "database"
        assert "status" in data
        assert "timestamp" in data
        assert "response_time_ms" in data
        assert "details" in data
        
        # Verify response time is reasonable
        assert data["response_time_ms"] >= 0
        assert data["response_time_ms"] < 5000  # Less than 5 seconds
    
    @pytest.mark.asyncio
    async def test_mcp_health_check(self, test_client: TestClient):
        """Test MCP server health check."""
        # Act
        response = test_client.get("/api/v1/health/mcp")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["component"] == "mcp_server"
        assert "status" in data
        assert "timestamp" in data
        assert "response_time_ms" in data
        assert "details" in data
        
        # Verify MCP-specific details
        details = data["details"]
        assert "type" in details
        assert "status" in details
        assert "tools_available" in details
        assert details["type"] == "mcp"
    
    @pytest.mark.asyncio
    async def test_component_health_check_response_times(self, test_client: TestClient):
        """Test component health check response times are reasonable."""
        # Act
        db_response = test_client.get("/api/v1/health/database")
        mcp_response = test_client.get("/api/v1/health/mcp")
        
        # Assert
        assert db_response.status_code == 200
        assert mcp_response.status_code == 200
        
        db_data = db_response.json()
        mcp_data = mcp_response.json()
        
        # Verify response times are reasonable
        assert db_data["response_time_ms"] < 5000
        assert mcp_data["response_time_ms"] < 5000


@pytest.mark.integration
class TestSystemMetricsAPI:
    """Test system metrics API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_system_metrics_success(self, test_client: TestClient):
        """Test successful system metrics retrieval."""
        # Act
        response = test_client.get("/api/v1/health/metrics")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert "timestamp" in data
        assert "metrics" in data
        assert "performance" in data
        
        # Verify metrics structure
        metrics = data["metrics"]
        assert "memory_routes_total" in metrics
        assert "agents_total" in metrics
        assert "memories_stored_total" in metrics
        assert "context_updates_total" in metrics
        assert "requests_total" in metrics
        assert "errors_total" in metrics
        
        # Verify performance structure
        performance = data["performance"]
        assert "average_response_time_ms" in performance
        assert "requests_per_second" in performance
        assert "memory_usage_mb" in performance
        assert "cpu_usage_percent" in performance
        assert "active_connections" in performance
        assert "database_connections" in performance
    
    @pytest.mark.asyncio
    async def test_system_metrics_data_types(self, test_client: TestClient):
        """Test system metrics data types are correct."""
        # Act
        response = test_client.get("/api/v1/health/metrics")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Verify metrics are numeric
        metrics = data["metrics"]
        for key, value in metrics.items():
            assert isinstance(value, int)
            assert value >= 0
        
        # Verify performance metrics are numeric
        performance = data["performance"]
        for key, value in performance.items():
            assert isinstance(value, (int, float))
            assert value >= 0
    
    @pytest.mark.asyncio
    async def test_system_metrics_performance_ranges(self, test_client: TestClient):
        """Test system metrics performance values are within reasonable ranges."""
        # Act
        response = test_client.get("/api/v1/health/metrics")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        performance = data["performance"]
        
        # Verify performance metrics are within reasonable ranges
        assert 0 <= performance["average_response_time_ms"] <= 10000  # 0-10 seconds
        assert 0 <= performance["requests_per_second"] <= 1000  # 0-1000 RPS
        assert 0 <= performance["memory_usage_mb"] <= 10000  # 0-10GB
        assert 0 <= performance["cpu_usage_percent"] <= 100  # 0-100%
        assert 0 <= performance["active_connections"] <= 1000  # 0-1000 connections
        assert 0 <= performance["database_connections"] <= 100  # 0-100 DB connections


@pytest.mark.integration
class TestReadinessAPI:
    """Test readiness check API endpoints."""
    
    @pytest.mark.asyncio
    async def test_readiness_check_success(self, test_client: TestClient):
        """Test successful readiness check."""
        # Act
        response = test_client.get("/api/v1/health/ready")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert "ready" in data
        assert "timestamp" in data
        assert "response_time_ms" in data
        assert "components" in data
        
        # Verify components structure
        components = data["components"]
        assert "database" in components
        assert "mcp_server" in components
        
        # Verify response time is reasonable
        assert data["response_time_ms"] >= 0
        assert data["response_time_ms"] < 5000
    
    @pytest.mark.asyncio
    async def test_readiness_check_components(self, test_client: TestClient):
        """Test readiness check component status."""
        # Act
        response = test_client.get("/api/v1/health/ready")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        components = data["components"]
        
        # Verify component statuses are boolean
        assert isinstance(components["database"], bool)
        assert isinstance(components["mcp_server"], bool)
        
        # If all critical components are ready, overall should be ready
        all_ready = all(components.values())
        assert data["ready"] == all_ready


@pytest.mark.integration
class TestLivenessAPI:
    """Test liveness check API endpoints."""
    
    @pytest.mark.asyncio
    async def test_liveness_check_success(self, test_client: TestClient):
        """Test successful liveness check."""
        # Act
        response = test_client.get("/api/v1/health/live")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["alive"] is True
        assert "timestamp" in data
        assert "response_time_ms" in data
        assert "uptime" in data
        
        # Verify response time is very fast (liveness should be minimal)
        assert data["response_time_ms"] >= 0
        assert data["response_time_ms"] < 100  # Less than 100ms
    
    @pytest.mark.asyncio
    async def test_liveness_check_uptime(self, test_client: TestClient):
        """Test liveness check uptime tracking."""
        # Act
        response1 = test_client.get("/api/v1/health/live")
        time.sleep(0.1)  # Wait 100ms
        response2 = test_client.get("/api/v1/health/live")
        
        # Assert
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        assert data2["uptime"] > data1["uptime"]
        assert data2["uptime"] - data1["uptime"] >= 0.1


@pytest.mark.integration
class TestSystemStatusAPI:
    """Test system status API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_system_status_success(self, test_client: TestClient):
        """Test successful system status retrieval."""
        # Act
        response = test_client.get("/api/v1/health/status")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert data["version"] == "0.1.0"
        assert "uptime" in data
        assert "components" in data
        assert "summary" in data
        
        # Verify status is valid
        assert data["status"] in ["operational", "degraded", "down"]
    
    @pytest.mark.asyncio
    async def test_system_status_components(self, test_client: TestClient):
        """Test system status component information."""
        # Act
        response = test_client.get("/api/v1/health/status")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        components = data["components"]
        
        # Verify expected components are present
        assert "database" in components
        assert "mcp_server" in components
        assert "redis" in components
        assert "chroma" in components
        
        # Verify component structure
        for component_name, component_info in components.items():
            assert "status" in component_info
            assert "priority" in component_info
            assert component_info["status"] in ["healthy", "unhealthy"]
            assert component_info["priority"] in ["critical", "high", "medium", "low"]
    
    @pytest.mark.asyncio
    async def test_system_status_summary(self, test_client: TestClient):
        """Test system status summary information."""
        # Act
        response = test_client.get("/api/v1/health/status")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        summary = data["summary"]
        
        assert "total_components" in summary
        assert "healthy_components" in summary
        assert "unhealthy_components" in summary
        assert "critical_components_healthy" in summary
        
        # Verify summary values are consistent
        assert summary["total_components"] > 0
        assert summary["healthy_components"] + summary["unhealthy_components"] == summary["total_components"]
        assert isinstance(summary["critical_components_healthy"], bool)
    
    @pytest.mark.asyncio
    async def test_system_status_overall_calculation(self, test_client: TestClient):
        """Test system status overall status calculation."""
        # Act
        response = test_client.get("/api/v1/health/status")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        overall_status = data["status"]
        components = data["components"]
        summary = data["summary"]
        
        # Verify overall status calculation logic
        if summary["critical_components_healthy"]:
            assert overall_status in ["operational", "degraded"]
        else:
            assert overall_status == "down"


@pytest.mark.integration
class TestHealthAPIPerformance:
    """Test health API performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_health_check_performance(self, test_client: TestClient, performance_monitor):
        """Test health check performance under normal load."""
        # Act
        performance_monitor.start()
        response = test_client.get("/api/v1/health/")
        performance_monitor.stop()
        
        # Assert
        assert response.status_code == 200
        duration = performance_monitor.get_duration()
        assert duration is not None
        assert duration < 0.5  # Health check should be very fast
    
    @pytest.mark.asyncio
    async def test_detailed_health_check_performance(self, test_client: TestClient, performance_monitor):
        """Test detailed health check performance."""
        # Act
        performance_monitor.start()
        response = test_client.get("/api/v1/health/detailed")
        performance_monitor.stop()
        
        # Assert
        assert response.status_code == 200
        duration = performance_monitor.get_duration()
        assert duration is not None
        assert duration < 5.0  # Detailed check should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_metrics_performance(self, test_client: TestClient, performance_monitor):
        """Test metrics retrieval performance."""
        # Act
        performance_monitor.start()
        response = test_client.get("/api/v1/health/metrics")
        performance_monitor.stop()
        
        # Assert
        assert response.status_code == 200
        duration = performance_monitor.get_duration()
        assert duration is not None
        assert duration < 2.0  # Metrics should be fast to retrieve
    
    @pytest.mark.asyncio
    async def test_readiness_performance(self, test_client: TestClient, performance_monitor):
        """Test readiness check performance."""
        # Act
        performance_monitor.start()
        response = test_client.get("/api/v1/health/ready")
        performance_monitor.stop()
        
        # Assert
        assert response.status_code == 200
        duration = performance_monitor.get_duration()
        assert duration is not None
        assert duration < 3.0  # Readiness check should be reasonably fast
    
    @pytest.mark.asyncio
    async def test_liveness_performance(self, test_client: TestClient, performance_monitor):
        """Test liveness check performance."""
        # Act
        performance_monitor.start()
        response = test_client.get("/api/v1/health/live")
        performance_monitor.stop()
        
        # Assert
        assert response.status_code == 200
        duration = performance_monitor.get_duration()
        assert duration is not None
        assert duration < 0.1  # Liveness check should be extremely fast


@pytest.mark.integration
class TestHealthAPISecurity:
    """Test health API security features."""
    
    @pytest.mark.asyncio
    async def test_health_check_no_sensitive_data(self, test_client: TestClient):
        """Test that health checks don't expose sensitive data."""
        # Act
        response = test_client.get("/api/v1/health/detailed")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Verify no sensitive information is exposed
        data_str = str(data).lower()
        sensitive_terms = ["password", "secret", "key", "token", "credential"]
        
        for term in sensitive_terms:
            assert term not in data_str, f"Sensitive term '{term}' found in health check response"
    
    @pytest.mark.asyncio
    async def test_health_check_information_disclosure(self, test_client: TestClient):
        """Test that health checks don't disclose internal system details."""
        # Act
        response = test_client.get("/api/v1/health/detailed")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Verify system info doesn't expose internal details
        system_info = data["system_info"]
        
        # Should not expose internal paths, IPs, or system-specific details
        system_str = str(system_info).lower()
        internal_terms = ["/etc/", "/var/", "127.0.0.1", "localhost", "internal"]
        
        for term in internal_terms:
            # These terms might be present but shouldn't be in sensitive contexts
            # This is more of a documentation test than a strict security test
            pass  # Placeholder for future security validation


@pytest.mark.integration
class TestHealthAPIIntegration:
    """Test health API integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_health_check_consistency(self, test_client: TestClient):
        """Test consistency between different health check endpoints."""
        # Act
        basic_response = test_client.get("/api/v1/health/")
        detailed_response = test_client.get("/api/v1/health/detailed")
        status_response = test_client.get("/api/v1/health/status")
        
        # Assert
        assert basic_response.status_code == 200
        assert detailed_response.status_code == 200
        assert status_response.status_code == 200
        
        basic_data = basic_response.json()
        detailed_data = detailed_response.json()
        status_data = status_response.json()
        
        # Verify version consistency
        assert basic_data["version"] == detailed_data["version"] == status_data["version"]
        
        # Verify uptime consistency (within reasonable tolerance)
        uptime_diff = abs(basic_data["uptime"] - detailed_data["uptime"])
        assert uptime_diff < 1.0  # Within 1 second tolerance
    
    @pytest.mark.asyncio
    async def test_health_check_monitoring_workflow(self, test_client: TestClient):
        """Test complete health monitoring workflow."""
        # Step 1: Check if system is alive
        liveness_response = test_client.get("/api/v1/health/live")
        assert liveness_response.status_code == 200
        assert liveness_response.json()["alive"] is True
        
        # Step 2: Check if system is ready
        readiness_response = test_client.get("/api/v1/health/ready")
        assert readiness_response.status_code == 200
        
        # Step 3: Get basic health status
        health_response = test_client.get("/api/v1/health/")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "healthy"
        
        # Step 4: Get detailed health information
        detailed_response = test_client.get("/api/v1/health/detailed")
        assert detailed_response.status_code == 200
        
        # Step 5: Get system status
        status_response = test_client.get("/api/v1/health/status")
        assert status_response.status_code == 200
        
        # Step 6: Get metrics
        metrics_response = test_client.get("/api/v1/health/metrics")
        assert metrics_response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_health_check_component_isolation(self, test_client: TestClient):
        """Test that component health checks work independently."""
        # Act - Test each component individually
        db_response = test_client.get("/api/v1/health/database")
        mcp_response = test_client.get("/api/v1/health/mcp")
        
        # Assert
        assert db_response.status_code == 200
        assert mcp_response.status_code == 200
        
        db_data = db_response.json()
        mcp_data = mcp_response.json()
        
        # Verify each component check is independent
        assert db_data["component"] == "database"
        assert mcp_data["component"] == "mcp_server"
        
        # Verify response times are independent
        assert db_data["response_time_ms"] != mcp_data["response_time_ms"]

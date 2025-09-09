"""
Integration tests for Context API endpoints.

This module tests the complete context management API including context updates,
retrieval, search, and conversation management functionality.
"""

import pytest
import time
from typing import Dict, Any, List
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.integration
class TestContextUpdateAPI:
    """Test context update API endpoints."""
    
    @pytest.mark.asyncio
    async def test_update_context_success(self, test_client: TestClient):
        """Test successful context update."""
        # Arrange
        update_request = {
            "conversation_id": "conv_123",
            "agent_id": "agent_001",
            "context_data": {
                "topic": "Project planning",
                "participants": ["agent_001", "agent_002"],
                "current_focus": "Timeline discussion",
                "priority": "high"
            },
            "update_type": "append"
        }
        
        # Act
        response = test_client.post("/api/v1/context/update", json=update_request)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["conversation_id"] == "conv_123"
        assert "context" in data
        assert "participants" in data
        assert "topics" in data
        assert "last_updated" in data
        assert "context_size" in data
        assert "metadata" in data
        
        # Verify context data is merged correctly
        assert data["context"]["topic"] == "Project planning"
        assert "agent_001" in data["participants"]
        assert "agent_002" in data["participants"]
    
    @pytest.mark.asyncio
    async def test_update_context_replace_type(self, test_client: TestClient):
        """Test context update with replace type."""
        # Arrange
        update_request = {
            "conversation_id": "conv_124",
            "agent_id": "agent_001",
            "context_data": {
                "topic": "New topic",
                "participants": ["agent_001"],
                "current_focus": "New focus"
            },
            "update_type": "replace"
        }
        
        # Act
        response = test_client.post("/api/v1/context/update", json=update_request)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["conversation_id"] == "conv_124"
        assert data["context"]["topic"] == "New topic"
        assert data["context"]["participants"] == ["agent_001"]
        assert data["context"]["current_focus"] == "New focus"
    
    @pytest.mark.asyncio
    async def test_update_context_merge_type(self, test_client: TestClient):
        """Test context update with merge type."""
        # Arrange
        update_request = {
            "conversation_id": "conv_125",
            "agent_id": "agent_001",
            "context_data": {
                "new_field": "new_value",
                "existing_field": "updated_value"
            },
            "update_type": "merge"
        }
        
        # Act
        response = test_client.post("/api/v1/context/update", json=update_request)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["conversation_id"] == "conv_125"
        assert data["context"]["new_field"] == "new_value"
        assert data["context"]["existing_field"] == "updated_value"
    
    @pytest.mark.asyncio
    async def test_update_context_validation_error(self, test_client: TestClient):
        """Test context update with invalid input data."""
        # Arrange
        invalid_request = {
            "conversation_id": "",  # Empty conversation ID
            "agent_id": "",  # Empty agent ID
            "context_data": {},  # Empty context data
            "update_type": "invalid_type"  # Invalid update type
        }
        
        # Act
        response = test_client.post("/api/v1/context/update", json=invalid_request)
        
        # Assert
        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert "Validation error" in data["error"]
    
    @pytest.mark.asyncio
    async def test_update_context_large_data(self, test_client: TestClient):
        """Test context update with large context data."""
        # Arrange
        large_context_data = {
            "topic": "Large context test",
            "participants": ["agent_001", "agent_002"],
            "large_field": "x" * 10000,  # 10KB field
            "nested_data": {
                "level1": {
                    "level2": {
                        "level3": "deep_nested_value"
                    }
                }
            }
        }
        
        update_request = {
            "conversation_id": "conv_large",
            "agent_id": "agent_001",
            "context_data": large_context_data,
            "update_type": "append"
        }
        
        # Act
        response = test_client.post("/api/v1/context/update", json=update_request)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["conversation_id"] == "conv_large"
        assert data["context"]["topic"] == "Large context test"
        assert len(data["context"]["large_field"]) == 10000
        assert data["context"]["nested_data"]["level1"]["level2"]["level3"] == "deep_nested_value"


@pytest.mark.integration
class TestContextRetrievalAPI:
    """Test context retrieval API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_context_success(self, test_client: TestClient):
        """Test successful context retrieval."""
        # Arrange
        conversation_id = "conv_123"
        
        # Act
        response = test_client.get(f"/api/v1/context/{conversation_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["conversation_id"] == conversation_id
        assert "context" in data
        assert "participants" in data
        assert "topics" in data
        assert "last_updated" in data
        assert "context_size" in data
        assert "metadata" in data
        
        # Verify context structure
        assert isinstance(data["context"], dict)
        assert isinstance(data["participants"], list)
        assert isinstance(data["topics"], list)
        assert isinstance(data["context_size"], int)
        assert data["context_size"] > 0
    
    @pytest.mark.asyncio
    async def test_get_context_invalid_id(self, test_client: TestClient):
        """Test context retrieval with invalid conversation ID."""
        # Arrange
        invalid_conversation_id = "nonexistent_conv"
        
        # Act
        response = test_client.get(f"/api/v1/context/{invalid_conversation_id}")
        
        # Assert
        assert response.status_code == 200  # API returns mock data for any conversation ID
        data = response.json()
        assert data["conversation_id"] == invalid_conversation_id
    
    @pytest.mark.asyncio
    async def test_get_context_special_characters(self, test_client: TestClient):
        """Test context retrieval with special characters in conversation ID."""
        # Arrange
        special_conv_id = "conv-with_special.chars@123"
        
        # Act
        response = test_client.get(f"/api/v1/context/{special_conv_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == special_conv_id


@pytest.mark.integration
class TestContextListingAPI:
    """Test context listing API endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_contexts_success(self, test_client: TestClient):
        """Test successful context listing."""
        # Act
        response = test_client.get("/api/v1/context/")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert "contexts" in data
        assert "total" in data
        assert "active_conversations" in data
        
        assert isinstance(data["contexts"], list)
        assert isinstance(data["total"], int)
        assert isinstance(data["active_conversations"], int)
        
        # Verify context structure
        if data["contexts"]:
            context = data["contexts"][0]
            assert "conversation_id" in context
            assert "agent_id" in context
            assert "context_data" in context
            assert "created_at" in context
            assert "updated_at" in context
            assert "metadata" in context
    
    @pytest.mark.asyncio
    async def test_list_contexts_with_agent_filter(self, test_client: TestClient):
        """Test context listing with agent filter."""
        # Act
        response = test_client.get("/api/v1/context/", params={"agent_id": "agent_1"})
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Verify all returned contexts belong to the specified agent
        for context in data["contexts"]:
            assert context["agent_id"] == "agent_1"
    
    @pytest.mark.asyncio
    async def test_list_contexts_with_pagination(self, test_client: TestClient):
        """Test context listing with pagination."""
        # Arrange
        limit = 3
        offset = 0
        
        # Act - Get first page
        response1 = test_client.get("/api/v1/context/", params={"limit": limit, "offset": offset})
        
        # Act - Get second page
        response2 = test_client.get("/api/v1/context/", params={"limit": limit, "offset": limit})
        
        # Assert
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Verify pagination
        assert len(data1["contexts"]) <= limit
        assert len(data2["contexts"]) <= limit
        
        # Verify no overlap between pages
        conv_ids_1 = {ctx["conversation_id"] for ctx in data1["contexts"]}
        conv_ids_2 = {ctx["conversation_id"] for ctx in data2["contexts"]}
        assert len(conv_ids_1.intersection(conv_ids_2)) == 0
    
    @pytest.mark.asyncio
    async def test_list_contexts_combined_filters(self, test_client: TestClient):
        """Test context listing with combined filters."""
        # Act
        response = test_client.get(
            "/api/v1/context/",
            params={
                "agent_id": "agent_1",
                "limit": 5
            }
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Verify all returned contexts match the filter
        for context in data["contexts"]:
            assert context["agent_id"] == "agent_1"
        
        # Verify limit is respected
        assert len(data["contexts"]) <= 5


@pytest.mark.integration
class TestContextSearchAPI:
    """Test context search API endpoints."""
    
    @pytest.mark.asyncio
    async def test_search_contexts_success(self, test_client: TestClient):
        """Test successful context search."""
        # Arrange
        search_request = {
            "query": "project planning",
            "agent_id": "agent_001",
            "limit": 10,
            "offset": 0
        }
        
        # Act
        response = test_client.post("/api/v1/context/search", json=search_request)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert "query" in data
        assert "results" in data
        assert "total_results" in data
        assert "search_time_ms" in data
        
        assert data["query"] == "project planning"
        assert isinstance(data["results"], list)
        assert data["total_results"] >= 0
        assert data["search_time_ms"] >= 0
    
    @pytest.mark.asyncio
    async def test_search_contexts_with_filters(self, test_client: TestClient):
        """Test context search with various filters."""
        # Arrange
        search_request = {
            "query": "timeline discussion",
            "agent_id": "agent_001",
            "conversation_id": "conv_123",
            "limit": 5
        }
        
        # Act
        response = test_client.post("/api/v1/context/search", json=search_request)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Verify search results match filters
        for result in data["results"]:
            assert result["agent_id"] == "agent_001"
            assert result["conversation_id"] == "conv_123"
    
    @pytest.mark.asyncio
    async def test_search_contexts_empty_query(self, test_client: TestClient):
        """Test context search with empty query."""
        # Arrange
        search_request = {
            "query": "",
            "limit": 10
        }
        
        # Act
        response = test_client.post("/api/v1/context/search", json=search_request)
        
        # Assert
        assert response.status_code == 422  # Validation error for empty query
    
    @pytest.mark.asyncio
    async def test_search_contexts_large_limit(self, test_client: TestClient):
        """Test context search with large limit."""
        # Arrange
        search_request = {
            "query": "test query",
            "limit": 1000  # Exceeds maximum allowed limit
        }
        
        # Act
        response = test_client.post("/api/v1/context/search", json=search_request)
        
        # Assert
        assert response.status_code == 422  # Validation error for limit too large
    
    @pytest.mark.asyncio
    async def test_search_contexts_relevance_scores(self, test_client: TestClient):
        """Test context search relevance scores."""
        # Arrange
        search_request = {
            "query": "test search query",
            "limit": 5
        }
        
        # Act
        response = test_client.post("/api/v1/context/search", json=search_request)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Verify relevance scores are valid
        for result in data["results"]:
            assert "relevance_score" in result
            assert 0 <= result["relevance_score"] <= 1
            assert "matched_content" in result
            assert "last_updated" in result


@pytest.mark.integration
class TestContextDeletionAPI:
    """Test context deletion API endpoints."""
    
    @pytest.mark.asyncio
    async def test_delete_context_success(self, test_client: TestClient):
        """Test successful context deletion."""
        # Arrange
        conversation_id = "conv_123"
        
        # Act
        response = test_client.delete(f"/api/v1/context/{conversation_id}")
        
        # Assert
        assert response.status_code == 204
        # No content should be returned for successful deletion
    
    @pytest.mark.asyncio
    async def test_delete_context_invalid_id(self, test_client: TestClient):
        """Test context deletion with invalid conversation ID."""
        # Arrange
        invalid_conversation_id = "nonexistent_conv"
        
        # Act
        response = test_client.delete(f"/api/v1/context/{invalid_conversation_id}")
        
        # Assert
        assert response.status_code == 204  # API returns success for any conversation ID
        # No content should be returned for successful deletion


@pytest.mark.integration
class TestContextParticipantsAPI:
    """Test context participants API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_conversation_participants_success(self, test_client: TestClient):
        """Test successful conversation participants retrieval."""
        # Arrange
        conversation_id = "conv_123"
        
        # Act
        response = test_client.get(f"/api/v1/context/{conversation_id}/participants")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["conversation_id"] == conversation_id
        assert "participants" in data
        assert "total_participants" in data
        assert "timestamp" in data
        
        assert isinstance(data["participants"], list)
        assert isinstance(data["total_participants"], int)
        assert data["total_participants"] == len(data["participants"])
        
        # Verify participant structure
        if data["participants"]:
            participant = data["participants"][0]
            assert "agent_id" in participant
            assert "agent_name" in participant
            assert "joined_at" in participant
            assert "last_active" in participant
            assert "role" in participant
    
    @pytest.mark.asyncio
    async def test_get_conversation_participants_invalid_id(self, test_client: TestClient):
        """Test conversation participants retrieval with invalid conversation ID."""
        # Arrange
        invalid_conversation_id = "nonexistent_conv"
        
        # Act
        response = test_client.get(f"/api/v1/context/{invalid_conversation_id}/participants")
        
        # Assert
        assert response.status_code == 200  # API returns mock data for any conversation ID
        data = response.json()
        assert data["conversation_id"] == invalid_conversation_id


@pytest.mark.integration
class TestContextHistoryAPI:
    """Test context history API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_conversation_history_success(self, test_client: TestClient):
        """Test successful conversation history retrieval."""
        # Arrange
        conversation_id = "conv_123"
        limit = 10
        
        # Act
        response = test_client.get(
            f"/api/v1/context/{conversation_id}/history",
            params={"limit": limit}
        )
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert data["conversation_id"] == conversation_id
        assert "history" in data
        assert "total_items" in data
        assert "timestamp" in data
        
        assert isinstance(data["history"], list)
        assert isinstance(data["total_items"], int)
        assert data["total_items"] == len(data["history"])
        assert len(data["history"]) <= limit
        
        # Verify history item structure
        if data["history"]:
            history_item = data["history"][0]
            assert "timestamp" in history_item
            assert "agent_id" in history_item
            assert "action" in history_item
            assert "description" in history_item
            assert "context_snippet" in history_item
    
    @pytest.mark.asyncio
    async def test_get_conversation_history_with_limit(self, test_client: TestClient):
        """Test conversation history retrieval with different limits."""
        # Arrange
        conversation_id = "conv_123"
        
        # Act - Get with small limit
        response1 = test_client.get(
            f"/api/v1/context/{conversation_id}/history",
            params={"limit": 3}
        )
        
        # Act - Get with larger limit
        response2 = test_client.get(
            f"/api/v1/context/{conversation_id}/history",
            params={"limit": 10}
        )
        
        # Assert
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Verify limits are respected
        assert len(data1["history"]) <= 3
        assert len(data2["history"]) <= 10
        
        # Verify larger limit returns more or equal items
        assert len(data2["history"]) >= len(data1["history"])
    
    @pytest.mark.asyncio
    async def test_get_conversation_history_invalid_id(self, test_client: TestClient):
        """Test conversation history retrieval with invalid conversation ID."""
        # Arrange
        invalid_conversation_id = "nonexistent_conv"
        
        # Act
        response = test_client.get(f"/api/v1/context/{invalid_conversation_id}/history")
        
        # Assert
        assert response.status_code == 200  # API returns mock data for any conversation ID
        data = response.json()
        assert data["conversation_id"] == invalid_conversation_id


@pytest.mark.integration
class TestContextStatsAPI:
    """Test context statistics API endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_context_stats_overview(self, test_client: TestClient):
        """Test retrieval of context statistics overview."""
        # Act
        response = test_client.get("/api/v1/context/stats/overview")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        assert "total_conversations" in data
        assert "active_conversations" in data
        assert "total_context_updates" in data
        assert "average_context_size" in data
        assert "topics" in data
        assert "participants_distribution" in data
        assert "timestamp" in data
        
        # Verify data types and ranges
        assert isinstance(data["total_conversations"], int)
        assert isinstance(data["active_conversations"], int)
        assert isinstance(data["total_context_updates"], int)
        assert isinstance(data["average_context_size"], int)
        assert isinstance(data["topics"], list)
        assert isinstance(data["participants_distribution"], dict)
        
        # Verify topics structure
        if data["topics"]:
            topic = data["topics"][0]
            assert "topic" in topic
            assert "count" in topic
            assert isinstance(topic["count"], int)
        
        # Verify participants distribution structure
        participants_dist = data["participants_distribution"]
        assert "2_participants" in participants_dist
        assert "3_participants" in participants_dist
        assert "4+_participants" in participants_dist


@pytest.mark.integration
class TestContextAPIPerformance:
    """Test context API performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_context_update_performance(self, test_client: TestClient, performance_monitor):
        """Test context update performance."""
        # Arrange
        update_request = {
            "conversation_id": "perf_test_conv",
            "agent_id": "agent_001",
            "context_data": {
                "topic": "Performance test",
                "participants": ["agent_001", "agent_002"],
                "current_focus": "Testing performance"
            },
            "update_type": "append"
        }
        
        # Act
        performance_monitor.start()
        response = test_client.post("/api/v1/context/update", json=update_request)
        performance_monitor.stop()
        
        # Assert
        assert response.status_code == 200
        duration = performance_monitor.get_duration()
        assert duration is not None
        assert duration < 1.0  # Context update should complete within 1 second
    
    @pytest.mark.asyncio
    async def test_context_search_performance(self, test_client: TestClient, performance_monitor):
        """Test context search performance."""
        # Arrange
        search_request = {
            "query": "performance test query",
            "limit": 10
        }
        
        # Act
        performance_monitor.start()
        response = test_client.post("/api/v1/context/search", json=search_request)
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
    
    @pytest.mark.asyncio
    async def test_context_listing_performance(self, test_client: TestClient, performance_monitor):
        """Test context listing performance."""
        # Act
        performance_monitor.start()
        response = test_client.get("/api/v1/context/")
        performance_monitor.stop()
        
        # Assert
        assert response.status_code == 200
        duration = performance_monitor.get_duration()
        assert duration is not None
        assert duration < 1.0  # Listing should be fast


@pytest.mark.integration
class TestContextAPISecurity:
    """Test context API security features."""
    
    @pytest.mark.asyncio
    async def test_context_update_sql_injection_protection(self, test_client: TestClient):
        """Test protection against SQL injection in context updates."""
        # Arrange
        malicious_request = {
            "conversation_id": "conv_001'; DROP TABLE contexts; --",
            "agent_id": "agent_001",
            "context_data": {
                "malicious_field": "'; DROP TABLE memories; --"
            },
            "update_type": "append"
        }
        
        # Act
        response = test_client.post("/api/v1/context/update", json=malicious_request)
        
        # Assert
        assert response.status_code == 200  # Should handle gracefully
        data = response.json()
        assert data["conversation_id"] == malicious_request["conversation_id"]  # Should be treated as literal string
    
    @pytest.mark.asyncio
    async def test_context_search_xss_protection(self, test_client: TestClient):
        """Test protection against XSS in context search."""
        # Arrange
        malicious_request = {
            "query": "<script>alert('xss')</script>",
            "limit": 10
        }
        
        # Act
        response = test_client.post("/api/v1/context/search", json=malicious_request)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        # The malicious script should be treated as literal text, not executed
        assert "<script>" in data["query"]
    
    @pytest.mark.asyncio
    async def test_context_data_size_limits(self, test_client: TestClient):
        """Test context data size limits."""
        # Arrange - Create very large context data (1MB)
        large_context_data = {
            "topic": "Large context test",
            "participants": ["agent_001"],
            "large_field": "x" * (1024 * 1024)
        }
        
        update_request = {
            "conversation_id": "large_context_conv",
            "agent_id": "agent_001",
            "context_data": large_context_data,
            "update_type": "append"
        }
        
        # Act
        response = test_client.post("/api/v1/context/update", json=update_request)
        
        # Assert
        # Should either succeed or fail gracefully with appropriate error
        assert response.status_code in [200, 413, 422]  # OK, Payload Too Large, or Validation Error


@pytest.mark.integration
class TestContextAPIIntegration:
    """Test context API integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_context_lifecycle_complete(self, test_client: TestClient):
        """Test complete context lifecycle: create -> update -> search -> delete."""
        # Arrange
        conversation_id = "lifecycle_test_conv"
        
        # Step 1: Create initial context
        initial_request = {
            "conversation_id": conversation_id,
            "agent_id": "agent_001",
            "context_data": {
                "topic": "Initial topic",
                "participants": ["agent_001"],
                "current_focus": "Initial focus"
            },
            "update_type": "append"
        }
        
        create_response = test_client.post("/api/v1/context/update", json=initial_request)
        assert create_response.status_code == 200
        
        # Step 2: Update context
        update_request = {
            "conversation_id": conversation_id,
            "agent_id": "agent_001",
            "context_data": {
                "topic": "Updated topic",
                "participants": ["agent_001", "agent_002"],
                "current_focus": "Updated focus",
                "new_field": "new_value"
            },
            "update_type": "merge"
        }
        
        update_response = test_client.post("/api/v1/context/update", json=update_request)
        assert update_response.status_code == 200
        
        # Step 3: Get context
        get_response = test_client.get(f"/api/v1/context/{conversation_id}")
        assert get_response.status_code == 200
        
        # Step 4: Search context
        search_request = {
            "query": "Updated topic",
            "conversation_id": conversation_id,
            "limit": 10
        }
        
        search_response = test_client.post("/api/v1/context/search", json=search_request)
        assert search_response.status_code == 200
        
        # Step 5: Get participants
        participants_response = test_client.get(f"/api/v1/context/{conversation_id}/participants")
        assert participants_response.status_code == 200
        
        # Step 6: Get history
        history_response = test_client.get(f"/api/v1/context/{conversation_id}/history")
        assert history_response.status_code == 200
        
        # Step 7: Delete context
        delete_response = test_client.delete(f"/api/v1/context/{conversation_id}")
        assert delete_response.status_code == 204
    
    @pytest.mark.asyncio
    async def test_multiple_conversations_management(self, test_client: TestClient):
        """Test management of multiple conversations simultaneously."""
        # Arrange
        conversations = [
            {
                "conversation_id": f"multi_conv_{i}",
                "agent_id": f"agent_{i % 3 + 1}",
                "context_data": {
                    "topic": f"Topic {i}",
                    "participants": [f"agent_{i % 3 + 1}"],
                    "current_focus": f"Focus {i}"
                },
                "update_type": "append"
            }
            for i in range(1, 6)
        ]
        
        # Act - Create multiple conversations
        for conv in conversations:
            response = test_client.post("/api/v1/context/update", json=conv)
            assert response.status_code == 200
        
        # Act - List all contexts
        list_response = test_client.get("/api/v1/context/")
        assert list_response.status_code == 200
        
        # Act - Search across all contexts
        search_request = {
            "query": "Topic",
            "limit": 10
        }
        
        search_response = test_client.post("/api/v1/context/search", json=search_request)
        assert search_response.status_code == 200
        
        # Act - Get statistics
        stats_response = test_client.get("/api/v1/context/stats/overview")
        assert stats_response.status_code == 200
        
        # Act - Delete all conversations
        for conv in conversations:
            response = test_client.delete(f"/api/v1/context/{conv['conversation_id']}")
            assert response.status_code == 204
    
    @pytest.mark.asyncio
    async def test_context_update_types_consistency(self, test_client: TestClient):
        """Test consistency between different context update types."""
        # Arrange
        conversation_id = "update_types_test_conv"
        base_context = {
            "topic": "Base topic",
            "participants": ["agent_001"],
            "existing_field": "existing_value"
        }
        
        new_context = {
            "topic": "New topic",
            "participants": ["agent_001", "agent_002"],
            "new_field": "new_value"
        }
        
        # Test append type
        append_request = {
            "conversation_id": f"{conversation_id}_append",
            "agent_id": "agent_001",
            "context_data": {**base_context, **new_context},
            "update_type": "append"
        }
        
        append_response = test_client.post("/api/v1/context/update", json=append_request)
        assert append_response.status_code == 200
        
        # Test replace type
        replace_request = {
            "conversation_id": f"{conversation_id}_replace",
            "agent_id": "agent_001",
            "context_data": new_context,
            "update_type": "replace"
        }
        
        replace_response = test_client.post("/api/v1/context/update", json=replace_request)
        assert replace_response.status_code == 200
        
        # Test merge type
        merge_request = {
            "conversation_id": f"{conversation_id}_merge",
            "agent_id": "agent_001",
            "context_data": new_context,
            "update_type": "merge"
        }
        
        merge_response = test_client.post("/api/v1/context/update", json=merge_request)
        assert merge_response.status_code == 200
        
        # Verify all update types work correctly
        assert append_response.json()["conversation_id"] == f"{conversation_id}_append"
        assert replace_response.json()["conversation_id"] == f"{conversation_id}_replace"
        assert merge_response.json()["conversation_id"] == f"{conversation_id}_merge"

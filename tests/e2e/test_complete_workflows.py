"""
End-to-end tests for complete workflows in AI Agent Memory Router.

This module tests complete user scenarios and workflows that span multiple
API endpoints and system components.
"""

import pytest
import time
from typing import Dict, Any, List
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.e2e
class TestAgentOnboardingWorkflow:
    """Test complete agent onboarding workflow."""
    
    @pytest.mark.asyncio
    async def test_agent_onboarding_complete_workflow(self, test_client: TestClient):
        """Test complete agent onboarding workflow from registration to active operation."""
        # Step 1: Register new agent
        registration_request = {
            "agent_id": "onboarding_agent_001",
            "agent_name": "Onboarding Test Agent",
            "capabilities": ["memory_routing", "context_management", "project_tracking"],
            "metadata": {
                "version": "1.0.0",
                "description": "Agent for testing complete onboarding workflow",
                "team": "testing"
            }
        }
        
        register_response = test_client.post("/api/v1/agents/register", json=registration_request)
        assert register_response.status_code == 201
        agent_data = register_response.json()
        assert agent_data["status"] == "active"
        
        # Step 2: Send initial heartbeat
        heartbeat = {
            "agent_id": "onboarding_agent_001",
            "status": "active",
            "capabilities": ["memory_routing", "context_management", "project_tracking"],
            "metadata": {
                "version": "1.0.0",
                "last_activity": "registration_complete"
            }
        }
        
        heartbeat_response = test_client.post(
            "/api/v1/agents/onboarding_agent_001/heartbeat",
            json=heartbeat
        )
        assert heartbeat_response.status_code == 200
        
        # Step 3: Verify agent appears in agent list
        list_response = test_client.get("/api/v1/agents/")
        assert list_response.status_code == 200
        agents_list = list_response.json()
        
        # Find our agent in the list
        our_agent = next(
            (agent for agent in agents_list["agents"] if agent["agent_id"] == "onboarding_agent_001"),
            None
        )
        assert our_agent is not None
        assert our_agent["status"] == "active"
        
        # Step 4: Get agent capabilities
        capabilities_response = test_client.get("/api/v1/agents/onboarding_agent_001/capabilities")
        assert capabilities_response.status_code == 200
        capabilities = capabilities_response.json()
        assert len(capabilities) > 0
        
        # Step 5: Update agent status to indicate readiness
        status_update = {
            "status": "ready",
            "capabilities": ["memory_routing", "context_management", "project_tracking"],
            "metadata": {
                "version": "1.0.0",
                "status": "ready_for_operations",
                "last_activity": "onboarding_complete"
            }
        }
        
        update_response = test_client.put(
            "/api/v1/agents/onboarding_agent_001/status",
            json=status_update
        )
        assert update_response.status_code == 200
        
        # Step 6: Verify agent statistics reflect the new agent
        stats_response = test_client.get("/api/v1/agents/stats/overview")
        assert stats_response.status_code == 200
        stats = stats_response.json()
        assert stats["total_agents"] > 0
        
        # Step 7: Clean up - deregister agent
        deregister_response = test_client.delete("/api/v1/agents/onboarding_agent_001")
        assert deregister_response.status_code == 204


@pytest.mark.e2e
class TestMemoryRoutingWorkflow:
    """Test complete memory routing workflow."""
    
    @pytest.mark.asyncio
    async def test_memory_routing_complete_workflow(self, test_client: TestClient):
        """Test complete memory routing workflow from creation to delivery."""
        # Step 1: Register source agent
        source_agent_request = {
            "agent_id": "source_agent_001",
            "agent_name": "Source Agent",
            "capabilities": ["memory_routing", "information_sharing"],
            "metadata": {"role": "information_provider"}
        }
        
        source_register_response = test_client.post("/api/v1/agents/register", json=source_agent_request)
        assert source_register_response.status_code == 201
        
        # Step 2: Register target agents
        target_agents = ["target_agent_001", "target_agent_002", "target_agent_003"]
        for agent_id in target_agents:
            target_agent_request = {
                "agent_id": agent_id,
                "agent_name": f"Target Agent {agent_id}",
                "capabilities": ["memory_routing", "information_processing"],
                "metadata": {"role": "information_consumer"}
            }
            
            target_register_response = test_client.post("/api/v1/agents/register", json=target_agent_request)
            assert target_register_response.status_code == 201
        
        # Step 3: Create conversation context
        context_request = {
            "conversation_id": "memory_routing_conv_001",
            "agent_id": "source_agent_001",
            "context_data": {
                "topic": "Project status update",
                "participants": ["source_agent_001"] + target_agents,
                "current_focus": "Sharing critical project information",
                "priority": "high"
            },
            "update_type": "append"
        }
        
        context_response = test_client.post("/api/v1/context/update", json=context_request)
        assert context_response.status_code == 200
        
        # Step 4: Route memory to target agents
        memory_route_request = {
            "source_agent_id": "source_agent_001",
            "target_agent_ids": target_agents,
            "memory_content": "Critical project milestone achieved: Phase 1 development completed successfully. All tests passing, ready for Phase 2 deployment.",
            "priority": "high",
            "context": {
                "project_id": "proj_123",
                "milestone": "phase_1_complete",
                "urgency": "immediate",
                "conversation_id": "memory_routing_conv_001"
            }
        }
        
        route_response = test_client.post("/api/v1/memory/route", json=memory_route_request)
        assert route_response.status_code == 201
        route_data = route_response.json()
        
        assert route_data["source_agent_id"] == "source_agent_001"
        assert route_data["target_agent_ids"] == target_agents
        assert route_data["status"] == "routed"
        assert route_data["priority"] == "high"
        
        # Step 5: Verify route details
        route_details_response = test_client.get(f"/api/v1/memory/route/{route_data['route_id']}")
        assert route_details_response.status_code == 200
        route_details = route_details_response.json()
        assert route_details["route_id"] == route_data["route_id"]
        
        # Step 6: Update context with routing information
        context_update_request = {
            "conversation_id": "memory_routing_conv_001",
            "agent_id": "source_agent_001",
            "context_data": {
                "last_route_id": route_data["route_id"],
                "routing_status": "completed",
                "target_agents_notified": target_agents,
                "routing_timestamp": route_data["timestamp"]
            },
            "update_type": "merge"
        }
        
        context_update_response = test_client.post("/api/v1/context/update", json=context_update_request)
        assert context_update_response.status_code == 200
        
        # Step 7: Verify memory statistics reflect the routing
        memory_stats_response = test_client.get("/api/v1/memory/stats/overview")
        assert memory_stats_response.status_code == 200
        memory_stats = memory_stats_response.json()
        assert memory_stats["total_routes"] > 0
        
        # Step 8: Clean up - deregister all agents
        for agent_id in ["source_agent_001"] + target_agents:
            deregister_response = test_client.delete(f"/api/v1/agents/{agent_id}")
            assert deregister_response.status_code == 204


@pytest.mark.e2e
class TestCipherIntegrationWorkflow:
    """Test complete Cipher integration workflow."""
    
    @pytest.mark.asyncio
    async def test_cipher_integration_complete_workflow(self, test_client: TestClient, mock_cipher_service):
        """Test complete Cipher integration workflow from project creation to memory management."""
        # Mock Cipher service responses
        mock_cipher_service.create_project.return_value = True
        mock_cipher_service.store_memory.return_value = "cipher_memory_123"
        mock_cipher_service.retrieve_memory.return_value = {
            "content": "Retrieved memory content",
            "metadata": {"retrieved": True},
            "from_cache": False
        }
        mock_cipher_service.search_memories.return_value = {
            "results": [
                {
                    "memory_id": "mem_1",
                    "content": "Search result 1",
                    "relevance_score": 0.9
                }
            ],
            "total": 1,
            "from_cache": False
        }
        mock_cipher_service.update_memory.return_value = True
        mock_cipher_service.delete_memory.return_value = True
        
        # Step 1: Create project in Cipher
        project_request = {
            "project_id": "cipher_test_project",
            "project_name": "Cipher Integration Test Project",
            "description": "A test project for Cipher integration workflow",
            "metadata": {
                "test": True,
                "workflow": "integration_test"
            }
        }
        
        project_response = test_client.post("/api/v1/memory/cipher/projects", json=project_request)
        assert project_response.status_code == 201
        project_data = project_response.json()
        assert project_data["status"] == "created"
        
        # Step 2: Verify project creation
        project_get_response = test_client.get("/api/v1/memory/cipher/projects/cipher_test_project")
        assert project_get_response.status_code == 200
        project_info = project_get_response.json()
        assert project_info["status"] == "found"
        
        # Step 3: Store memory in Cipher
        memory_store_request = {
            "project_id": "cipher_test_project",
            "agent_id": "cipher_test_agent",
            "memory_content": "Important project information: All systems operational, ready for next phase.",
            "memory_type": "project_status",
            "tags": ["status", "operational", "ready"],
            "metadata": {
                "phase": "testing",
                "priority": "high"
            },
            "priority": 8
        }
        
        store_response = test_client.post("/api/v1/memory/cipher/store", json=memory_store_request)
        assert store_response.status_code == 201
        store_data = store_response.json()
        assert store_data["status"] == "stored"
        memory_id = store_data["memory_id"]
        
        # Step 4: Retrieve memory from Cipher
        retrieve_response = test_client.get(
            f"/api/v1/memory/cipher/retrieve/cipher_test_project/{memory_id}"
        )
        assert retrieve_response.status_code == 200
        retrieve_data = retrieve_response.json()
        assert retrieve_data["status"] == "retrieved"
        assert retrieve_data["memory_data"] is not None
        
        # Step 5: Search memories in Cipher
        search_request = {
            "project_id": "cipher_test_project",
            "query": "operational systems",
            "agent_id": "cipher_test_agent",
            "limit": 10
        }
        
        search_response = test_client.post("/api/v1/memory/cipher/search", json=search_request)
        assert search_response.status_code == 200
        search_data = search_response.json()
        assert len(search_data["results"]) > 0
        
        # Step 6: Update memory in Cipher
        update_request = {
            "content": "Updated project information: All systems operational and optimized, ready for production deployment.",
            "tags": ["status", "operational", "optimized", "production_ready"],
            "metadata": {
                "phase": "production_ready",
                "priority": "critical"
            }
        }
        
        update_response = test_client.put(
            f"/api/v1/memory/cipher/update/cipher_test_project/{memory_id}",
            json=update_request
        )
        assert update_response.status_code == 200
        update_data = update_response.json()
        assert update_data["status"] == "updated"
        
        # Step 7: Verify Cipher health
        health_response = test_client.get("/api/v1/memory/cipher/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"
        
        # Step 8: Delete memory from Cipher
        delete_response = test_client.delete(
            f"/api/v1/memory/cipher/delete/cipher_test_project/{memory_id}"
        )
        assert delete_response.status_code == 200
        delete_data = delete_response.json()
        assert delete_data["status"] == "deleted"


@pytest.mark.e2e
class TestMultiAgentCollaborationWorkflow:
    """Test multi-agent collaboration workflow."""
    
    @pytest.mark.asyncio
    async def test_multi_agent_collaboration_workflow(self, test_client: TestClient):
        """Test complete multi-agent collaboration workflow."""
        # Step 1: Register multiple agents with different roles
        agents = [
            {
                "agent_id": "project_manager_001",
                "agent_name": "Project Manager Agent",
                "capabilities": ["project_tracking", "coordination", "memory_routing"],
                "metadata": {"role": "project_manager", "team": "management"}
            },
            {
                "agent_id": "developer_001",
                "agent_name": "Developer Agent",
                "capabilities": ["development", "code_review", "memory_routing"],
                "metadata": {"role": "developer", "team": "engineering"}
            },
            {
                "agent_id": "tester_001",
                "agent_name": "Tester Agent",
                "capabilities": ["testing", "quality_assurance", "memory_routing"],
                "metadata": {"role": "tester", "team": "qa"}
            },
            {
                "agent_id": "analyst_001",
                "agent_name": "Analyst Agent",
                "capabilities": ["analysis", "reporting", "memory_routing"],
                "metadata": {"role": "analyst", "team": "analytics"}
            }
        ]
        
        for agent in agents:
            register_response = test_client.post("/api/v1/agents/register", json=agent)
            assert register_response.status_code == 201
        
        # Step 2: Create shared conversation context
        conversation_id = "multi_agent_collab_001"
        context_request = {
            "conversation_id": conversation_id,
            "agent_id": "project_manager_001",
            "context_data": {
                "topic": "Sprint Planning and Coordination",
                "participants": [agent["agent_id"] for agent in agents],
                "current_focus": "Planning next sprint deliverables",
                "sprint_number": 5,
                "project": "AI Agent Memory Router"
            },
            "update_type": "append"
        }
        
        context_response = test_client.post("/api/v1/context/update", json=context_request)
        assert context_response.status_code == 200
        
        # Step 3: Project Manager shares sprint goals
        sprint_goals_route = {
            "source_agent_id": "project_manager_001",
            "target_agent_ids": ["developer_001", "tester_001", "analyst_001"],
            "memory_content": "Sprint 5 Goals: 1) Complete memory routing optimization 2) Implement advanced search features 3) Add performance monitoring 4) Prepare for production deployment",
            "priority": "high",
            "context": {
                "conversation_id": conversation_id,
                "sprint": 5,
                "type": "sprint_goals"
            }
        }
        
        goals_route_response = test_client.post("/api/v1/memory/route", json=sprint_goals_route)
        assert goals_route_response.status_code == 201
        
        # Step 4: Developer shares progress update
        developer_progress_route = {
            "source_agent_id": "developer_001",
            "target_agent_ids": ["project_manager_001", "tester_001"],
            "memory_content": "Development Progress: Memory routing optimization 80% complete, advanced search features 60% complete. Estimated completion: 2 days ahead of schedule.",
            "priority": "normal",
            "context": {
                "conversation_id": conversation_id,
                "sprint": 5,
                "type": "progress_update"
            }
        }
        
        progress_route_response = test_client.post("/api/v1/memory/route", json=developer_progress_route)
        assert progress_route_response.status_code == 201
        
        # Step 5: Tester shares test results
        test_results_route = {
            "source_agent_id": "tester_001",
            "target_agent_ids": ["project_manager_001", "developer_001"],
            "memory_content": "Test Results: All existing functionality tests passing. New optimization features tested successfully. Performance improvements confirmed: 25% faster routing, 40% reduced memory usage.",
            "priority": "normal",
            "context": {
                "conversation_id": conversation_id,
                "sprint": 5,
                "type": "test_results"
            }
        }
        
        test_route_response = test_client.post("/api/v1/memory/route", json=test_results_route)
        assert test_route_response.status_code == 201
        
        # Step 6: Analyst shares performance analysis
        analysis_route = {
            "source_agent_id": "analyst_001",
            "target_agent_ids": ["project_manager_001", "developer_001", "tester_001"],
            "memory_content": "Performance Analysis: System metrics show significant improvements. User satisfaction scores increased by 30%. Recommendation: Proceed with production deployment as planned.",
            "priority": "high",
            "context": {
                "conversation_id": conversation_id,
                "sprint": 5,
                "type": "performance_analysis"
            }
        }
        
        analysis_route_response = test_client.post("/api/v1/memory/route", json=analysis_route)
        assert analysis_route_response.status_code == 201
        
        # Step 7: Update conversation context with final decisions
        final_context_update = {
            "conversation_id": conversation_id,
            "agent_id": "project_manager_001",
            "context_data": {
                "final_decisions": [
                    "Proceed with production deployment",
                    "Schedule deployment for next week",
                    "All team members ready for production support"
                ],
                "sprint_status": "completed_successfully",
                "next_sprint_planning": "scheduled_for_next_week"
            },
            "update_type": "merge"
        }
        
        final_context_response = test_client.post("/api/v1/context/update", json=final_context_update)
        assert final_context_response.status_code == 200
        
        # Step 8: Verify conversation participants
        participants_response = test_client.get(f"/api/v1/context/{conversation_id}/participants")
        assert participants_response.status_code == 200
        participants_data = participants_response.json()
        assert len(participants_data["participants"]) == 4
        
        # Step 9: Verify conversation history
        history_response = test_client.get(f"/api/v1/context/{conversation_id}/history")
        assert history_response.status_code == 200
        history_data = history_response.json()
        assert len(history_data["history"]) > 0
        
        # Step 10: Search conversation context
        context_search_request = {
            "query": "sprint goals production deployment",
            "conversation_id": conversation_id,
            "limit": 10
        }
        
        context_search_response = test_client.post("/api/v1/context/search", json=context_search_request)
        assert context_search_response.status_code == 200
        context_search_data = context_search_response.json()
        assert len(context_search_data["results"]) > 0
        
        # Step 11: Get final statistics
        memory_stats_response = test_client.get("/api/v1/memory/stats/overview")
        assert memory_stats_response.status_code == 200
        
        agent_stats_response = test_client.get("/api/v1/agents/stats/overview")
        assert agent_stats_response.status_code == 200
        
        context_stats_response = test_client.get("/api/v1/context/stats/overview")
        assert context_stats_response.status_code == 200
        
        # Step 12: Clean up - deregister all agents
        for agent in agents:
            deregister_response = test_client.delete(f"/api/v1/agents/{agent['agent_id']}")
            assert deregister_response.status_code == 204


@pytest.mark.e2e
class TestSystemHealthMonitoringWorkflow:
    """Test system health monitoring workflow."""
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring_workflow(self, test_client: TestClient):
        """Test complete system health monitoring workflow."""
        # Step 1: Check basic health
        health_response = test_client.get("/api/v1/health/")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"
        
        # Step 2: Check detailed health
        detailed_health_response = test_client.get("/api/v1/health/detailed")
        assert detailed_health_response.status_code == 200
        detailed_health_data = detailed_health_response.json()
        assert detailed_health_data["overall_status"] in ["healthy", "degraded"]
        
        # Step 3: Check component health individually
        db_health_response = test_client.get("/api/v1/health/database")
        assert db_health_response.status_code == 200
        
        mcp_health_response = test_client.get("/api/v1/health/mcp")
        assert mcp_health_response.status_code == 200
        
        # Step 4: Check readiness
        readiness_response = test_client.get("/api/v1/health/ready")
        assert readiness_response.status_code == 200
        readiness_data = readiness_response.json()
        assert readiness_data["ready"] is True
        
        # Step 5: Check liveness
        liveness_response = test_client.get("/api/v1/health/live")
        assert liveness_response.status_code == 200
        liveness_data = liveness_response.json()
        assert liveness_data["alive"] is True
        
        # Step 6: Get system metrics
        metrics_response = test_client.get("/api/v1/health/metrics")
        assert metrics_response.status_code == 200
        metrics_data = metrics_response.json()
        assert "metrics" in metrics_data
        assert "performance" in metrics_data
        
        # Step 7: Get system status
        status_response = test_client.get("/api/v1/health/status")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["status"] in ["operational", "degraded", "down"]
        
        # Step 8: Perform some operations to generate metrics
        # Register a test agent
        agent_request = {
            "agent_id": "health_monitor_agent",
            "agent_name": "Health Monitor Agent",
            "capabilities": ["monitoring", "health_checking"],
            "metadata": {"role": "monitor"}
        }
        
        agent_response = test_client.post("/api/v1/agents/register", json=agent_request)
        assert agent_response.status_code == 201
        
        # Route some memory
        memory_route_request = {
            "source_agent_id": "health_monitor_agent",
            "target_agent_ids": ["health_monitor_agent"],
            "memory_content": "Health monitoring test memory",
            "priority": "normal"
        }
        
        memory_route_response = test_client.post("/api/v1/memory/route", json=memory_route_request)
        assert memory_route_response.status_code == 201
        
        # Update context
        context_request = {
            "conversation_id": "health_monitor_conv",
            "agent_id": "health_monitor_agent",
            "context_data": {
                "topic": "Health monitoring",
                "participants": ["health_monitor_agent"],
                "current_focus": "System health verification"
            },
            "update_type": "append"
        }
        
        context_response = test_client.post("/api/v1/context/update", json=context_request)
        assert context_response.status_code == 200
        
        # Step 9: Check metrics again to see changes
        updated_metrics_response = test_client.get("/api/v1/health/metrics")
        assert updated_metrics_response.status_code == 200
        updated_metrics_data = updated_metrics_response.json()
        
        # Verify metrics have been updated
        assert updated_metrics_data["metrics"]["requests_total"] > 0
        assert updated_metrics_data["metrics"]["agents_total"] > 0
        
        # Step 10: Clean up
        deregister_response = test_client.delete("/api/v1/agents/health_monitor_agent")
        assert deregister_response.status_code == 204


@pytest.mark.e2e
class TestErrorHandlingAndRecoveryWorkflow:
    """Test error handling and recovery workflow."""
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery_workflow(self, test_client: TestClient):
        """Test system behavior under error conditions and recovery."""
        # Step 1: Test invalid agent registration
        invalid_agent_request = {
            "agent_id": "",  # Invalid empty ID
            "agent_name": "Invalid Agent",
            "capabilities": []
        }
        
        invalid_register_response = test_client.post("/api/v1/agents/register", json=invalid_agent_request)
        assert invalid_register_response.status_code == 422  # Validation error
        
        # Step 2: Test invalid memory routing
        invalid_route_request = {
            "source_agent_id": "nonexistent_agent",
            "target_agent_ids": [],
            "memory_content": "",
            "priority": "invalid_priority"
        }
        
        invalid_route_response = test_client.post("/api/v1/memory/route", json=invalid_route_request)
        assert invalid_route_response.status_code == 422  # Validation error
        
        # Step 3: Test invalid context update
        invalid_context_request = {
            "conversation_id": "",
            "agent_id": "",
            "context_data": {},
            "update_type": "invalid_type"
        }
        
        invalid_context_response = test_client.post("/api/v1/context/update", json=invalid_context_request)
        assert invalid_context_response.status_code == 422  # Validation error
        
        # Step 4: Test system recovery - verify system is still healthy after errors
        health_response = test_client.get("/api/v1/health/")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"
        
        # Step 5: Test valid operations still work after errors
        valid_agent_request = {
            "agent_id": "recovery_test_agent",
            "agent_name": "Recovery Test Agent",
            "capabilities": ["testing", "recovery"],
            "metadata": {"role": "recovery_tester"}
        }
        
        valid_register_response = test_client.post("/api/v1/agents/register", json=valid_agent_request)
        assert valid_register_response.status_code == 201
        
        # Step 6: Test valid memory routing
        valid_route_request = {
            "source_agent_id": "recovery_test_agent",
            "target_agent_ids": ["recovery_test_agent"],
            "memory_content": "Recovery test memory content",
            "priority": "normal"
        }
        
        valid_route_response = test_client.post("/api/v1/memory/route", json=valid_route_request)
        assert valid_route_response.status_code == 201
        
        # Step 7: Test valid context update
        valid_context_request = {
            "conversation_id": "recovery_test_conv",
            "agent_id": "recovery_test_agent",
            "context_data": {
                "topic": "Recovery testing",
                "participants": ["recovery_test_agent"],
                "current_focus": "System recovery verification"
            },
            "update_type": "append"
        }
        
        valid_context_response = test_client.post("/api/v1/context/update", json=valid_context_request)
        assert valid_context_response.status_code == 200
        
        # Step 8: Verify system metrics show error handling
        metrics_response = test_client.get("/api/v1/health/metrics")
        assert metrics_response.status_code == 200
        metrics_data = metrics_response.json()
        
        # System should have handled errors gracefully
        assert metrics_data["metrics"]["requests_total"] > 0
        
        # Step 9: Clean up
        deregister_response = test_client.delete("/api/v1/agents/recovery_test_agent")
        assert deregister_response.status_code == 204


@pytest.mark.e2e
class TestPerformanceUnderLoadWorkflow:
    """Test system performance under load."""
    
    @pytest.mark.asyncio
    async def test_performance_under_load_workflow(self, test_client: TestClient, performance_monitor):
        """Test system performance under simulated load."""
        # Step 1: Register multiple agents quickly
        agents = []
        for i in range(5):
            agent_request = {
                "agent_id": f"load_test_agent_{i}",
                "agent_name": f"Load Test Agent {i}",
                "capabilities": ["load_testing", "performance"],
                "metadata": {"role": "load_tester", "index": i}
            }
            
            performance_monitor.start()
            register_response = test_client.post("/api/v1/agents/register", json=agent_request)
            performance_monitor.stop()
            
            assert register_response.status_code == 201
            duration = performance_monitor.get_duration()
            assert duration < 1.0  # Each registration should be fast
            
            agents.append(agent_request["agent_id"])
        
        # Step 2: Send multiple memory routes quickly
        for i in range(10):
            route_request = {
                "source_agent_id": f"load_test_agent_{i % 5}",
                "target_agent_ids": [f"load_test_agent_{(i + 1) % 5}"],
                "memory_content": f"Load test memory content {i}",
                "priority": "normal"
            }
            
            performance_monitor.start()
            route_response = test_client.post("/api/v1/memory/route", json=route_request)
            performance_monitor.stop()
            
            assert route_response.status_code == 201
            duration = performance_monitor.get_duration()
            assert duration < 2.0  # Each route should be reasonably fast
        
        # Step 3: Update multiple contexts quickly
        for i in range(5):
            context_request = {
                "conversation_id": f"load_test_conv_{i}",
                "agent_id": f"load_test_agent_{i}",
                "context_data": {
                    "topic": f"Load test topic {i}",
                    "participants": [f"load_test_agent_{i}"],
                    "current_focus": f"Load testing focus {i}"
                },
                "update_type": "append"
            }
            
            performance_monitor.start()
            context_response = test_client.post("/api/v1/context/update", json=context_request)
            performance_monitor.stop()
            
            assert context_response.status_code == 200
            duration = performance_monitor.get_duration()
            assert duration < 1.0  # Each context update should be fast
        
        # Step 4: Perform multiple searches quickly
        for i in range(5):
            search_request = {
                "query": f"load test query {i}",
                "limit": 10
            }
            
            performance_monitor.start()
            search_response = test_client.post("/api/v1/memory/search", json=search_request)
            performance_monitor.stop()
            
            assert search_response.status_code == 200
            duration = performance_monitor.get_duration()
            assert duration < 2.0  # Each search should be reasonably fast
        
        # Step 5: Check system health under load
        health_response = test_client.get("/api/v1/health/")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["status"] == "healthy"
        
        # Step 6: Check system metrics under load
        metrics_response = test_client.get("/api/v1/health/metrics")
        assert metrics_response.status_code == 200
        metrics_data = metrics_response.json()
        
        # Verify system handled the load
        assert metrics_data["metrics"]["requests_total"] > 0
        assert metrics_data["metrics"]["agents_total"] >= 5
        
        # Step 7: Clean up - deregister all agents
        for agent_id in agents:
            deregister_response = test_client.delete(f"/api/v1/agents/{agent_id}")
            assert deregister_response.status_code == 204

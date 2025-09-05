#!/usr/bin/env python3
"""
Test script for Weaviate integration with AI Agent Memory Router.

This script tests the complete Weaviate integration including:
- Weaviate client functionality
- Memory storage and retrieval
- Semantic search
- Cross-project knowledge sharing
- Memory deduplication
- SQLite metadata integration
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from uuid import uuid4

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.config import get_settings
from app.core.weaviate_client import WeaviateClient
from app.core.sqlite_storage import SQLiteUnifiedStorage
from app.services.weaviate_service import WeaviateMemoryService
from app.services.cross_project_service import CrossProjectSharingService
from app.services.deduplication_service import DeduplicationService
from app.models.memory import MemoryStoreCreate, MemoryContent, MemorySearch
from app.models.memory import AgentSource

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeaviateIntegrationTester:
    """Test class for Weaviate integration."""
    
    def __init__(self):
        """Initialize the tester."""
        self.settings = get_settings()
        self.weaviate_client = None
        self.sqlite_storage = None
        self.weaviate_service = None
        self.cross_project_service = None
        self.deduplication_service = None
        self.test_results = {}
    
    async def setup(self):
        """Setup test environment."""
        try:
            logger.info("Setting up test environment...")
            
            # Initialize SQLite storage
            db_path = "data/test_weaviate_integration.db"
            self.sqlite_storage = SQLiteUnifiedStorage(db_path, max_connections=5)
            await self.sqlite_storage.initialize()
            
            # Initialize Weaviate client
            self.weaviate_client = WeaviateClient()
            await self.weaviate_client.initialize()
            
            # Note: Clear operation not working with current Weaviate client version
            # await self.weaviate_client.clear_all_memories()
            
            # Initialize services
            self.weaviate_service = WeaviateMemoryService(self.sqlite_storage)
            await self.weaviate_service.initialize()
            
            self.cross_project_service = CrossProjectSharingService(
                self.weaviate_service, self.sqlite_storage
            )
            await self.cross_project_service.initialize()
            
            self.deduplication_service = DeduplicationService(
                self.weaviate_service, self.sqlite_storage
            )
            await self.deduplication_service.initialize()
            
            # Create test agents
            await self._create_test_agents()
            
            logger.info("Test environment setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
            return False
    
    async def _create_test_agents(self):
        """Create test agents for integration testing."""
        try:
            logger.info("Creating test agents...")
            
            # Define test agents with their properties
            test_agents = [
                {
                    "id": "test-agent-1",
                    "name": "Test Agent 1",
                    "type": "assistant",
                    "version": "1.0.0"
                },
                {
                    "id": "test-agent-2", 
                    "name": "Test Agent 2",
                    "type": "assistant",
                    "version": "1.0.0"
                },
                {
                    "id": "test-agent-3",
                    "name": "Test Agent 3", 
                    "type": "tool",
                    "version": "2.0.0"
                },
                {
                    "id": "project1-agent",
                    "name": "Project 1 Agent",
                    "type": "assistant",
                    "version": "1.0.0"
                },
                {
                    "id": "project2-agent",
                    "name": "Project 2 Agent",
                    "type": "assistant", 
                    "version": "1.0.0"
                }
            ]
            
            # Create each test agent
            for agent_data in test_agents:
                success = await self.sqlite_storage.agent.create_agent_if_not_exists(
                    agent_id=agent_data["id"],
                    agent_name=agent_data["name"],
                    agent_type=agent_data["type"],
                    version=agent_data["version"]
                )
                if success:
                    logger.info(f"Created test agent: {agent_data['id']}")
                else:
                    logger.warning(f"Failed to create test agent: {agent_data['id']}")
            
            logger.info("Test agent creation completed")
            
        except Exception as e:
            logger.error(f"Failed to create test agents: {e}")
            raise
    
    async def _cleanup_test_data(self):
        """Clean up test data after tests complete."""
        try:
            logger.info("Cleaning up test data...")
            
            # Clean up test agents
            test_agent_ids = [
                "test-agent-1", "test-agent-2", "test-agent-3",
                "project1-agent", "project2-agent"
            ]
            
            for agent_id in test_agent_ids:
                try:
                    await self.sqlite_storage.agent.delete_agent(agent_id)
                    logger.debug(f"Cleaned up test agent: {agent_id}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup agent {agent_id}: {e}")
            
            logger.info("Test data cleanup completed")
            
        except Exception as e:
            logger.error(f"Failed to cleanup test data: {e}")
    
    async def test_weaviate_client(self):
        """Test Weaviate client functionality."""
        try:
            logger.info("Testing Weaviate client...")
            
            # Test collection stats
            stats = await self.weaviate_client.get_collection_stats()
            logger.info(f"Collection stats: {stats}")
            
            # Test storing a memory
            test_memory_id = str(uuid4())
            success = await self.weaviate_client.store_memory(
                memory_id=test_memory_id,
                content="This is a test memory for Weaviate integration testing.",
                agent_id="test-agent-1",
                memory_type="knowledge",
                importance=8,
                tags=["test", "integration"],
                metadata={"test": True},
                project_id="test-project"
            )
            
            if not success:
                raise Exception("Failed to store test memory")
            
            # Test retrieving the memory
            retrieved_memory = await self.weaviate_client.get_memory(test_memory_id)
            if not retrieved_memory:
                raise Exception("Failed to retrieve test memory")
            
            # Test semantic search
            search_results = await self.weaviate_client.search_memories(
                query="test memory integration",
                limit=5
            )
            
            if not search_results:
                raise Exception("No search results found")
            
            # Clean up test memory
            await self.weaviate_client.delete_memory(test_memory_id)
            
            self.test_results["weaviate_client"] = "PASSED"
            logger.info("Weaviate client test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"Weaviate client test FAILED: {e}")
            self.test_results["weaviate_client"] = f"FAILED: {e}"
            return False
    
    async def test_memory_service(self):
        """Test Weaviate memory service."""
        try:
            logger.info("Testing Weaviate memory service...")
            
            # Create test memory
            test_memory = MemoryStoreCreate(
                content=MemoryContent(
                    text="This is a test memory for the Weaviate memory service.",
                    language="en",
                    format="text",
                    encoding="utf-8"
                ),
                source=AgentSource(
                    agent_id="test-agent-2",
                    project_id="test-project"
                ),
                memory_type="knowledge",
                importance=9
            )
            
            # Store memory
            stored_memory = await self.weaviate_service.store_memory(test_memory)
            if not stored_memory:
                raise Exception("Failed to store memory via service")
            
            # Retrieve memory
            retrieved_memory = await self.weaviate_service.get_memory(stored_memory.id)
            if not retrieved_memory:
                raise Exception("Failed to retrieve memory via service")
            
            # Search memories with a more specific query to avoid old corrupted data
            search_query = MemorySearch(
                query="This is a test memory for the Weaviate memory service",
                agent_id="test-agent-2",  # Filter by the specific agent
                limit=10
            )
            search_results = await self.weaviate_service.search_memories(search_query)
            if not search_results.results:
                raise Exception("No search results found via service")
            
            # Update memory
            update_success = await self.weaviate_service.update_memory(
                stored_memory.id,
                {"importance": 9}
            )
            if not update_success:
                raise Exception("Failed to update memory via service")
            
            # Delete memory
            delete_success = await self.weaviate_service.delete_memory(stored_memory.id)
            if not delete_success:
                raise Exception("Failed to delete memory via service")
            
            self.test_results["memory_service"] = "PASSED"
            logger.info("Weaviate memory service test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"Weaviate memory service test FAILED: {e}")
            self.test_results["memory_service"] = f"FAILED: {e}"
            return False
    
    async def test_cross_project_sharing(self):
        """Test cross-project knowledge sharing."""
        try:
            logger.info("Testing cross-project knowledge sharing...")
            
            # Create memories in different projects
            project1_memory = MemoryStoreCreate(
                content=MemoryContent(
                    text="This is knowledge from project 1 about machine learning algorithms.",
                    tags=["ml", "algorithms", "project1"],
                    metadata={"project": "project1"}
                ),
                source=AgentSource(
                    agent_id="agent-project1",
                    project_id="project1"
                ),
                memory_type="knowledge",
                importance=8
            )
            
            project2_memory = MemoryStoreCreate(
                content=MemoryContent(
                    text="This is knowledge from project 2 about data preprocessing techniques.",
                    tags=["data", "preprocessing", "project2"],
                    metadata={"project": "project2"}
                ),
                source=AgentSource(
                    agent_id="agent-project2",
                    project_id="project2"
                ),
                memory_type="knowledge",
                importance=7
            )
            
            # Store memories
            stored_memory1 = await self.weaviate_service.store_memory(project1_memory)
            stored_memory2 = await self.weaviate_service.store_memory(project2_memory)
            
            # Test cross-project knowledge sharing
            sharing_result = await self.cross_project_service.share_knowledge_across_projects(
                source_project_id="project1",
                query="machine learning data preprocessing",
                target_project_ids=["project2"],
                sharing_level="standard",
                max_results=5
            )
            
            if "error" in sharing_result:
                raise Exception(f"Cross-project sharing failed: {sharing_result['error']}")
            
            # Test knowledge discovery
            discovery_results = await self.cross_project_service.discover_related_knowledge(
                project_id="project1",
                memory_content="machine learning algorithms",
                similarity_threshold=0.6,
                max_results=3
            )
            
            # Test project knowledge summary
            summary = await self.cross_project_service.get_project_knowledge_summary(
                project_id="project1",
                include_cross_project=True
            )
            
            if "error" in summary:
                raise Exception(f"Knowledge summary failed: {summary['error']}")
            
            # Clean up test memories
            await self.weaviate_service.delete_memory(stored_memory1.id)
            await self.weaviate_service.delete_memory(stored_memory2.id)
            
            self.test_results["cross_project_sharing"] = "PASSED"
            logger.info("Cross-project knowledge sharing test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"Cross-project knowledge sharing test FAILED: {e}")
            self.test_results["cross_project_sharing"] = f"FAILED: {e}"
            return False
    
    async def test_deduplication(self):
        """Test memory deduplication."""
        try:
            logger.info("Testing memory deduplication...")
            
            # Create duplicate memories
            duplicate_memory1 = MemoryStoreCreate(
                content=MemoryContent(
                    text="This is a test memory about Python programming.",
                    tags=["python", "programming"],
                    metadata={"test": True}
                ),
                source=AgentSource(
                    agent_id="test-agent-3",
                    project_id="test-project"
                ),
                memory_type="knowledge",
                importance=8
            )
            
            duplicate_memory2 = MemoryStoreCreate(
                content=MemoryContent(
                    text="This is a test memory about Python programming.",  # Same content
                    tags=["python", "programming"],
                    metadata={"test": True}
                ),
                source=AgentSource(
                    agent_id="test-agent-3",
                    project_id="test-project"
                ),
                memory_type="knowledge",
                importance=7
            )
            
            # Store duplicate memories
            stored_memory1 = await self.weaviate_service.store_memory(duplicate_memory1)
            stored_memory2 = await self.weaviate_service.store_memory(duplicate_memory2)
            
            # Test duplicate detection
            duplicate_analysis = await self.deduplication_service.find_duplicates(
                agent_id="test-agent-3",
                similarity_threshold=0.9
            )
            
            if "error" in duplicate_analysis:
                raise Exception(f"Duplicate detection failed: {duplicate_analysis['error']}")
            
            # Test auto-deduplication
            auto_dedup_result = await self.deduplication_service.auto_deduplicate(
                agent_id="test-agent-3",
                similarity_threshold=0.95,
                auto_resolve=True
            )
            
            if "error" in auto_dedup_result:
                raise Exception(f"Auto-deduplication failed: {auto_dedup_result['error']}")
            
            # Clean up remaining test memories
            await self.weaviate_service.delete_memory(stored_memory1.id)
            await self.weaviate_service.delete_memory(stored_memory2.id)
            
            self.test_results["deduplication"] = "PASSED"
            logger.info("Memory deduplication test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"Memory deduplication test FAILED: {e}")
            self.test_results["deduplication"] = f"FAILED: {e}"
            return False
    
    async def test_sqlite_integration(self):
        """Test SQLite metadata integration."""
        try:
            logger.info("Testing SQLite metadata integration...")
            
            # Test storing memory with SQLite metadata
            test_memory = MemoryStoreCreate(
                content=MemoryContent(
                    text="This memory tests SQLite metadata integration.",
                    tags=["sqlite", "metadata", "integration"],
                    metadata={"sqlite_test": True}
                ),
                source=AgentSource(
                    agent_id="test-agent-4",
                    project_id="test-project"
                ),
                memory_type="knowledge",
                importance=6
            )
            
            # Store memory (should store in both Weaviate and SQLite)
            stored_memory = await self.weaviate_service.store_memory(test_memory)
            
            # Verify SQLite metadata was stored
            metadata = await self.sqlite_storage.metadata.get_metadata(stored_memory.id)
            if not metadata:
                raise Exception("Metadata not found in SQLite")
            
            # Verify metadata fields
            if metadata.source != "weaviate":
                raise Exception("Incorrect metadata source")
            
            if metadata.vector_dimension != self.settings.weaviate.vector_dimension:
                raise Exception("Incorrect vector dimension")
            
            # Test updating metadata
            update_success = await self.sqlite_storage.metadata.update_metadata(
                stored_memory.id,
                {"confidence": 0.9}
            )
            if not update_success:
                raise Exception("Failed to update SQLite metadata")
            
            # Clean up
            await self.weaviate_service.delete_memory(stored_memory.id)
            
            self.test_results["sqlite_integration"] = "PASSED"
            logger.info("SQLite metadata integration test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"SQLite metadata integration test FAILED: {e}")
            self.test_results["sqlite_integration"] = f"FAILED: {e}"
            return False
    
    async def cleanup(self):
        """Cleanup test environment."""
        try:
            logger.info("Cleaning up test environment...")
            
            # Clean up test data first
            await self._cleanup_test_data()
            
            if self.deduplication_service:
                await self.deduplication_service.close()
            
            if self.cross_project_service:
                await self.cross_project_service.close()
            
            if self.weaviate_service:
                await self.weaviate_service.close()
            
            if self.weaviate_client:
                await self.weaviate_client.close()
            
            if self.sqlite_storage:
                await self.sqlite_storage.close()
            
            logger.info("Test environment cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def run_all_tests(self):
        """Run all integration tests."""
        try:
            logger.info("Starting Weaviate integration tests...")
            
            # Setup
            if not await self.setup():
                logger.error("Failed to setup test environment")
                return False
            
            # Run tests
            tests = [
                ("Weaviate Client", self.test_weaviate_client),
                ("Memory Service", self.test_memory_service),
                ("Cross-Project Sharing", self.test_cross_project_sharing),
                ("Deduplication", self.test_deduplication),
                ("SQLite Integration", self.test_sqlite_integration)
            ]
            
            passed_tests = 0
            total_tests = len(tests)
            
            for test_name, test_func in tests:
                logger.info(f"Running {test_name} test...")
                try:
                    success = await test_func()
                    if success:
                        passed_tests += 1
                except Exception as e:
                    logger.error(f"{test_name} test failed with exception: {e}")
            
            # Print results
            logger.info("\n" + "="*50)
            logger.info("WEAVIATE INTEGRATION TEST RESULTS")
            logger.info("="*50)
            
            for test_name, result in self.test_results.items():
                status = "✅ PASSED" if result == "PASSED" else f"❌ {result}"
                logger.info(f"{test_name}: {status}")
            
            logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
            
            if passed_tests == total_tests:
                logger.info("🎉 All tests PASSED! Weaviate integration is working correctly.")
                return True
            else:
                logger.error("❌ Some tests FAILED. Please check the logs above.")
                return False
                
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False
        
        finally:
            await self.cleanup()


async def main():
    """Main test function."""
    tester = WeaviateIntegrationTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 Weaviate integration test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Weaviate integration test failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

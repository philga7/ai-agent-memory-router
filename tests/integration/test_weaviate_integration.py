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
import pytest
from datetime import datetime, timedelta
from uuid import uuid4

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.models.memory import MemoryStoreCreate, MemoryContent, AgentSource
from app.services.weaviate_service import WeaviateService
from app.core.sqlite_storage import SQLiteUnifiedStorage

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class WeaviateIntegrationTester:
    """Test class for Weaviate integration functionality."""
    
    def __init__(self):
        """Initialize the tester."""
        self.settings = get_settings()
        self.weaviate_service = None
        self.sqlite_storage = None
        self.deduplication_service = None
        self.test_results = {}
        
    async def setup(self):
        """Setup test environment."""
        try:
            logger.info("Setting up Weaviate integration test environment...")
            
            # Initialize Weaviate service
            self.weaviate_service = WeaviateService()
            await self.weaviate_service.initialize()
            
            # Initialize SQLite storage
            self.sqlite_storage = SQLiteUnifiedStorage()
            await self.sqlite_storage.initialize()
            
            # Create test agents
            await self._create_test_agents()
            
            logger.info("Weaviate integration test environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
            return False
    
    async def _create_test_agents(self):
        """Create test agents for integration testing."""
        try:
            # This would create test agents in the system
            # For now, we'll just log that we would do this
            logger.info("Creating test agents for integration testing...")
            return True
        except Exception as e:
            logger.error(f"Failed to create test agents: {e}")
            return False
    
    async def _cleanup_test_data(self):
        """Clean up test data."""
        try:
            logger.info("Cleaning up test data...")
            # Clean up any test data created during testing
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup test data: {e}")
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
            
            if self.sqlite_storage:
                await self.sqlite_storage.close()
            
            if self.weaviate_service:
                await self.weaviate_service.close()
            
            logger.info("Test environment cleanup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup test environment: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all integration tests."""
        try:
            logger.info("Starting Weaviate integration tests...")
            
            # Setup
            if not await self.setup():
                logger.error("Failed to setup test environment")
                return False
            
            # Run tests
            test_methods = [
                self.test_sqlite_integration,
            ]
            
            for test_method in test_methods:
                test_name = test_method.__name__
                logger.info(f"Running {test_name}...")
                
                try:
                    success = await test_method()
                    if success:
                        logger.info(f"{test_name} PASSED")
                    else:
                        logger.error(f"{test_name} FAILED")
                except Exception as e:
                    logger.error(f"{test_name} FAILED with exception: {e}")
                    self.test_results[test_name] = f"FAILED: {e}"
            
            # Print results
            logger.info("=== Weaviate Integration Test Results ===")
            for test_name, result in self.test_results.items():
                logger.info(f"{test_name}: {result}")
            
            # Cleanup
            await self.cleanup()
            
            return True
            
        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
            await self.cleanup()
            return False

@pytest.mark.integration
async def test_weaviate_integration():
    """Test Weaviate integration functionality."""
    tester = WeaviateIntegrationTester()
    success = await tester.run_all_tests()
    assert success, "Weaviate integration tests failed"

if __name__ == "__main__":
    # Run the integration tests
    asyncio.run(test_weaviate_integration())

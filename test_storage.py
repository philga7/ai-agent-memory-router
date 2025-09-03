#!/usr/bin/env python3
"""
Simple test script for the SQLite storage layer.

This script tests the basic functionality of the storage abstraction
layer and SQLite implementation.
"""

import asyncio
import tempfile
import os
from pathlib import Path
import uuid
from datetime import datetime

# Add the app directory to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.core.sqlite_storage import SQLiteUnifiedStorage
from app.models.memory import MemoryItem, MemoryMetadata
from app.models.agent import Agent, AgentCapabilities, AgentStatus
from app.models.context import ConversationContext, SimpleContext


async def test_storage_layer():
    """Test the storage layer functionality."""
    print("🧪 Testing SQLite Storage Layer")
    print("=" * 50)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_file:
        db_path = temp_file.name
    
    try:
        # Initialize storage
        print("📁 Initializing storage...")
        storage = SQLiteUnifiedStorage(db_path, max_connections=5)
        await storage.manager.initialize()
        
        # Test agent storage
        print("\n🤖 Testing agent storage...")
        agent = Agent(
            id=str(uuid.uuid4()),
            name="Test Agent",
            description="A test agent for storage testing",
            agent_type="assistant",
            version="1.0.0",
            capabilities=AgentCapabilities(
                reasoning=True,
                memory=True,
                learning=True,
                communication=True,
                tool_use=True,
                planning=True,
                creativity=True,
                problem_solving=True
            ),
            status=AgentStatus(
                status="online",
                last_heartbeat=datetime.now(),
                uptime=0.0,
                error_count=0,
                performance_metrics={}
            ),
            metadata={"test": True}
        )
        
        agent_id = await storage.agent.store_agent(agent)
        print(f"✅ Agent stored with ID: {agent_id}")
        
        retrieved_agent = await storage.agent.get_agent(agent_id)
        if retrieved_agent and retrieved_agent.name == "Test Agent":
            print("✅ Agent retrieval successful")
        else:
            print("❌ Agent retrieval failed")
        
        # Test memory storage
        print("\n🧠 Testing memory storage...")
        memory = MemoryItem(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            content="This is a test memory for storage testing",
            memory_type="knowledge",
            priority=2
        )
        
        memory_id = await storage.memory.store_memory(memory)
        print(f"✅ Memory stored with ID: {memory_id}")
        
        retrieved_memory = await storage.memory.get_memory(memory_id)
        if retrieved_memory and retrieved_memory.content == memory.content:
            print("✅ Memory retrieval successful")
        else:
            print("❌ Memory retrieval failed")
        
        # Test metadata storage
        print("\n🏷️ Testing metadata storage...")
        metadata = MemoryMetadata(
            id=str(uuid.uuid4()),
            memory_id=memory_id,
            tags=["test", "storage", "metadata"],
            source="test_script",
            confidence=0.95,
            embedding_vector="test_vector",
            vector_dimension=128
        )
        
        metadata_id = await storage.metadata.store_metadata(metadata)
        print(f"✅ Metadata stored with ID: {metadata_id}")
        
        retrieved_metadata = await storage.metadata.get_metadata(memory_id)
        if retrieved_metadata and retrieved_metadata.tags == metadata.tags:
            print("✅ Metadata retrieval successful")
        else:
            print("❌ Metadata retrieval failed")
        
        # Test context storage
        print("\n💬 Testing context storage...")
        context = SimpleContext(
            id=str(uuid.uuid4()),
            conversation_id="test_conversation",
            agent_id=agent_id,
            context_data={"topic": "testing", "status": "active"},
            context_type="conversation"
        )
        
        context_id = await storage.context.store_context(context)
        print(f"✅ Context stored with ID: {context_id}")
        
        retrieved_context = await storage.context.get_context(context_id)
        if retrieved_context and retrieved_context.context_data == context.context_data:
            print("✅ Context retrieval successful")
        else:
            print("❌ Context retrieval failed")
        
        # Test search functionality
        print("\n🔍 Testing search functionality...")
        search_results = await storage.memory.search_memories("test memory")
        if search_results and len(search_results) > 0:
            print(f"✅ Search successful: found {len(search_results)} results")
        else:
            print("❌ Search failed")
        
        # Test agent listing
        print("\n📋 Testing agent listing...")
        agents = await storage.agent.list_agents()
        if agents and len(agents) > 0:
            print(f"✅ Agent listing successful: found {len(agents)} agents")
        else:
            print("❌ Agent listing failed")
        
        # Test health check
        print("\n💚 Testing health check...")
        health_status = await storage.manager.health_check()
        if health_status:
            print("✅ Health check passed")
        else:
            print("❌ Health check failed")
        
        # Test backup functionality
        print("\n💾 Testing backup functionality...")
        backup_path = await storage.manager.backup("test_backup.db")
        if backup_path:
            print(f"✅ Backup created: {backup_path}")
        else:
            print("❌ Backup failed")
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        await storage.manager.close()
        
        # Remove temporary files
        if os.path.exists(db_path):
            os.unlink(db_path)
        if os.path.exists("test_backup.db"):
            os.unlink("test_backup.db")
        
        print("✅ Cleanup completed")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_storage_layer())

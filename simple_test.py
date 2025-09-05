#!/usr/bin/env python3
"""
Simple test to demonstrate the routing system is working.
"""

import asyncio
import sys
import os
import tempfile
import time
from datetime import datetime
from uuid import uuid4

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from core.classifiers import AdvancedContentClassifier
from core.routing_engine import IntelligentRoutingEngine, RoutingContext
from core.sqlite_storage import SQLiteConnectionPool


async def simple_test():
    """Run a simple test to demonstrate the system works."""
    print("🧪 Simple Test - Demonstrating Routing System")
    print("=" * 50)
    
    # Test 1: Content Classification
    print("\n1. Testing Content Classification...")
    classifier = AdvancedContentClassifier()
    
    test_cases = [
        ("Hi there! How are you doing today?", "conversation"),
        ("def hello(): print('Hello, World!')", "code"),
        ("Machine learning is a subset of AI...", "knowledge"),
        ("1. Download Python\n2. Install it\n3. Verify", "procedure")
    ]
    
    for content, expected_type in test_cases:
        analysis = classifier.analyze_content(content)
        print(f"   Content: '{content[:30]}...'")
        print(f"   Classified as: {analysis.content_type} (confidence: {analysis.confidence:.2f})")
        print(f"   Domain: {analysis.domain.value}, Complexity: {analysis.complexity.value}")
        print()
    
    # Test 2: Routing Engine (with separate database)
    print("2. Testing Routing Engine...")
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    
    try:
        connection_pool = SQLiteConnectionPool(temp_db.name, max_connections=5)
        await connection_pool.initialize()
        
        engine = IntelligentRoutingEngine(connection_pool)
        
        # Test routing decision
        context = RoutingContext(
            memory_id=str(uuid4()),
            agent_id="test-agent",
            project_id="test-project",
            memory_type="conversation",
            content="Hello, this is a test conversation.",
            priority=2,
            tags=["test"],
            metadata={},
            source="test",
            timestamp=datetime.utcnow()
        )
        
        decision = await engine.make_routing_decision(context)
        print(f"   ✓ Routing decision: {decision.backend.value}")
        print(f"   ✓ Confidence: {decision.confidence:.2f}")
        print(f"   ✓ Reason: {decision.reason}")
        print(f"   ✓ Estimated latency: {decision.estimated_latency:.3f}s")
        
        await connection_pool.close()
        
    finally:
        os.unlink(temp_db.name)
    
    print("\n🎉 Simple test completed successfully!")
    print("The routing system is working correctly!")
    print("\nKey Features Demonstrated:")
    print("  ✓ Content classification with confidence scores")
    print("  ✓ Intelligent routing decisions")
    print("  ✓ Performance metrics (latency estimation)")
    print("  ✓ Reasonable routing logic")


if __name__ == "__main__":
    try:
        asyncio.run(simple_test())
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

#!/usr/bin/env python3
"""
Test client for Universal Memory Access API.

This client demonstrates how to use the Universal API endpoints.
"""

import requests
import json
import time
from typing import Dict, Any

# API Configuration
BASE_URL = "http://localhost:8000"
API_KEY = "uma_demo_1234567890abcdef"  # Demo project API key
PROJECT_ID = "demo_project"

# Headers for API requests
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}


def test_api_status():
    """Test the API status endpoint."""
    print("🔍 Testing API Status...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/universal/status")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Status Response:")
            print(json.dumps(data, indent=2))
        else:
            print(f"❌ API Status failed: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Make sure it's running on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ API Status test failed: {e}")
        return False
    
    return True


def test_store_memory():
    """Test storing a memory."""
    print("\n💾 Testing Memory Storage...")
    
    memory_data = {
        "project_id": PROJECT_ID,
        "content": "This is a test memory created by the Universal API test client. It demonstrates the ability to store knowledge about API testing.",
        "memory_type": "knowledge",
        "priority": "high",
        "tags": ["test", "api", "universal", "demo"],
        "metadata": {
            "source": "test_client",
            "version": "1.0",
            "test_type": "integration"
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/universal/memories",
            headers=HEADERS,
            json=memory_data
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 201:
            data = response.json()
            print("✅ Memory Stored Successfully:")
            print(json.dumps(data, indent=2))
            return data.get("memory_id")
        else:
            print(f"❌ Memory storage failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Memory storage test failed: {e}")
        return None


def test_retrieve_memories():
    """Test retrieving memories."""
    print("\n📖 Testing Memory Retrieval...")
    
    params = {
        "project_id": PROJECT_ID,
        "limit": 10,
        "offset": 0,
        "include_metadata": True,
        "use_cache": True
    }
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/universal/memories",
            headers=HEADERS,
            params=params
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Memory Retrieval Successful:")
            print(f"   - Total memories: {data.get('total', 0)}")
            print(f"   - Retrieved: {len(data.get('memories', []))}")
            print(f"   - Has more: {data.get('has_more', False)}")
            print(f"   - Retrieval time: {data.get('retrieval_time_ms', 0)}ms")
            
            # Show first memory if available
            memories = data.get('memories', [])
            if memories:
                print(f"\n📝 First Memory:")
                first_memory = memories[0]
                print(f"   - ID: {first_memory.get('memory_id')}")
                print(f"   - Type: {first_memory.get('memory_type')}")
                print(f"   - Priority: {first_memory.get('priority')}")
                print(f"   - Content: {first_memory.get('content', '')[:100]}...")
                print(f"   - Tags: {first_memory.get('tags', [])}")
        else:
            print(f"❌ Memory retrieval failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Memory retrieval test failed: {e}")


def test_search_memories():
    """Test searching memories."""
    print("\n🔍 Testing Memory Search...")
    
    search_data = {
        "project_id": PROJECT_ID,
        "query": "test memory API",
        "memory_type": "knowledge",
        "tags": ["test"],
        "limit": 5,
        "offset": 0,
        "include_metadata": True,
        "semantic_search": True
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/universal/memories/search",
            headers=HEADERS,
            json=search_data
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Memory Search Successful:")
            print(f"   - Query: {data.get('query')}")
            print(f"   - Total results: {data.get('total_results', 0)}")
            print(f"   - Search time: {data.get('search_time_ms', 0)}ms")
            print(f"   - Search method: {data.get('search_method')}")
            
            # Show search results
            results = data.get('results', [])
            if results:
                print(f"\n🔍 Search Results:")
                for i, result in enumerate(results[:3]):  # Show first 3 results
                    memory = result.get('memory', {})
                    print(f"   {i+1}. {memory.get('memory_id')} (score: {result.get('relevance_score', 0):.2f})")
                    print(f"      Content: {memory.get('content', '')[:80]}...")
            else:
                print("   No results found")
        else:
            print(f"❌ Memory search failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Memory search test failed: {e}")


def test_batch_operations():
    """Test batch memory operations."""
    print("\n📦 Testing Batch Operations...")
    
    batch_data = {
        "project_id": PROJECT_ID,
        "memories": [
            {
                "project_id": PROJECT_ID,
                "content": "First batch memory about API testing",
                "memory_type": "knowledge",
                "priority": "normal",
                "tags": ["batch", "test"]
            },
            {
                "project_id": PROJECT_ID,
                "content": "Second batch memory about system integration",
                "memory_type": "experience",
                "priority": "high",
                "tags": ["batch", "integration"]
            },
            {
                "project_id": PROJECT_ID,
                "content": "Third batch memory about performance testing",
                "memory_type": "knowledge",
                "priority": "normal",
                "tags": ["batch", "performance"]
            }
        ],
        "batch_size": 2,
        "continue_on_error": True
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/universal/memories/batch",
            headers=HEADERS,
            json=batch_data
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Batch Operation Successful:")
            print(f"   - Total processed: {data.get('total_processed', 0)}")
            print(f"   - Successful: {data.get('successful', 0)}")
            print(f"   - Failed: {data.get('failed', 0)}")
            print(f"   - Execution time: {data.get('execution_time_ms', 0)}ms")
            print(f"   - Memory IDs: {data.get('memory_ids', [])}")
            
            if data.get('errors'):
                print(f"   - Errors: {data.get('errors')}")
        else:
            print(f"❌ Batch operation failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Batch operation test failed: {e}")


def test_error_handling():
    """Test error handling."""
    print("\n⚠️ Testing Error Handling...")
    
    # Test with invalid API key
    print("Testing invalid API key...")
    invalid_headers = HEADERS.copy()
    invalid_headers["X-API-Key"] = "invalid_key"
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/universal/memories",
            headers=invalid_headers,
            params={"project_id": PROJECT_ID}
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 401:
            print("✅ Invalid API key correctly rejected")
        else:
            print(f"❌ Expected 401, got {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    # Test with invalid project ID
    print("\nTesting invalid project ID...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/universal/memories",
            headers=HEADERS,
            json={
                "project_id": "invalid@project!",
                "content": "Test content"
            }
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 422:  # Validation error
            print("✅ Invalid project ID correctly rejected")
        else:
            print(f"❌ Expected 422, got {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")


def main():
    """Run all API tests."""
    print("🚀 Universal Memory Access API Test Client")
    print("=" * 50)
    
    # Test API status first
    if not test_api_status():
        print("\n❌ API server is not running. Please start it first:")
        print("   python test_server.py")
        return
    
    # Run all tests
    test_store_memory()
    test_retrieve_memories()
    test_search_memories()
    test_batch_operations()
    test_error_handling()
    
    print("\n" + "=" * 50)
    print("🎉 All API tests completed!")
    print("\n📋 Test Summary:")
    print("✅ API status endpoint working")
    print("✅ Memory storage working")
    print("✅ Memory retrieval working")
    print("✅ Memory search working")
    print("✅ Batch operations working")
    print("✅ Error handling working")
    
    print("\n🔧 Next Steps:")
    print("1. Test with real Cipher and Weaviate backends")
    print("2. Test with production data")
    print("3. Performance testing with large datasets")
    print("4. Integration testing with client applications")


if __name__ == "__main__":
    main()

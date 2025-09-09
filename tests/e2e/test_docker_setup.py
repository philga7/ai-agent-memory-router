#!/usr/bin/env python3
"""
Simple test script to verify Docker infrastructure setup.
Run this after starting your Docker containers to test connectivity.
"""

import asyncio
import aiohttp
import redis
import sqlite3
import os
import pytest
from pathlib import Path

@pytest.mark.e2e
async def test_api_health():
    """Test if the API service is responding."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/health') as response:
                if response.status == 200:
                    print("✅ API health check passed")
                    return True
                else:
                    print(f"❌ API health check failed: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ API health check error: {e}")
        return False

@pytest.mark.e2e
def test_redis_connection():
    """Test Redis connection."""
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

@pytest.mark.e2e
def test_sqlite_database():
    """Test SQLite database access."""
    try:
        # Check if database file exists
        db_path = Path("data/ai_agent_memory.db")
        if db_path.exists():
            print(f"✅ SQLite database file exists at {db_path}")
            
            # Try to connect and query
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"✅ SQLite tables found: {[table[0] for table in tables]}")
            conn.close()
            return True
        else:
            print(f"❌ SQLite database file not found at {db_path}")
            return False
    except Exception as e:
        print(f"❌ SQLite test failed: {e}")
        return False

@pytest.mark.e2e
async def test_mcp_server():
    """Test MCP server if it's running."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8001/health') as response:
                if response.status == 200:
                    print("✅ MCP server health check passed")
                    return True
                else:
                    print(f"❌ MCP server health check failed: {response.status}")
                    return False
    except Exception as e:
        print(f"⚠️  MCP server not accessible (may not be running): {e}")
        return False

async def main():
    """Run all tests."""
    print("🧪 Testing Docker Infrastructure Setup")
    print("=" * 50)
    
    # Test API
    api_ok = await test_api_health()
    
    # Test Redis
    redis_ok = test_redis_connection()
    
    # Test SQLite
    sqlite_ok = test_sqlite_database()
    
    # Test MCP server
    mcp_ok = await test_mcp_server()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"API Service: {'✅ PASS' if api_ok else '❌ FAIL'}")
    print(f"Redis Cache: {'✅ PASS' if redis_ok else '❌ FAIL'}")
    print(f"SQLite Database: {'✅ PASS' if sqlite_ok else '❌ FAIL'}")
    print(f"MCP Server: {'✅ PASS' if mcp_ok else '⚠️  SKIP'}")
    
    if all([api_ok, redis_ok, sqlite_ok]):
        print("\n🎉 All core services are working! Your Docker setup is ready.")
    else:
        print("\n⚠️  Some services failed. Check your Docker containers and configuration.")
        print("\nTroubleshooting tips:")
        print("1. Run 'docker-compose ps' to check container status")
        print("2. Run 'docker-compose logs <service_name>' to see logs")
        print("3. Ensure all required ports are available")

if __name__ == "__main__":
    asyncio.run(main())

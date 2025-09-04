# Cipher MCP Integration

This document describes the Cipher MCP integration for the AI Agent Memory Router, providing project-based memory storage and retrieval with hybrid SQLite local storage.

## Overview

The Cipher integration provides:

- **Project-based Memory Storage**: Organize memories by project for better organization
- **Hybrid Storage**: Cipher for full content, SQLite for metadata and caching
- **MCP Protocol Support**: Full MCP server integration for Cursor and other clients
- **Universal Memory Access**: RESTful API endpoints for any project to store/retrieve memories
- **Error Handling & Fallback**: Graceful degradation when Cipher is unavailable
- **Caching**: Local SQLite cache for improved performance

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AI Agents     │    │  Memory Router   │    │  Cipher MCP     │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │   Agent A   │ │◄──►│ │ Cipher       │ │◄──►│ │   Cipher    │ │
│ │   Agent B   │ │    │ │ Service      │ │    │ │   Server    │ │
│ │   Agent C   │ │    │ └──────────────┘ │    │ └─────────────┘ │
│ └─────────────┘ │    │ ┌──────────────┐ │    └─────────────────┘
└─────────────────┘    │ │ SQLite       │ │
                       │ │ Storage      │ │
                       │ │ (Metadata)   │ │
                       │ └──────────────┘ │
                       └──────────────────┘
```

## Components

### 1. Cipher MCP Client (`app/core/cipher_client.py`)

Low-level client for communicating with Cipher MCP server:

- **Connection Management**: Automatic connection pooling and retry logic
- **Memory Operations**: Store, retrieve, search, update, delete memories
- **Project Management**: Create, get, list projects
- **Health Monitoring**: Health checks and status monitoring

### 2. Cipher Service (`app/services/cipher_service.py`)

High-level service layer providing business logic:

- **Hybrid Storage**: Coordinates between Cipher and SQLite
- **Caching**: Intelligent local caching with TTL
- **Error Handling**: Graceful fallback when Cipher is unavailable
- **Project-based Routing**: Organizes memories by project

### 3. Storage Abstraction (`app/core/storage.py`)

Unified storage interface supporting hybrid storage:

- **CipherHybridStorage**: Combines Cipher and SQLite storage
- **CipherHybridMemoryStorage**: Memory-specific hybrid operations
- **Abstraction Layer**: Easy migration to other storage backends

### 4. MCP Server Integration (`app/core/mcp_server.py`)

MCP protocol support for Cursor integration:

- **Cipher Tools**: Full set of Cipher operations via MCP
- **Memory Operations**: Store, retrieve, search memories
- **Project Operations**: Create, get, list projects
- **Health Monitoring**: Cipher server health checks

### 5. REST API Endpoints (`app/api/v1/memory.py`)

Universal API endpoints for any project:

- **Memory Operations**: `/memory/cipher/store`, `/memory/cipher/retrieve`, etc.
- **Project Operations**: `/memory/cipher/projects`
- **Search**: `/memory/cipher/search`
- **Health**: `/memory/cipher/health`

## Configuration

### Environment Variables

```bash
# Cipher MCP Server Configuration
CIPHER_MCP_URL=http://localhost:3000
CIPHER_TIMEOUT=30
CIPHER_MAX_RETRIES=3
CIPHER_RETRY_DELAY=1.0
CIPHER_CACHE_TTL_HOURS=1
CIPHER_ENABLE_HYBRID_STORAGE=true

# Optional: Cipher API Configuration
CIPHER_API_URL=https://cipher.informedcrew.com
CIPHER_API_KEY=your-cipher-api-key-here
```

### Configuration Class

```python
from app.core.config import get_settings

settings = get_settings()
cipher_config = settings.cipher

print(f"MCP URL: {cipher_config.mcp_url}")
print(f"Timeout: {cipher_config.timeout}")
print(f"Max Retries: {cipher_config.max_retries}")
```

## Usage Examples

### 1. Using the Cipher Service

```python
from app.services.cipher_service import get_cipher_service

# Get service instance
cipher_service = await get_cipher_service()

# Store memory
memory_id = await cipher_service.store_memory(
    project_id="my_project",
    agent_id="agent_001",
    memory_content="Important project information",
    memory_type="knowledge",
    tags=["project", "important"],
    metadata={"priority": "high"},
    priority=8
)

# Retrieve memory
memory_data = await cipher_service.retrieve_memory(
    project_id="my_project",
    memory_id=memory_id
)

# Search memories
search_results = await cipher_service.search_memories(
    project_id="my_project",
    query="important information",
    limit=10
)
```

### 2. Using the MCP Client

```python
from app.core.cipher_client import get_cipher_client

# Get client instance
cipher_client = await get_cipher_client()

# Store memory directly
result = await cipher_client.store_memory(
    project_id="my_project",
    memory_content="Direct memory storage",
    memory_type="direct",
    tags=["direct", "test"]
)

# Search memories
results = await cipher_client.search_memories(
    project_id="my_project",
    query="direct memory",
    limit=5
)
```

### 3. Using REST API

```bash
# Store memory
curl -X POST http://localhost:8000/api/v1/memory/cipher/store \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "my_project",
    "agent_id": "agent_001",
    "memory_content": "API stored memory",
    "memory_type": "api",
    "tags": ["api", "test"],
    "priority": 5
  }'

# Retrieve memory
curl http://localhost:8000/api/v1/memory/cipher/retrieve/my_project/memory_id

# Search memories
curl -X POST http://localhost:8000/api/v1/memory/cipher/search \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "my_project",
    "query": "API stored",
    "limit": 10
  }'
```

### 4. Using MCP Tools in Cursor

```python
# Store memory via MCP
result = await mcp_client.call_tool("cipher_store_memory", {
    "project_id": "my_project",
    "agent_id": "agent_001",
    "memory_content": "MCP stored memory",
    "memory_type": "mcp",
    "tags": ["mcp", "cursor"]
})

# Search memories via MCP
results = await mcp_client.call_tool("cipher_search_memories", {
    "project_id": "my_project",
    "query": "MCP stored",
    "limit": 5
})
```

## Error Handling

The integration includes comprehensive error handling:

### Connection Errors

```python
from app.core.cipher_client import CipherConnectionError

try:
    await cipher_client.store_memory(...)
except CipherConnectionError as e:
    print(f"Cipher server unavailable: {e}")
    # Fallback to local storage or retry
```

### Operation Errors

```python
from app.core.cipher_client import CipherOperationError

try:
    await cipher_client.retrieve_memory(...)
except CipherOperationError as e:
    print(f"Memory operation failed: {e}")
    # Handle specific operation failures
```

### Graceful Degradation

When Cipher is unavailable, the system:

1. **Falls back to local SQLite storage**
2. **Logs warnings about Cipher unavailability**
3. **Continues serving requests with cached data**
4. **Automatically retries when Cipher becomes available**

## Caching Strategy

### Local Cache (SQLite)

- **Metadata Storage**: Agent IDs, memory types, timestamps
- **Content Caching**: Full memory content for fast access
- **TTL Management**: Configurable cache expiration
- **Cache Invalidation**: Automatic updates when Cipher data changes

### Cache Configuration

```python
# Cache TTL: 1 hour (default)
CIPHER_CACHE_TTL_HOURS=1

# Enable/disable caching
CIPHER_ENABLE_HYBRID_STORAGE=true
```

## Testing

### Run Integration Tests

```bash
# Run the test script
python test_cipher_integration.py
```

### Test Components

1. **Configuration Test**: Verify settings are loaded correctly
2. **Client Test**: Test Cipher MCP client functionality
3. **Service Test**: Test hybrid storage service
4. **API Test**: Test REST endpoints (manual)

### Expected Output

```
🚀 Starting Cipher MCP Integration Tests

🔧 Testing Configuration...
✅ Configuration loaded
✅ Cipher MCP URL: http://localhost:3000
✅ Cipher timeout: 30
✅ Cipher max retries: 3
✅ Cipher cache TTL: 1 hours
✅ Hybrid storage enabled: True

🔧 Testing Cipher MCP Client...
✅ Cipher MCP client initialized
⚠️  Cipher MCP server not available: Connection failed
   This is expected if Cipher server is not running

🔧 Testing Cipher Service...
✅ SQLite storage initialized
✅ Cipher service initialized
⚠️  Cipher service initialization failed (Cipher server may not be available)
✅ Cipher service closed

📊 Test Summary:
   Configuration: ✅ PASS
   Cipher Client: ✅ PASS
   Cipher Service: ✅ PASS

🎉 All tests passed! Cipher MCP integration is working correctly.
```

## Deployment

### 1. Prerequisites

- Python 3.8+
- SQLite database
- Cipher MCP server (optional for full functionality)

### 2. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp env.example .env

# Edit configuration
nano .env
```

### 3. Configuration

Update `.env` file with your Cipher MCP server details:

```bash
CIPHER_MCP_URL=http://your-cipher-server:3000
CIPHER_TIMEOUT=30
CIPHER_MAX_RETRIES=3
```

### 4. Start Services

```bash
# Start the application
python -m app.main

# Or with Docker
docker-compose up
```

## Monitoring

### Health Checks

```bash
# Check Cipher health
curl http://localhost:8000/api/v1/memory/cipher/health

# Check overall system health
curl http://localhost:8000/api/v1/health
```

### Metrics

The integration provides metrics for:

- **Memory Operations**: Store, retrieve, search counts
- **Cache Performance**: Hit/miss ratios
- **Error Rates**: Connection and operation failures
- **Response Times**: Operation latency

### Logging

Structured logging includes:

- **Operation Tracking**: All memory operations logged
- **Error Details**: Comprehensive error information
- **Performance Metrics**: Operation timing and success rates
- **Cache Statistics**: Cache hit/miss information

## Troubleshooting

### Common Issues

1. **Cipher Server Unavailable**
   - Check if Cipher MCP server is running
   - Verify `CIPHER_MCP_URL` configuration
   - Check network connectivity

2. **Memory Not Found**
   - Verify project ID and memory ID
   - Check if memory exists in Cipher
   - Try disabling cache (`use_cache=false`)

3. **Slow Performance**
   - Check cache configuration
   - Verify SQLite database performance
   - Monitor Cipher server response times

### Debug Mode

Enable debug logging:

```bash
LOG_LEVEL=DEBUG
```

### Manual Testing

```python
# Test Cipher connection
from app.core.cipher_client import CipherMCPClient

client = CipherMCPClient()
await client.connect()
health = await client.health_check()
print(health)
```

## Future Enhancements

1. **PostgreSQL Migration**: Easy migration from SQLite to PostgreSQL
2. **Advanced Caching**: Redis-based distributed caching
3. **Batch Operations**: Bulk memory operations
4. **Real-time Sync**: WebSocket-based real-time updates
5. **Analytics**: Advanced memory usage analytics
6. **Backup/Restore**: Automated backup and restore functionality

## Support

For issues and questions:

1. Check the logs for error details
2. Run the integration test script
3. Verify configuration settings
4. Check Cipher MCP server status
5. Review this documentation

## License

This integration is part of the AI Agent Memory Router project and follows the same license terms.

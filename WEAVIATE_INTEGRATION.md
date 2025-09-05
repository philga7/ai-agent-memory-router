# Weaviate Integration for AI Agent Memory Router

This document describes the Weaviate integration implementation for the AI Agent Memory Router, providing vector-based memory storage and semantic search capabilities.

## 🎯 Overview

The Weaviate integration enables:
- **Vector-based memory storage** using Weaviate's built-in vectorization
- **Semantic search** across all stored memories
- **Cross-project knowledge sharing** between different AI agent projects
- **Memory deduplication** using similarity analysis
- **SQLite metadata storage** for routing decisions and local metadata

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AI Agents     │    │  Memory Router  │    │   Weaviate      │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │   Agent 1   │ │───▶│ │ Weaviate    │ │───▶│ │ Vector DB   │ │
│ │   Agent 2   │ │    │ │ Service     │ │    │ │ + Search    │ │
│ │   Agent 3   │ │    │ └─────────────┘ │    │ └─────────────┘ │
│ └─────────────┘ │    │ ┌─────────────┐ │    └─────────────────┘
└─────────────────┘    │ │ SQLite      │ │
                       │ │ Metadata    │ │
                       │ └─────────────┘ │
                       └─────────────────┘
```

## 🚀 Quick Start

### 1. Start Weaviate

```bash
# Start Weaviate with Docker Compose
./start_weaviate.sh

# Or manually
docker-compose -f docker-compose.weaviate.yml up -d
```

### 2. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy environment template
cp env.example .env

# Edit .env file with your Weaviate settings
WEAVIATE_API_URL=http://localhost:8080
WEAVIATE_COLLECTION_NAME=AgentMemories
WEAVIATE_VECTOR_DIMENSION=384
```

### 4. Run Integration Tests

```bash
# Test the complete integration
python test_weaviate_integration.py
```

## 📁 File Structure

```
app/
├── core/
│   ├── weaviate_client.py      # Core Weaviate client
│   ├── embeddings.py           # Optional embedding service
│   └── config.py               # Weaviate configuration
├── services/
│   ├── weaviate_service.py     # High-level Weaviate service
│   ├── cross_project_service.py # Cross-project knowledge sharing
│   └── deduplication_service.py # Memory deduplication
└── models/
    └── memory.py               # Memory data models
```

## 🔧 Configuration

### Weaviate Settings

```python
# app/core/config.py
class WeaviateSettings(BaseSettings):
    api_url: str = "http://localhost:8080"
    api_key: Optional[str] = None
    timeout: int = 30
    collection_name: str = "AgentMemories"
    vector_dimension: int = 384
    enable_hybrid_search: bool = True
    similarity_threshold: float = 0.7
    max_search_results: int = 100
    batch_size: int = 100
```

### Environment Variables

```bash
# Weaviate Configuration
WEAVIATE_API_URL=http://localhost:8080
WEAVIATE_API_KEY=                    # Optional API key
WEAVIATE_TIMEOUT=30
WEAVIATE_MAX_RETRIES=3
WEAVIATE_RETRY_DELAY=1.0
WEAVIATE_COLLECTION_NAME=AgentMemories
WEAVIATE_VECTOR_DIMENSION=384
WEAVIATE_ENABLE_HYBRID_SEARCH=true
WEAVIATE_SIMILARITY_THRESHOLD=0.7
WEAVIATE_MAX_SEARCH_RESULTS=100
WEAVIATE_BATCH_SIZE=100
```

## 💻 Usage Examples

### Basic Memory Storage

```python
from app.services.weaviate_service import get_weaviate_service
from app.models.memory import MemoryStoreCreate, MemoryContent
from app.models.agent import AgentSource

# Initialize service
weaviate_service = await get_weaviate_service()

# Create memory
memory_data = MemoryStoreCreate(
    content=MemoryContent(
        text="Python is a versatile programming language for AI development.",
        tags=["python", "programming", "ai"],
        metadata={"category": "knowledge"}
    ),
    source=AgentSource(
        agent_id="agent-1",
        project_id="project-1"
    ),
    memory_type="knowledge",
    importance=0.8
)

# Store memory
stored_memory = await weaviate_service.store_memory(memory_data)
print(f"Memory stored with ID: {stored_memory.id}")
```

### Semantic Search

```python
from app.models.memory import MemorySearch

# Search for memories
search_query = MemorySearch(
    query="Python programming best practices",
    filters={
        "memory_type": "knowledge",
        "min_importance": 0.5
    },
    limit=10
)

search_results = await weaviate_service.search_memories(search_query)

for result in search_results.results:
    print(f"Found: {result.content.text}")
    print(f"Relevance: {result.relevance_score}")
    print(f"Importance: {result.importance}")
```

### Cross-Project Knowledge Sharing

```python
from app.services.cross_project_service import get_cross_project_service

# Initialize cross-project service
cross_project_service = await get_cross_project_service()

# Share knowledge across projects
sharing_result = await cross_project_service.share_knowledge_across_projects(
    source_project_id="project-1",
    query="machine learning algorithms",
    target_project_ids=["project-2", "project-3"],
    sharing_level="standard",
    max_results=5
)

print(f"Shared {sharing_result['memories_shared']} memories")
```

### Memory Deduplication

```python
from app.services.deduplication_service import get_deduplication_service

# Initialize deduplication service
dedup_service = await get_deduplication_service()

# Find duplicates
duplicate_analysis = await dedup_service.find_duplicates(
    agent_id="agent-1",
    similarity_threshold=0.9
)

print(f"Found {duplicate_analysis['duplicate_groups_found']} duplicate groups")

# Auto-resolve duplicates
auto_dedup_result = await dedup_service.auto_deduplicate(
    agent_id="agent-1",
    similarity_threshold=0.95,
    auto_resolve=True
)
```

## 🔍 Key Features

### 1. Built-in Vectorization

Weaviate automatically generates embeddings using `text2vec-transformers`:
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension**: 384
- **No pre-computation needed**

### 2. Hybrid Search

Combines vector search with keyword search:
- **Vector similarity** for semantic matching
- **Keyword matching** for exact terms
- **Configurable thresholds** for result filtering

### 3. Cross-Project Knowledge Sharing

Enables knowledge transfer between projects:
- **Project isolation** with selective sharing
- **Similarity-based discovery** of related knowledge
- **Sharing level controls** (standard, limited, full)

### 4. Memory Deduplication

Identifies and handles duplicate memories:
- **Content similarity** analysis
- **Semantic similarity** using vector embeddings
- **Hash-based exact matching**
- **Multiple resolution strategies**

### 5. SQLite Metadata Integration

Maintains local metadata for routing decisions:
- **Weaviate object references**
- **Project associations**
- **Similarity thresholds**
- **Routing metadata**

## 🧪 Testing

### Run Integration Tests

```bash
# Test complete integration
python test_weaviate_integration.py

# Test specific components
python -m pytest tests/test_weaviate_client.py
python -m pytest tests/test_weaviate_service.py
```

### Test Coverage

The integration test covers:
- ✅ Weaviate client functionality
- ✅ Memory storage and retrieval
- ✅ Semantic search
- ✅ Cross-project knowledge sharing
- ✅ Memory deduplication
- ✅ SQLite metadata integration

## 🐳 Docker Setup

### Weaviate with Transformers

```yaml
# docker-compose.weaviate.yml
version: '3.8'
services:
  weaviate:
    image: semitechnologies/weaviate:1.21.0
    ports:
      - "8080:8080"
    environment:
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      DEFAULT_VECTORIZER_MODULE: 'text2vec-transformers'
      ENABLE_MODULES: 'text2vec-transformers'
      TRANSFORMERS_INFERENCE_API: 'http://t2v-transformers:8080'
  
  t2v-transformers:
    image: semitechnologies/transformers-inference:sentence-transformers-multi-qa-MiniLM-L6-cos-v1
    environment:
      ENABLE_CUDA: '0'
```

## 📊 Performance Considerations

### Vector Search Performance

- **Batch operations** for bulk memory storage
- **Connection pooling** for Weaviate client
- **Async operations** throughout the stack
- **Configurable timeouts** and retry logic

### Memory Management

- **Automatic cleanup** of expired memories
- **Deduplication** to reduce storage overhead
- **Efficient indexing** in SQLite for metadata
- **Connection management** for both Weaviate and SQLite

## 🔒 Security

### Access Control

- **API key authentication** for Weaviate (optional)
- **Project-based isolation** for knowledge sharing
- **Configurable sharing levels** for sensitive data
- **SQLite metadata protection** for routing decisions

### Data Privacy

- **Local metadata storage** in SQLite
- **Configurable data retention** policies
- **Secure memory deletion** from both systems
- **Audit logging** for knowledge sharing

## 🚨 Troubleshooting

### Common Issues

1. **Weaviate Connection Failed**
   ```bash
   # Check if Weaviate is running
   curl http://localhost:8080/v1/.well-known/ready
   
   # Check Docker logs
   docker-compose -f docker-compose.weaviate.yml logs weaviate
   ```

2. **Vectorization Errors**
   ```bash
   # Check transformers service
   curl http://localhost:8080/health
   
   # Restart transformers service
   docker-compose -f docker-compose.weaviate.yml restart t2v-transformers
   ```

3. **SQLite Metadata Issues**
   ```bash
   # Check database file permissions
   ls -la data/ai_agent_memory.db
   
   # Recreate database
   rm data/ai_agent_memory.db
   python -c "from app.core.sqlite_storage import SQLiteUnifiedStorage; import asyncio; asyncio.run(SQLiteUnifiedStorage('data/ai_agent_memory.db').initialize())"
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python test_weaviate_integration.py --verbose
```

## 📈 Monitoring

### Health Checks

```python
# Check Weaviate health
weaviate_client = await get_weaviate_client()
stats = await weaviate_client.get_collection_stats()
print(f"Weaviate status: {stats}")

# Check service health
weaviate_service = await get_weaviate_service()
service_stats = await weaviate_service.get_service_stats()
print(f"Service status: {service_stats}")
```

### Metrics

- **Memory storage rate** (memories per second)
- **Search response time** (milliseconds)
- **Cross-project sharing frequency**
- **Deduplication effectiveness**
- **Vector search accuracy**

## 🔄 Migration

### From SQLite-only to Weaviate

1. **Backup existing data**
2. **Start Weaviate services**
3. **Run migration script**
4. **Verify data integrity**
5. **Update configuration**

### Future Migrations

The architecture supports easy migration to:
- **PostgreSQL** for metadata storage
- **Different vector databases** (Pinecone, Qdrant)
- **Cloud-based Weaviate** instances
- **Custom embedding models**

## 📚 Additional Resources

- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Sentence Transformers](https://www.sbert.net/)
- [Vector Database Comparison](https://weaviate.io/blog/vector-database-comparison)
- [AI Agent Memory Router Documentation](./README.md)

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Add tests for new functionality**
4. **Run integration tests**
5. **Submit a pull request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

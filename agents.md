# AI Agent Guidelines for AI Agent Memory Router

## 🎯 Project Overview

The AI Agent Memory Router is a sophisticated system designed to facilitate intelligent communication and knowledge sharing between AI agents through centralized memory management and intelligent routing capabilities. This document provides comprehensive guidelines for AI agents working with this project.

## 🏗️ Project Architecture

### Core Components

- **Memory Router Engine**: Central routing logic for directing memory requests between agents
- **Agent Communication Hub**: Manages inter-agent communication protocols and message routing
- **Memory Storage Layer**: Persistent storage for agent memories, knowledge, and conversation context
- **Routing Intelligence**: AI-powered decision making for optimal memory routing and distribution
- **Integration APIs**: RESTful and MCP interfaces for external systems and agent frameworks
- **Context Management**: Maintains conversation context and agent state across sessions

### System Flow

```
AI Agent → Memory Router → Context Analysis → Intelligent Routing → Target Agent(s)
    ↓           ↓              ↓              ↓              ↓
Memory ← Response ← Context Update ← Route Decision ← Memory Storage
```

## 🤖 AI Agent Responsibilities

### Primary Responsibilities

1. **Memory Management**
   - Store and retrieve agent memories efficiently
   - Maintain conversation context across sessions
   - Implement intelligent memory routing algorithms

2. **Code Generation**
   - Generate clean, maintainable Python code
   - Follow FastAPI best practices for API development
   - Implement proper error handling and validation

3. **MCP Integration**
   - Maintain MCP server compatibility
   - Ensure proper tool registration and discovery
   - Handle MCP protocol communication

4. **Configuration Management**
   - Manage environment variables and configuration files
   - Ensure proper database connection handling
   - Maintain security best practices

### Secondary Responsibilities

1. **Testing and Quality Assurance**
   - Write comprehensive test suites
   - Ensure code coverage and quality
   - Implement proper error handling

2. **Documentation**
   - Maintain API documentation
   - Update project documentation
   - Create clear code comments

## 💻 Development Patterns

### Code Structure Standards

#### Python Code Patterns

```python
# ALWAYS: Use async/await for database operations
async def get_agent_memory(agent_id: str) -> Optional[AgentMemory]:
    async with get_db_session() as session:
        result = await session.execute(
            select(AgentMemory).where(AgentMemory.agent_id == agent_id)
        )
        return result.scalar_one_or_none()

# ALWAYS: Use Pydantic models for data validation
class MemoryRouteRequest(BaseModel):
    source_agent_id: str
    target_agent_ids: List[str]
    memory_content: str
    priority: Priority = Priority.NORMAL
    context: Optional[Dict[str, Any]] = None

# ALWAYS: Implement proper error handling
async def route_memory(request: MemoryRouteRequest) -> MemoryRouteResponse:
    try:
        # Implementation logic
        pass
    except AgentNotFoundError as e:
        logger.error(f"Agent not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

#### API Endpoint Patterns

```python
# ALWAYS: Use proper HTTP status codes
@router.post("/memory/route", response_model=MemoryRouteResponse)
async def route_memory(
    request: MemoryRouteRequest,
    current_user: User = Depends(get_current_user)
) -> MemoryRouteResponse:
    """Route memory between AI agents with intelligent decision making."""
    return await memory_service.route_memory(request)

# ALWAYS: Implement proper validation and error handling
@router.get("/memory/{agent_id}", response_model=List[MemoryItem])
async def get_agent_memories(
    agent_id: str,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
) -> List[MemoryItem]:
    """Retrieve memories for a specific agent with pagination."""
    return await memory_service.get_agent_memories(agent_id, limit, offset)
```

### Database Patterns

```python
# ALWAYS: Use async database operations
async def create_memory_route(route: MemoryRoute) -> MemoryRoute:
    async with get_db_session() as session:
        session.add(route)
        await session.commit()
        await session.refresh(route)
        return route

# ALWAYS: Use proper transaction handling
async def update_agent_context(agent_id: str, context: Dict[str, Any]) -> bool:
    async with get_db_session() as session:
        async with session.begin():
            agent = await session.get(Agent, agent_id)
            if not agent:
                return False
            agent.context = context
            await session.commit()
            return True
```

## 📁 File Interaction Standards

### File Organization

```
app/
├── api/                    # API endpoints and routers
│   ├── v1/                # API version 1
│   │   ├── memory.py      # Memory-related endpoints
│   │   ├── agents.py      # Agent management endpoints
│   │   └── context.py     # Context management endpoints
├── core/                   # Core business logic
│   ├── memory_router.py   # Main routing engine
│   ├── agent_manager.py   # Agent lifecycle management
│   └── context_manager.py # Context management
├── models/                 # Data models and schemas
│   ├── memory.py          # Memory-related models
│   ├── agent.py           # Agent models
│   └── context.py         # Context models
├── services/               # Business logic services
│   ├── memory_service.py  # Memory operations
│   ├── routing_service.py # Routing logic
│   └── agent_service.py   # Agent operations
└── utils/                  # Utility functions
    ├── database.py        # Database utilities
    ├── security.py        # Security utilities
    └── logging.py         # Logging configuration
```

### File Naming Conventions

- **ALWAYS**: Use snake_case for Python files and functions
- **ALWAYS**: Use PascalCase for Python classes
- **ALWAYS**: Use descriptive, clear names that indicate purpose
- **NEVER**: Use abbreviations or unclear acronyms

### Import Standards

```python
# ALWAYS: Group imports in this order
# 1. Standard library imports
import asyncio
import logging
from typing import List, Optional, Dict, Any

# 2. Third-party imports
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

# 3. Local application imports
from app.core.memory_router import MemoryRouter
from app.models.memory import MemoryRoute, MemoryRouteRequest
from app.services.memory_service import MemoryService
from app.utils.database import get_db_session
```

## 🎯 Decision-Making Guidelines

### Priority Order for AI Decisions

1. **Security First**: Always prioritize security and data protection
2. **Performance**: Optimize for speed and efficiency in memory routing
3. **Scalability**: Design for horizontal scaling and high availability
4. **Maintainability**: Write clean, readable, and well-documented code
5. **User Experience**: Ensure smooth and intuitive agent interactions

### Common Decision Points

#### Memory Routing Decisions

```python
# ALWAYS: Consider agent capabilities when routing
async def determine_target_agents(
    memory_content: str,
    source_agent_id: str,
    available_agents: List[Agent]
) -> List[str]:
    """Intelligently determine which agents should receive the memory."""
    
    # Priority 1: Security - check agent permissions
    authorized_agents = await filter_authorized_agents(available_agents, source_agent_id)
    
    # Priority 2: Performance - use semantic similarity for routing
    relevant_agents = await find_semantically_relevant_agents(memory_content, authorized_agents)
    
    # Priority 3: Load balancing - distribute memory across available agents
    return await balance_memory_distribution(relevant_agents, memory_content)
```

#### Context Management Decisions

```python
# ALWAYS: Maintain conversation context for intelligent routing
async def update_conversation_context(
    agent_id: str,
    conversation_id: str,
    new_context: Dict[str, Any]
) -> bool:
    """Update conversation context for better routing decisions."""
    
    # Priority 1: Security - validate context data
    if not is_valid_context(new_context):
        raise ValueError("Invalid context data")
    
    # Priority 2: Performance - merge with existing context efficiently
    existing_context = await get_conversation_context(conversation_id)
    merged_context = merge_contexts(existing_context, new_context)
    
    # Priority 3: Persistence - store updated context
    return await store_conversation_context(conversation_context, merged_context)
```

## 🔍 Code Review Checklist

### Before Submitting Code

- [ ] **Security**: No hardcoded secrets or credentials
- [ ] **Performance**: Async operations used where appropriate
- [ ] **Error Handling**: Proper exception handling implemented
- [ ] **Validation**: Input validation using Pydantic models
- [ ] **Testing**: Unit tests written and passing
- [ ] **Documentation**: Code comments and docstrings added
- [ ] **Logging**: Appropriate logging statements included
- [ ] **Type Hints**: Type annotations used throughout

### Code Quality Standards

- [ ] **Readability**: Code is clear and easy to understand
- [ ] **Consistency**: Follows established patterns and conventions
- [ ] **Efficiency**: No unnecessary database queries or operations
- [ ] **Maintainability**: Code is modular and well-structured
- [ ] **Scalability**: Design supports future growth and changes

## 🚫 Prohibited Actions

### Security Violations

- **NEVER**: Hardcode API keys, passwords, or sensitive data
- **NEVER**: Skip input validation or sanitization
- **NEVER**: Expose internal system information in error messages
- **NEVER**: Implement authentication bypasses or workarounds

### Code Quality Violations

- **NEVER**: Use synchronous operations where async is available
- **NEVER**: Skip error handling or logging
- **NEVER**: Use unclear variable names or abbreviations
- **NEVER**: Duplicate code without proper abstraction

### Performance Violations

- **NEVER**: Make unnecessary database queries
- **NEVER**: Load entire datasets into memory unnecessarily
- **NEVER**: Skip connection pooling or resource management
- **NEVER**: Implement blocking operations in async contexts

## 🔧 AI Agent Communication Standards

### Inter-Agent Communication

```python
# ALWAYS: Use structured message formats
class AgentMessage(BaseModel):
    sender_id: str
    recipient_ids: List[str]
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime
    priority: Priority = Priority.NORMAL
    context: Optional[Dict[str, Any]] = None

# ALWAYS: Implement proper message routing
async def route_agent_message(message: AgentMessage) -> bool:
    """Route message between AI agents with intelligent decision making."""
    
    # Validate message format and permissions
    if not await validate_message_permissions(message):
        return False
    
    # Determine optimal routing path
    route_path = await calculate_optimal_route(message)
    
    # Execute routing with error handling
    try:
        return await execute_message_routing(message, route_path)
    except Exception as e:
        logger.error(f"Message routing failed: {e}")
        return False
```

### Memory Sharing Protocols

```python
# ALWAYS: Implement secure memory sharing
async def share_memory_between_agents(
    source_agent_id: str,
    target_agent_ids: List[str],
    memory_data: MemoryData,
    sharing_level: SharingLevel = SharingLevel.STANDARD
) -> bool:
    """Securely share memory between AI agents."""
    
    # Security check: verify sharing permissions
    if not await verify_sharing_permissions(source_agent_id, target_agent_ids, sharing_level):
        raise PermissionError("Insufficient permissions for memory sharing")
    
    # Privacy check: ensure data classification allows sharing
    if not await validate_sharing_classification(memory_data, sharing_level):
        raise ValueError("Memory classification incompatible with sharing level")
    
    # Execute sharing with audit logging
    return await execute_memory_sharing(source_agent_id, target_agent_ids, memory_data, sharing_level)
```

## 🧠 Memory-Specific Guidelines

### Knowledge Storage

```python
# ALWAYS: Use semantic indexing for efficient retrieval
async def store_agent_memory(
    agent_id: str,
    memory_content: str,
    memory_type: MemoryType,
    tags: List[str] = None,
    metadata: Dict[str, Any] = None
) -> str:
    """Store agent memory with semantic indexing."""
    
    # Create memory record
    memory = AgentMemory(
        agent_id=agent_id,
        content=memory_content,
        memory_type=memory_type,
        tags=tags or [],
        metadata=metadata or {},
        created_at=datetime.utcnow()
    )
    
    # Store in database
    memory_id = await memory_service.store_memory(memory)
    
    # Generate and store semantic embeddings
    embeddings = await generate_semantic_embeddings(memory_content)
    await vector_store.store_embeddings(memory_id, embeddings)
    
    return memory_id
```

### Knowledge Retrieval

```python
# ALWAYS: Implement intelligent search with context awareness
async def search_agent_memories(
    query: str,
    agent_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    limit: int = 10
) -> List[MemorySearchResult]:
    """Search agent memories with intelligent context awareness."""
    
    # Generate query embeddings
    query_embeddings = await generate_semantic_embeddings(query)
    
    # Perform semantic search
    semantic_results = await vector_store.semantic_search(
        query_embeddings,
        agent_id=agent_id,
        limit=limit * 2  # Get more results for filtering
    )
    
    # Apply context-aware filtering
    if context:
        semantic_results = await apply_context_filtering(semantic_results, context)
    
    # Return top results
    return semantic_results[:limit]
```

### Knowledge Updates

```python
# ALWAYS: Maintain data integrity during updates
async def update_agent_memory(
    memory_id: str,
    updates: Dict[str, Any],
    agent_id: str
) -> bool:
    """Update agent memory with integrity checks."""
    
    # Verify ownership and permissions
    memory = await memory_service.get_memory(memory_id)
    if not memory or memory.agent_id != agent_id:
        raise PermissionError("Cannot update memory owned by another agent")
    
    # Validate update data
    if not await validate_memory_updates(updates):
        raise ValueError("Invalid memory update data")
    
    # Update memory record
    success = await memory_service.update_memory(memory_id, updates)
    
    # Update semantic embeddings if content changed
    if success and "content" in updates:
        new_embeddings = await generate_semantic_embeddings(updates["content"])
        await vector_store.update_embeddings(memory_id, new_embeddings)
    
    return success
```

## 🔌 Integration Standards

### MCP Integration

```python
# ALWAYS: Implement proper MCP tool registration
class MemoryRouterMCPServer:
    def __init__(self):
        self.tools = {
            "memory_route": self.memory_route,
            "agent_register": self.agent_register,
            "memory_search": self.memory_search,
            "context_get": self.context_get
        }
    
    async def memory_route(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Route memory between AI agents via MCP."""
        try:
            request = MemoryRouteRequest(**params)
            result = await memory_service.route_memory(request)
            return {"success": True, "result": result.dict()}
        except Exception as e:
            logger.error(f"MCP memory_route failed: {e}")
            return {"success": False, "error": str(e)}

# ALWAYS: Handle MCP errors gracefully
async def handle_mcp_request(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP requests with proper error handling."""
    try:
        if method not in MCP_TOOLS:
            return {"success": False, "error": f"Unknown method: {method}"}
        
        return await MCP_TOOLS[method](params)
    except Exception as e:
        logger.error(f"MCP request failed: {e}")
        return {"success": False, "error": str(e)}
```

### API Integration

```python
# ALWAYS: Implement proper API versioning
router = APIRouter(prefix="/api/v1")

# ALWAYS: Use consistent response formats
class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# ALWAYS: Implement proper error handling
@router.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=APIResponse(
            success=False,
            error="Internal server error"
        ).dict()
    )
```

## 🧪 Testing Standards

### Test Structure

```python
# ALWAYS: Use descriptive test names
async def test_memory_routing_with_multiple_target_agents():
    """Test memory routing to multiple target agents with load balancing."""
    # Test implementation

# ALWAYS: Test both success and failure scenarios
async def test_memory_routing_with_invalid_agent_id():
    """Test memory routing fails gracefully with invalid agent ID."""
    # Test implementation

# ALWAYS: Use proper test fixtures
@pytest.fixture
async def memory_router():
    """Provide configured memory router for testing."""
    return MemoryRouter(test_config)
```

### Test Coverage Requirements

- **Unit Tests**: Minimum 90% code coverage
- **Integration Tests**: Test all API endpoints
- **Error Handling**: Test all error scenarios
- **Performance**: Test with realistic data volumes
- **Security**: Test authentication and authorization

## 📊 Performance Guidelines

### Database Optimization

```python
# ALWAYS: Use connection pooling
async def get_db_session() -> AsyncSession:
    return await db_session_pool.acquire()

# ALWAYS: Implement query optimization
async def get_agent_memories_optimized(agent_id: str, limit: int) -> List[MemoryItem]:
    """Optimized query for retrieving agent memories."""
    async with get_db_session() as session:
        # Use proper indexing and limit results
        result = await session.execute(
            select(AgentMemory)
            .where(AgentMemory.agent_id == agent_id)
            .order_by(AgentMemory.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()
```

### Caching Strategies

```python
# ALWAYS: Implement intelligent caching
async def get_agent_context_cached(agent_id: str) -> Optional[AgentContext]:
    """Get agent context with intelligent caching."""
    
    # Check cache first
    cache_key = f"agent_context:{agent_id}"
    cached_context = await cache.get(cache_key)
    if cached_context:
        return AgentContext(**cached_context)
    
    # Fetch from database if not cached
    context = await context_service.get_agent_context(agent_id)
    if context:
        # Cache for 5 minutes
        await cache.set(cache_key, context.dict(), expire=300)
    
    return context
```

## 🔒 Security Considerations

### Authentication and Authorization

```python
# ALWAYS: Implement proper authentication
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current authenticated user from token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = await user_service.get_user(user_id)
    if user is None:
        raise credentials_exception
    return user

# ALWAYS: Check permissions before operations
async def verify_agent_permissions(agent_id: str, user: User) -> bool:
    """Verify user has permissions to access agent."""
    return await permission_service.check_agent_access(user.id, agent_id)
```

### Data Protection

```python
# ALWAYS: Encrypt sensitive data
async def store_sensitive_memory(memory_data: SensitiveMemoryData) -> str:
    """Store sensitive memory data with encryption."""
    
    # Encrypt sensitive content
    encrypted_content = encrypt_data(memory_data.content, ENCRYPTION_KEY)
    
    # Store encrypted data
    memory = SensitiveMemory(
        content=encrypted_content,
        encryption_version=ENCRYPTION_VERSION,
        created_at=datetime.utcnow()
    )
    
    return await memory_service.store_sensitive_memory(memory)
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### Memory Routing Failures

```python
# ALWAYS: Implement comprehensive logging for debugging
async def debug_memory_routing(source_agent_id: str, target_agent_ids: List[str]):
    """Debug memory routing issues with comprehensive logging."""
    
    logger.info(f"Starting memory routing debug for {source_agent_id}")
    
    # Check source agent status
    source_agent = await agent_service.get_agent(source_agent_id)
    if not source_agent:
        logger.error(f"Source agent {source_agent_id} not found")
        return False
    
    # Check target agents
    for target_id in target_agent_ids:
        target_agent = await agent_service.get_agent(target_id)
        if not target_agent:
            logger.error(f"Target agent {target_id} not found")
            continue
        
        logger.info(f"Target agent {target_id} status: {target_agent.status}")
    
    # Check routing configuration
    routing_config = await routing_service.get_routing_config()
    logger.info(f"Routing configuration: {routing_config}")
    
    return True
```

#### Performance Issues

```python
# ALWAYS: Monitor and log performance metrics
async def monitor_memory_routing_performance():
    """Monitor memory routing performance metrics."""
    
    start_time = time.time()
    
    # Execute routing operation
    result = await execute_memory_routing()
    
    execution_time = time.time() - start_time
    
    # Log performance metrics
    logger.info(f"Memory routing completed in {execution_time:.2f} seconds")
    
    # Alert if performance is below threshold
    if execution_time > PERFORMANCE_THRESHOLD:
        logger.warning(f"Memory routing performance below threshold: {execution_time:.2f}s")
    
    return result
```

## 📚 Additional Resources

### Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy Async Documentation](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)

### Best Practices

- [Python Async Best Practices](https://docs.python.org/3/library/asyncio-dev.html)
- [API Design Best Practices](https://restfulapi.net/)
- [Database Design Best Practices](https://www.postgresql.org/docs/current/ddl.html)

---

**Last Updated**: September 2025  
**Version**: 1.0.0  
**Maintainer**: AI Agent Memory Router Team

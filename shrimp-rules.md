# Shrimp Rules for AI Agent Memory Router

## 🎯 Project Overview

The AI Agent Memory Router is a sophisticated system designed to facilitate intelligent communication and knowledge sharing between AI agents through centralized memory management and intelligent routing capabilities. This document provides project-specific development standards and rules that AI agents must follow when working with this codebase.

## 🏗️ Architecture

### Core Architecture Principles

- **Microservices Architecture**: Modular design with clear separation of concerns
- **Event-Driven Communication**: Asynchronous message passing between components
- **Scalable Storage**: Multi-layered storage with caching and persistence
- **Intelligent Routing**: AI-powered decision making for optimal memory distribution
- **Security First**: Comprehensive security measures at every layer

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway  │    │  Memory Router  │    │  Agent Manager  │
│   (FastAPI)    │◄──►│     Engine      │◄──►│   (Lifecycle)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Server    │    │ Context Manager │    │ Memory Storage  │
│  (Integration)  │    │  (State Mgmt)   │    │  (Database)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 💻 Code Standards

### Python Code Standards

#### ALWAYS Use These Patterns

```python
# ALWAYS: Use async/await for all I/O operations
async def get_agent_memory(agent_id: str) -> Optional[AgentMemory]:
    async with get_db_session() as session:
        result = await session.execute(
            select(AgentMemory).where(AgentMemory.agent_id == agent_id)
        )
        return result.scalar_one_or_none()

# ALWAYS: Use Pydantic models for data validation
class MemoryRouteRequest(BaseModel):
    source_agent_id: str = Field(..., description="Source agent identifier")
    target_agent_ids: List[str] = Field(..., description="Target agent identifiers")
    memory_content: str = Field(..., min_length=1, description="Memory content to route")
    priority: Priority = Field(Priority.NORMAL, description="Routing priority")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

# ALWAYS: Implement comprehensive error handling
async def route_memory(request: MemoryRouteRequest) -> MemoryRouteResponse:
    try:
        # Validate source agent exists
        source_agent = await agent_service.get_agent(request.source_agent_id)
        if not source_agent:
            raise AgentNotFoundError(f"Source agent {request.source_agent_id} not found")
        
        # Validate target agents exist
        target_agents = await agent_service.get_agents(request.target_agent_ids)
        if len(target_agents) != len(request.target_agent_ids):
            missing_agents = set(request.target_agent_ids) - {a.id for a in target_agents}
            raise AgentNotFoundError(f"Target agents not found: {missing_agents}")
        
        # Execute routing logic
        return await memory_service.route_memory(request)
        
    except AgentNotFoundError as e:
        logger.error(f"Agent not found during memory routing: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        logger.error(f"Validation error during memory routing: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during memory routing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

#### NEVER Use These Patterns

```python
# NEVER: Use synchronous database operations
def get_agent_memory_sync(agent_id: str) -> Optional[AgentMemory]:
    session = get_db_session_sync()  # ❌ Synchronous operation
    result = session.execute(...)    # ❌ Blocking call
    return result.scalar_one_or_none()

# NEVER: Skip input validation
async def route_memory_unsafe(request: dict) -> dict:
    # ❌ No validation of input data
    return await memory_service.route_memory(request)

# NEVER: Use generic exception handling without logging
async def route_memory_no_logging(request: MemoryRouteRequest) -> MemoryRouteResponse:
    try:
        return await memory_service.route_memory(request)
    except Exception as e:
        # ❌ No logging, generic exception handling
        raise HTTPException(status_code=500, detail="Error occurred")
```

### API Endpoint Standards

#### ALWAYS Follow These Patterns

```python
# ALWAYS: Use proper HTTP status codes and response models
@router.post("/memory/route", 
             response_model=MemoryRouteResponse,
             status_code=201,
             summary="Route memory between AI agents",
             description="Intelligently route memory content between specified AI agents")
async def route_memory(
    request: MemoryRouteRequest,
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks
) -> MemoryRouteResponse:
    """Route memory between AI agents with intelligent decision making."""
    
    # Add background task for logging and analytics
    background_tasks.add_task(log_memory_routing_attempt, request, current_user.id)
    
    # Execute routing logic
    result = await memory_service.route_memory(request)
    
    # Add background task for notification
    background_tasks.add_task(notify_target_agents, request.target_agent_ids, result)
    
    return result

# ALWAYS: Implement proper pagination and filtering
@router.get("/memory/{agent_id}",
            response_model=PaginatedMemoryResponse,
            summary="Retrieve agent memories",
            description="Get paginated list of memories for a specific agent")
async def get_agent_memories(
    agent_id: str,
    limit: int = Query(10, ge=1, le=100, description="Number of memories to return"),
    offset: int = Query(0, ge=0, description="Number of memories to skip"),
    memory_type: Optional[MemoryType] = Query(None, description="Filter by memory type"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    current_user: User = Depends(get_current_user)
) -> PaginatedMemoryResponse:
    """Retrieve memories for a specific agent with pagination and filtering."""
    
    # Verify user has access to agent
    if not await permission_service.can_access_agent(current_user.id, agent_id):
        raise HTTPException(status_code=403, detail="Access denied to agent")
    
    # Get memories with filtering
    memories = await memory_service.get_agent_memories(
        agent_id=agent_id,
        limit=limit,
        offset=offset,
        memory_type=memory_type,
        tags=tags
    )
    
    # Get total count for pagination
    total_count = await memory_service.get_agent_memory_count(
        agent_id=agent_id,
        memory_type=memory_type,
        tags=tags
    )
    
    return PaginatedMemoryResponse(
        items=memories,
        total=total_count,
        limit=limit,
        offset=offset,
        has_more=offset + limit < total_count
    )
```

### Database Standards

#### ALWAYS Use These Patterns

```python
# ALWAYS: Use proper transaction handling
async def create_memory_route_with_context(route: MemoryRoute, context: Dict[str, Any]) -> MemoryRoute:
    async with get_db_session() as session:
        async with session.begin():
            # Create memory route
            session.add(route)
            await session.flush()  # Get the ID
            
            # Create context record
            context_record = RouteContext(
                route_id=route.id,
                context_data=context,
                created_at=datetime.utcnow()
            )
            session.add(context_record)
            
            await session.commit()
            await session.refresh(route)
            return route

# ALWAYS: Use proper indexing and query optimization
async def get_agent_memories_optimized(agent_id: str, limit: int, offset: int) -> List[MemoryItem]:
    """Optimized query for retrieving agent memories with proper indexing."""
    async with get_db_session() as session:
        # Use proper indexing on agent_id and created_at
        result = await session.execute(
            select(AgentMemory)
            .where(AgentMemory.agent_id == agent_id)
            .order_by(AgentMemory.created_at.desc())
            .limit(limit)
            .offset(offset)
            .options(selectinload(AgentMemory.tags))  # Eager load related data
        )
        return result.scalars().all()

# ALWAYS: Implement connection pooling
async def get_db_session() -> AsyncSession:
    """Get database session from connection pool."""
    return await db_session_pool.acquire()

async def release_db_session(session: AsyncSession):
    """Release database session back to pool."""
    await db_session_pool.release(session)
```

## 🧩 Component Standards

### Memory Router Component

#### ALWAYS Implement These Features

```python
class MemoryRouter:
    """Core memory routing engine with intelligent decision making."""
    
    def __init__(self, config: RouterConfig):
        self.config = config
        self.agent_manager = AgentManager()
        self.context_manager = ContextManager()
        self.routing_engine = RoutingEngine()
        self.logger = logging.getLogger(__name__)
    
    async def route_memory(self, request: MemoryRouteRequest) -> MemoryRouteResponse:
        """Route memory between AI agents with intelligent decision making."""
        
        # Step 1: Validate and preprocess request
        validated_request = await self._validate_request(request)
        
        # Step 2: Analyze routing context
        routing_context = await self._analyze_routing_context(validated_request)
        
        # Step 3: Determine optimal routing strategy
        routing_strategy = await self._determine_routing_strategy(routing_context)
        
        # Step 4: Execute routing with load balancing
        routing_result = await self._execute_routing(routing_strategy)
        
        # Step 5: Update context and return response
        await self._update_routing_context(routing_result)
        
        return routing_result
    
    async def _validate_request(self, request: MemoryRouteRequest) -> ValidatedMemoryRouteRequest:
        """Validate memory routing request with comprehensive checks."""
        # Implementation details...
        pass
    
    async def _analyze_routing_context(self, request: ValidatedMemoryRouteRequest) -> RoutingContext:
        """Analyze routing context for intelligent decision making."""
        # Implementation details...
        pass
    
    async def _determine_routing_strategy(self, context: RoutingContext) -> RoutingStrategy:
        """Determine optimal routing strategy based on context analysis."""
        # Implementation details...
        pass
    
    async def _execute_routing(self, strategy: RoutingStrategy) -> MemoryRouteResponse:
        """Execute routing strategy with load balancing and error handling."""
        # Implementation details...
        pass
```

### Agent Manager Component

#### ALWAYS Implement These Features

```python
class AgentManager:
    """Manages AI agent lifecycle and capabilities."""
    
    async def register_agent(self, agent_data: AgentRegistrationData) -> Agent:
        """Register new AI agent with capability assessment."""
        
        # Validate agent capabilities
        capabilities = await self._assess_agent_capabilities(agent_data)
        
        # Create agent record
        agent = Agent(
            id=agent_data.id,
            name=agent_data.name,
            capabilities=capabilities,
            status=AgentStatus.ACTIVE,
            created_at=datetime.utcnow()
        )
        
        # Store agent
        await self.agent_service.store_agent(agent)
        
        # Initialize agent context
        await self.context_manager.initialize_agent_context(agent.id)
        
        return agent
    
    async def _assess_agent_capabilities(self, agent_data: AgentRegistrationData) -> AgentCapabilities:
        """Assess agent capabilities for intelligent routing."""
        # Implementation details...
        pass
```

### Context Manager Component

#### ALWAYS Implement These Features

```python
class ContextManager:
    """Manages conversation context and agent state."""
    
    async def update_conversation_context(
        self, 
        conversation_id: str, 
        agent_id: str, 
        context_update: Dict[str, Any]
    ) -> bool:
        """Update conversation context for better routing decisions."""
        
        # Validate context data
        if not self._is_valid_context_update(context_update):
            raise ValueError("Invalid context update data")
        
        # Get existing context
        existing_context = await self._get_conversation_context(conversation_id)
        
        # Merge contexts intelligently
        merged_context = await self._merge_contexts(existing_context, context_update)
        
        # Store updated context
        success = await self._store_conversation_context(conversation_id, merged_context)
        
        # Update agent state
        if success:
            await self._update_agent_state(agent_id, conversation_id, merged_context)
        
        return success
    
    async def _merge_contexts(
        self, 
        existing: Optional[Dict[str, Any]], 
        update: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Intelligently merge context updates with existing context."""
        # Implementation details...
        pass
```

## 🔧 Service Layer Standards

### Memory Service

#### ALWAYS Implement These Patterns

```python
class MemoryService:
    """Service layer for memory operations with business logic."""
    
    async def store_memory(self, memory_data: MemoryData) -> str:
        """Store agent memory with semantic indexing."""
        
        # Create memory record
        memory = AgentMemory(
            agent_id=memory_data.agent_id,
            content=memory_data.content,
            memory_type=memory_data.memory_type,
            tags=memory_data.tags or [],
            metadata=memory_data.metadata or {},
            created_at=datetime.utcnow()
        )
        
        # Store in database
        memory_id = await self.memory_repository.store(memory)
        
        # Generate semantic embeddings
        embeddings = await self.embedding_service.generate_embeddings(memory_data.content)
        
        # Store embeddings in vector database
        await self.vector_store.store_embeddings(memory_id, embeddings)
        
        # Update search index
        await self.search_index.update_index(memory_id, memory_data.content)
        
        return memory_id
    
    async def search_memories(
        self, 
        query: str, 
        filters: Optional[MemoryFilters] = None,
        limit: int = 10
    ) -> List[MemorySearchResult]:
        """Search memories with semantic similarity and filtering."""
        
        # Generate query embeddings
        query_embeddings = await self.embedding_service.generate_embeddings(query)
        
        # Perform semantic search
        semantic_results = await self.vector_store.semantic_search(
            query_embeddings,
            filters=filters,
            limit=limit * 2  # Get more results for filtering
        )
        
        # Apply business logic filters
        filtered_results = await self._apply_business_filters(semantic_results, filters)
        
        # Return top results
        return filtered_results[:limit]
```

### Routing Service

#### ALWAYS Implement These Patterns

```python
class RoutingService:
    """Service layer for intelligent memory routing."""
    
    async def calculate_optimal_route(
        self, 
        source_agent_id: str, 
        target_agent_ids: List[str],
        memory_content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingPlan:
        """Calculate optimal routing plan using AI-powered decision making."""
        
        # Analyze memory content
        content_analysis = await self._analyze_memory_content(memory_content)
        
        # Assess agent capabilities and availability
        agent_assessment = await self._assess_target_agents(target_agent_ids)
        
        # Consider routing context
        routing_context = await self._build_routing_context(source_agent_id, context)
        
        # Generate routing plan
        routing_plan = await self._generate_routing_plan(
            content_analysis,
            agent_assessment,
            routing_context
        )
        
        return routing_plan
    
    async def _analyze_memory_content(self, content: str) -> ContentAnalysis:
        """Analyze memory content for routing decisions."""
        # Implementation details...
        pass
    
    async def _assess_target_agents(self, agent_ids: List[str]) -> AgentAssessment:
        """Assess target agents for routing decisions."""
        # Implementation details...
        pass
```

## 🎣 Custom Hook Standards

### Memory Routing Hooks

#### ALWAYS Implement These Hooks

```python
class MemoryRoutingHooks:
    """Custom hooks for memory routing operations."""
    
    async def pre_routing_hook(self, request: MemoryRouteRequest) -> MemoryRouteRequest:
        """Hook executed before memory routing."""
        
        # Log routing attempt
        await self._log_routing_attempt(request)
        
        # Validate routing permissions
        await self._validate_routing_permissions(request)
        
        # Enrich request with additional context
        enriched_request = await self._enrich_request_context(request)
        
        return enriched_request
    
    async def post_routing_hook(self, request: MemoryRouteRequest, result: MemoryRouteResponse):
        """Hook executed after memory routing."""
        
        # Log routing success
        await self._log_routing_success(request, result)
        
        # Update routing statistics
        await self._update_routing_statistics(request, result)
        
        # Trigger post-routing notifications
        await self._trigger_post_routing_notifications(request, result)
    
    async def routing_error_hook(self, request: MemoryRouteRequest, error: Exception):
        """Hook executed when routing fails."""
        
        # Log routing error
        await self._log_routing_error(request, error)
        
        # Update error statistics
        await self._update_error_statistics(request, error)
        
        # Trigger error notifications
        await self._trigger_error_notifications(request, error)
```

### Context Management Hooks

#### ALWAYS Implement These Hooks

```python
class ContextManagementHooks:
    """Custom hooks for context management operations."""
    
    async def pre_context_update_hook(
        self, 
        conversation_id: str, 
        context_update: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Hook executed before context update."""
        
        # Validate context update
        validated_update = await self._validate_context_update(context_update)
        
        # Check context size limits
        if await self._exceeds_context_limits(conversation_id, validated_update):
            validated_update = await self._truncate_context_update(validated_update)
        
        return validated_update
    
    async def post_context_update_hook(
        self, 
        conversation_id: str, 
        updated_context: Dict[str, Any]
    ):
        """Hook executed after context update."""
        
        # Update context metadata
        await self._update_context_metadata(conversation_id, updated_context)
        
        # Trigger context change notifications
        await self._trigger_context_change_notifications(conversation_id, updated_context)
```

## 📊 Data Management

### Memory Storage Patterns

#### ALWAYS Use These Storage Patterns

```python
# ALWAYS: Use multi-layered storage approach
class MultiLayeredMemoryStorage:
    """Multi-layered memory storage with caching and persistence."""
    
    def __init__(self):
        self.cache_layer = RedisCache()
        self.database_layer = PostgreSQLDatabase()
        self.vector_layer = VectorDatabase()
        self.search_layer = ElasticsearchIndex()
    
    async def store_memory(self, memory: AgentMemory) -> str:
        """Store memory across all storage layers."""
        
        # Store in database (primary storage)
        memory_id = await self.database_layer.store(memory)
        
        # Store in cache for quick access
        await self.cache_layer.store(f"memory:{memory_id}", memory.dict(), expire=3600)
        
        # Store embeddings in vector database
        embeddings = await self.embedding_service.generate_embeddings(memory.content)
        await self.vector_layer.store_embeddings(memory_id, embeddings)
        
        # Update search index
        await self.search_layer.index_memory(memory_id, memory.content, memory.tags)
        
        return memory_id
    
    async def retrieve_memory(self, memory_id: str) -> Optional[AgentMemory]:
        """Retrieve memory with cache-first approach."""
        
        # Try cache first
        cached_memory = await self.cache_layer.get(f"memory:{memory_id}")
        if cached_memory:
            return AgentMemory(**cached_memory)
        
        # Fall back to database
        memory = await self.database_layer.get(memory_id)
        if memory:
            # Update cache
            await self.cache_layer.store(f"memory:{memory_id}", memory.dict(), expire=3600)
        
        return memory
```

### Data Validation Patterns

#### ALWAYS Use These Validation Patterns

```python
# ALWAYS: Implement comprehensive data validation
class MemoryDataValidator:
    """Comprehensive data validation for memory operations."""
    
    async def validate_memory_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate memory data with multiple validation layers."""
        
        # Basic validation
        basic_validation = await self._validate_basic_fields(data)
        if not basic_validation.is_valid:
            return basic_validation
        
        # Content validation
        content_validation = await self._validate_content(data.get("content", ""))
        if not content_validation.is_valid:
            return content_validation
        
        # Security validation
        security_validation = await self._validate_security(data)
        if not security_validation.is_valid:
            return security_validation
        
        # Business rule validation
        business_validation = await self._validate_business_rules(data)
        if not business_validation.is_valid:
            return business_validation
        
        return ValidationResult(is_valid=True, errors=[])
    
    async def _validate_basic_fields(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate basic required fields."""
        required_fields = ["agent_id", "content", "memory_type"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return ValidationResult(
                is_valid=False,
                errors=[f"Missing required fields: {missing_fields}"]
            )
        
        return ValidationResult(is_valid=True, errors=[])
    
    async def _validate_content(self, content: str) -> ValidationResult:
        """Validate memory content."""
        if not content or not content.strip():
            return ValidationResult(
                is_valid=False,
                errors=["Memory content cannot be empty"]
            )
        
        if len(content) > MAX_CONTENT_LENGTH:
            return ValidationResult(
                is_valid=False,
                errors=[f"Memory content exceeds maximum length of {MAX_CONTENT_LENGTH} characters"]
            )
        
        # Check for malicious content
        if await self._contains_malicious_content(content):
            return ValidationResult(
                is_valid=False,
                errors=["Memory content contains potentially malicious content"]
            )
        
        return ValidationResult(is_valid=True, errors=[])
```

## 🎨 UI/UX Standards

### API Response Standards

#### ALWAYS Use These Response Patterns

```python
# ALWAYS: Use consistent API response format
class StandardAPIResponse(BaseModel):
    """Standard API response format for all endpoints."""
    
    success: bool = Field(..., description="Whether the operation was successful")
    data: Optional[Any] = Field(None, description="Response data when successful")
    error: Optional[str] = Field(None, description="Error message when unsuccessful")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: str = Field(..., description="Unique request identifier for tracking")

# ALWAYS: Implement proper error responses
@router.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors with consistent format."""
    return JSONResponse(
        status_code=400,
        content=StandardAPIResponse(
            success=False,
            error=f"Validation error: {exc.errors()}",
            request_id=request.headers.get("X-Request-ID", "unknown")
        ).dict()
    )

@router.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=StandardAPIResponse(
            success=False,
            error=exc.detail,
            request_id=request.headers.get("X-Request-ID", "unknown")
        ).dict()
    )
```

### Logging Standards

#### ALWAYS Use These Logging Patterns

```python
# ALWAYS: Implement structured logging
class StructuredLogger:
    """Structured logging for consistent log format."""
    
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
        self._setup_structured_logging()
    
    def _setup_structured_logging(self):
        """Setup structured logging with JSON format."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s", '
            '"extra": %(extra)s}'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_memory_routing(
        self, 
        source_agent_id: str, 
        target_agent_ids: List[str], 
        success: bool,
        duration_ms: float,
        error: Optional[str] = None
    ):
        """Log memory routing operation with structured data."""
        log_data = {
            "operation": "memory_routing",
            "source_agent_id": source_agent_id,
            "target_agent_ids": target_agent_ids,
            "success": success,
            "duration_ms": duration_ms,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if error:
            log_data["error"] = error
            self.logger.error("Memory routing failed", extra=log_data)
        else:
            self.logger.info("Memory routing completed", extra=log_data)
```

## 🔄 Development Workflow

### Feature Development Process

#### ALWAYS Follow These Steps

1. **Analysis Phase**
   - Analyze requirements and create detailed specifications
   - Identify affected components and dependencies
   - Plan testing strategy and acceptance criteria

2. **Implementation Phase**
   - Follow established code patterns and standards
   - Implement comprehensive error handling and logging
   - Write unit tests for all new functionality

3. **Testing Phase**
   - Ensure all tests pass (minimum 90% coverage)
   - Test error scenarios and edge cases
   - Validate performance and security requirements

4. **Review Phase**
   - Self-review code against standards
   - Update documentation and comments
   - Prepare for code review

5. **Integration Phase**
   - Integrate with existing components
   - Run integration tests
   - Validate system behavior

### Code Review Process

#### ALWAYS Check These Items

- [ ] **Code Standards**: Follows established patterns and conventions
- [ ] **Error Handling**: Comprehensive exception handling implemented
- [ ] **Logging**: Appropriate logging statements included
- [ ] **Testing**: Unit tests written and passing
- [ ] **Documentation**: Code comments and docstrings added
- [ ] **Security**: No security vulnerabilities introduced
- [ ] **Performance**: No performance regressions introduced
- [ ] **Integration**: Properly integrates with existing components

## 🚫 Prohibited Actions

### Security Violations

- **NEVER**: Hardcode API keys, passwords, or sensitive data
- **NEVER**: Skip input validation or sanitization
- **NEVER**: Expose internal system information in error messages
- **NEVER**: Implement authentication bypasses or workarounds
- **NEVER**: Store sensitive data without encryption

### Code Quality Violations

- **NEVER**: Use synchronous operations where async is available
- **NEVER**: Skip error handling or logging
- **NEVER**: Use unclear variable names or abbreviations
- **NEVER**: Duplicate code without proper abstraction
- **NEVER**: Skip type hints or documentation

### Performance Violations

- **NEVER**: Make unnecessary database queries
- **NEVER**: Load entire datasets into memory unnecessarily
- **NEVER**: Skip connection pooling or resource management
- **NEVER**: Implement blocking operations in async contexts
- **NEVER**: Skip caching where appropriate

## 🎯 AI Decision-Making Standards

### Priority Order for Decisions

1. **Security First**: Always prioritize security and data protection
2. **Data Integrity**: Ensure data consistency and reliability
3. **Performance**: Optimize for speed and efficiency
4. **Scalability**: Design for horizontal scaling and high availability
5. **Maintainability**: Write clean, readable, and well-documented code
6. **User Experience**: Ensure smooth and intuitive agent interactions

### Common Decision Points

#### Memory Routing Decisions

```python
# ALWAYS: Consider multiple factors for routing decisions
async def make_routing_decision(
    memory_content: str,
    source_agent_id: str,
    available_agents: List[Agent],
    context: Optional[Dict[str, Any]] = None
) -> RoutingDecision:
    """Make intelligent routing decision based on multiple factors."""
    
    # Factor 1: Security and permissions
    authorized_agents = await filter_authorized_agents(available_agents, source_agent_id)
    if not authorized_agents:
        return RoutingDecision(
            success=False,
            error="No authorized agents available for routing"
        )
    
    # Factor 2: Content relevance
    relevant_agents = await find_semantically_relevant_agents(memory_content, authorized_agents)
    if not relevant_agents:
        return RoutingDecision(
            success=False,
            error="No agents found relevant to memory content"
        )
    
    # Factor 3: Agent availability and load
    available_agents = await filter_available_agents(relevant_agents)
    if not available_agents:
        return RoutingDecision(
            success=False,
            error="No agents currently available for routing"
        )
    
    # Factor 4: Load balancing
    balanced_agents = await balance_agent_load(available_agents)
    
    # Factor 5: Context optimization
    optimized_agents = await optimize_for_context(balanced_agents, context)
    
    return RoutingDecision(
        success=True,
        target_agents=optimized_agents,
        routing_strategy="intelligent_load_balanced"
    )
```

#### Context Management Decisions

```python
# ALWAYS: Maintain context for intelligent decision making
async def make_context_decision(
    conversation_id: str,
    agent_id: str,
    context_update: Dict[str, Any]
) -> ContextDecision:
    """Make intelligent context management decision."""
    
    # Decision 1: Context size management
    if await self._exceeds_context_limits(conversation_id, context_update):
        context_update = await self._optimize_context_size(context_update)
    
    # Decision 2: Context relevance
    if not await self._is_context_relevant(context_update):
        return ContextDecision(
            success=False,
            error="Context update not relevant to current conversation"
        )
    
    # Decision 3: Context merging strategy
    merge_strategy = await self._determine_merge_strategy(conversation_id, context_update)
    
    # Decision 4: Context persistence
    persistence_strategy = await self._determine_persistence_strategy(context_update)
    
    return ContextDecision(
        success=True,
        optimized_update=context_update,
        merge_strategy=merge_strategy,
        persistence_strategy=persistence_strategy
    )
```

## 🔍 Code Review Checklist

### Before Submitting Code

- [ ] **Security**: No hardcoded secrets or credentials
- [ ] **Performance**: Async operations used where appropriate
- [ ] **Error Handling**: Comprehensive exception handling implemented
- [ ] **Validation**: Input validation using Pydantic models
- [ ] **Testing**: Unit tests written and passing (90%+ coverage)
- [ ] **Documentation**: Code comments and docstrings added
- [ ] **Logging**: Appropriate logging statements included
- [ ] **Type Hints**: Type annotations used throughout
- [ ] **Code Standards**: Follows established patterns and conventions

### Code Quality Standards

- [ ] **Readability**: Code is clear and easy to understand
- [ ] **Consistency**: Follows established patterns and conventions
- [ ] **Efficiency**: No unnecessary database queries or operations
- [ ] **Maintainability**: Code is modular and well-structured
- [ ] **Scalability**: Design supports future growth and changes
- [ ] **Integration**: Properly integrates with existing components

### Security Standards

- [ ] **Input Validation**: All inputs are properly validated
- [ ] **Authentication**: Proper authentication mechanisms implemented
- [ ] **Authorization**: Proper authorization checks implemented
- [ ] **Data Protection**: Sensitive data is properly protected
- [ ] **Error Handling**: No sensitive information exposed in errors

---

**Last Updated**: September 2025  
**Version**: 1.0.0  
**Maintainer**: AI Agent Memory Router Team

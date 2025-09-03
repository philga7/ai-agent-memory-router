# AI Agent Memory Router - Project Structure

This document outlines the current structure of the AI Agent Memory Router project.

## Project Overview

The AI Agent Memory Router is a sophisticated system designed to facilitate intelligent communication and knowledge sharing between AI agents. It provides memory routing, storage, search, and context management capabilities through a REST API and MCP (Microservice Control Plane) server.

## Directory Structure

```
ai-agent-memory-router/
├── .cursor/
│   └── mcp.json                    # MCP server configuration
├── app/
│   ├── __init__.py                 # Application package
│   ├── main.py                     # FastAPI application entry point
│   ├── api/
│   │   └── v1/
│   │       ├── __init__.py         # API v1 package
│   │       ├── router.py           # Main API router
│   │       ├── memory.py           # Memory endpoints
│   │       ├── agents.py           # Agent management endpoints
│   │       ├── context.py          # Context management endpoints
│   │       └── health.py           # Health monitoring endpoints
│   ├── core/
│   │   ├── __init__.py             # Core package
│   │   ├── config.py               # Configuration management
│   │   ├── database.py             # Database connection management
│   │   ├── logging.py              # Logging configuration
│   │   ├── mcp_server.py           # MCP server implementation
│   │   └── metrics.py              # Prometheus metrics
│   ├── models/
│   │   ├── __init__.py             # Models package
│   │   ├── common.py               # Common base models
│   │   ├── memory.py               # Memory-related models
│   │   ├── agent.py                # Agent-related models
│   │   └── context.py              # Context-related models
│   └── services/
│       ├── __init__.py             # Services package
│       └── memory_service.py       # Memory business logic service
├── docs/                           # Documentation (to be created)
├── tests/                          # Test suite (to be created)
├── scripts/                        # Utility scripts (to be created)
├── .env.example                    # Environment configuration template
├── .gitignore                      # Git ignore rules
├── docker-compose.yml              # Docker services configuration
├── Dockerfile                      # Application container definition
├── requirements.txt                 # Production dependencies
├── requirements-dev.txt             # Development dependencies
├── README.md                       # Project overview and setup
├── agents.md                       # AI agent guidelines
├── shrimp-rules.md                 # Project development standards
└── PROJECT_STRUCTURE.md            # This file
```

## Core Components

### 1. Application Entry Point (`app/main.py`)
- FastAPI application initialization
- Middleware configuration (CORS, TrustedHost)
- Database and MCP server lifecycle management
- Global exception handlers
- Health check and root endpoints

### 2. Configuration (`app/core/config.py`)
- Environment-based configuration using Pydantic
- Nested settings for all components
- Environment and log level validation
- Support for multiple environments (dev, staging, prod)

### 3. Database Management (`app/core/database.py`)
- Asynchronous SQLAlchemy setup
- Connection pooling and session management
- Health checks and monitoring
- Placeholder for migrations and backup/restore

### 4. Logging (`app/core/logging.py`)
- Structured JSON logging
- Console and rotating file handlers
- Performance metrics decorators
- Configurable log levels and formats

### 5. MCP Server (`app/core/mcp_server.py`)
- Microservice Control Plane server implementation
- Client connection management
- Tool implementations for memory routing, agent management, etc.
- Mock implementations for development

### 6. Metrics (`app/core/metrics.py`)
- Prometheus metrics collection
- Counters, gauges, and histograms
- Automatic tracking decorators
- Performance and operational metrics

## API Structure

### Version 1 API (`app/api/v1/`)
- **Memory Endpoints** (`/api/v1/memory/`)
  - Store, retrieve, search memories
  - Memory routing and statistics
  
- **Agent Endpoints** (`/api/v1/agents/`)
  - Agent registration and management
  - Status updates and heartbeats
  - Capability management
  
- **Context Endpoints** (`/api/v1/context/`)
  - Conversation context management
  - Participant and message handling
  - Context search and statistics
  
- **Health Endpoints** (`/api/v1/health/`)
  - System health checks
  - Component status monitoring
  - Performance metrics

## Data Models

### Common Models (`app/models/common.py`)
- Base classes and mixins
- Pagination and search utilities
- Error and success response models
- Generic data structures

### Memory Models (`app/models/memory.py`)
- Memory content and storage
- Memory routing and access control
- Search and filtering capabilities
- Statistics and batch operations

### Agent Models (`app/models/agent.py`)
- AI agent definitions
- Capabilities and permissions
- Status tracking and heartbeats
- Group management and batch operations

### Context Models (`app/models/context.py`)
- Conversation context management
- Participant and message handling
- Search and filtering
- Access control and statistics

## Services Layer

### Memory Service (`app/services/memory_service.py`)
- Memory storage and retrieval
- Search and filtering logic
- Statistics calculation
- Route management
- Mock implementation for development

## Infrastructure

### Docker Configuration
- Multi-service environment with PostgreSQL, Redis, ChromaDB
- Health checks and volume management
- Optional monitoring stack (Prometheus, Grafana)
- Custom network configuration

### Dependencies
- **Core Framework**: FastAPI, Pydantic, SQLAlchemy
- **Database**: PostgreSQL, Redis, ChromaDB
- **Monitoring**: Prometheus, Grafana
- **Development**: pytest, black, flake8, mypy

## Current Status

### ✅ Completed
- Project structure and documentation
- Core application setup
- API endpoint definitions
- Data models and validation
- Basic service implementations
- Docker configuration
- Health monitoring endpoints

### 🚧 In Progress
- Service layer implementation
- Database schema design
- MCP tool implementations
- Testing framework setup

### 📋 Planned
- Authentication and authorization
- Vector database integration
- Advanced search algorithms
- Performance optimization
- Comprehensive testing
- CI/CD pipeline
- Production deployment

## Development Guidelines

### Code Standards
- Follow PEP 8 and project-specific rules in `shrimp-rules.md`
- Use type hints throughout
- Implement comprehensive error handling
- Follow async/await patterns
- Use structured logging

### Testing Strategy
- Unit tests for all services
- Integration tests for API endpoints
- End-to-end tests for critical flows
- Performance and load testing
- Security testing

### Documentation
- API documentation with OpenAPI/Swagger
- Code documentation and examples
- Architecture decision records
- Deployment and operations guides

## Next Steps

1. **Complete Service Layer**: Implement remaining business logic services
2. **Database Integration**: Set up actual database connections and schemas
3. **Testing Framework**: Establish comprehensive test coverage
4. **Authentication**: Implement security and access control
5. **Performance**: Optimize search and routing algorithms
6. **Monitoring**: Enhance observability and alerting
7. **Deployment**: Prepare for production deployment

## Contributing

Please refer to `CONTRIBUTING.md` for development guidelines and contribution instructions. The project follows the standards outlined in `shrimp-rules.md` and `agents.md`.

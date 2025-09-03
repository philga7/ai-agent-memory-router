# AI Agent Memory Router

A sophisticated memory routing system designed to facilitate intelligent communication and knowledge sharing between AI agents through centralized memory management and intelligent routing capabilities.

## 🚀 Project Overview

The AI Agent Memory Router serves as a central hub for managing and routing memory and knowledge between multiple AI agents. It provides intelligent memory routing, context-aware communication, and seamless integration with various AI agent frameworks.

## 🏗️ Architecture

### Core Components

- **Memory Router Engine**: Central routing logic for directing memory requests
- **Agent Communication Hub**: Manages inter-agent communication protocols
- **Memory Storage Layer**: Persistent storage for agent memories and knowledge
- **Routing Intelligence**: AI-powered decision making for optimal memory routing
- **Integration APIs**: RESTful and MCP interfaces for external systems

### Key Features

- **Intelligent Memory Routing**: AI-powered decision making for optimal memory distribution
- **Multi-Agent Support**: Handle communication between multiple AI agents simultaneously
- **Context Awareness**: Maintain conversation context and agent state
- **Scalable Architecture**: Designed for high-performance, distributed deployments
- **MCP Integration**: Native support for Microservice Control Plane protocols
- **Memory Persistence**: Long-term storage and retrieval of agent knowledge

## 🛠️ Technology Stack

- **Backend**: Python 3.11+
- **Framework**: FastAPI for REST APIs
- **Database**: PostgreSQL with async support
- **Memory Storage**: Vector database for semantic search
- **MCP Integration**: Native MCP server implementation
- **Containerization**: Docker with docker-compose
- **Testing**: pytest with async support

## 📋 Prerequisites

- Python 3.11 or higher
- PostgreSQL 14+
- Docker and docker-compose
- Node.js 18+ (for development tools)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai-agent-memory-router
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 4. Start Services

```bash
# Start database and core services
docker-compose up -d

# Run the application
python -m uvicorn app.main:app --reload
```

### 5. Access the Application

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **MCP Server**: Available on configured port

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@localhost/ai_agent_memory` |
| `MCP_PORT` | MCP server port | `8001` |
| `API_PORT` | REST API port | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `ENVIRONMENT` | Environment (dev/staging/prod) | `development` |

### MCP Configuration

The project includes a pre-configured MCP server for seamless integration with Cursor and other MCP clients. Configuration is located in `.cursor/mcp.json`.

## 📚 API Documentation

### Core Endpoints

- `POST /api/v1/memory/route` - Route memory between agents
- `GET /api/v1/memory/{agent_id}` - Retrieve agent memories
- `POST /api/v1/memory/{agent_id}` - Store agent memory
- `GET /api/v1/agents` - List available agents
- `POST /api/v1/agents` - Register new agent

### MCP Tools

- `memory_route` - Route memory between agents
- `agent_register` - Register new AI agent
- `memory_search` - Search agent memories
- `context_get` - Get conversation context

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_memory_router.py
```

## 📦 Development

### Project Structure

```
ai-agent-memory-router/
├── app/                    # Main application code
│   ├── api/               # API endpoints
│   ├── core/              # Core business logic
│   ├── models/            # Data models
│   ├── services/          # Business services
│   └── utils/             # Utility functions
├── tests/                 # Test suite
├── docker/                # Docker configuration
├── docs/                  # Documentation
├── scripts/               # Utility scripts
└── .cursor/               # Cursor IDE configuration
```

### Development Workflow

1. **Feature Development**: Create feature branches from `main`
2. **Testing**: Ensure all tests pass before submitting PR
3. **Code Review**: All changes require review and approval
4. **Documentation**: Update docs for any API changes
5. **Release**: Follow semantic versioning for releases

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run linting
flake8 app/ tests/
black app/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/ai-agent-memory-router/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/ai-agent-memory-router/discussions)

## 🔗 Related Projects

- [Cipher](https://github.com/your-org/cipher) - Memory-powered AI agent framework
- [Shrimp Task Manager](https://github.com/your-org/shrimp-task-manager) - Task management system

---

**Version**: 0.1.0  
**Last Updated**: September 2025

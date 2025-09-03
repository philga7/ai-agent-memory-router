# Docker Infrastructure Setup Guide

This guide explains how to set up and use the Docker infrastructure for the AI Agent Memory Router project.

## 🏗️ Architecture Overview

Our Docker setup uses a **SQLite-first approach** that makes it easy to get started while maintaining the option to migrate to PostgreSQL later.

### Services

- **API Service** (`:8000`) - Main FastAPI application with SQLite database
- **MCP Server** (`:8001`) - Model Context Protocol server
- **Redis** (`:6379`) - Caching and session management
- **SQLite** - Local database (runs inside API container)

### Why SQLite-First?

1. **Quick Setup** - No external database server needed
2. **Development Friendly** - Perfect for local development and testing
3. **Easy Migration** - Storage abstraction layer makes switching to PostgreSQL simple
4. **Portable** - Database file travels with your project

## 🚀 Quick Start

### Prerequisites

- Docker Desktop installed and running
- Python 3.12+ (for running test scripts)

### 1. Start All Services

```bash
./start_docker.sh start
```

This will:
- Start all Docker containers
- Create necessary directories
- Wait for services to be healthy
- Show service status

### 2. Test Your Setup

```bash
./start_docker.sh test
```

Or manually:

```bash
python test_docker_setup.py
```

### 3. Check Service Status

```bash
./start_docker.sh status
```

## 📋 Available Commands

The `start_docker.sh` script provides several useful commands:

```bash
./start_docker.sh start      # Start all services
./start_docker.sh stop       # Stop all services
./start_docker.sh restart    # Restart all services
./start_docker.sh status     # Show service status
./start_docker.sh logs       # Show all logs
./start_docker.sh logs api   # Show logs for specific service
./start_docker.sh test       # Run test script
./start_docker.sh cleanup    # Remove all containers and volumes
./start_docker.sh help       # Show help
```

## 🔧 Manual Docker Commands

If you prefer to use Docker commands directly:

### Start Services
```bash
docker-compose up -d
```

### Stop Services
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f          # All services
docker-compose logs -f api      # Specific service
```

### Check Status
```bash
docker-compose ps
```

## 🗄️ Database Management

### SQLite Database

The SQLite database is automatically created at `data/ai_agent_memory.db` when you first start the services.

**Location**: `./data/ai_agent_memory.db`

**Initialization**: The database schema is automatically created using `docker/sqlite/init.sql`

### Database Schema

The initial schema includes:

- **agents** - AI agent information
- **memories** - Agent memories and knowledge
- **memory_routes** - Memory routing decisions
- **contexts** - Conversation context

### Backup and Restore

```bash
# Backup database
cp data/ai_agent_memory.db data/backup_$(date +%Y%m%d_%H%M%S).db

# Restore database
cp data/backup_YYYYMMDD_HHMMSS.db data/ai_agent_memory.db
```

## 🔍 Monitoring and Debugging

### Health Checks

All services include health checks that Docker automatically monitors:

- **API**: `http://localhost:8000/health`
- **MCP Server**: `http://localhost:8001/health`
- **Redis**: Internal ping command

### Logs

View logs for troubleshooting:

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f redis
docker-compose logs -f mcp

# Last 100 lines
docker-compose logs --tail=100 api
```

### Common Issues

#### Port Already in Use
```bash
# Check what's using port 8000
lsof -i :8000

# Kill the process or change the port in docker-compose.yml
```

#### Database Connection Issues
```bash
# Check if database file exists
ls -la data/

# Check database permissions
ls -la data/ai_agent_memory.db
```

#### Redis Connection Issues
```bash
# Test Redis connection
redis-cli -h localhost -p 6379 ping

# Check Redis logs
docker-compose logs redis
```

## 🔄 Migration to PostgreSQL

When you're ready to migrate to PostgreSQL:

### 1. Update Environment Variables

In `docker-compose.yml`, change:

```yaml
environment:
  - DATABASE_TYPE=postgresql
  - DATABASE_URL=postgresql+asyncpg://user:password@postgres:5432/dbname
```

### 2. Add PostgreSQL Service

```yaml
postgres:
  image: postgres:15-alpine
  environment:
    POSTGRES_DB: ai_agent_memory
    POSTGRES_USER: ai_agent_user
    POSTGRES_PASSWORD: ai_agent_password
  volumes:
    - postgres_data:/var/lib/postgresql/data
  ports:
    - "5432:5432"
```

### 3. Implement PostgreSQL Backend

Create `app/core/postgres_backend.py` implementing the `StorageBackend` interface.

### 4. Update Storage Manager

The storage abstraction layer will automatically use the new backend.

## 🧪 Testing

### Automated Tests

Run the comprehensive test suite:

```bash
./start_docker.sh test
```

### Manual Testing

Test individual services:

```bash
# Test API
curl http://localhost:8000/health

# Test Redis
redis-cli -h localhost -p 6379 ping

# Test MCP Server
curl http://localhost:8001/health
```

### Load Testing

For basic load testing:

```bash
# Install Apache Bench
brew install httpd  # macOS
sudo apt-get install apache2-utils  # Ubuntu

# Test API endpoint
ab -n 100 -c 10 http://localhost:8000/health
```

## 📊 Performance Tuning

### Redis Configuration

Edit `docker/redis/redis.conf` to optimize for your use case:

- **Memory**: Adjust `maxmemory` based on available RAM
- **Persistence**: Modify `save` intervals for your backup needs
- **Networking**: Adjust `tcp-backlog` for high concurrency

### SQLite Optimization

SQLite is already optimized for our use case, but you can:

- Add more indexes for specific query patterns
- Use `PRAGMA` statements for performance tuning
- Consider WAL mode for concurrent access

## 🔒 Security Considerations

### Development vs Production

**Development** (current setup):
- No authentication required
- SQLite database file accessible
- Redis no password

**Production** (recommended):
- Add authentication middleware
- Use PostgreSQL with proper user management
- Set Redis password
- Use HTTPS
- Implement rate limiting

### Environment Variables

Never commit sensitive information. Use `.env` files:

```bash
# .env (not committed to git)
DATABASE_PASSWORD=your_secure_password
REDIS_PASSWORD=your_redis_password
SECRET_KEY=your_secret_key
```

## 🚀 Deployment

### Local Development

```bash
./start_docker.sh start
```

### Staging/Production

1. Set environment variables
2. Use production Docker images
3. Configure reverse proxy (Nginx)
4. Set up monitoring (Prometheus/Grafana)
5. Configure backups
6. Set up SSL certificates

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [Redis Documentation](https://redis.io/documentation)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🤝 Getting Help

If you encounter issues:

1. Check the logs: `./start_docker.sh logs`
2. Verify service status: `./start_docker.sh status`
3. Run tests: `./start_docker.sh test`
4. Check this documentation
5. Review the project issues on GitHub

---

**Happy Docker-ing! 🐳**

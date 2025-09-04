"""
Configuration management for AI Agent Memory Router.

This module provides centralized configuration management using Pydantic settings
with environment variable support and validation.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(
        default="postgresql+asyncpg://user:password@localhost/ai_agent_memory",
        description="Database connection URL"
    )
    pool_size: int = Field(default=20, description="Database connection pool size")
    max_overflow: int = Field(default=30, description="Maximum database connections")
    pool_timeout: int = Field(default=30, description="Database connection timeout")
    
    model_config = {"env_prefix": "DATABASE_"}


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    pool_size: int = Field(default=10, description="Redis connection pool size")
    
    model_config = {"env_prefix": "REDIS_"}


class ChromaSettings(BaseSettings):
    """Chroma vector database configuration settings."""
    
    host: str = Field(default="localhost", description="Chroma server host")
    port: int = Field(default=8000, description="Chroma server port")
    collection_name: str = Field(
        default="agent_memories",
        description="Default collection name for agent memories"
    )
    
    model_config = {"env_prefix": "CHROMA_"}


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    secret_key: str = Field(
        default="your-secret-key-here-change-in-production",
        description="Secret key for JWT tokens"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration time in minutes"
    )
    refresh_token_expire_days: int = Field(
        default=7,
        description="Refresh token expiration time in days"
    )
    
    model_config = {"env_prefix": ""}


class CipherSettings(BaseSettings):
    """Cipher integration configuration settings."""
    
    api_url: str = Field(
        default="https://cipher.informedcrew.com",
        description="Cipher API base URL"
    )
    api_key: str = Field(
        default="your-cipher-api-key-here",
        description="Cipher API key"
    )
    timeout: int = Field(
        default=30,
        description="Cipher API request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for Cipher operations"
    )
    retry_delay: float = Field(
        default=1.0,
        description="Delay between retry attempts in seconds"
    )
    cache_ttl_hours: int = Field(
        default=1,
        description="Cache TTL for local memory cache in hours"
    )
    enable_hybrid_storage: bool = Field(
        default=True,
        description="Enable hybrid storage (Cipher + SQLite)"
    )
    
    model_config = {"env_prefix": "CIPHER_"}


class WeaviateSettings(BaseSettings):
    """Weaviate integration configuration settings."""
    
    api_url: str = Field(
        default="https://weaviate.informedcrew.com",
        description="Weaviate API base URL"
    )
    mcp_url: str = Field(
        default="https://weaviate.informedcrew.com/mcp/http",
        description="Weaviate MCP server URL"
    )
    
    model_config = {"env_prefix": "WEAVIATE_"}


class MemorySettings(BaseSettings):
    """Memory management configuration settings."""
    
    max_content_length: int = Field(
        default=10000,
        description="Maximum memory content length in characters"
    )
    max_context_size: int = Field(
        default=50000,
        description="Maximum context size in characters"
    )
    retention_days: int = Field(
        default=365,
        description="Memory retention period in days"
    )
    
    model_config = {"env_prefix": "MEMORY_"}


class AgentSettings(BaseSettings):
    """Agent management configuration settings."""
    
    max_agents_per_route: int = Field(
        default=10,
        description="Maximum number of agents per memory route"
    )
    heartbeat_interval: int = Field(
        default=30,
        description="Agent heartbeat interval in seconds"
    )
    timeout_seconds: int = Field(
        default=300,
        description="Agent timeout in seconds"
    )
    
    model_config = {"env_prefix": "AGENT_"}


class RoutingSettings(BaseSettings):
    """Routing configuration settings."""
    
    strategy: str = Field(
        default="intelligent_load_balanced",
        description="Default routing strategy"
    )
    max_attempts: int = Field(
        default=3,
        description="Maximum routing attempts"
    )
    timeout_seconds: int = Field(
        default=60,
        description="Routing timeout in seconds"
    )
    
    model_config = {"env_prefix": "ROUTING_"}


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    format: str = Field(default="json", description="Log format")
    file_path: str = Field(
        default="logs/app.log",
        description="Log file path"
    )
    max_size: str = Field(default="100MB", description="Maximum log file size")
    backup_count: int = Field(default=5, description="Number of log backups")
    
    model_config = {"env_prefix": "LOG_"}


class MonitoringSettings(BaseSettings):
    """Monitoring configuration settings."""
    
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics server port")
    health_check_interval: int = Field(
        default=30,
        description="Health check interval in seconds"
    )
    
    model_config = {"env_prefix": "MONITORING_"}


class CacheSettings(BaseSettings):
    """Caching configuration settings."""
    
    ttl_seconds: int = Field(
        default=3600,
        description="Cache TTL in seconds"
    )
    max_size: int = Field(
        default=1000,
        description="Maximum cache size"
    )
    enable_cache: bool = Field(default=True, description="Enable caching")
    
    model_config = {"env_prefix": "CACHE_"}


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: str = Field(default="development", description="Application environment")
    debug: bool = Field(default=True, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_prefix: str = Field(default="/api/v1", description="API prefix")
    
    # MCP Server Configuration
    mcp_host: str = Field(default="0.0.0.0", description="MCP server host")
    mcp_port: int = Field(default=8001, description="MCP server port")
    mcp_timeout: int = Field(default=60000, description="MCP server timeout")
    
    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    
    # Trusted Hosts
    allowed_hosts: List[str] = Field(
        default=["*"],
        description="Allowed host headers"
    )
    
    # Component Settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    cipher: CipherSettings = Field(default_factory=CipherSettings)
    weaviate: WeaviateSettings = Field(default_factory=WeaviateSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "allow"
    }
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment setting."""
        allowed_environments = ["development", "staging", "production"]
        if v not in allowed_environments:
            raise ValueError(f"Environment must be one of: {allowed_environments}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level setting."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"
    
    @property
    def is_staging(self) -> bool:
        """Check if running in staging mode."""
        return self.environment == "staging"


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings instance."""
    global _settings
    
    if _settings is None:
        _settings = Settings()
    
    return _settings


def reload_settings() -> Settings:
    """Reload application settings."""
    global _settings
    
    _settings = Settings()
    return _settings

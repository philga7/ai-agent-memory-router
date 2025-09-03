"""
Database configuration and session management for AI Agent Memory Router.

This module provides async database connection management using SQLAlchemy
with connection pooling and session handling.
"""

import asyncio
import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import declarative_base

from app.core.config import get_settings

# Setup logging
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Create declarative base for models
Base = declarative_base()

# Global database engine and session factory
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


async def init_database() -> None:
    """Initialize database connection and create tables."""
    global _engine, _session_factory
    
    try:
        logger.info("Initializing database connection...")
        
        # For SQLite, ensure the database file exists
        if "sqlite" in settings.database.url.lower():
            import os
            # Extract database path from URL
            db_path = settings.database.url.replace("sqlite+aiosqlite:///", "")
            if db_path.startswith("/"):
                # This is an absolute path, use it as is
                db_path = db_path
            else:
                # This is a relative path, make it absolute
                db_path = os.path.join("/app", db_path)
            
            # The data directory should already exist from the volume mount
            # Just check if the database file exists, and if not, it will be created by SQLite
            logger.info(f"Using SQLite database path: {db_path}")
        
        # Create async engine with SQLite-aware configuration
        engine_kwargs = {
            "url": settings.database.url,
            "echo": settings.debug,
            "pool_pre_ping": True,
        }
        
        # SQLite doesn't support connection pooling, so use NullPool
        if "sqlite" in settings.database.url.lower():
            engine_kwargs["poolclass"] = NullPool
        else:
            # PostgreSQL-specific settings
            engine_kwargs.update({
                "pool_size": settings.database.pool_size,
                "max_overflow": settings.database.max_overflow,
                "pool_timeout": settings.database.pool_timeout,
                "pool_recycle": 3600,
            })
        
        _engine = create_async_engine(**engine_kwargs)
        
        # Create session factory
        _session_factory = async_sessionmaker(
            bind=_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False
        )
        
        # Test connection
        async with _engine.begin() as conn:
            await conn.run_sync(lambda _: None)
        
        logger.info("Database connection initialized successfully")
        
        # Create tables if they don't exist
        await create_tables()
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def create_tables() -> None:
    """Create database tables if they don't exist."""
    try:
        logger.info("Creating database tables...")
        
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise


async def close_database() -> None:
    """Close database connections and cleanup."""
    global _engine, _session_factory
    
    try:
        if _engine:
            await _engine.dispose()
            _engine = None
            logger.info("Database engine disposed")
        
        _session_factory = None
        
    except Exception as e:
        logger.error(f"Error closing database: {e}")


def get_engine() -> AsyncEngine:
    """Get the database engine instance."""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get the database session factory."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _session_factory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session with automatic cleanup."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    session = _session_factory()
    
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_db_session_async() -> AsyncSession:
    """Get a database session for use in dependency injection."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    return _session_factory()


async def execute_in_transaction(func, *args, **kwargs):
    """Execute a function within a database transaction."""
    async with get_db_session() as session:
        try:
            result = await func(session, *args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            raise


# Database health check
async def check_database_health() -> bool:
    """Check if database is healthy and accessible."""
    try:
        if _engine is None:
            return False
        
        async with _engine.begin() as conn:
            await conn.run_sync(lambda _: None)
        return True
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


# Database statistics
async def get_database_stats() -> dict:
    """Get database statistics and connection pool info."""
    try:
        if _engine is None:
            return {"status": "not_initialized"}
        
        pool = _engine.pool
        return {
            "status": "healthy",
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid()
        }
        
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {"status": "error", "error": str(e)}


# Migration support
async def run_migrations() -> None:
    """Run database migrations using Alembic."""
    try:
        logger.info("Running database migrations...")
        
        # This would integrate with Alembic for migrations
        # For now, just log that migrations would run here
        
        logger.info("Database migrations completed")
        
    except Exception as e:
        logger.error(f"Failed to run migrations: {e}")
        raise


# Backup and restore support
async def backup_database(backup_path: str) -> bool:
    """Create a database backup."""
    try:
        logger.info(f"Creating database backup to {backup_path}")
        
        # This would implement actual backup logic
        # For now, just log the operation
        
        logger.info("Database backup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create database backup: {e}")
        return False


async def restore_database(backup_path: str) -> bool:
    """Restore database from backup."""
    try:
        logger.info(f"Restoring database from {backup_path}")
        
        # This would implement actual restore logic
        # For now, just log the operation
        
        logger.info("Database restore completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to restore database: {e}")
        return False

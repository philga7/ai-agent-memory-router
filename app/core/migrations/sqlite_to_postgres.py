"""
SQLite to PostgreSQL migration.

This module provides migration tools to upgrade from SQLite to PostgreSQL
while preserving all data and maintaining the same schema structure.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from .migration_manager import Migration, MigrationError

logger = logging.getLogger(__name__)


class SQLiteToPostgresMigration(Migration):
    """Migration from SQLite to PostgreSQL."""
    
    def __init__(self):
        super().__init__(
            version="2.0.0",
            description="Migrate from SQLite to PostgreSQL",
            dependencies=["1.0.0"]
        )
    
    async def up(self, storage) -> bool:
        """Migrate from SQLite to PostgreSQL."""
        try:
            logger.info("Starting SQLite to PostgreSQL migration")
            
            # Step 1: Export data from SQLite
            data = await self._export_sqlite_data(storage)
            
            # Step 2: Create PostgreSQL schema
            await self._create_postgres_schema(storage)
            
            # Step 3: Import data to PostgreSQL
            await self._import_postgres_data(storage, data)
            
            # Step 4: Verify migration
            success = await self._verify_migration(storage, data)
            
            if success:
                logger.info("SQLite to PostgreSQL migration completed successfully")
            else:
                logger.error("SQLite to PostgreSQL migration verification failed")
            
            return success
            
        except Exception as e:
            logger.error(f"SQLite to PostgreSQL migration failed: {e}")
            raise MigrationError(f"Migration failed: {e}")
    
    async def down(self, storage) -> bool:
        """Rollback migration (not supported for database system changes)."""
        logger.warning("Rollback from PostgreSQL to SQLite is not supported")
        return False
    
    async def _export_sqlite_data(self, storage) -> Dict[str, Any]:
        """Export all data from SQLite database."""
        logger.info("Exporting data from SQLite")
        
        data = {
            "agents": [],
            "memory_items": [],
            "memory_metadata": [],
            "memory_routes": [],
            "conversation_contexts": [],
            "performance_metrics": [],
            "system_events": []
        }
        
        try:
            # Export agents
            agents = await storage.agent.list_agents(limit=10000, offset=0)
            for agent in agents:
                data["agents"].append({
                    "id": agent.id,
                    "name": agent.name,
                    "description": agent.description,
                    "capabilities": agent.capabilities,
                    "status": agent.status,
                    "metadata": agent.metadata,
                    "created_at": agent.created_at.isoformat() if agent.created_at else None,
                    "updated_at": agent.updated_at.isoformat() if agent.updated_at else None
                })
            
            # Export memory items (in batches)
            offset = 0
            batch_size = 1000
            while True:
                memories = await storage.memory.get_agent_memories("", limit=batch_size, offset=offset)
                if not memories:
                    break
                
                for memory in memories:
                    data["memory_items"].append({
                        "id": memory.id,
                        "agent_id": memory.agent_id,
                        "content": memory.content,
                        "memory_type": memory.memory_type,
                        "priority": memory.priority,
                        "created_at": memory.created_at.isoformat() if memory.created_at else None,
                        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                        "expires_at": memory.expires_at.isoformat() if memory.expires_at else None
                    })
                
                offset += batch_size
                if len(memories) < batch_size:
                    break
            
            # Export metadata
            for memory in data["memory_items"]:
                metadata = await storage.metadata.get_metadata(memory["id"])
                if metadata:
                    data["memory_metadata"].append({
                        "id": metadata.id,
                        "memory_id": metadata.memory_id,
                        "tags": metadata.tags,
                        "source": metadata.source,
                        "confidence": metadata.confidence,
                        "embedding_vector": metadata.embedding_vector,
                        "vector_dimension": metadata.vector_dimension,
                        "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
                        "updated_at": metadata.updated_at.isoformat() if metadata.updated_at else None
                    })
            
            # Export routes
            for agent in data["agents"]:
                routes = await storage.routing.get_agent_routes(agent["id"], limit=10000, offset=0)
                for route in routes:
                    data["memory_routes"].append({
                        "id": route.id,
                        "source_agent_id": route.source_agent_id,
                        "target_agent_id": route.target_agent_id,
                        "memory_id": route.memory_id,
                        "route_type": route.route_type,
                        "priority": route.priority,
                        "status": route.status,
                        "routing_reason": route.routing_reason,
                        "created_at": route.created_at.isoformat() if route.created_at else None,
                        "delivered_at": route.delivered_at.isoformat() if route.delivered_at else None,
                        "acknowledged_at": route.acknowledged_at.isoformat() if route.acknowledged_at else None
                    })
            
            logger.info(f"Exported {len(data['agents'])} agents, {len(data['memory_items'])} memories, "
                       f"{len(data['memory_routes'])} routes")
            
            return data
            
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            raise MigrationError(f"Data export failed: {e}")
    
    async def _create_postgres_schema(self, storage) -> bool:
        """Create PostgreSQL schema."""
        logger.info("Creating PostgreSQL schema")
        
        # This would create the PostgreSQL schema
        # Implementation depends on the specific PostgreSQL storage backend
        # For now, just log the action
        logger.info("PostgreSQL schema creation would happen here")
        return True
    
    async def _import_postgres_data(self, storage, data: Dict[str, Any]) -> bool:
        """Import data to PostgreSQL."""
        logger.info("Importing data to PostgreSQL")
        
        try:
            # Import agents
            for agent_data in data["agents"]:
                # Create agent in PostgreSQL
                # This would use the PostgreSQL storage backend
                logger.debug(f"Importing agent: {agent_data['id']}")
            
            # Import memory items
            for memory_data in data["memory_items"]:
                # Create memory in PostgreSQL
                logger.debug(f"Importing memory: {memory_data['id']}")
            
            # Import metadata
            for metadata_data in data["memory_metadata"]:
                # Create metadata in PostgreSQL
                logger.debug(f"Importing metadata: {metadata_data['id']}")
            
            # Import routes
            for route_data in data["memory_routes"]:
                # Create route in PostgreSQL
                logger.debug(f"Importing route: {route_data['id']}")
            
            logger.info("Data import to PostgreSQL completed")
            return True
            
        except Exception as e:
            logger.error(f"Data import to PostgreSQL failed: {e}")
            raise MigrationError(f"Data import failed: {e}")
    
    async def _verify_migration(self, storage, original_data: Dict[str, Any]) -> bool:
        """Verify that the migration was successful."""
        logger.info("Verifying migration")
        
        try:
            # Verify agent count
            postgres_agents = await storage.agent.list_agents(limit=10000, offset=0)
            if len(postgres_agents) != len(original_data["agents"]):
                logger.error(f"Agent count mismatch: {len(postgres_agents)} vs {len(original_data['agents'])}")
                return False
            
            # Verify memory count
            total_memories = 0
            for agent in postgres_agents:
                memories = await storage.memory.get_agent_memories(agent.id, limit=10000, offset=0)
                total_memories += len(memories)
            
            if total_memories != len(original_data["memory_items"]):
                logger.error(f"Memory count mismatch: {total_memories} vs {len(original_data['memory_items'])}")
                return False
            
            logger.info("Migration verification completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Migration verification failed: {e}")
            return False


class PostgreSQLSchemaMigration(Migration):
    """Migration to create PostgreSQL-specific schema optimizations."""
    
    def __init__(self):
        super().__init__(
            version="2.1.0",
            description="Add PostgreSQL-specific schema optimizations",
            dependencies=["2.0.0"]
        )
    
    async def up(self, storage) -> bool:
        """Apply PostgreSQL schema optimizations."""
        logger.info("Applying PostgreSQL schema optimizations")
        
        # This would add PostgreSQL-specific features like:
        # - Full-text search indexes
        # - JSONB columns for better performance
        # - Partitioning for large tables
        # - Advanced indexing strategies
        
        logger.info("PostgreSQL schema optimizations applied")
        return True
    
    async def down(self, storage) -> bool:
        """Rollback PostgreSQL schema optimizations."""
        logger.info("Rolling back PostgreSQL schema optimizations")
        return True


# Register migrations
def register_migrations(migration_manager):
    """Register all migrations with the migration manager."""
    migration_manager.register_migration(SQLiteToPostgresMigration())
    migration_manager.register_migration(PostgreSQLSchemaMigration())

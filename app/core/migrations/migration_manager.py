"""
Migration manager for database schema upgrades.

This module provides a framework for managing database migrations
between different versions and database systems.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Exception raised during migration operations."""
    pass


class Migration:
    """Base class for database migrations."""
    
    def __init__(self, version: str, description: str, dependencies: List[str] = None):
        self.version = version
        self.description = description
        self.dependencies = dependencies or []
        self.applied_at: Optional[datetime] = None
    
    async def up(self, storage) -> bool:
        """Apply the migration."""
        raise NotImplementedError("Subclasses must implement up()")
    
    async def down(self, storage) -> bool:
        """Rollback the migration."""
        raise NotImplementedError("Subclasses must implement down()")
    
    def __str__(self):
        return f"Migration {self.version}: {self.description}"


class MigrationManager:
    """Manages database migrations."""
    
    def __init__(self, storage):
        self.storage = storage
        self.migrations: List[Migration] = []
        self.applied_migrations: Dict[str, datetime] = {}
    
    def register_migration(self, migration: Migration):
        """Register a migration."""
        self.migrations.append(migration)
        logger.info(f"Registered migration: {migration}")
    
    async def get_applied_migrations(self) -> Dict[str, datetime]:
        """Get list of applied migrations from the database."""
        # This would query a migrations table in the database
        # For now, return empty dict
        return {}
    
    async def apply_migration(self, migration: Migration) -> bool:
        """Apply a single migration."""
        try:
            logger.info(f"Applying migration: {migration}")
            
            # Check dependencies
            for dep in migration.dependencies:
                if dep not in self.applied_migrations:
                    raise MigrationError(f"Migration {migration.version} depends on {dep} which is not applied")
            
            # Apply migration
            success = await migration.up(self.storage)
            if success:
                self.applied_migrations[migration.version] = datetime.utcnow()
                logger.info(f"Successfully applied migration: {migration}")
                return True
            else:
                logger.error(f"Failed to apply migration: {migration}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying migration {migration.version}: {e}")
            raise MigrationError(f"Migration {migration.version} failed: {e}")
    
    async def rollback_migration(self, migration: Migration) -> bool:
        """Rollback a single migration."""
        try:
            logger.info(f"Rolling back migration: {migration}")
            
            # Check if migration is applied
            if migration.version not in self.applied_migrations:
                logger.warning(f"Migration {migration.version} is not applied, skipping rollback")
                return True
            
            # Rollback migration
            success = await migration.down(self.storage)
            if success:
                del self.applied_migrations[migration.version]
                logger.info(f"Successfully rolled back migration: {migration}")
                return True
            else:
                logger.error(f"Failed to rollback migration: {migration}")
                return False
                
        except Exception as e:
            logger.error(f"Error rolling back migration {migration.version}: {e}")
            raise MigrationError(f"Rollback of {migration.version} failed: {e}")
    
    async def migrate_to_version(self, target_version: str) -> bool:
        """Migrate to a specific version."""
        try:
            logger.info(f"Starting migration to version: {target_version}")
            
            # Get current version
            current_version = await self.get_current_version()
            logger.info(f"Current version: {current_version}")
            
            # Determine migration path
            if target_version == current_version:
                logger.info("Already at target version")
                return True
            
            # Sort migrations by version
            sorted_migrations = sorted(self.migrations, key=lambda m: m.version)
            
            if target_version > current_version:
                # Forward migration
                return await self._migrate_forward(current_version, target_version, sorted_migrations)
            else:
                # Backward migration
                return await self._migrate_backward(current_version, target_version, sorted_migrations)
                
        except Exception as e:
            logger.error(f"Migration to version {target_version} failed: {e}")
            raise MigrationError(f"Migration failed: {e}")
    
    async def _migrate_forward(self, current_version: str, target_version: str, migrations: List[Migration]) -> bool:
        """Migrate forward to a newer version."""
        for migration in migrations:
            if migration.version <= current_version:
                continue
            
            if migration.version > target_version:
                break
            
            if migration.version not in self.applied_migrations:
                success = await self.apply_migration(migration)
                if not success:
                    return False
        
        return True
    
    async def _migrate_backward(self, current_version: str, target_version: str, migrations: List[Migration]) -> bool:
        """Migrate backward to an older version."""
        # Reverse migrations for backward migration
        reverse_migrations = sorted(migrations, key=lambda m: m.version, reverse=True)
        
        for migration in reverse_migrations:
            if migration.version > current_version:
                continue
            
            if migration.version <= target_version:
                break
            
            if migration.version in self.applied_migrations:
                success = await self.rollback_migration(migration)
                if not success:
                    return False
        
        return True
    
    async def get_current_version(self) -> str:
        """Get the current database version."""
        # This would query the database for the current version
        # For now, return a default version
        return "1.0.0"
    
    async def get_pending_migrations(self) -> List[Migration]:
        """Get list of pending migrations."""
        applied = await self.get_applied_migrations()
        return [m for m in self.migrations if m.version not in applied]
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        current_version = await self.get_current_version()
        applied = await self.get_applied_migrations()
        pending = await self.get_pending_migrations()
        
        return {
            "current_version": current_version,
            "applied_migrations": list(applied.keys()),
            "pending_migrations": [m.version for m in pending],
            "total_migrations": len(self.migrations),
            "applied_count": len(applied),
            "pending_count": len(pending)
        }
    
    def create_migration_table(self):
        """Create the migrations table in the database."""
        # This would create a table to track applied migrations
        # Implementation depends on the specific database system
        pass

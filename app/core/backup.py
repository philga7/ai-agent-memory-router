"""
Backup and recovery system for AI Agent Memory Router.

This module provides comprehensive backup and recovery functionality
for the SQLite database, including automated backups, integrity checks,
and disaster recovery procedures.
"""

import os
import shutil
import sqlite3
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib
import gzip
import tempfile

logger = logging.getLogger(__name__)


class BackupError(Exception):
    """Exception raised during backup operations."""
    pass


class RecoveryError(Exception):
    """Exception raised during recovery operations."""
    pass


class BackupManager:
    """Manages database backup and recovery operations."""
    
    def __init__(self, database_path: str, backup_directory: str = "backups"):
        self.database_path = Path(database_path)
        self.backup_directory = Path(backup_directory)
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        
        # Ensure backup directory exists
        if not self.backup_directory.exists():
            self.backup_directory.mkdir(parents=True, exist_ok=True)
    
    async def create_backup(self, backup_name: Optional[str] = None, 
                           compress: bool = True, verify: bool = True) -> str:
        """Create a backup of the database."""
        try:
            # Generate backup name if not provided
            if not backup_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"backup_{timestamp}.db"
            
            backup_path = self.backup_directory / backup_name
            
            # Create backup
            logger.info(f"Creating backup: {backup_path}")
            
            # Use SQLite's backup API for integrity
            await self._create_sqlite_backup(backup_path)
            
            # Compress if requested
            if compress:
                compressed_path = await self._compress_backup(backup_path)
                backup_path = compressed_path
            
            # Verify backup if requested
            if verify:
                await self._verify_backup(backup_path)
            
            # Create backup metadata
            metadata = await self._create_backup_metadata(backup_path)
            await self._save_backup_metadata(backup_path, metadata)
            
            logger.info(f"Backup created successfully: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            raise BackupError(f"Backup creation failed: {e}")
    
    async def _create_sqlite_backup(self, backup_path: Path) -> None:
        """Create backup using SQLite's backup API."""
        try:
            # Create a temporary connection for backup
            source_conn = sqlite3.connect(self.database_path)
            backup_conn = sqlite3.connect(backup_path)
            
            # Use SQLite's backup API
            source_conn.backup(backup_conn)
            
            # Close connections
            backup_conn.close()
            source_conn.close()
            
        except Exception as e:
            logger.error(f"SQLite backup failed: {e}")
            raise BackupError(f"SQLite backup failed: {e}")
    
    async def _compress_backup(self, backup_path: Path) -> Path:
        """Compress a backup file using gzip."""
        try:
            compressed_path = backup_path.with_suffix(backup_path.suffix + '.gz')
            
            with open(backup_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove uncompressed backup
            backup_path.unlink()
            
            logger.info(f"Backup compressed: {compressed_path}")
            return compressed_path
            
        except Exception as e:
            logger.error(f"Backup compression failed: {e}")
            raise BackupError(f"Backup compression failed: {e}")
    
    async def _verify_backup(self, backup_path: Path) -> bool:
        """Verify backup integrity."""
        try:
            if backup_path.suffix.endswith('.gz'):
                # Decompress temporarily for verification
                with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(temp_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Verify the decompressed file
                success = await self._verify_sqlite_file(temp_path)
                
                # Clean up
                temp_path.unlink()
                
                return success
            else:
                return await self._verify_sqlite_file(backup_path)
                
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False
    
    async def _verify_sqlite_file(self, db_path: Path) -> bool:
        """Verify SQLite file integrity."""
        try:
            conn = sqlite3.connect(db_path)
            
            # Check if database is valid
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            
            if result and result[0] == "ok":
                # Check if we can read from main tables
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]
                
                if table_count > 0:
                    logger.info(f"Backup verification passed: {table_count} tables found")
                    conn.close()
                    return True
            
            conn.close()
            logger.error("Backup verification failed: integrity check failed")
            return False
            
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False
    
    async def _create_backup_metadata(self, backup_path: Path) -> Dict[str, Any]:
        """Create metadata for the backup."""
        try:
            # Get file information
            stat = backup_path.stat()
            
            # Calculate checksum
            checksum = await self._calculate_file_checksum(backup_path)
            
            # Get database information
            db_info = await self._get_database_info()
            
            metadata = {
                "backup_name": backup_path.name,
                "backup_path": str(backup_path),
                "created_at": datetime.now().isoformat(),
                "file_size": stat.st_size,
                "checksum": checksum,
                "compressed": backup_path.suffix.endswith('.gz'),
                "database_info": db_info,
                "backup_version": "1.0.0"
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to create backup metadata: {e}")
            return {}
    
    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        try:
            hash_sha256 = hashlib.sha256()
            
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            
            return hash_sha256.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to calculate checksum: {e}")
            return ""
    
    async def _get_database_info(self) -> Dict[str, Any]:
        """Get information about the source database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get table information
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get row counts for main tables
            table_counts = {}
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    table_counts[table] = count
                except:
                    table_counts[table] = 0
            
            # Get database size
            db_size = self.database_path.stat().st_size
            
            conn.close()
            
            return {
                "tables": tables,
                "table_counts": table_counts,
                "database_size": db_size,
                "backup_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {}
    
    async def _save_backup_metadata(self, backup_path: Path, metadata: Dict[str, Any]) -> None:
        """Save backup metadata to a JSON file."""
        try:
            metadata_path = backup_path.with_suffix('.json')
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug(f"Backup metadata saved: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save backup metadata: {e}")
    
    async def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups."""
        try:
            backups = []
            
            for backup_file in self.backup_directory.glob("*.db*"):
                if backup_file.name.startswith("backup_"):
                    metadata_path = backup_file.with_suffix('.json')
                    
                    metadata = {}
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                        except:
                            pass
                    
                    backup_info = {
                        "filename": backup_file.name,
                        "path": str(backup_file),
                        "size": backup_file.stat().st_size,
                        "created_at": backup_file.stat().st_mtime,
                        "metadata": metadata
                    }
                    
                    backups.append(backup_info)
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x["created_at"], reverse=True)
            
            return backups
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []
    
    async def restore_backup(self, backup_path: str, 
                           verify_after_restore: bool = True) -> bool:
        """Restore database from a backup."""
        try:
            backup_path = Path(backup_path)
            
            if not backup_path.exists():
                raise RecoveryError(f"Backup file not found: {backup_path}")
            
            logger.info(f"Starting restore from backup: {backup_path}")
            
            # Create backup of current database before restore
            current_backup = await self.create_backup(
                backup_name=f"pre_restore_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db",
                compress=False,
                verify=False
            )
            logger.info(f"Pre-restore backup created: {current_backup}")
            
            # Restore the backup
            if backup_path.suffix.endswith('.gz'):
                await self._restore_compressed_backup(backup_path)
            else:
                await self._restore_uncompressed_backup(backup_path)
            
            # Verify restore if requested
            if verify_after_restore:
                success = await self._verify_restore()
                if not success:
                    # Restore failed, try to recover
                    await self._recover_from_failed_restore(current_backup)
                    raise RecoveryError("Restore verification failed, database recovered from backup")
            
            logger.info("Database restore completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            raise RecoveryError(f"Database restore failed: {e}")
    
    async def _restore_compressed_backup(self, backup_path: Path) -> None:
        """Restore from a compressed backup."""
        try:
            # Decompress to temporary location
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            
            with gzip.open(backup_path, 'rb') as f_in:
                with open(temp_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Restore from temporary file
            await self._restore_uncompressed_backup(temp_path)
            
            # Clean up
            temp_path.unlink()
            
        except Exception as e:
            logger.error(f"Compressed backup restore failed: {e}")
            raise RecoveryError(f"Compressed backup restore failed: {e}")
    
    async def _restore_uncompressed_backup(self, backup_path: Path) -> None:
        """Restore from an uncompressed backup."""
        try:
            # Stop any active connections (this would be done by the application)
            # For now, we'll just copy the file
            
            # Create a backup of the current database
            current_backup = self.database_path.with_suffix('.db.backup')
            if self.database_path.exists():
                shutil.copy2(self.database_path, current_backup)
            
            # Copy backup to database location
            shutil.copy2(backup_path, self.database_path)
            
            # Remove temporary backup
            if current_backup.exists():
                current_backup.unlink()
            
        except Exception as e:
            logger.error(f"Uncompressed backup restore failed: {e}")
            raise RecoveryError(f"Uncompressed backup restore failed: {e}")
    
    async def _verify_restore(self) -> bool:
        """Verify that the restore was successful."""
        try:
            # Check if database is accessible
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Check if main tables exist
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            
            if table_count == 0:
                logger.error("Restore verification failed: no tables found")
                conn.close()
                return False
            
            # Check if we can read from main tables
            try:
                cursor.execute("SELECT COUNT(*) FROM agents")
                agent_count = cursor.fetchone()[0]
                logger.info(f"Restore verification: {agent_count} agents found")
            except:
                logger.warning("Restore verification: agents table not accessible")
            
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Restore verification failed: {e}")
            return False
    
    async def _recover_from_failed_restore(self, backup_path: str) -> None:
        """Recover from a failed restore operation."""
        try:
            logger.info(f"Recovering from failed restore using backup: {backup_path}")
            
            # Restore from the pre-restore backup
            await self._restore_uncompressed_backup(Path(backup_path))
            
            logger.info("Recovery from failed restore completed")
            
        except Exception as e:
            logger.error(f"Recovery from failed restore failed: {e}")
            raise RecoveryError(f"Recovery failed: {e}")
    
    async def cleanup_old_backups(self, keep_days: int = 30, 
                                 keep_count: int = 10) -> int:
        """Clean up old backups based on age and count."""
        try:
            backups = await self.list_backups()
            
            if not backups:
                return 0
            
            deleted_count = 0
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            
            # Delete backups older than keep_days
            for backup in backups:
                backup_date = datetime.fromtimestamp(backup["created_at"])
                
                if backup_date < cutoff_date:
                    backup_path = Path(backup["path"])
                    metadata_path = backup_path.with_suffix('.json')
                    
                    try:
                        if backup_path.exists():
                            backup_path.unlink()
                        if metadata_path.exists():
                            metadata_path.unlink()
                        
                        deleted_count += 1
                        logger.info(f"Deleted old backup: {backup['filename']}")
                        
                    except Exception as e:
                        logger.error(f"Failed to delete backup {backup['filename']}: {e}")
            
            # Keep only the most recent keep_count backups
            if len(backups) - deleted_count > keep_count:
                remaining_backups = [b for b in backups if b["created_at"] > cutoff_date.timestamp()]
                remaining_backups.sort(key=lambda x: x["created_at"], reverse=True)
                
                for backup in remaining_backups[keep_count:]:
                    backup_path = Path(backup["path"])
                    metadata_path = backup_path.with_suffix('.json')
                    
                    try:
                        if backup_path.exists():
                            backup_path.unlink()
                        if metadata_path.exists():
                            metadata_path.unlink()
                        
                        deleted_count += 1
                        logger.info(f"Deleted excess backup: {backup['filename']}")
                        
                    except Exception as e:
                        logger.error(f"Failed to delete backup {backup['filename']}: {e}")
            
            logger.info(f"Backup cleanup completed: {deleted_count} backups deleted")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            return 0
    
    async def get_backup_status(self) -> Dict[str, Any]:
        """Get backup system status."""
        try:
            backups = await self.list_backups()
            
            total_size = sum(b["size"] for b in backups)
            oldest_backup = min(backups, key=lambda x: x["created_at"]) if backups else None
            newest_backup = max(backups, key=lambda x: x["created_at"]) if backups else None
            
            return {
                "total_backups": len(backups),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "oldest_backup": oldest_backup["created_at"] if oldest_backup else None,
                "newest_backup": newest_backup["created_at"] if newest_backup else None,
                "backup_directory": str(self.backup_directory),
                "database_path": str(self.database_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get backup status: {e}")
            return {}

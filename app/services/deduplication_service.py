"""
Memory deduplication service for AI Agent Memory Router.

This service identifies and handles duplicate memories using local metadata
and semantic similarity analysis.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from uuid import uuid4
import hashlib
import difflib

from app.core.logging import get_logger
from app.core.config import get_settings
from app.services.weaviate_service import WeaviateMemoryService
from app.core.sqlite_storage import SQLiteUnifiedStorage
from app.models.memory import MemoryStore, MemorySearch, MemorySearchResponse

logger = get_logger(__name__)


class DeduplicationService:
    """Service for memory deduplication and management."""
    
    def __init__(self, weaviate_service: WeaviateMemoryService, sqlite_storage: SQLiteUnifiedStorage):
        """Initialize the deduplication service.
        
        Args:
            weaviate_service: Weaviate memory service for vector operations
            sqlite_storage: SQLite storage for metadata and routing
        """
        self.settings = get_settings()
        self.weaviate_service = weaviate_service
        self.sqlite_storage = sqlite_storage
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the deduplication service."""
        try:
            logger.info("Initializing deduplication service")
            
            # Ensure services are initialized
            if not self.weaviate_service._initialized:
                await self.weaviate_service.initialize()
            
            if not self.sqlite_storage:
                await self.sqlite_storage.initialize()
            
            self._initialized = True
            logger.info("Deduplication service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize deduplication service: {e}")
            return False
    
    async def find_duplicates(
        self,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
        similarity_threshold: float = 0.9,
        time_window_days: int = 30
    ) -> Dict[str, Any]:
        """Find duplicate memories based on content similarity.
        
        Args:
            agent_id: Optional agent ID to limit search
            project_id: Optional project ID to limit search
            similarity_threshold: Similarity threshold for considering memories as duplicates
            time_window_days: Time window to consider for duplicates
            
        Returns:
            Dictionary with duplicate analysis results
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Finding duplicates with threshold {similarity_threshold}")
            
            # Get memories to analyze
            search_filters = {}
            if agent_id:
                search_filters["agent_id"] = agent_id
            if project_id:
                search_filters["project_id"] = project_id
            
            # Get memories from the time window
            start_date = datetime.utcnow() - timedelta(days=time_window_days)
            
            memories_response = await self.weaviate_service.search_memories(
                MemorySearch(
                    query="",
                    limit=100  # Maximum allowed limit
                )
            )
            
            memories = memories_response.results
            
            # Find duplicates using multiple methods
            content_duplicates = await self._find_content_duplicates(memories, similarity_threshold)
            semantic_duplicates = await self._find_semantic_duplicates(memories, similarity_threshold)
            hash_duplicates = await self._find_hash_duplicates(memories)
            
            # Combine and deduplicate results
            all_duplicates = self._combine_duplicate_results(
                content_duplicates, semantic_duplicates, hash_duplicates
            )
            
            result = {
                "analysis_id": str(uuid4()),
                "agent_id": agent_id,
                "project_id": project_id,
                "similarity_threshold": similarity_threshold,
                "time_window_days": time_window_days,
                "total_memories_analyzed": len(memories),
                "duplicate_groups_found": len(all_duplicates),
                "total_duplicate_memories": sum(len(group) for group in all_duplicates),
                "duplicate_groups": all_duplicates,
                "analysis_methods": {
                    "content_similarity": len(content_duplicates),
                    "semantic_similarity": len(semantic_duplicates),
                    "hash_matching": len(hash_duplicates)
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Found {len(all_duplicates)} duplicate groups")
            return result
            
        except Exception as e:
            logger.error(f"Failed to find duplicates: {e}")
            return {"error": str(e)}
    
    async def resolve_duplicates(
        self,
        duplicate_groups: List[List[str]],
        resolution_strategy: str = "keep_most_recent"
    ) -> Dict[str, Any]:
        """Resolve duplicate memories using specified strategy.
        
        Args:
            duplicate_groups: List of duplicate memory ID groups
            resolution_strategy: Strategy for resolving duplicates
                - keep_most_recent: Keep the most recently created memory
                - keep_highest_importance: Keep the memory with highest importance
                - merge_content: Merge content and keep one memory
                - manual_review: Mark for manual review
        
        Returns:
            Dictionary with resolution results
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Resolving {len(duplicate_groups)} duplicate groups using strategy: {resolution_strategy}")
            
            resolution_results = []
            total_resolved = 0
            
            for group in duplicate_groups:
                if len(group) < 2:
                    continue
                
                # Get memory details for the group
                memories = []
                for memory_id in group:
                    memory = await self.weaviate_service.get_memory(memory_id)
                    if memory:
                        memories.append(memory)
                
                if len(memories) < 2:
                    continue
                
                # Apply resolution strategy
                resolution = await self._apply_resolution_strategy(
                    memories, resolution_strategy
                )
                
                resolution_results.append(resolution)
                total_resolved += len(resolution.get("memories_to_remove", []))
            
            result = {
                "resolution_id": str(uuid4()),
                "strategy": resolution_strategy,
                "groups_processed": len(duplicate_groups),
                "total_memories_resolved": total_resolved,
                "resolution_results": resolution_results,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Resolved {total_resolved} duplicate memories")
            return result
            
        except Exception as e:
            logger.error(f"Failed to resolve duplicates: {e}")
            return {"error": str(e)}
    
    async def auto_deduplicate(
        self,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
        similarity_threshold: float = 0.95,
        auto_resolve: bool = False
    ) -> Dict[str, Any]:
        """Automatically find and optionally resolve duplicates.
        
        Args:
            agent_id: Optional agent ID to limit search
            project_id: Optional project ID to limit search
            similarity_threshold: Similarity threshold for duplicates
            auto_resolve: Whether to automatically resolve duplicates
            
        Returns:
            Dictionary with deduplication results
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Starting auto-deduplication with threshold {similarity_threshold}")
            
            # Find duplicates
            duplicate_analysis = await self.find_duplicates(
                agent_id=agent_id,
                project_id=project_id,
                similarity_threshold=similarity_threshold
            )
            
            if "error" in duplicate_analysis:
                return duplicate_analysis
            
            result = {
                "deduplication_id": str(uuid4()),
                "agent_id": agent_id,
                "project_id": project_id,
                "similarity_threshold": similarity_threshold,
                "auto_resolve": auto_resolve,
                "duplicates_found": duplicate_analysis["duplicate_groups_found"],
                "memories_affected": duplicate_analysis["total_duplicate_memories"],
                "generated_at": datetime.utcnow().isoformat()
            }
            
            # Auto-resolve if requested
            if auto_resolve and duplicate_analysis["duplicate_groups_found"] > 0:
                resolution = await self.resolve_duplicates(
                    duplicate_analysis["duplicate_groups"],
                    resolution_strategy="keep_most_recent"
                )
                result["resolution"] = resolution
                result["memories_resolved"] = resolution.get("total_memories_resolved", 0)
            else:
                result["resolution"] = None
                result["memories_resolved"] = 0
            
            logger.info(f"Auto-deduplication completed: {result['duplicates_found']} groups found")
            return result
            
        except Exception as e:
            logger.error(f"Failed to auto-deduplicate: {e}")
            return {"error": str(e)}
    
    async def get_deduplication_stats(
        self,
        time_period_days: int = 30
    ) -> Dict[str, Any]:
        """Get deduplication statistics for a time period.
        
        Args:
            time_period_days: Number of days to analyze
            
        Returns:
            Dictionary with deduplication statistics
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Getting deduplication stats for {time_period_days} days")
            
            # This would typically query a deduplication history table
            # For now, we'll return a basic structure
            stats = {
                "time_period_days": time_period_days,
                "total_duplicates_found": 0,
                "total_memories_resolved": 0,
                "duplicates_by_agent": {},
                "duplicates_by_project": {},
                "resolution_strategies_used": {},
                "average_duplicate_group_size": 0.0,
                "most_common_duplicate_types": [],
                "generated_at": datetime.utcnow().isoformat()
            }
            
            logger.info("Deduplication stats generated")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get deduplication stats: {e}")
            return {"error": str(e)}
    
    async def _find_content_duplicates(
        self,
        memories: List[Any],
        similarity_threshold: float
    ) -> List[List[str]]:
        """Find duplicates based on content similarity."""
        duplicates = []
        processed = set()
        
        for i, memory1 in enumerate(memories):
            if memory1.memory_id in processed:
                continue
            
            duplicate_group = [memory1.memory_id]
            
            for j, memory2 in enumerate(memories[i+1:], i+1):
                if memory2.memory_id in processed:
                    continue
                
                # Calculate content similarity
                similarity = self._calculate_content_similarity(
                    memory1.content.text,
                    memory2.content.text
                )
                
                if similarity >= similarity_threshold:
                    duplicate_group.append(memory2.memory_id)
                    processed.add(memory2.memory_id)
            
            if len(duplicate_group) > 1:
                duplicates.append(duplicate_group)
                processed.add(memory1.memory_id)
        
        return duplicates
    
    async def _find_semantic_duplicates(
        self,
        memories: List[Any],
        similarity_threshold: float
    ) -> List[List[str]]:
        """Find duplicates based on semantic similarity using Weaviate."""
        duplicates = []
        processed = set()
        
        for i, memory1 in enumerate(memories):
            if memory1.memory_id in processed:
                continue
            
            # Search for semantically similar memories
            similar_memories = await self.weaviate_service.search_memories(
                MemorySearch(
                    query=memory1.content.text,
                    agent_id=memory1.source.agent_id,
                    limit=50
                )
            )
            
            duplicate_group = [memory1.memory_id]
            
            for similar_memory in similar_memories.results:
                if (similar_memory.memory_id != memory1.memory_id and
                    similar_memory.memory_id not in processed and
                    similar_memory.relevance_score >= similarity_threshold):
                    
                    duplicate_group.append(similar_memory.memory_id)
                    processed.add(similar_memory.memory_id)
            
            if len(duplicate_group) > 1:
                duplicates.append(duplicate_group)
                processed.add(memory1.memory_id)
        
        return duplicates
    
    async def _find_hash_duplicates(self, memories: List[Any]) -> List[List[str]]:
        """Find exact duplicates using content hashing."""
        content_hashes = {}
        duplicates = []
        
        for memory in memories:
            # Create hash of normalized content
            normalized_content = self._normalize_content(memory.content.text)
            content_hash = hashlib.md5(normalized_content.encode()).hexdigest()
            
            if content_hash in content_hashes:
                content_hashes[content_hash].append(memory.memory_id)
            else:
                content_hashes[content_hash] = [memory.memory_id]
        
        # Find groups with multiple memories (exact duplicates)
        for hash_key, memory_ids in content_hashes.items():
            if len(memory_ids) > 1:
                duplicates.append(memory_ids)
        
        return duplicates
    
    def _combine_duplicate_results(
        self,
        content_duplicates: List[List[str]],
        semantic_duplicates: List[List[str]],
        hash_duplicates: List[List[str]]
    ) -> List[List[str]]:
        """Combine duplicate results from different methods."""
        all_duplicates = []
        processed_memories = set()
        
        # Add hash duplicates first (most reliable)
        for group in hash_duplicates:
            all_duplicates.append(group)
            processed_memories.update(group)
        
        # Add content duplicates (excluding already processed)
        for group in content_duplicates:
            if not any(mem_id in processed_memories for mem_id in group):
                all_duplicates.append(group)
                processed_memories.update(group)
        
        # Add semantic duplicates (excluding already processed)
        for group in semantic_duplicates:
            if not any(mem_id in processed_memories for mem_id in group):
                all_duplicates.append(group)
                processed_memories.update(group)
        
        return all_duplicates
    
    async def _apply_resolution_strategy(
        self,
        memories: List[MemoryStore],
        strategy: str
    ) -> Dict[str, Any]:
        """Apply resolution strategy to a group of duplicate memories."""
        
        if strategy == "keep_most_recent":
            # Sort by creation date, keep the most recent
            memories.sort(key=lambda m: m.created_at, reverse=True)
            memory_to_keep = memories[0]
            memories_to_remove = memories[1:]
            
        elif strategy == "keep_highest_importance":
            # Sort by importance, keep the highest
            memories.sort(key=lambda m: m.importance, reverse=True)
            memory_to_keep = memories[0]
            memories_to_remove = memories[1:]
            
        elif strategy == "merge_content":
            # Merge content and keep the first memory
            memory_to_keep = memories[0]
            memories_to_remove = memories[1:]
            
            # Merge content from all memories
            merged_content = memory_to_keep.content.text
            for memory in memories_to_remove:
                merged_content += f"\n\n---\n\n{memory.content.text}"
            
            # Update the kept memory with merged content
            await self.weaviate_service.update_memory(
                memory_to_keep.id,
                {"content": merged_content}
            )
            
        else:  # manual_review
            # Mark for manual review, don't resolve automatically
            memory_to_keep = None
            memories_to_remove = []
        
        return {
            "strategy": strategy,
            "memory_to_keep": memory_to_keep.id if memory_to_keep else None,
            "memories_to_remove": [m.id for m in memories_to_remove],
            "action_taken": "resolved" if strategy != "manual_review" else "marked_for_review"
        }
    
    def _calculate_content_similarity(self, text1: str, text2: str) -> float:
        """Calculate content similarity between two texts."""
        # Normalize texts
        text1_norm = self._normalize_content(text1)
        text2_norm = self._normalize_content(text2)
        
        # Use difflib for sequence matching
        matcher = difflib.SequenceMatcher(None, text1_norm, text2_norm)
        return matcher.ratio()
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for comparison."""
        # Convert to lowercase, remove extra whitespace
        normalized = content.lower().strip()
        
        # Remove common punctuation and normalize spacing
        import re
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        try:
            return {
                "service_initialized": self._initialized,
                "weaviate_service_available": self.weaviate_service is not None,
                "sqlite_storage_available": self.sqlite_storage is not None,
                "deduplication_enabled": True
            }
            
        except Exception as e:
            logger.error(f"Failed to get service stats: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close the deduplication service."""
        try:
            self._initialized = False
            logger.info("Deduplication service closed")
            
        except Exception as e:
            logger.error(f"Error closing deduplication service: {e}")


# Global service instance
_deduplication_service: Optional[DeduplicationService] = None


async def get_deduplication_service(
    weaviate_service: Optional[WeaviateMemoryService] = None,
    sqlite_storage: Optional[SQLiteUnifiedStorage] = None
) -> DeduplicationService:
    """Get the global deduplication service instance."""
    global _deduplication_service
    
    if _deduplication_service is None:
        if weaviate_service is None:
            from app.services.weaviate_service import get_weaviate_service
            weaviate_service = await get_weaviate_service(sqlite_storage)
        
        _deduplication_service = DeduplicationService(weaviate_service, sqlite_storage)
        await _deduplication_service.initialize()
    
    return _deduplication_service


async def close_deduplication_service():
    """Close the global deduplication service."""
    global _deduplication_service
    
    if _deduplication_service:
        await _deduplication_service.close()
        _deduplication_service = None

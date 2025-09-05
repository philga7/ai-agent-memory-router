"""
Cross-project knowledge sharing service for AI Agent Memory Router.

This service enables knowledge sharing between different projects by leveraging
Weaviate's semantic search capabilities and SQLite metadata for routing decisions.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from uuid import uuid4

from app.core.logging import get_logger
from app.core.config import get_settings
from app.services.weaviate_service import WeaviateMemoryService
from app.core.sqlite_storage import SQLiteUnifiedStorage
from app.models.memory import MemorySearch, MemorySearchResponse, MemorySearchResult
from app.models.memory import AgentSource

logger = get_logger(__name__)


class CrossProjectSharingService:
    """Service for cross-project knowledge sharing."""
    
    def __init__(self, weaviate_service: WeaviateMemoryService, sqlite_storage: SQLiteUnifiedStorage):
        """Initialize the cross-project sharing service.
        
        Args:
            weaviate_service: Weaviate memory service for vector operations
            sqlite_storage: SQLite storage for metadata and routing
        """
        self.settings = get_settings()
        self.weaviate_service = weaviate_service
        self.sqlite_storage = sqlite_storage
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the cross-project sharing service."""
        try:
            logger.info("Initializing cross-project sharing service")
            
            # Ensure services are initialized
            if not self.weaviate_service._initialized:
                await self.weaviate_service.initialize()
            
            if not self.sqlite_storage:
                await self.sqlite_storage.initialize()
            
            self._initialized = True
            logger.info("Cross-project sharing service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize cross-project sharing service: {e}")
            return False
    
    async def share_knowledge_across_projects(
        self,
        source_project_id: str,
        query: str,
        target_project_ids: Optional[List[str]] = None,
        sharing_level: str = "standard",
        max_results: int = 10
    ) -> Dict[str, Any]:
        """Share knowledge from one project to others.
        
        Args:
            source_project_id: Source project ID
            query: Knowledge query to share
            target_project_ids: List of target project IDs (None for all)
            sharing_level: Level of sharing (standard, limited, full)
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with sharing results
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Sharing knowledge from project {source_project_id} with query: {query}")
            
            # Get cross-project memories
            cross_project_memories = await self.weaviate_service.get_cross_project_memories(
                query=query,
                source_project_id=source_project_id,
                target_project_ids=target_project_ids,
                limit=max_results
            )
            
            # Filter by sharing level
            filtered_memories = await self._filter_by_sharing_level(
                cross_project_memories, sharing_level
            )
            
            # Create sharing record
            sharing_id = str(uuid4())
            sharing_record = {
                "id": sharing_id,
                "source_project_id": source_project_id,
                "target_project_ids": target_project_ids or [],
                "query": query,
                "sharing_level": sharing_level,
                "memories_shared": len(filtered_memories),
                "created_at": datetime.utcnow().isoformat(),
                "status": "completed"
            }
            
            # Store sharing record in SQLite
            await self._store_sharing_record(sharing_record)
            
            result = {
                "sharing_id": sharing_id,
                "source_project_id": source_project_id,
                "target_project_ids": target_project_ids,
                "query": query,
                "sharing_level": sharing_level,
                "memories_found": len(cross_project_memories),
                "memories_shared": len(filtered_memories),
                "memories": filtered_memories,
                "created_at": sharing_record["created_at"],
                "status": "completed"
            }
            
            logger.info(f"Knowledge sharing completed: {len(filtered_memories)} memories shared")
            return result
            
        except Exception as e:
            logger.error(f"Failed to share knowledge across projects: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    async def discover_related_knowledge(
        self,
        project_id: str,
        memory_content: str,
        similarity_threshold: float = 0.7,
        max_results: int = 5
    ) -> List[MemorySearchResult]:
        """Discover related knowledge from other projects.
        
        Args:
            project_id: Current project ID
            memory_content: Memory content to find related knowledge for
            similarity_threshold: Minimum similarity threshold
            max_results: Maximum number of results
            
        Returns:
            List of related memories from other projects
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Discovering related knowledge for project {project_id}")
            
            # Search for related memories in other projects
            related_memories = await self.weaviate_service.get_cross_project_memories(
                query=memory_content,
                source_project_id=project_id,
                target_project_ids=None,  # Search all projects
                limit=max_results * 2  # Get more for filtering
            )
            
            # Filter by similarity threshold
            filtered_memories = [
                memory for memory in related_memories
                if memory.relevance_score >= similarity_threshold
            ]
            
            # Sort by relevance and limit results
            filtered_memories.sort(key=lambda m: m.relevance_score, reverse=True)
            filtered_memories = filtered_memories[:max_results]
            
            logger.info(f"Found {len(filtered_memories)} related memories from other projects")
            return filtered_memories
            
        except Exception as e:
            logger.error(f"Failed to discover related knowledge: {e}")
            return []
    
    async def get_project_knowledge_summary(
        self,
        project_id: str,
        include_cross_project: bool = True
    ) -> Dict[str, Any]:
        """Get a summary of knowledge for a project.
        
        Args:
            project_id: Project ID to summarize
            include_cross_project: Whether to include cross-project insights
            
        Returns:
            Dictionary with project knowledge summary
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Generating knowledge summary for project {project_id}")
            
            # Get project-specific memories
            project_memories = await self.weaviate_service.search_memories(
                MemorySearch(
                    query="",
                    limit=100
                )
            )
            
            # Analyze memory types and importance
            memory_types = {}
            importance_distribution = {"low": 0, "medium": 0, "high": 0}
            total_memories = len(project_memories.results)
            
            for memory in project_memories.results:
                # Count memory types
                memory_type = memory.memory_type
                memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
                
                # Categorize importance
                if memory.importance < 0.3:
                    importance_distribution["low"] += 1
                elif memory.importance < 0.7:
                    importance_distribution["medium"] += 1
                else:
                    importance_distribution["high"] += 1
            
            summary = {
                "project_id": project_id,
                "total_memories": total_memories,
                "memory_types": memory_types,
                "importance_distribution": importance_distribution,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            # Add cross-project insights if requested
            if include_cross_project:
                cross_project_insights = await self._get_cross_project_insights(project_id)
                summary["cross_project_insights"] = cross_project_insights
            
            logger.info(f"Knowledge summary generated for project {project_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate knowledge summary: {e}")
            return {"error": str(e)}
    
    async def suggest_knowledge_transfer(
        self,
        source_project_id: str,
        target_project_id: str,
        transfer_type: str = "automatic"
    ) -> Dict[str, Any]:
        """Suggest knowledge that could be transferred between projects.
        
        Args:
            source_project_id: Source project ID
            target_project_id: Target project ID
            transfer_type: Type of transfer (automatic, manual, selective)
            
        Returns:
            Dictionary with transfer suggestions
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Suggesting knowledge transfer from {source_project_id} to {target_project_id}")
            
            # Get high-importance memories from source project
            source_memories = await self.weaviate_service.search_memories(
                MemorySearch(
                    query="",
                    limit=50
                )
            )
            
            # Get target project's existing knowledge
            target_memories = await self.weaviate_service.search_memories(
                MemorySearch(
                    query="",
                    limit=100
                )
            )
            
            # Find knowledge gaps and transfer opportunities
            transfer_suggestions = []
            target_knowledge_set = set()
            
            # Build knowledge set from target project
            for memory in target_memories.results:
                target_knowledge_set.add(memory.content.text.lower())
            
            # Find transferable knowledge
            for source_memory in source_memories.results:
                source_content = source_memory.content.text.lower()
                
                # Check if this knowledge already exists in target project
                if not any(source_content in target_knowledge for target_knowledge in target_knowledge_set):
                    transfer_suggestions.append({
                        "memory_id": source_memory.memory_id,
                        "content": source_memory.content.text,
                        "memory_type": source_memory.memory_type,
                        "importance": source_memory.importance,
                        "relevance_score": source_memory.relevance_score,
                        "transfer_reason": "high_importance_knowledge_gap"
                    })
            
            # Sort by importance and relevance
            transfer_suggestions.sort(
                key=lambda x: (x["importance"], x["relevance_score"]),
                reverse=True
            )
            
            result = {
                "source_project_id": source_project_id,
                "target_project_id": target_project_id,
                "transfer_type": transfer_type,
                "suggestions_count": len(transfer_suggestions),
                "suggestions": transfer_suggestions[:10],  # Top 10 suggestions
                "generated_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Generated {len(transfer_suggestions)} transfer suggestions")
            return result
            
        except Exception as e:
            logger.error(f"Failed to suggest knowledge transfer: {e}")
            return {"error": str(e)}
    
    async def track_knowledge_flow(
        self,
        project_id: str,
        time_period_days: int = 30
    ) -> Dict[str, Any]:
        """Track knowledge flow for a project over time.
        
        Args:
            project_id: Project ID to track
            time_period_days: Number of days to look back
            
        Returns:
            Dictionary with knowledge flow metrics
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Tracking knowledge flow for project {project_id}")
            
            # Get sharing records for the time period
            start_date = datetime.utcnow() - timedelta(days=time_period_days)
            
            # This would require additional database queries to get sharing history
            # For now, we'll return a basic structure
            flow_metrics = {
                "project_id": project_id,
                "time_period_days": time_period_days,
                "start_date": start_date.isoformat(),
                "end_date": datetime.utcnow().isoformat(),
                "knowledge_shared_out": 0,  # Would be calculated from sharing records
                "knowledge_received_in": 0,  # Would be calculated from sharing records
                "cross_project_interactions": 0,  # Would be calculated from sharing records
                "most_active_partner_projects": [],  # Would be calculated from sharing records
                "generated_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Knowledge flow tracking completed for project {project_id}")
            return flow_metrics
            
        except Exception as e:
            logger.error(f"Failed to track knowledge flow: {e}")
            return {"error": str(e)}
    
    async def _filter_by_sharing_level(
        self,
        memories: List[MemorySearchResult],
        sharing_level: str
    ) -> List[MemorySearchResult]:
        """Filter memories by sharing level."""
        
        if sharing_level == "full":
            return memories
        elif sharing_level == "limited":
            # Only share high-importance memories
            return [m for m in memories if m.importance >= 0.7]
        elif sharing_level == "standard":
            # Share medium and high importance memories
            return [m for m in memories if m.importance >= 0.5]
        else:
            # Default: only high importance
            return [m for m in memories if m.importance >= 0.8]
    
    async def _store_sharing_record(self, sharing_record: Dict[str, Any]) -> bool:
        """Store a knowledge sharing record in SQLite."""
        try:
            # This would store the sharing record in a dedicated table
            # For now, we'll log it
            logger.info(f"Knowledge sharing record: {sharing_record['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store sharing record: {e}")
            return False
    
    async def _get_cross_project_insights(self, project_id: str) -> Dict[str, Any]:
        """Get cross-project insights for a project."""
        try:
            # Get recent cross-project interactions
            insights = {
                "recent_cross_project_searches": 0,
                "knowledge_shared_recently": 0,
                "knowledge_received_recently": 0,
                "most_related_projects": [],
                "knowledge_gaps_identified": 0
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get cross-project insights: {e}")
            return {"error": str(e)}
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        try:
            return {
                "service_initialized": self._initialized,
                "weaviate_service_available": self.weaviate_service is not None,
                "sqlite_storage_available": self.sqlite_storage is not None,
                "cross_project_sharing_enabled": True
            }
            
        except Exception as e:
            logger.error(f"Failed to get service stats: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close the cross-project sharing service."""
        try:
            self._initialized = False
            logger.info("Cross-project sharing service closed")
            
        except Exception as e:
            logger.error(f"Error closing cross-project sharing service: {e}")


# Global service instance
_cross_project_service: Optional[CrossProjectSharingService] = None


async def get_cross_project_service(
    weaviate_service: Optional[WeaviateMemoryService] = None,
    sqlite_storage: Optional[SQLiteUnifiedStorage] = None
) -> CrossProjectSharingService:
    """Get the global cross-project sharing service instance."""
    global _cross_project_service
    
    if _cross_project_service is None:
        if weaviate_service is None:
            from app.services.weaviate_service import get_weaviate_service
            weaviate_service = await get_weaviate_service(sqlite_storage)
        
        _cross_project_service = CrossProjectSharingService(weaviate_service, sqlite_storage)
        await _cross_project_service.initialize()
    
    return _cross_project_service


async def close_cross_project_service():
    """Close the global cross-project sharing service."""
    global _cross_project_service
    
    if _cross_project_service:
        await _cross_project_service.close()
        _cross_project_service = None

"""
Core Weaviate client implementation for AI Agent Memory Router.

This module provides the core Weaviate client functionality with built-in vectorization
for semantic search and memory storage.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class WeaviateClient:
    """Core Weaviate client for vector operations."""
    
    def __init__(self):
        """Initialize the Weaviate client."""
        self.settings = get_settings()
        self.client: Optional[weaviate.Client] = None
        self.collection = None
        self._initialized = False
        self._collection_name = self.settings.weaviate.collection_name
    
    async def initialize(self) -> bool:
        """Initialize the Weaviate client and create collection if needed."""
        try:
            logger.info(f"Initializing Weaviate client for {self.settings.weaviate.api_url}")
            
            # Check if this is a local or cloud connection
            from urllib.parse import urlparse
            parsed_url = urlparse(self.settings.weaviate.api_url)
            
            if parsed_url.hostname in ["localhost", "127.0.0.1"] or parsed_url.scheme == "http":
                # Local connection
                host = parsed_url.hostname or "localhost"
                port = parsed_url.port or 8080
                
                auth_credentials = None
                if self.settings.weaviate.api_key:
                    auth_credentials = weaviate.classes.init.Auth.api_key(self.settings.weaviate.api_key)
                
                self.client = weaviate.connect_to_local(
                    host=host,
                    port=port,
                    auth_credentials=auth_credentials
                )
            else:
                # Cloud connection
                auth_credentials = None
                if self.settings.weaviate.api_key:
                    auth_credentials = weaviate.classes.init.Auth.api_key(self.settings.weaviate.api_key)
                
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.settings.weaviate.api_url,
                    auth_credentials=auth_credentials,
                    skip_init_checks=True
                )
            
            # Test connection
            try:
                if self.client.is_ready():
                    logger.info("Weaviate client connected successfully")
                else:
                    logger.error("Weaviate client is not ready")
                    return False
            except Exception as e:
                logger.error(f"Failed to connect to Weaviate: {e}")
                return False
            
            # Create or get collection
            await self._ensure_collection_exists()
            
            self._initialized = True
            logger.info("Weaviate client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate client: {e}")
            return False
    
    async def _ensure_collection_exists(self):
        """Ensure the collection exists with proper schema."""
        try:
            # Check if collection exists
            if self.client.collections.exists(self._collection_name):
                logger.info(f"Collection '{self._collection_name}' already exists")
                self.collection = self.client.collections.get(self._collection_name)
                return
            
            # Create collection with schema
            logger.info(f"Creating collection '{self._collection_name}'")
            
            self.collection = self.client.collections.create(
                name=self._collection_name,
                properties=[
                    Property(
                        name="content",
                        data_type=DataType.TEXT,
                        description="The memory content text"
                    ),
                    Property(
                        name="agent_id",
                        data_type=DataType.TEXT,
                        description="ID of the agent that created this memory"
                    ),
                    Property(
                        name="memory_type",
                        data_type=DataType.TEXT,
                        description="Type of memory (fact, experience, skill, etc.)"
                    ),
                    Property(
                        name="importance",
                        data_type=DataType.NUMBER,
                        description="Importance score of the memory (0.0 to 1.0)"
                    ),
                    Property(
                        name="tags",
                        data_type=DataType.TEXT_ARRAY,
                        description="Tags associated with the memory"
                    ),
                    Property(
                        name="metadata",
                        data_type=DataType.TEXT,
                        description="Additional metadata as JSON string"
                    ),
                    Property(
                        name="project_id",
                        data_type=DataType.TEXT,
                        description="ID of the project this memory belongs to"
                    ),
                    Property(
                        name="created_at",
                        data_type=DataType.DATE,
                        description="When the memory was created"
                    ),
                    Property(
                        name="updated_at",
                        data_type=DataType.DATE,
                        description="When the memory was last updated"
                    ),
                    Property(
                        name="expires_at",
                        data_type=DataType.DATE,
                        description="When the memory expires (optional)"
                    )
                ],
                # Use built-in text2vec-transformers for automatic vectorization
                # Note: Vector configuration is handled by Weaviate's built-in modules
            )
            
            logger.info(f"Collection '{self._collection_name}' created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    async def store_memory(
        self,
        memory_id: str,
        content: str,
        agent_id: str,
        memory_type: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> bool:
        """Store a memory in Weaviate.
        
        Args:
            memory_id: Unique identifier for the memory
            content: The memory content text
            agent_id: ID of the agent that created this memory
            memory_type: Type of memory
            importance: Importance score (0.0 to 1.0)
            tags: Optional tags
            metadata: Optional metadata
            project_id: Optional project ID
            expires_at: Optional expiration date
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Ensure collection is available
            if not self.collection:
                if self.client.collections.exists(self._collection_name):
                    self.collection = self.client.collections.get(self._collection_name)
                else:
                    logger.error("Collection not available")
                    return False
            
            # Prepare data object
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            data_object = {
                "content": content,
                "agent_id": agent_id,
                "memory_type": memory_type,
                "importance": importance,
                "tags": tags or [],
                "metadata": json.dumps(metadata or {}),
                "project_id": project_id or "default",
                "created_at": now,
                "updated_at": now,
                "expires_at": expires_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if expires_at else None
            }
            
            # Store in Weaviate
            result = self.collection.data.insert(
                properties=data_object,
                uuid=memory_id
            )
            
            if result:
                logger.info(f"Memory stored successfully in Weaviate: {memory_id}")
                return True
            else:
                logger.error(f"Failed to store memory in Weaviate: {memory_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing memory in Weaviate: {e}")
            return False
    
    async def search_memories(
        self,
        query: str,
        agent_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        project_id: Optional[str] = None,
        min_importance: Optional[float] = None,
        limit: int = 10,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search memories using semantic search.
        
        Args:
            query: Search query text
            agent_id: Filter by agent ID
            memory_type: Filter by memory type
            project_id: Filter by project ID
            min_importance: Minimum importance threshold
            limit: Maximum number of results
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of search results
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Ensure collection is available
            if not self.collection:
                if self.client.collections.exists(self._collection_name):
                    self.collection = self.client.collections.get(self._collection_name)
                else:
                    logger.error("Collection not available")
                    return []
            
            # Build where clause for filters
            where_clause = {}
            
            if agent_id:
                where_clause["agent_id"] = {"equal": agent_id}
            
            if memory_type:
                where_clause["memory_type"] = {"equal": memory_type}
            
            if project_id:
                where_clause["project_id"] = {"equal": project_id}
            
            if min_importance is not None:
                where_clause["importance"] = {"greaterThanEqual": min_importance}
            
            # Perform semantic search
            if where_clause:
                logger.info(f"Searching with where_clause: {where_clause}")
                # For now, perform search without filters and filter results in Python
                # This is less efficient but works with the current Weaviate client version
                response = self.collection.query.near_text(
                    query=query,
                    limit=limit * 3,  # Get more results to filter from
                    return_metadata=MetadataQuery(distance=True)
                )
                
                logger.info(f"Found {len(response.objects)} objects before filtering")
                
                # Filter results based on where_clause
                filtered_objects = []
                for obj in response.objects:
                    # Check if object matches the where clause
                    matches = True
                    for field, condition in where_clause.items():
                        if field in obj.properties:
                            obj_value = obj.properties[field]
                            logger.info(f"Checking field {field}: obj_value={obj_value}, condition={condition}")
                            if condition.get("equal") and obj_value != condition["equal"]:
                                matches = False
                                logger.info(f"Field {field} does not match: {obj_value} != {condition['equal']}")
                                break
                            elif condition.get("greaterThanEqual") and obj_value < condition["greaterThanEqual"]:
                                matches = False
                                logger.info(f"Field {field} does not meet threshold: {obj_value} < {condition['greaterThanEqual']}")
                                break
                        else:
                            matches = False
                            logger.info(f"Field {field} not found in object properties")
                            break
                    
                    if matches:
                        filtered_objects.append(obj)
                        logger.info(f"Object {obj.uuid} matches filter criteria")
                        if len(filtered_objects) >= limit:
                            break
                    else:
                        logger.info(f"Object {obj.uuid} does not match filter criteria")
                
                logger.info(f"Found {len(filtered_objects)} objects after filtering")
                
                # Create a mock response object with filtered results
                class MockResponse:
                    def __init__(self, objects):
                        self.objects = objects
                
                response = MockResponse(filtered_objects)
            else:
                # Use simple near_text search
                response = self.collection.query.near_text(
                    query=query,
                    limit=limit,
                    return_metadata=MetadataQuery(distance=True)
                )
            
            # Process results
            results = []
            threshold = similarity_threshold or self.settings.weaviate.similarity_threshold
            
            for obj in response.objects:
                # Convert distance to similarity (1 - distance)
                distance = obj.metadata.distance
                similarity = 1.0 - distance if distance is not None else 1.0
                
                # Filter by similarity threshold
                if similarity >= threshold:
                    result = {
                        "id": str(obj.uuid),
                        "content": obj.properties.get("content", ""),
                        "agent_id": obj.properties.get("agent_id", ""),
                        "memory_type": obj.properties.get("memory_type", ""),
                        "importance": obj.properties.get("importance", 0.0),
                        "tags": obj.properties.get("tags", []),
                        "metadata": json.loads(obj.properties.get("metadata", "{}")),
                        "project_id": obj.properties.get("project_id", ""),
                        "created_at": obj.properties.get("created_at", ""),
                        "updated_at": obj.properties.get("updated_at", ""),
                        "expires_at": obj.properties.get("expires_at"),
                        "similarity": similarity,
                        "distance": distance
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} memories for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by ID.
        
        Args:
            memory_id: Memory ID to retrieve
            
        Returns:
            Memory data or None if not found
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Ensure collection is available
            if not self.collection:
                if self.client.collections.exists(self._collection_name):
                    self.collection = self.client.collections.get(self._collection_name)
                else:
                    logger.error("Collection not available")
                    return None
            
            # Get memory by UUID
            response = self.collection.query.fetch_object_by_id(
                uuid=memory_id
            )
            
            if response:
                return {
                    "id": str(response.uuid),
                    "content": response.properties.get("content", ""),
                    "agent_id": response.properties.get("agent_id", ""),
                    "memory_type": response.properties.get("memory_type", ""),
                    "importance": response.properties.get("importance", 0.0),
                    "tags": response.properties.get("tags", []),
                    "metadata": json.loads(response.properties.get("metadata", "{}")),
                    "project_id": response.properties.get("project_id", ""),
                    "created_at": response.properties.get("created_at", ""),
                    "updated_at": response.properties.get("updated_at", ""),
                    "expires_at": response.properties.get("expires_at")
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None
    
    async def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update a memory in Weaviate.
        
        Args:
            memory_id: Memory ID to update
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Ensure collection is available
            if not self.collection:
                if self.client.collections.exists(self._collection_name):
                    self.collection = self.client.collections.get(self._collection_name)
                else:
                    logger.error("Collection not available")
                    return False
            
            # Add updated_at timestamp
            updates["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            
            # Update memory
            result = self.collection.data.update(
                uuid=memory_id,
                properties=updates
            )
            
            if result:
                logger.info(f"Memory updated successfully: {memory_id}")
                return True
            else:
                logger.error(f"Failed to update memory: {memory_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {e}")
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from Weaviate.
        
        Args:
            memory_id: Memory ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Ensure collection is available
            if not self.collection:
                if self.client.collections.exists(self._collection_name):
                    self.collection = self.client.collections.get(self._collection_name)
                else:
                    logger.error("Collection not available")
                    return False
            
            # Delete memory
            result = self.collection.data.delete_by_id(memory_id)
            
            if result:
                logger.info(f"Memory deleted successfully: {memory_id}")
                return True
            else:
                logger.error(f"Failed to delete memory: {memory_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Ensure collection is available
            if not self.collection:
                # Try to get the collection again
                if self.client.collections.exists(self._collection_name):
                    self.collection = self.client.collections.get(self._collection_name)
                else:
                    return {"error": "Collection not available"}
            
            # Get collection info
            collection_info = self.collection.config.get()
            
            return {
                "name": collection_info.name,
                "vectorizer": collection_info.vectorizer_config.vectorizer,
                "properties": len(collection_info.properties),
                "vectorize_properties": getattr(collection_info.vectorizer_config, 'vectorize_properties', [])
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    async def clear_all_memories(self) -> bool:
        """Clear all memories from the collection."""
        try:
            logger.info("Starting to clear all memories from collection...")
            if not self._initialized:
                await self.initialize()
            
            if not self.collection:
                logger.error("Collection not available")
                return False
            
            # Get all objects first
            all_objects = self.collection.query.get(limit=1000)
            
            if not all_objects.objects:
                logger.info("No objects found in collection")
                return True
            
            logger.info(f"Found {len(all_objects.objects)} objects to delete")
            
            # Delete each object individually
            deleted_count = 0
            for obj in all_objects.objects:
                try:
                    result = self.collection.data.delete_by_id(obj.uuid)
                    if result:
                        deleted_count += 1
                    else:
                        logger.warning(f"Failed to delete object {obj.uuid}")
                except Exception as e:
                    logger.warning(f"Error deleting object {obj.uuid}: {e}")
            
            logger.info(f"Successfully deleted {deleted_count} objects from collection")
            return deleted_count > 0
                
        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
            return False
    
    async def close(self):
        """Close the Weaviate client."""
        try:
            if self.client:
                self.client.close()
                self.client = None
            
            self.collection = None
            self._initialized = False
            logger.info("Weaviate client closed")
            
        except Exception as e:
            logger.error(f"Error closing Weaviate client: {e}")


# Global Weaviate client instance
_weaviate_client: Optional[WeaviateClient] = None


async def get_weaviate_client() -> WeaviateClient:
    """Get the global Weaviate client instance."""
    global _weaviate_client
    
    if _weaviate_client is None:
        _weaviate_client = WeaviateClient()
        await _weaviate_client.initialize()
    
    return _weaviate_client


async def close_weaviate_client():
    """Close the global Weaviate client."""
    global _weaviate_client
    
    if _weaviate_client:
        await _weaviate_client.close()
        _weaviate_client = None

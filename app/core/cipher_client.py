"""
Cipher MCP client implementation for AI Agent Memory Router.

This module provides a client interface to interact with Cipher MCP server
for project-based memory storage and retrieval operations.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from contextlib import asynccontextmanager
import aiohttp
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger

# Get settings and logger
settings = get_settings()
logger = get_logger(__name__)


class CipherMCPError(Exception):
    """Base exception for Cipher MCP operations."""
    pass


class CipherConnectionError(CipherMCPError):
    """Exception raised for connection issues with Cipher MCP."""
    pass


class CipherOperationError(CipherMCPError):
    """Exception raised for Cipher MCP operation failures."""
    pass


class CipherAPIClient:
    """Client for interacting with Cipher API."""
    
    def __init__(self, base_url: str = None, api_key: str = None, timeout: int = None):
        """Initialize Cipher API client.
        
        Args:
            base_url: Base URL for Cipher API
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or settings.cipher.api_url
        self.api_key = api_key or settings.cipher.api_key
        self.timeout = timeout or settings.cipher.timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self._connection_retries = settings.cipher.max_retries
        self._retry_delay = settings.cipher.retry_delay
        
        logger.info(f"Cipher API client initialized with base URL: {self.base_url}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> None:
        """Establish connection to Cipher MCP server."""
        try:
            if self.session is None or self.session.closed:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                self.session = aiohttp.ClientSession(
                    timeout=timeout,
                    headers={
                        'Content-Type': 'application/json',
                        'User-Agent': 'AI-Agent-Memory-Router/1.0'
                    }
                )
            
            # Test connection
            await self._test_connection()
            logger.info("Connected to Cipher MCP server successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Cipher MCP server: {e}")
            raise CipherConnectionError(f"Connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close connection to Cipher MCP server."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Disconnected from Cipher MCP server")
    
    async def _test_connection(self) -> None:
        """Test connection to Cipher API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            async with self.session.get(f"{self.base_url}/health", headers=headers) as response:
                if response.status != 200:
                    raise CipherConnectionError(f"Health check failed with status {response.status}")
        except aiohttp.ClientError as e:
            raise CipherConnectionError(f"Connection test failed: {e}")
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        retries: int = None
    ) -> Dict[str, Any]:
        """Make HTTP request to Cipher API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            data: Request data
            retries: Number of retries (defaults to self._connection_retries)
            
        Returns:
            Response data as dictionary
            
        Raises:
            CipherConnectionError: If connection fails
            CipherOperationError: If operation fails
        """
        if retries is None:
            retries = self._connection_retries
        
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(retries + 1):
            try:
                logger.debug(f"Making {method} request to {url} (attempt {attempt + 1})")
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                async with self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=headers
                ) as response:
                    response_data = await response.json()
                    
                    if response.status >= 400:
                        error_msg = response_data.get('error', f'HTTP {response.status}')
                        raise CipherOperationError(f"Request failed: {error_msg}")
                    
                    logger.debug(f"Request successful: {response.status}")
                    return response_data
                    
            except aiohttp.ClientError as e:
                if attempt < retries:
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {self._retry_delay}s: {e}")
                    await asyncio.sleep(self._retry_delay)
                    continue
                else:
                    logger.error(f"Request failed after {retries + 1} attempts: {e}")
                    raise CipherConnectionError(f"Request failed: {e}")
            
            except Exception as e:
                logger.error(f"Unexpected error during request: {e}")
                raise CipherOperationError(f"Request failed: {e}")
    
    # Memory Operations
    
    async def store_memory(
        self, 
        project_id: str,
        memory_content: str,
        memory_type: str = "general",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Store memory in Cipher for a specific project.
        
        Args:
            project_id: Project identifier
            memory_content: Memory content to store
            memory_type: Type of memory
            tags: Optional tags for the memory
            metadata: Optional metadata
            
        Returns:
            Dictionary containing memory ID and storage details
        """
        try:
            logger.info(f"Storing memory for project {project_id}")
            
            data = {
                "project_id": project_id,
                "content": memory_content,
                "memory_type": memory_type,
                "tags": tags or [],
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = await self._make_request("POST", "/memories", data)
            
            logger.info(f"Memory stored successfully: {response.get('memory_id')}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise CipherOperationError(f"Memory storage failed: {e}")
    
    async def retrieve_memory(
        self, 
        project_id: str,
        memory_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve memory from Cipher by ID.
        
        Args:
            project_id: Project identifier
            memory_id: Memory identifier
            
        Returns:
            Memory data or None if not found
        """
        try:
            logger.debug(f"Retrieving memory {memory_id} for project {project_id}")
            
            response = await self._make_request(
                "GET", 
                f"/memories/{memory_id}",
                {"project_id": project_id}
            )
            
            if response.get("found"):
                logger.debug(f"Memory retrieved successfully: {memory_id}")
                return response.get("memory")
            else:
                logger.debug(f"Memory not found: {memory_id}")
                return None
                
        except CipherOperationError as e:
            if "not found" in str(e).lower():
                return None
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve memory: {e}")
            raise CipherOperationError(f"Memory retrieval failed: {e}")
    
    async def search_memories(
        self,
        project_id: str,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Search memories in Cipher for a specific project.
        
        Args:
            project_id: Project identifier
            query: Search query
            filters: Optional search filters
            limit: Maximum number of results
            offset: Result offset for pagination
            
        Returns:
            Search results with memories and metadata
        """
        try:
            logger.info(f"Searching memories for project {project_id} with query: {query}")
            
            data = {
                "project_id": project_id,
                "query": query,
                "filters": filters or {},
                "limit": limit,
                "offset": offset
            }
            
            response = await self._make_request("POST", "/memories/search", data)
            
            logger.info(f"Memory search completed: {len(response.get('results', []))} results")
            return response
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            raise CipherOperationError(f"Memory search failed: {e}")
    
    async def update_memory(
        self,
        project_id: str,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update memory in Cipher.
        
        Args:
            project_id: Project identifier
            memory_id: Memory identifier
            updates: Updates to apply
            
        Returns:
            Updated memory data
        """
        try:
            logger.info(f"Updating memory {memory_id} for project {project_id}")
            
            data = {
                "project_id": project_id,
                "updates": updates,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = await self._make_request(
                "PUT", 
                f"/memories/{memory_id}",
                data
            )
            
            logger.info(f"Memory updated successfully: {memory_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
            raise CipherOperationError(f"Memory update failed: {e}")
    
    async def delete_memory(
        self,
        project_id: str,
        memory_id: str
    ) -> bool:
        """Delete memory from Cipher.
        
        Args:
            project_id: Project identifier
            memory_id: Memory identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            logger.info(f"Deleting memory {memory_id} for project {project_id}")
            
            await self._make_request(
                "DELETE", 
                f"/memories/{memory_id}",
                {"project_id": project_id}
            )
            
            logger.info(f"Memory deleted successfully: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            raise CipherOperationError(f"Memory deletion failed: {e}")
    
    # Project Operations
    
    async def create_project(
        self,
        project_id: str,
        project_name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new project in Cipher.
        
        Args:
            project_id: Project identifier
            project_name: Project name
            description: Optional project description
            metadata: Optional project metadata
            
        Returns:
            Project creation details
        """
        try:
            logger.info(f"Creating project {project_id}: {project_name}")
            
            data = {
                "project_id": project_id,
                "name": project_name,
                "description": description,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat()
            }
            
            response = await self._make_request("POST", "/projects", data)
            
            logger.info(f"Project created successfully: {project_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            raise CipherOperationError(f"Project creation failed: {e}")
    
    async def get_project(
        self,
        project_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get project information from Cipher.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Project data or None if not found
        """
        try:
            logger.debug(f"Getting project {project_id}")
            
            response = await self._make_request("GET", f"/projects/{project_id}")
            
            if response.get("found"):
                logger.debug(f"Project retrieved successfully: {project_id}")
                return response.get("project")
            else:
                logger.debug(f"Project not found: {project_id}")
                return None
                
        except CipherOperationError as e:
            if "not found" in str(e).lower():
                return None
            raise
        except Exception as e:
            logger.error(f"Failed to get project: {e}")
            raise CipherOperationError(f"Project retrieval failed: {e}")
    
    async def list_projects(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List all projects in Cipher.
        
        Args:
            limit: Maximum number of projects to return
            offset: Result offset for pagination
            
        Returns:
            List of projects with metadata
        """
        try:
            logger.debug("Listing projects")
            
            response = await self._make_request(
                "GET", 
                "/projects",
                {"limit": limit, "offset": offset}
            )
            
            logger.debug(f"Projects listed: {len(response.get('projects', []))} found")
            return response
            
        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            raise CipherOperationError(f"Project listing failed: {e}")
    
    # Health and Status
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Cipher API health.
        
        Returns:
            Health status information
        """
        try:
            response = await self._make_request("GET", "/health")
            return response
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise CipherOperationError(f"Health check failed: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Cipher MCP server statistics.
        
        Returns:
            Server statistics
        """
        try:
            response = await self._make_request("GET", "/api/stats")
            return response
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise CipherOperationError(f"Stats retrieval failed: {e}")


# Global client instance
_cipher_client: Optional[CipherAPIClient] = None


async def get_cipher_client() -> CipherAPIClient:
    """Get or create global Cipher API client instance."""
    global _cipher_client
    
    if _cipher_client is None:
        _cipher_client = CipherAPIClient()
        await _cipher_client.connect()
    
    return _cipher_client


async def close_cipher_client() -> None:
    """Close global Cipher MCP client instance."""
    global _cipher_client
    
    if _cipher_client:
        await _cipher_client.disconnect()
        _cipher_client = None


@asynccontextmanager
async def cipher_client_context():
    """Context manager for Cipher MCP client."""
    client = CipherMCPClient()
    try:
        await client.connect()
        yield client
    finally:
        await client.disconnect()

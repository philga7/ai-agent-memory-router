"""
MCP (Microservice Control Plane) server for AI Agent Memory Router.

This module provides MCP protocol implementation for integrating with
Cursor and other MCP clients, exposing memory routing tools and capabilities.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from app.core.config import get_settings
from app.core.logging import get_logger

# Get settings and logger
settings = get_settings()
logger = get_logger(__name__)


class MCPServer:
    """MCP server implementation for AI Agent Memory Router."""
    
    def __init__(self):
        """Initialize MCP server with tools and capabilities."""
        self.tools = {
            "memory_route": self.memory_route,
            "agent_register": self.agent_register,
            "memory_search": self.memory_search,
            "context_get": self.context_get,
            "memory_store": self.memory_store,
            "agent_status": self.agent_status,
            "routing_stats": self.routing_stats,
            # Cipher integration tools
            "cipher_store_memory": self.cipher_store_memory,
            "cipher_retrieve_memory": self.cipher_retrieve_memory,
            "cipher_search_memories": self.cipher_search_memories,
            "cipher_update_memory": self.cipher_update_memory,
            "cipher_delete_memory": self.cipher_delete_memory,
            "cipher_create_project": self.cipher_create_project,
            "cipher_get_project": self.cipher_get_project,
            "cipher_list_projects": self.cipher_list_projects,
            "cipher_health_check": self.cipher_health_check
        }
        
        self.server_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info("MCP server initialized with tools", extra={
            "tools": list(self.tools.keys())
        })
    
    async def start(self, host: str = None, port: int = None) -> None:
        """Start the MCP server."""
        host = host or settings.mcp_host
        port = port or settings.mcp_port
        
        try:
            logger.info(f"Starting MCP server on {host}:{port}")
            
            # Create server
            server = await asyncio.start_server(
                self.handle_client,
                host=host,
                port=port,
                reuse_address=True,
                reuse_port=True
            )
            
            self.is_running = True
            self.server_task = asyncio.create_task(self._run_server(server))
            
            logger.info("MCP server started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the MCP server."""
        try:
            logger.info("Stopping MCP server...")
            
            self.is_running = False
            
            if self.server_task:
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("MCP server stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping MCP server: {e}")
    
    async def _run_server(self, server) -> None:
        """Run the MCP server."""
        async with server:
            await server.serve_forever()
    
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle MCP client connection."""
        client_addr = writer.get_extra_info('peername')
        logger.info(f"New MCP client connected: {client_addr}")
        
        try:
            while self.is_running:
                # Read request
                data = await reader.read(4096)
                if not data:
                    break
                
                # Parse request
                try:
                    request = json.loads(data.decode('utf-8'))
                    response = await self.process_request(request)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from client {client_addr}: {e}")
                    response = self.create_error_response("Invalid JSON", -32700)
                except Exception as e:
                    logger.error(f"Error processing request from {client_addr}: {e}")
                    response = self.create_error_response(str(e), -32603)
                
                # Send response
                writer.write(json.dumps(response).encode('utf-8'))
                await writer.drain()
                
        except Exception as e:
            logger.error(f"Error handling client {client_addr}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            logger.info(f"MCP client disconnected: {client_addr}")
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process MCP request and return response."""
        try:
            # Validate request format
            if not self._is_valid_request(request):
                return self.create_error_response("Invalid request format", -32600)
            
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            logger.debug(f"Processing MCP request: {method}", extra={
                "method": method,
                "params": params,
                "request_id": request_id
            })
            
            # Route to appropriate tool
            if method in self.tools:
                try:
                    result = await self.tools[method](**params)
                    return self.create_success_response(result, request_id)
                except Exception as e:
                    logger.error(f"Tool {method} failed: {e}")
                    return self.create_error_response(str(e), -32603, request_id)
            else:
                return self.create_error_response(f"Unknown method: {method}", -32601, request_id)
                
        except Exception as e:
            logger.error(f"Error processing MCP request: {e}")
            return self.create_error_response(str(e), -32603)
    
    def _is_valid_request(self, request: Dict[str, Any]) -> bool:
        """Validate MCP request format."""
        required_fields = ["jsonrpc", "method", "id"]
        return all(field in request for field in required_fields)
    
    def create_success_response(self, result: Any, request_id: Any) -> Dict[str, Any]:
        """Create successful MCP response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }
    
    def create_error_response(self, message: str, code: int, request_id: Any = None) -> Dict[str, Any]:
        """Create error MCP response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
    
    # MCP Tool Implementations
    
    async def memory_route(
        self,
        source_agent_id: str,
        target_agent_ids: List[str],
        memory_content: str,
        priority: str = "normal",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Route memory between AI agents via MCP."""
        try:
            logger.info(f"Routing memory from {source_agent_id} to {target_agent_ids}")
            
            # This would integrate with the actual memory routing service
            # For now, return a mock response
            result = {
                "route_id": f"route_{asyncio.get_event_loop().time()}",
                "source_agent_id": source_agent_id,
                "target_agent_ids": target_agent_ids,
                "status": "routed",
                "timestamp": asyncio.get_event_loop().time(),
                "priority": priority
            }
            
            if context:
                result["context"] = context
            
            logger.info(f"Memory routing completed: {result['route_id']}")
            return result
            
        except Exception as e:
            logger.error(f"MCP memory_route failed: {e}")
            raise
    
    async def agent_register(
        self,
        agent_id: str,
        agent_name: str,
        capabilities: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Register new AI agent via MCP."""
        try:
            logger.info(f"Registering agent: {agent_id} ({agent_name})")
            
            # This would integrate with the actual agent management service
            result = {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "capabilities": capabilities,
                "status": "registered",
                "timestamp": asyncio.get_event_loop().time()
            }
            
            if metadata:
                result["metadata"] = metadata
            
            logger.info(f"Agent registration completed: {agent_id}")
            return result
            
        except Exception as e:
            logger.error(f"MCP agent_register failed: {e}")
            raise
    
    async def memory_search(
        self,
        query: str,
        agent_id: Optional[str] = None,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search agent memories via MCP."""
        try:
            logger.info(f"Searching memories with query: {query}")
            
            # This would integrate with the actual memory search service
            # For now, return a mock response
            result = {
                "query": query,
                "results": [
                    {
                        "memory_id": f"mem_{i}",
                        "content": f"Sample memory content {i}",
                        "agent_id": agent_id or "unknown",
                        "relevance_score": 0.9 - (i * 0.1),
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    for i in range(min(limit, 5))
                ],
                "total_results": 5,
                "search_time_ms": 150.0
            }
            
            if filters:
                result["filters"] = filters
            
            logger.info(f"Memory search completed: {len(result['results'])} results")
            return result
            
        except Exception as e:
            logger.error(f"MCP memory_search failed: {e}")
            raise
    
    async def context_get(
        self,
        conversation_id: str,
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get conversation context via MCP."""
        try:
            logger.info(f"Getting context for conversation: {conversation_id}")
            
            # This would integrate with the actual context management service
            result = {
                "conversation_id": conversation_id,
                "context": {
                    "participants": [agent_id] if agent_id else [],
                    "topics": ["AI agents", "Memory routing"],
                    "last_updated": asyncio.get_event_loop().time(),
                    "context_size": 1024
                },
                "timestamp": asyncio.get_event_loop().time()
            }
            
            logger.info(f"Context retrieval completed: {conversation_id}")
            return result
            
        except Exception as e:
            logger.error(f"MCP context_get failed: {e}")
            raise
    
    async def memory_store(
        self,
        agent_id: str,
        memory_content: str,
        memory_type: str = "general",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Store agent memory via MCP."""
        try:
            logger.info(f"Storing memory for agent: {agent_id}")
            
            # This would integrate with the actual memory storage service
            result = {
                "memory_id": f"mem_{asyncio.get_event_loop().time()}",
                "agent_id": agent_id,
                "memory_type": memory_type,
                "status": "stored",
                "timestamp": asyncio.get_event_loop().time()
            }
            
            if tags:
                result["tags"] = tags
            if metadata:
                result["metadata"] = metadata
            
            logger.info(f"Memory storage completed: {result['memory_id']}")
            return result
            
        except Exception as e:
            logger.error(f"MCP memory_store failed: {e}")
            raise
    
    async def agent_status(
        self,
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get agent status via MCP."""
        try:
            logger.info(f"Getting agent status for: {agent_id or 'all'}")
            
            # This would integrate with the actual agent management service
            if agent_id:
                result = {
                    "agent_id": agent_id,
                    "status": "active",
                    "last_heartbeat": asyncio.get_event_loop().time(),
                    "capabilities": ["memory_routing", "context_management"],
                    "memory_count": 150
                }
            else:
                result = {
                    "total_agents": 5,
                    "active_agents": 4,
                    "inactive_agents": 1,
                    "agents": [
                        {
                            "agent_id": f"agent_{i}",
                            "status": "active" if i < 4 else "inactive",
                            "last_heartbeat": asyncio.get_event_loop().time() - (i * 10)
                        }
                        for i in range(5)
                    ]
                }
            
            logger.info(f"Agent status retrieval completed")
            return result
            
        except Exception as e:
            logger.error(f"MCP agent_status failed: {e}")
            raise
    
    async def routing_stats(
        self,
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """Get routing statistics via MCP."""
        try:
            logger.info(f"Getting routing stats for: {time_range}")
            
            # This would integrate with the actual routing service
            result = {
                "time_range": time_range,
                "total_routes": 1250,
                "successful_routes": 1180,
                "failed_routes": 70,
                "success_rate": 0.944,
                "average_routing_time_ms": 45.2,
                "top_source_agents": [
                    {"agent_id": "agent_1", "routes": 150},
                    {"agent_id": "agent_2", "routes": 120},
                    {"agent_id": "agent_3", "routes": 95}
                ],
                "timestamp": asyncio.get_event_loop().time()
            }
            
            logger.info(f"Routing stats retrieval completed")
            return result
            
        except Exception as e:
            logger.error(f"MCP routing_stats failed: {e}")
            raise
    
    # Cipher Integration MCP Tools
    
    async def cipher_store_memory(
        self,
        project_id: str,
        agent_id: str,
        memory_content: str,
        memory_type: str = "general",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: int = 1
    ) -> Dict[str, Any]:
        """Store memory in Cipher via MCP."""
        try:
            logger.info(f"Storing memory in Cipher for project {project_id}, agent {agent_id}")
            
            # Import here to avoid circular imports
            from app.services.cipher_service import get_cipher_service
            
            cipher_service = await get_cipher_service()
            
            memory_id = await cipher_service.store_memory(
                project_id=project_id,
                agent_id=agent_id,
                memory_content=memory_content,
                memory_type=memory_type,
                tags=tags,
                metadata=metadata,
                priority=priority
            )
            
            result = {
                "memory_id": memory_id,
                "project_id": project_id,
                "agent_id": agent_id,
                "memory_type": memory_type,
                "status": "stored",
                "timestamp": asyncio.get_event_loop().time()
            }
            
            if tags:
                result["tags"] = tags
            if metadata:
                result["metadata"] = metadata
            
            logger.info(f"Cipher memory storage completed: {memory_id}")
            return result
            
        except Exception as e:
            logger.error(f"MCP cipher_store_memory failed: {e}")
            raise
    
    async def cipher_retrieve_memory(
        self,
        project_id: str,
        memory_id: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Retrieve memory from Cipher via MCP."""
        try:
            logger.info(f"Retrieving memory {memory_id} from Cipher for project {project_id}")
            
            from app.services.cipher_service import get_cipher_service
            
            cipher_service = await get_cipher_service()
            
            memory_data = await cipher_service.retrieve_memory(
                project_id=project_id,
                memory_id=memory_id,
                use_cache=use_cache
            )
            
            if memory_data:
                result = {
                    "memory_id": memory_id,
                    "project_id": project_id,
                    "memory_data": memory_data,
                    "status": "retrieved",
                    "from_cache": memory_data.get("from_cache", False),
                    "timestamp": asyncio.get_event_loop().time()
                }
            else:
                result = {
                    "memory_id": memory_id,
                    "project_id": project_id,
                    "status": "not_found",
                    "timestamp": asyncio.get_event_loop().time()
                }
            
            logger.info(f"Cipher memory retrieval completed: {memory_id}")
            return result
            
        except Exception as e:
            logger.error(f"MCP cipher_retrieve_memory failed: {e}")
            raise
    
    async def cipher_search_memories(
        self,
        project_id: str,
        query: str,
        agent_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Search memories in Cipher via MCP."""
        try:
            logger.info(f"Searching memories in Cipher for project {project_id} with query: {query}")
            
            from app.services.cipher_service import get_cipher_service
            
            cipher_service = await get_cipher_service()
            
            search_results = await cipher_service.search_memories(
                project_id=project_id,
                query=query,
                agent_id=agent_id,
                memory_type=memory_type,
                tags=tags,
                limit=limit,
                offset=offset
            )
            
            result = {
                "project_id": project_id,
                "query": query,
                "results": search_results.get("results", []),
                "total": search_results.get("total", 0),
                "from_cache": search_results.get("from_cache", False),
                "timestamp": asyncio.get_event_loop().time()
            }
            
            if agent_id:
                result["agent_id"] = agent_id
            if memory_type:
                result["memory_type"] = memory_type
            if tags:
                result["tags"] = tags
            
            logger.info(f"Cipher memory search completed: {len(result['results'])} results")
            return result
            
        except Exception as e:
            logger.error(f"MCP cipher_search_memories failed: {e}")
            raise
    
    async def cipher_update_memory(
        self,
        project_id: str,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update memory in Cipher via MCP."""
        try:
            logger.info(f"Updating memory {memory_id} in Cipher for project {project_id}")
            
            from app.services.cipher_service import get_cipher_service
            
            cipher_service = await get_cipher_service()
            
            success = await cipher_service.update_memory(
                project_id=project_id,
                memory_id=memory_id,
                updates=updates
            )
            
            result = {
                "memory_id": memory_id,
                "project_id": project_id,
                "updates": updates,
                "status": "updated" if success else "failed",
                "timestamp": asyncio.get_event_loop().time()
            }
            
            logger.info(f"Cipher memory update completed: {memory_id}")
            return result
            
        except Exception as e:
            logger.error(f"MCP cipher_update_memory failed: {e}")
            raise
    
    async def cipher_delete_memory(
        self,
        project_id: str,
        memory_id: str
    ) -> Dict[str, Any]:
        """Delete memory from Cipher via MCP."""
        try:
            logger.info(f"Deleting memory {memory_id} from Cipher for project {project_id}")
            
            from app.services.cipher_service import get_cipher_service
            
            cipher_service = await get_cipher_service()
            
            success = await cipher_service.delete_memory(
                project_id=project_id,
                memory_id=memory_id
            )
            
            result = {
                "memory_id": memory_id,
                "project_id": project_id,
                "status": "deleted" if success else "failed",
                "timestamp": asyncio.get_event_loop().time()
            }
            
            logger.info(f"Cipher memory deletion completed: {memory_id}")
            return result
            
        except Exception as e:
            logger.error(f"MCP cipher_delete_memory failed: {e}")
            raise
    
    async def cipher_create_project(
        self,
        project_id: str,
        project_name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create project in Cipher via MCP."""
        try:
            logger.info(f"Creating project {project_id}: {project_name} in Cipher")
            
            from app.services.cipher_service import get_cipher_service
            
            cipher_service = await get_cipher_service()
            
            success = await cipher_service.create_project(
                project_id=project_id,
                project_name=project_name,
                description=description,
                metadata=metadata
            )
            
            result = {
                "project_id": project_id,
                "project_name": project_name,
                "status": "created" if success else "failed",
                "timestamp": asyncio.get_event_loop().time()
            }
            
            if description:
                result["description"] = description
            if metadata:
                result["metadata"] = metadata
            
            logger.info(f"Cipher project creation completed: {project_id}")
            return result
            
        except Exception as e:
            logger.error(f"MCP cipher_create_project failed: {e}")
            raise
    
    async def cipher_get_project(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """Get project information from Cipher via MCP."""
        try:
            logger.info(f"Getting project {project_id} from Cipher")
            
            from app.services.cipher_service import get_cipher_service
            
            cipher_service = await get_cipher_service()
            
            project_info = await cipher_service.get_project_info(project_id)
            
            if project_info:
                result = {
                    "project_id": project_id,
                    "project_info": project_info,
                    "status": "found",
                    "timestamp": asyncio.get_event_loop().time()
                }
            else:
                result = {
                    "project_id": project_id,
                    "status": "not_found",
                    "timestamp": asyncio.get_event_loop().time()
                }
            
            logger.info(f"Cipher project retrieval completed: {project_id}")
            return result
            
        except Exception as e:
            logger.error(f"MCP cipher_get_project failed: {e}")
            raise
    
    async def cipher_list_projects(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List projects in Cipher via MCP."""
        try:
            logger.info("Listing projects in Cipher")
            
            from app.core.cipher_client import get_cipher_client
            
            cipher_client = await get_cipher_client()
            
            projects_data = await cipher_client.list_projects(limit=limit, offset=offset)
            
            result = {
                "projects": projects_data.get("projects", []),
                "total": projects_data.get("total", 0),
                "limit": limit,
                "offset": offset,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            logger.info(f"Cipher project listing completed: {len(result['projects'])} projects")
            return result
            
        except Exception as e:
            logger.error(f"MCP cipher_list_projects failed: {e}")
            raise
    
    async def cipher_health_check(self) -> Dict[str, Any]:
        """Check Cipher MCP server health via MCP."""
        try:
            logger.info("Checking Cipher MCP server health")
            
            from app.core.cipher_client import get_cipher_client
            
            cipher_client = await get_cipher_client()
            
            health_data = await cipher_client.health_check()
            
            result = {
                "status": "healthy" if health_data.get("status") == "ok" else "unhealthy",
                "health_data": health_data,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            logger.info(f"Cipher health check completed: {result['status']}")
            return result
            
        except Exception as e:
            logger.error(f"MCP cipher_health_check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }


# Global MCP server instance
_mcp_server: Optional[MCPServer] = None


async def start_mcp_server() -> None:
    """Start the MCP server."""
    global _mcp_server
    
    try:
        _mcp_server = MCPServer()
        await _mcp_server.start()
        
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        raise


async def stop_mcp_server() -> None:
    """Stop the MCP server."""
    global _mcp_server
    
    try:
        if _mcp_server:
            await _mcp_server.stop()
            _mcp_server = None
            
    except Exception as e:
        logger.error(f"Failed to stop MCP server: {e}")


def get_mcp_server() -> Optional[MCPServer]:
    """Get the MCP server instance."""
    return _mcp_server


async def health_check() -> bool:
    """Check MCP server health."""
    return _mcp_server is not None and _mcp_server.is_running

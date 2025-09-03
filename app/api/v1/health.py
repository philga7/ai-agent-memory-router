"""
Health monitoring endpoints for AI Agent Memory Router.

This module provides REST API endpoints for health checks, system status,
and monitoring information.
"""

import time
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.database import check_database_health, get_database_stats
from app.core.mcp_server import health_check as mcp_health_check
from app.core.metrics import record_request, record_request_duration
from app.core.logging import get_logger

# Setup logger
logger = get_logger(__name__)

# Create router
router = APIRouter()


# Response Models

class HealthStatus(BaseModel):
    """Model for health status."""
    
    status: str = Field(..., description="Overall health status")
    timestamp: float = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Application version")
    uptime: float = Field(..., description="Application uptime in seconds")


class ComponentHealth(BaseModel):
    """Model for component health."""
    
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Component status")
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    last_check: float = Field(..., description="Last health check timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")


class DetailedHealthResponse(BaseModel):
    """Response model for detailed health check."""
    
    overall_status: str = Field(..., description="Overall system health status")
    timestamp: float = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Application version")
    uptime: float = Field(..., description="Application uptime in seconds")
    components: List[ComponentHealth] = Field(..., description="Component health status")
    system_info: Dict[str, Any] = Field(..., description="System information")


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""
    
    timestamp: float = Field(..., description="Metrics timestamp")
    metrics: Dict[str, Any] = Field(..., description="System metrics")
    performance: Dict[str, Any] = Field(..., description="Performance metrics")


# Global variables for tracking
_start_time = time.time()


# Endpoints

@router.get(
    "/",
    response_model=HealthStatus,
    summary="Basic health check",
    description="Simple health check endpoint for load balancers and basic monitoring"
)
async def health_check() -> HealthStatus:
    """Basic health check endpoint."""
    
    start_time = time.time()
    
    try:
        logger.debug("Basic health check requested")
        
        uptime = time.time() - _start_time
        
        result = HealthStatus(
            status="healthy",
            timestamp=time.time(),
            version="0.1.0",
            uptime=uptime
        )
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration(duration, "GET", "/health/", 200)
        record_request("GET", "/health/", 200)
        
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration(duration, "GET", "/health/", 500)
        record_request("GET", "/health/", 200)
        
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get(
    "/detailed",
    response_model=DetailedHealthResponse,
    summary="Detailed health check",
    description="Comprehensive health check including all system components"
)
async def detailed_health_check() -> DetailedHealthResponse:
    """Detailed health check endpoint."""
    
    start_time = time.time()
    
    try:
        logger.info("Detailed health check requested")
        
        uptime = time.time() - _start_time
        components = []
        
        # Check database health
        db_start = time.time()
        db_healthy = await check_database_health()
        db_response_time = (time.time() - db_start) * 1000
        
        components.append(ComponentHealth(
            name="database",
            status="healthy" if db_healthy else "unhealthy",
            response_time_ms=db_response_time,
            last_check=time.time(),
            details={"type": "postgresql", "status": "connected" if db_healthy else "disconnected"}
        ))
        
        # Check MCP server health
        mcp_start = time.time()
        mcp_healthy = await mcp_health_check()
        mcp_response_time = (time.time() - mcp_start) * 1000
        
        components.append(ComponentHealth(
            name="mcp_server",
            status="healthy" if mcp_healthy else "unhealthy",
            response_time_ms=mcp_response_time,
            last_check=time.time(),
            details={"type": "mcp", "status": "running" if mcp_healthy else "stopped"}
        ))
        
        # Check Redis health (mock for now)
        redis_start = time.time()
        redis_healthy = True  # Mock implementation
        redis_response_time = (time.time() - redis_start) * 1000
        
        components.append(ComponentHealth(
            name="redis",
            status="healthy" if redis_healthy else "unhealthy",
            response_time_ms=redis_response_time,
            last_check=time.time(),
            details={"type": "redis", "status": "connected" if redis_healthy else "disconnected"}
        ))
        
        # Check Chroma health (mock for now)
        chroma_start = time.time()
        chroma_healthy = True  # Mock implementation
        chroma_response_time = (time.time() - chroma_start) * 1000
        
        components.append(ComponentHealth(
            name="chroma",
            status="healthy" if chroma_healthy else "unhealthy",
            response_time_ms=chroma_response_time,
            last_check=time.time(),
            details={"type": "vector_database", "status": "connected" if chroma_healthy else "disconnected"}
        ))
        
        # Determine overall status
        all_healthy = all(c.status == "healthy" for c in components)
        overall_status = "healthy" if all_healthy else "degraded"
        
        # System information
        system_info = {
            "python_version": "3.11.0",
            "platform": "linux",
            "memory_usage": "128MB",
            "cpu_usage": "5%",
            "active_connections": 12
        }
        
        result = DetailedHealthResponse(
            overall_status=overall_status,
            timestamp=time.time(),
            version="0.1.0",
            uptime=uptime,
            components=components,
            system_info=system_info
        )
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration(duration, "GET", "/health/detailed", 200)
        record_request("GET", "/health/detailed", 200)
        
        logger.info(f"Detailed health check completed: {overall_status}")
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration(duration, "GET", "/health/detailed", 500)
        record_request("GET", "/health/detailed", 500)
        
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail="Detailed health check failed")


@router.get(
    "/database",
    summary="Database health check",
    description="Detailed health check for database component"
)
async def database_health_check():
    """Database health check endpoint."""
    
    start_time = time.time()
    
    try:
        logger.info("Database health check requested")
        
        # Check database health
        db_healthy = await check_database_health()
        db_stats = await get_database_stats()
        
        result = {
            "component": "database",
            "status": "healthy" if db_healthy else "unhealthy",
            "timestamp": time.time(),
            "response_time_ms": (time.time() - start_time) * 1000,
            "details": db_stats
        }
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration(duration, "GET", "/health/database", 200)
        record_request("GET", "/health/database", 200)
        
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration(duration, "GET", "/health/database", 500)
        record_request("GET", "/health/database", 500)
        
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(status_code=500, detail="Database health check failed")


@router.get(
    "/mcp",
    summary="MCP server health check",
    description="Health check for MCP server component"
)
async def mcp_health_check():
    """MCP server health check endpoint."""
    
    start_time = time.time()
    
    try:
        logger.info("MCP server health check requested")
        
        # Check MCP server health
        mcp_healthy = await mcp_health_check()
        
        result = {
            "component": "mcp_server",
            "status": "healthy" if mcp_healthy else "unhealthy",
            "timestamp": time.time(),
            "response_time_ms": (time.time() - start_time) * 1000,
            "details": {
                "type": "mcp",
                "status": "running" if mcp_healthy else "stopped",
                "tools_available": 7 if mcp_healthy else 0
            }
        }
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration(duration, "GET", "/health/mcp", 200)
        record_request("GET", "/health/mcp", 200)
        
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration(duration, "GET", "/health/mcp", 500)
        record_request("GET", "/health/mcp", 500)
        
        logger.error(f"MCP server health check failed: {e}")
        raise HTTPException(status_code=500, detail="MCP server health check failed")


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="System metrics",
    description="Retrieve system metrics and performance information"
)
async def get_system_metrics() -> MetricsResponse:
    """Get system metrics endpoint."""
    
    start_time = time.time()
    
    try:
        logger.info("System metrics requested")
        
        # Mock metrics for now
        metrics = {
            "memory_routes_total": 1250,
            "agents_total": 25,
            "memories_stored_total": 8900,
            "context_updates_total": 3450,
            "requests_total": 15600,
            "errors_total": 45
        }
        
        performance = {
            "average_response_time_ms": 45.2,
            "requests_per_second": 12.5,
            "memory_usage_mb": 128.5,
            "cpu_usage_percent": 5.2,
            "active_connections": 12,
            "database_connections": 8
        }
        
        result = MetricsResponse(
            timestamp=time.time(),
            metrics=metrics,
            performance=performance
        )
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration(duration, "GET", "/health/metrics", 200)
        record_request("GET", "/health/metrics", 200)
        
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration(duration, "GET", "/health/metrics", 500)
        record_request("GET", "/health/metrics", 500)
        
        logger.error(f"System metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="System metrics retrieval failed")


@router.get(
    "/ready",
    summary="Readiness check",
    description="Check if the application is ready to receive traffic"
)
async def readiness_check():
    """Readiness check endpoint."""
    
    start_time = time.time()
    
    try:
        logger.debug("Readiness check requested")
        
        # Check critical components
        db_ready = await check_database_health()
        mcp_ready = await mcp_health_check()
        
        # Application is ready if critical components are healthy
        ready = db_ready and mcp_ready
        
        result = {
            "ready": ready,
            "timestamp": time.time(),
            "response_time_ms": (time.time() - start_time) * 1000,
            "components": {
                "database": db_ready,
                "mcp_server": mcp_ready
            }
        }
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration(duration, "GET", "/health/ready", 200)
        record_request("GET", "/health/ready", 200)
        
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration(duration, "GET", "/health/ready", 500)
        record_request("GET", "/health/ready", 500)
        
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=500, detail="Readiness check failed")


@router.get(
    "/live",
    summary="Liveness check",
    description="Check if the application is alive and running"
)
async def liveness_check():
    """Liveness check endpoint."""
    
    start_time = time.time()
    
    try:
        logger.debug("Liveness check requested")
        
        # Simple liveness check - just verify the application is responding
        result = {
            "alive": True,
            "timestamp": time.time(),
            "response_time_ms": (time.time() - start_time) * 1000,
            "uptime": time.time() - _start_time
        }
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration(duration, "GET", "/health/live", 200)
        record_request("GET", "/health/live", 200)
        
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration(duration, "GET", "/health/live", 500)
        record_request("GET", "/health/live", 500)
        
        logger.error(f"Liveness check failed: {e}")
        raise HTTPException(status_code=500, detail="Liveness check failed")


@router.get(
    "/status",
    summary="System status",
    description="Get comprehensive system status information"
)
async def get_system_status():
    """Get system status endpoint."""
    
    start_time = time.time()
    
    try:
        logger.info("System status requested")
        
        # Get component statuses
        db_healthy = await check_database_health()
        mcp_healthy = await mcp_health_check()
        
        # Mock other component statuses
        redis_healthy = True
        chroma_healthy = True
        
        # Calculate overall status
        components = {
            "database": {"status": "healthy" if db_healthy else "unhealthy", "priority": "critical"},
            "mcp_server": {"status": "healthy" if mcp_healthy else "unhealthy", "priority": "high"},
            "redis": {"status": "healthy" if redis_healthy else "unhealthy", "priority": "medium"},
            "chroma": {"status": "healthy" if chroma_healthy else "unhealthy", "priority": "medium"}
        }
        
        critical_components = [c for c, info in components.items() if info["priority"] == "critical"]
        critical_healthy = all(components[c]["status"] == "healthy" for c in critical_components)
        
        if critical_healthy:
            overall_status = "operational"
        elif any(components[c]["status"] == "healthy" for c in critical_components):
            overall_status = "degraded"
        else:
            overall_status = "down"
        
        result = {
            "status": overall_status,
            "timestamp": time.time(),
            "version": "0.1.0",
            "uptime": time.time() - _start_time,
            "components": components,
            "summary": {
                "total_components": len(components),
                "healthy_components": len([c for c in components.values() if c["status"] == "healthy"]),
                "unhealthy_components": len([c for c in components.values() if c["status"] == "unhealthy"]),
                "critical_components_healthy": critical_healthy
            }
        }
        
        # Record metrics
        duration = time.time() - start_time
        record_request_duration(duration, "GET", "/health/status", 200)
        record_request("GET", "/health/status", 200)
        
        return result
        
    except Exception as e:
        # Record error metrics
        duration = time.time() - start_time
        record_request_duration(duration, "GET", "/health/status", 500)
        record_request("GET", "/health/status", 500)
        
        logger.error(f"System status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="System status retrieval failed")

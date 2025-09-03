"""
AI Agent Memory Router - Main Application

A sophisticated memory routing system designed to facilitate intelligent communication
and knowledge sharing between AI agents through centralized memory management and
intelligent routing capabilities.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import get_settings
from app.core.database import init_database, close_database
from app.core.logging import setup_logging
from app.api.v1.router import api_router
from app.core.mcp_server import start_mcp_server, stop_mcp_server
from app.core.metrics import setup_metrics

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting AI Agent Memory Router...")
    
    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized successfully")
        
        # Setup metrics
        setup_metrics()
        logger.info("Metrics setup completed")
        
        # Start MCP server in background
        mcp_task = asyncio.create_task(start_mcp_server())
        logger.info("MCP server started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down AI Agent Memory Router...")
        
        try:
            # Stop MCP server
            await stop_mcp_server()
            logger.info("MCP server stopped successfully")
            
            # Close database connections
            await close_database()
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="AI Agent Memory Router",
    description=(
        "A sophisticated memory routing system designed to facilitate intelligent "
        "communication and knowledge sharing between AI agents through centralized "
        "memory management and intelligent routing capabilities."
    ),
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts
)

# Include API router
app.include_router(api_router, prefix=settings.api_prefix)


# Global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Validation error",
            "details": exc.errors(),
            "timestamp": asyncio.get_event_loop().time()
        }
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": asyncio.get_event_loop().time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": asyncio.get_event_loop().time()
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "AI Agent Memory Router",
        "version": "0.1.0",
        "timestamp": asyncio.get_event_loop().time()
    }


# Root endpoint
@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with service information."""
    return {
        "service": "AI Agent Memory Router",
        "description": (
            "A sophisticated memory routing system for AI agents with "
            "intelligent communication and knowledge sharing capabilities."
        ),
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "api": f"{settings.api_prefix}/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )

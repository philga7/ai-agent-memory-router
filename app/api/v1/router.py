"""
Main API router for AI Agent Memory Router.

This module provides the main API router that includes all endpoint
modules and handles API versioning and documentation.
"""

from fastapi import APIRouter

from app.api.v1 import memory, agents, context, health, universal

# Create main API router
api_router = APIRouter()

# Include all endpoint modules
api_router.include_router(
    memory.router,
    prefix="/memory",
    tags=["Memory Management"]
)

api_router.include_router(
    agents.router,
    prefix="/agents",
    tags=["Agent Management"]
)

api_router.include_router(
    context.router,
    prefix="/context",
    tags=["Context Management"]
)

api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health & Monitoring"]
)

api_router.include_router(
    universal.router,
    prefix="/universal",
    tags=["Universal Memory Access"]
)

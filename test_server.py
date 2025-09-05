#!/usr/bin/env python3
"""
Simple test server for Universal Memory Access API.

This server runs without database dependencies to test the API endpoints.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import our Universal API
from app.api.v1.universal import router as universal_router

# Create FastAPI app
app = FastAPI(
    title="Universal Memory Access API - Test Server",
    description="Test server for Universal Memory Access API without database dependencies",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the universal router
app.include_router(universal_router, prefix="/api/v1/universal")

# Add a simple health check
@app.get("/")
async def root():
    return {
        "message": "Universal Memory Access API Test Server",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "status": "/api/v1/universal/status",
            "store_memory": "POST /api/v1/universal/memories",
            "retrieve_memories": "GET /api/v1/universal/memories",
            "search_memories": "POST /api/v1/universal/memories/search"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Test server is running"}

if __name__ == "__main__":
    print("🚀 Starting Universal Memory Access API Test Server")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔧 Test endpoints without database dependencies")
    print("-" * 50)
    
    uvicorn.run(
        "test_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

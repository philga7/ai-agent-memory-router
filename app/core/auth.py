"""
Project authentication and validation for Universal Memory Access API.

This module provides authentication, authorization, and validation services
for projects using the universal memory access API.
"""

import hashlib
import hmac
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import jwt
from passlib.context import CryptContext

from app.core.logging import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

# Security configuration
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
JWT_SECRET_KEY = settings.security.secret_key or "your-secret-key-change-in-production"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# API Key Configuration
API_KEY_HEADER = "X-API-Key"
API_KEY_PREFIX = "uma_"  # Universal Memory Access


class ProjectCredentials(BaseModel):
    """Project credentials model."""
    
    project_id: str = Field(..., description="Project identifier")
    api_key: str = Field(..., description="API key")
    secret_key: Optional[str] = Field(None, description="Secret key for HMAC signing")
    permissions: List[str] = Field(default_factory=list, description="Project permissions")
    rate_limits: Dict[str, int] = Field(default_factory=dict, description="Rate limits")
    quotas: Dict[str, int] = Field(default_factory=dict, description="Project quotas")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    is_active: bool = Field(default=True, description="Whether credentials are active")


class ProjectAuthRequest(BaseModel):
    """Project authentication request."""
    
    project_id: str = Field(..., description="Project identifier")
    api_key: str = Field(..., description="API key")
    timestamp: Optional[int] = Field(None, description="Request timestamp for HMAC validation")
    signature: Optional[str] = Field(None, description="HMAC signature")


class ProjectAuthResponse(BaseModel):
    """Project authentication response."""
    
    project_id: str = Field(..., description="Project identifier")
    authenticated: bool = Field(..., description="Authentication status")
    permissions: List[str] = Field(..., description="Project permissions")
    rate_limits: Dict[str, int] = Field(..., description="Rate limits")
    quotas: Dict[str, int] = Field(..., description="Project quotas")
    expires_at: Optional[datetime] = Field(None, description="Token expiration")
    token: Optional[str] = Field(None, description="JWT token")


class ProjectPermission(BaseModel):
    """Project permission model."""
    
    permission: str = Field(..., description="Permission name")
    resource: str = Field(..., description="Resource type")
    action: str = Field(..., description="Action allowed")
    conditions: Optional[Dict[str, Any]] = Field(None, description="Permission conditions")


class ProjectValidator:
    """Project validation service."""
    
    def __init__(self):
        self.valid_projects: Dict[str, ProjectCredentials] = {}
        self._load_default_projects()
    
    def _load_default_projects(self):
        """Load default project configurations."""
        # Default project for testing
        default_project = ProjectCredentials(
            project_id="demo_project",
            api_key="uma_demo_1234567890abcdef",
            secret_key="demo_secret_key_change_in_production",
            permissions=["memory:read", "memory:write", "memory:search", "project:read"],
            rate_limits={
                "requests_per_minute": 100,
                "requests_per_hour": 1000,
                "requests_per_day": 10000
            },
            quotas={
                "max_memories": 10000,
                "max_storage_bytes": 100 * 1024 * 1024  # 100MB
            }
        )
        self.valid_projects[default_project.project_id] = default_project
        
        logger.info(f"Loaded {len(self.valid_projects)} default projects")
    
    def validate_project_id(self, project_id: str) -> bool:
        """Validate project ID format."""
        if not project_id:
            return False
        
        # Project ID should be alphanumeric with hyphens and underscores
        if not project_id.replace('_', '').replace('-', '').isalnum():
            return False
        
        # Length constraints
        if len(project_id) < 3 or len(project_id) > 100:
            return False
        
        return True
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key format."""
        if not api_key:
            return False
        
        # API key should start with prefix and be long enough
        if not api_key.startswith(API_KEY_PREFIX):
            return False
        
        if len(api_key) < 20:
            return False
        
        return True
    
    def authenticate_project(self, project_id: str, api_key: str) -> Optional[ProjectCredentials]:
        """Authenticate project using API key."""
        if not self.validate_project_id(project_id) or not self.validate_api_key(api_key):
            return None
        
        credentials = self.valid_projects.get(project_id)
        if not credentials:
            return None
        
        if not credentials.is_active:
            return None
        
        if credentials.api_key != api_key:
            return None
        
        # Check expiration
        if credentials.expires_at and credentials.expires_at < datetime.utcnow():
            return None
        
        return credentials
    
    def validate_hmac_signature(self, project_id: str, api_key: str, timestamp: int, signature: str) -> bool:
        """Validate HMAC signature for enhanced security."""
        credentials = self.authenticate_project(project_id, api_key)
        if not credentials or not credentials.secret_key:
            return False
        
        # Check timestamp (within 5 minutes)
        current_time = int(time.time())
        if abs(current_time - timestamp) > 300:  # 5 minutes
            return False
        
        # Generate expected signature
        message = f"{project_id}:{api_key}:{timestamp}"
        expected_signature = hmac.new(
            credentials.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    def check_permission(self, project_id: str, permission: str) -> bool:
        """Check if project has specific permission."""
        credentials = self.valid_projects.get(project_id)
        if not credentials or not credentials.is_active:
            return False
        
        return permission in credentials.permissions
    
    def get_project_credentials(self, project_id: str) -> Optional[ProjectCredentials]:
        """Get project credentials."""
        return self.valid_projects.get(project_id)
    
    def create_project_credentials(self, project_id: str, permissions: List[str] = None) -> ProjectCredentials:
        """Create new project credentials."""
        if not self.validate_project_id(project_id):
            raise ValueError("Invalid project ID format")
        
        if project_id in self.valid_projects:
            raise ValueError("Project already exists")
        
        # Generate API key
        api_key = f"{API_KEY_PREFIX}{project_id}_{hashlib.sha256(project_id.encode()).hexdigest()[:16]}"
        
        # Generate secret key
        secret_key = hashlib.sha256(f"{project_id}_{time.time()}".encode()).hexdigest()
        
        credentials = ProjectCredentials(
            project_id=project_id,
            api_key=api_key,
            secret_key=secret_key,
            permissions=permissions or ["memory:read", "memory:write", "memory:search"],
            rate_limits={
                "requests_per_minute": 60,
                "requests_per_hour": 1000,
                "requests_per_day": 10000
            },
            quotas={
                "max_memories": 5000,
                "max_storage_bytes": 50 * 1024 * 1024  # 50MB
            }
        )
        
        self.valid_projects[project_id] = credentials
        logger.info(f"Created credentials for project: {project_id}")
        
        return credentials


class JWTManager:
    """JWT token management."""
    
    @staticmethod
    def create_token(project_id: str, permissions: List[str], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT token for project."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
        
        payload = {
            "project_id": project_id,
            "permissions": permissions,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "project_access"
        }
        
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return token
    
    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None
    
    @staticmethod
    def get_token_expiration(token: str) -> Optional[datetime]:
        """Get token expiration time."""
        payload = JWTManager.verify_token(token)
        if payload and "exp" in payload:
            return datetime.fromtimestamp(payload["exp"])
        return None


class RateLimiter:
    """Rate limiting service."""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = {}
    
    def is_rate_limited(self, project_id: str, rate_limits: Dict[str, int]) -> bool:
        """Check if project is rate limited."""
        current_time = time.time()
        
        # Clean old requests
        if project_id in self.requests:
            self.requests[project_id] = [
                req_time for req_time in self.requests[project_id]
                if current_time - req_time < 3600  # Keep last hour
            ]
        else:
            self.requests[project_id] = []
        
        # Check rate limits
        for period, limit in rate_limits.items():
            if period == "requests_per_minute":
                recent_requests = [
                    req_time for req_time in self.requests[project_id]
                    if current_time - req_time < 60
                ]
                if len(recent_requests) >= limit:
                    return True
            
            elif period == "requests_per_hour":
                recent_requests = [
                    req_time for req_time in self.requests[project_id]
                    if current_time - req_time < 3600
                ]
                if len(recent_requests) >= limit:
                    return True
            
            elif period == "requests_per_day":
                recent_requests = [
                    req_time for req_time in self.requests[project_id]
                    if current_time - req_time < 86400
                ]
                if len(recent_requests) >= limit:
                    return True
        
        # Record this request
        self.requests[project_id].append(current_time)
        return False
    
    def get_usage_stats(self, project_id: str) -> Dict[str, int]:
        """Get current usage statistics."""
        current_time = time.time()
        
        if project_id not in self.requests:
            return {"requests_per_minute": 0, "requests_per_hour": 0, "requests_per_day": 0}
        
        requests = self.requests[project_id]
        
        return {
            "requests_per_minute": len([r for r in requests if current_time - r < 60]),
            "requests_per_hour": len([r for r in requests if current_time - r < 3600]),
            "requests_per_day": len([r for r in requests if current_time - r < 86400])
        }


# Global instances
project_validator = ProjectValidator()
rate_limiter = RateLimiter()


# FastAPI Dependencies

async def get_current_project(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> ProjectCredentials:
    """Get current authenticated project."""
    
    # Try JWT token first
    try:
        payload = JWTManager.verify_token(credentials.credentials)
        if payload and payload.get("type") == "project_access":
            project_id = payload.get("project_id")
            if project_id:
                project_creds = project_validator.get_project_credentials(project_id)
                if project_creds and project_creds.is_active:
                    return project_creds
    except Exception as e:
        logger.debug(f"JWT authentication failed: {e}")
    
    # Try API key authentication
    api_key = request.headers.get(API_KEY_HEADER)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # Extract project ID from request (could be in path, query, or body)
    project_id = None
    
    # Try to get from path parameters
    if hasattr(request, 'path_params') and 'project_id' in request.path_params:
        project_id = request.path_params['project_id']
    
    # Try to get from query parameters
    if not project_id and 'project_id' in request.query_params:
        project_id = request.query_params['project_id']
    
    if not project_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Project ID required"
        )
    
    # Authenticate project
    project_creds = project_validator.authenticate_project(project_id, api_key)
    if not project_creds:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid project credentials"
        )
    
    # Check rate limits
    if rate_limiter.is_rate_limited(project_id, project_creds.rate_limits):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    return project_creds


async def require_permission(permission: str):
    """Dependency to require specific permission."""
    async def permission_checker(project: ProjectCredentials = Depends(get_current_project)):
        if not project_validator.check_permission(project.project_id, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return project
    
    return permission_checker


async def get_project_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[ProjectCredentials]:
    """Get current project (optional authentication)."""
    if not credentials:
        return None
    
    try:
        return await get_current_project(request, credentials)
    except HTTPException:
        return None


# Utility functions

def create_project_auth_response(project_creds: ProjectCredentials) -> ProjectAuthResponse:
    """Create authentication response."""
    token = JWTManager.create_token(
        project_creds.project_id,
        project_creds.permissions
    )
    
    return ProjectAuthResponse(
        project_id=project_creds.project_id,
        authenticated=True,
        permissions=project_creds.permissions,
        rate_limits=project_creds.rate_limits,
        quotas=project_creds.quotas,
        expires_at=JWTManager.get_token_expiration(token),
        token=token
    )


def validate_project_access(project_id: str, required_permissions: List[str]) -> bool:
    """Validate project has required permissions."""
    project_creds = project_validator.get_project_credentials(project_id)
    if not project_creds or not project_creds.is_active:
        return False
    
    return all(
        project_validator.check_permission(project_id, permission)
        for permission in required_permissions
    )


def get_project_usage_stats(project_id: str) -> Dict[str, Any]:
    """Get project usage statistics."""
    usage_stats = rate_limiter.get_usage_stats(project_id)
    project_creds = project_validator.get_project_credentials(project_id)
    
    if not project_creds:
        return usage_stats
    
    return {
        **usage_stats,
        "rate_limits": project_creds.rate_limits,
        "quotas": project_creds.quotas
    }

"""
Rate limiting and quota management for Universal Memory Access API.

This module provides comprehensive rate limiting and quota management
capabilities to ensure fair usage and prevent abuse of the API.
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import hashlib

from app.core.logging import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class RateLimitWindow:
    """Rate limit window for tracking requests."""
    
    window_start: float
    request_count: int = 0
    requests: deque = field(default_factory=deque)
    
    def add_request(self, timestamp: float) -> None:
        """Add a request to the window."""
        self.requests.append(timestamp)
        self.request_count += 1
    
    def cleanup_old_requests(self, window_duration: float, current_time: float) -> None:
        """Remove requests outside the window."""
        cutoff_time = current_time - window_duration
        while self.requests and self.requests[0] < cutoff_time:
            self.requests.popleft()
            self.request_count -= 1


@dataclass
class QuotaUsage:
    """Quota usage tracking."""
    
    memory_count: int = 0
    storage_bytes: int = 0
    last_reset: datetime = field(default_factory=datetime.utcnow)
    
    def reset_if_needed(self, reset_period: timedelta) -> bool:
        """Reset quota if reset period has passed."""
        if datetime.utcnow() - self.last_reset >= reset_period:
            self.memory_count = 0
            self.storage_bytes = 0
            self.last_reset = datetime.utcnow()
            return True
        return False


class RateLimiter:
    """Advanced rate limiter with multiple window support."""
    
    def __init__(self):
        self.rate_limits: Dict[str, Dict[str, RateLimitWindow]] = defaultdict(dict)
        self.quota_usage: Dict[str, QuotaUsage] = {}
        self.blocked_ips: Dict[str, float] = {}  # IP -> unblock_time
        self.suspicious_activity: Dict[str, List[float]] = defaultdict(list)
        
        # Rate limit configurations
        self.default_limits = {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "requests_per_day": 10000
        }
        
        # Quota configurations
        self.default_quotas = {
            "max_memories": 10000,
            "max_storage_bytes": 100 * 1024 * 1024,  # 100MB
            "quota_reset_period_hours": 24
        }
        
        # Cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        try:
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        except RuntimeError:
            # No event loop running, skip cleanup task
            pass
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                await self._cleanup_expired_data()
                await asyncio.sleep(60)  # Cleanup every minute
            except Exception as e:
                logger.error(f"Rate limiter cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_expired_data(self):
        """Clean up expired rate limit data."""
        current_time = time.time()
        
        # Clean up expired rate limit windows
        for project_id in list(self.rate_limits.keys()):
            for limit_type in list(self.rate_limits[project_id].keys()):
                window = self.rate_limits[project_id][limit_type]
                window_duration = self._get_window_duration(limit_type)
                window.cleanup_old_requests(window_duration, current_time)
                
                # Remove empty windows
                if window.request_count == 0:
                    del self.rate_limits[project_id][limit_type]
            
            # Remove empty project entries
            if not self.rate_limits[project_id]:
                del self.rate_limits[project_id]
        
        # Clean up expired blocked IPs
        expired_ips = [
            ip for ip, unblock_time in self.blocked_ips.items()
            if current_time > unblock_time
        ]
        for ip in expired_ips:
            del self.blocked_ips[ip]
        
        # Clean up old suspicious activity
        cutoff_time = current_time - 3600  # 1 hour
        for ip in list(self.suspicious_activity.keys()):
            self.suspicious_activity[ip] = [
                timestamp for timestamp in self.suspicious_activity[ip]
                if timestamp > cutoff_time
            ]
            if not self.suspicious_activity[ip]:
                del self.suspicious_activity[ip]
    
    def _get_window_duration(self, limit_type: str) -> float:
        """Get window duration for limit type."""
        durations = {
            "requests_per_minute": 60,
            "requests_per_hour": 3600,
            "requests_per_day": 86400
        }
        return durations.get(limit_type, 60)
    
    def is_rate_limited(
        self,
        project_id: str,
        rate_limits: Dict[str, int],
        client_ip: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """Check if project is rate limited.
        
        Returns:
            Tuple of (is_limited, limit_type_that_triggered)
        """
        current_time = time.time()
        
        # Check if IP is blocked
        if client_ip and client_ip in self.blocked_ips:
            if current_time < self.blocked_ips[client_ip]:
                return True, "ip_blocked"
            else:
                del self.blocked_ips[client_ip]
        
        # Check rate limits
        for limit_type, limit_value in rate_limits.items():
            if not self._check_rate_limit(project_id, limit_type, limit_value, current_time):
                # Track suspicious activity
                if client_ip:
                    self._track_suspicious_activity(client_ip, current_time)
                return True, limit_type
        
        return False, None
    
    def _check_rate_limit(
        self,
        project_id: str,
        limit_type: str,
        limit_value: int,
        current_time: float
    ) -> bool:
        """Check if a specific rate limit is exceeded."""
        window_duration = self._get_window_duration(limit_type)
        
        # Get or create window
        if limit_type not in self.rate_limits[project_id]:
            self.rate_limits[project_id][limit_type] = RateLimitWindow(
                window_start=current_time
            )
        
        window = self.rate_limits[project_id][limit_type]
        
        # Clean up old requests
        window.cleanup_old_requests(window_duration, current_time)
        
        # Check if limit exceeded
        if window.request_count >= limit_value:
            return False
        
        # Add current request
        window.add_request(current_time)
        return True
    
    def _track_suspicious_activity(self, client_ip: str, timestamp: float):
        """Track suspicious activity from an IP."""
        self.suspicious_activity[client_ip].append(timestamp)
        
        # If too many violations in short time, block IP
        recent_violations = [
            t for t in self.suspicious_activity[client_ip]
            if timestamp - t < 300  # 5 minutes
        ]
        
        if len(recent_violations) >= 10:  # 10 violations in 5 minutes
            self.blocked_ips[client_ip] = timestamp + 3600  # Block for 1 hour
            logger.warning(f"Blocked IP {client_ip} for 1 hour due to excessive rate limit violations")
    
    def is_quota_exceeded(
        self,
        project_id: str,
        quotas: Dict[str, int],
        memory_size_bytes: int = 0
    ) -> Tuple[bool, Optional[str]]:
        """Check if project quota is exceeded.
        
        Returns:
            Tuple of (is_exceeded, quota_type_that_triggered)
        """
        # Get or create quota usage
        if project_id not in self.quota_usage:
            self.quota_usage[project_id] = QuotaUsage()
        
        usage = self.quota_usage[project_id]
        
        # Reset quota if needed
        reset_period = timedelta(hours=self.default_quotas.get("quota_reset_period_hours", 24))
        usage.reset_if_needed(reset_period)
        
        # Check memory count quota
        max_memories = quotas.get("max_memories", self.default_quotas["max_memories"])
        if usage.memory_count >= max_memories:
            return True, "max_memories"
        
        # Check storage quota
        max_storage = quotas.get("max_storage_bytes", self.default_quotas["max_storage_bytes"])
        if usage.storage_bytes + memory_size_bytes > max_storage:
            return True, "max_storage_bytes"
        
        return False, None
    
    def record_quota_usage(
        self,
        project_id: str,
        memory_count: int = 1,
        storage_bytes: int = 0
    ):
        """Record quota usage for a project."""
        if project_id not in self.quota_usage:
            self.quota_usage[project_id] = QuotaUsage()
        
        usage = self.quota_usage[project_id]
        usage.memory_count += memory_count
        usage.storage_bytes += storage_bytes
    
    def get_usage_stats(self, project_id: str) -> Dict[str, Any]:
        """Get usage statistics for a project."""
        current_time = time.time()
        
        # Rate limit stats
        rate_limit_stats = {}
        for limit_type in self.default_limits.keys():
            if limit_type in self.rate_limits[project_id]:
                window = self.rate_limits[project_id][limit_type]
                window_duration = self._get_window_duration(limit_type)
                window.cleanup_old_requests(window_duration, current_time)
                rate_limit_stats[limit_type] = window.request_count
            else:
                rate_limit_stats[limit_type] = 0
        
        # Quota stats
        quota_stats = {}
        if project_id in self.quota_usage:
            usage = self.quota_usage[project_id]
            quota_stats = {
                "memory_count": usage.memory_count,
                "storage_bytes": usage.storage_bytes,
                "last_reset": usage.last_reset.isoformat()
            }
        else:
            quota_stats = {
                "memory_count": 0,
                "storage_bytes": 0,
                "last_reset": datetime.utcnow().isoformat()
            }
        
        return {
            "rate_limits": rate_limit_stats,
            "quotas": quota_stats,
            "timestamp": current_time
        }
    
    def get_reset_times(self, project_id: str) -> Dict[str, datetime]:
        """Get reset times for rate limits and quotas."""
        reset_times = {}
        current_time = datetime.utcnow()
        
        # Rate limit reset times (next window)
        for limit_type in self.default_limits.keys():
            window_duration = self._get_window_duration(limit_type)
            if limit_type in self.rate_limits[project_id]:
                window = self.rate_limits[project_id][limit_type]
                next_reset = datetime.fromtimestamp(window.window_start + window_duration)
                reset_times[limit_type] = next_reset
            else:
                reset_times[limit_type] = current_time
        
        # Quota reset time
        if project_id in self.quota_usage:
            usage = self.quota_usage[project_id]
            reset_period = timedelta(hours=self.default_quotas.get("quota_reset_period_hours", 24))
            reset_times["quota"] = usage.last_reset + reset_period
        else:
            reset_times["quota"] = current_time + timedelta(hours=24)
        
        return reset_times
    
    def set_custom_limits(
        self,
        project_id: str,
        rate_limits: Optional[Dict[str, int]] = None,
        quotas: Optional[Dict[str, int]] = None
    ):
        """Set custom rate limits and quotas for a project."""
        if rate_limits:
            # Store custom rate limits (would be persisted in real implementation)
            logger.info(f"Set custom rate limits for project {project_id}: {rate_limits}")
        
        if quotas:
            # Store custom quotas (would be persisted in real implementation)
            logger.info(f"Set custom quotas for project {project_id}: {quotas}")
    
    def get_blocked_ips(self) -> List[str]:
        """Get list of currently blocked IPs."""
        current_time = time.time()
        return [
            ip for ip, unblock_time in self.blocked_ips.items()
            if current_time < unblock_time
        ]
    
    def unblock_ip(self, client_ip: str) -> bool:
        """Manually unblock an IP address."""
        if client_ip in self.blocked_ips:
            del self.blocked_ips[client_ip]
            logger.info(f"Manually unblocked IP: {client_ip}")
            return True
        return False
    
    def get_suspicious_ips(self) -> List[Dict[str, Any]]:
        """Get list of IPs with suspicious activity."""
        current_time = time.time()
        suspicious = []
        
        for ip, timestamps in self.suspicious_activity.items():
            recent_violations = [
                t for t in timestamps
                if current_time - t < 3600  # Last hour
            ]
            
            if recent_violations:
                suspicious.append({
                    "ip": ip,
                    "violation_count": len(recent_violations),
                    "last_violation": max(recent_violations),
                    "is_blocked": ip in self.blocked_ips
                })
        
        return suspicious


class QuotaManager:
    """Advanced quota management with storage tracking."""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.storage_tracking: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def check_memory_quota(
        self,
        project_id: str,
        quotas: Dict[str, int],
        memory_size_bytes: int
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Check if adding a memory would exceed quota."""
        
        # Check if quota exceeded
        is_exceeded, quota_type = self.rate_limiter.is_quota_exceeded(
            project_id, quotas, memory_size_bytes
        )
        
        if is_exceeded:
            return True, quota_type, {}
        
        # Get current usage
        current_usage = self.rate_limiter.get_usage_stats(project_id)
        
        # Calculate projected usage
        projected_usage = {
            "memory_count": current_usage["quotas"]["memory_count"] + 1,
            "storage_bytes": current_usage["quotas"]["storage_bytes"] + memory_size_bytes
        }
        
        # Calculate usage percentages
        usage_percentages = {
            "memory_count": (projected_usage["memory_count"] / quotas.get("max_memories", 10000)) * 100,
            "storage_bytes": (projected_usage["storage_bytes"] / quotas.get("max_storage_bytes", 100 * 1024 * 1024)) * 100
        }
        
        return False, None, {
            "current_usage": current_usage["quotas"],
            "projected_usage": projected_usage,
            "usage_percentages": usage_percentages,
            "quotas": quotas
        }
    
    def record_memory_creation(
        self,
        project_id: str,
        memory_size_bytes: int
    ):
        """Record memory creation in quota tracking."""
        self.rate_limiter.record_quota_usage(
            project_id,
            memory_count=1,
            storage_bytes=memory_size_bytes
        )
        
        # Track storage by type
        self.storage_tracking[project_id]["total_bytes"] += memory_size_bytes
        self.storage_tracking[project_id]["memory_count"] += 1
    
    def record_memory_deletion(
        self,
        project_id: str,
        memory_size_bytes: int
    ):
        """Record memory deletion in quota tracking."""
        # Note: In a real implementation, you'd need to track individual memory sizes
        # For now, we'll just decrement the count
        self.rate_limiter.record_quota_usage(
            project_id,
            memory_count=-1,
            storage_bytes=-memory_size_bytes
        )
        
        # Update storage tracking
        self.storage_tracking[project_id]["total_bytes"] = max(0, 
            self.storage_tracking[project_id]["total_bytes"] - memory_size_bytes
        )
        self.storage_tracking[project_id]["memory_count"] = max(0,
            self.storage_tracking[project_id]["memory_count"] - 1
        )
    
    def get_storage_breakdown(self, project_id: str) -> Dict[str, Any]:
        """Get detailed storage breakdown for a project."""
        if project_id not in self.storage_tracking:
            return {
                "total_bytes": 0,
                "memory_count": 0,
                "average_memory_size": 0
            }
        
        storage_data = self.storage_tracking[project_id]
        memory_count = storage_data["memory_count"]
        
        return {
            "total_bytes": storage_data["total_bytes"],
            "memory_count": memory_count,
            "average_memory_size": storage_data["total_bytes"] / memory_count if memory_count > 0 else 0
        }


# Global instances
_rate_limiter: Optional[RateLimiter] = None
_quota_manager: Optional[QuotaManager] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter instance."""
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
        logger.info("Rate limiter initialized")
    
    return _rate_limiter


def get_quota_manager() -> QuotaManager:
    """Get or create global quota manager instance."""
    global _quota_manager
    
    if _quota_manager is None:
        rate_limiter = get_rate_limiter()
        _quota_manager = QuotaManager(rate_limiter)
        logger.info("Quota manager initialized")
    
    return _quota_manager


# Utility functions

def extract_client_ip(request) -> Optional[str]:
    """Extract client IP from request."""
    # Check for forwarded headers first
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct connection
    if hasattr(request, "client") and request.client:
        return request.client.host
    
    return None


def calculate_memory_size(content: str, metadata: Dict[str, Any] = None) -> int:
    """Calculate memory size in bytes."""
    size = len(content.encode('utf-8'))
    
    if metadata:
        size += len(json.dumps(metadata).encode('utf-8'))
    
    return size


async def check_rate_limit_and_quota(
    project_id: str,
    rate_limits: Dict[str, int],
    quotas: Dict[str, int],
    request,
    memory_size_bytes: int = 0
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Check both rate limits and quotas.
    
    Returns:
        Tuple of (is_limited, limit_type, quota_info)
    """
    rate_limiter = get_rate_limiter()
    quota_manager = get_quota_manager()
    
    # Extract client IP
    client_ip = extract_client_ip(request)
    
    # Check rate limits
    is_rate_limited, limit_type = rate_limiter.is_rate_limited(
        project_id, rate_limits, client_ip
    )
    
    if is_rate_limited:
        return True, limit_type, None
    
    # Check quotas
    is_quota_exceeded, quota_type, quota_info = quota_manager.check_memory_quota(
        project_id, quotas, memory_size_bytes
    )
    
    if is_quota_exceeded:
        return True, quota_type, quota_info
    
    return False, None, quota_info

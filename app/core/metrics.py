"""
Simplified metrics collection for AI Agent Memory Router.

This module provides basic metrics collection without external dependencies.
Prometheus integration can be added later when needed.
"""

import time
from typing import Dict, Any, Optional
from app.core.config import get_settings

# Get settings
settings = get_settings()


class MetricsCollector:
    """Simplified metrics collector for AI Agent Memory Router."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup basic metrics."""
        # Initialize basic counters and gauges
        self.memory_routes_total = 0
        self.active_memory_routes = 0
        self.agents_total = 0
        self.memories_stored_total = 0
        
    def record_memory_route(self, source_agent_id: str, target_agent_id: str, status: str, priority: str):
        """Record a memory route."""
        self.memory_routes_total += 1
        
    def record_memory_routing_duration(self, duration: float, source_agent_id: str, target_agent_id: str, priority: str):
        """Record memory routing duration."""
        # For now, just log the duration
        pass
        
    def set_active_memory_routes(self, count: int):
        """Set the number of active memory routes."""
        self.active_memory_routes = count
        
    def set_agents_total(self, count: int):
        """Set the total number of agents."""
        self.agents_total = count
        
    def record_memory_stored(self, agent_id: str, memory_type: str):
        """Record a memory being stored."""
        self.memories_stored_total += 1
        
    def record_memory_storage_duration(self, duration: float, agent_id: str, memory_type: str):
        """Record memory storage duration."""
        # For now, just log the duration
        pass
        
    def record_memory_search_duration(self, duration: float, query_type: str):
        """Record memory search duration."""
        # For now, just log the duration
        pass
        
    def record_context_update(self, conversation_id: str, agent_id: str):
        """Record a context update."""
        # For now, just log the update
        pass
        
    def set_context_size(self, conversation_id: str, size_bytes: int):
        """Set the size of conversation context."""
        # For now, just log the size
        pass


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    
    return _metrics_collector


def record_request(method: str, endpoint: str, status_code: int):
    """Record an API request."""
    # For now, just log the request
    pass


def record_request_duration(method: str, endpoint: str, duration: float):
    """Record API request duration."""
    # For now, just log the duration
    pass


def record_memory_operation(operation: str, agent_id: str, success: bool):
    """Record a memory operation."""
    # For now, just log the operation
    pass


def record_agent_operation(operation: str, agent_id: str, success: bool):
    """Record an agent operation."""
    # For now, just log the operation
    pass


def record_routing_decision(source_agent_id: str, target_agent_ids: list, decision: str):
    """Record a routing decision."""
    # For now, just log the decision
    pass


def record_error(error_type: str, error_message: str, context: Optional[Dict[str, Any]] = None):
    """Record an error."""
    # For now, just log the error
    pass


def get_metrics_summary() -> Dict[str, Any]:
    """Get a summary of current metrics."""
    collector = get_metrics_collector()
    return {
        "memory_routes_total": collector.memory_routes_total,
        "active_memory_routes": collector.active_memory_routes,
        "agents_total": collector.agents_total,
        "memories_stored_total": collector.memories_stored_total
    }

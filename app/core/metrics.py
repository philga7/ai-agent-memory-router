"""
Metrics collection for AI Agent Memory Router.

This module provides Prometheus metrics collection for monitoring
application performance, memory routing statistics, and system health.
"""

import time
from typing import Dict, Any, Optional
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info,
    generate_latest, CONTENT_TYPE_LATEST
)

from app.core.config import get_settings

# Get settings
settings = get_settings()


class MetricsCollector:
    """Metrics collector for AI Agent Memory Router."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup Prometheus metrics."""
        
        # Memory routing metrics
        self.memory_routes_total = Counter(
            'memory_routes_total',
            'Total number of memory routes',
            ['source_agent_id', 'target_agent_id', 'status', 'priority']
        )
        
        self.memory_routing_duration = Histogram(
            'memory_routing_duration_seconds',
            'Memory routing duration in seconds',
            ['source_agent_id', 'target_agent_id', 'priority'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        self.active_memory_routes = Gauge(
            'active_memory_routes',
            'Number of currently active memory routes'
        )
        
        # Agent metrics
        self.agents_total = Gauge(
            'agents_total',
            'Total number of registered agents',
            ['status']
        )
        
        self.agent_heartbeats = Counter(
            'agent_heartbeats_total',
            'Total number of agent heartbeats',
            ['agent_id', 'status']
        )
        
        self.agent_memory_count = Gauge(
            'agent_memory_count',
            'Number of memories per agent',
            ['agent_id']
        )
        
        # Memory storage metrics
        self.memories_stored_total = Counter(
            'memories_stored_total',
            'Total number of memories stored',
            ['agent_id', 'memory_type']
        )
        
        self.memory_storage_duration = Histogram(
            'memory_storage_duration_seconds',
            'Memory storage duration in seconds',
            ['agent_id', 'memory_type'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        self.memory_search_duration = Histogram(
            'memory_search_duration_seconds',
            'Memory search duration in seconds',
            ['query_type'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        # Context management metrics
        self.context_updates_total = Counter(
            'context_updates_total',
            'Total number of context updates',
            ['conversation_id', 'agent_id']
        )
        
        self.context_size_bytes = Gauge(
            'context_size_bytes',
            'Size of conversation context in bytes',
            ['conversation_id']
        )
        
        # System metrics
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint', 'status_code'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        self.requests_total = Counter(
            'http_requests_total',
            'Total number of HTTP requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            ['connection_type']
        )
        
        # Error metrics
        self.errors_total = Counter(
            'errors_total',
            'Total number of errors',
            ['error_type', 'component']
        )
        
        # Application info
        self.app_info = Info(
            'ai_agent_memory_router',
            'AI Agent Memory Router application information'
        )
        self.app_info.info({
            'version': '0.1.0',
            'environment': settings.environment,
            'description': 'AI Agent Memory Router'
        })
    
    def record_memory_route(
        self,
        source_agent_id: str,
        target_agent_id: str,
        status: str,
        priority: str = "normal"
    ):
        """Record memory route metric."""
        self.memory_routes_total.labels(
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
            status=status,
            priority=priority
        ).inc()
    
    def record_memory_routing_duration(
        self,
        duration: float,
        source_agent_id: str,
        target_agent_id: str,
        priority: str = "normal"
    ):
        """Record memory routing duration metric."""
        self.memory_routing_duration.labels(
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
            priority=priority
        ).observe(duration)
    
    def set_active_memory_routes(self, count: int):
        """Set active memory routes gauge."""
        self.active_memory_routes.set(count)
    
    def record_agent_registration(self, status: str):
        """Record agent registration metric."""
        self.agents_total.labels(status=status).inc()
    
    def record_agent_heartbeat(self, agent_id: str, status: str):
        """Record agent heartbeat metric."""
        self.agent_heartbeats.labels(
            agent_id=agent_id,
            status=status
        ).inc()
    
    def set_agent_memory_count(self, agent_id: str, count: int):
        """Set agent memory count gauge."""
        self.agent_memory_count.labels(agent_id=agent_id).set(count)
    
    def record_memory_stored(
        self,
        agent_id: str,
        memory_type: str
    ):
        """Record memory stored metric."""
        self.memories_stored_total.labels(
            agent_id=agent_id,
            memory_type=memory_type
        ).inc()
    
    def record_memory_storage_duration(
        self,
        duration: float,
        agent_id: str,
        memory_type: str
    ):
        """Record memory storage duration metric."""
        self.memory_storage_duration.labels(
            agent_id=agent_id,
            memory_type=memory_type
        ).observe(duration)
    
    def record_memory_search_duration(
        self,
        duration: float,
        query_type: str
    ):
        """Record memory search duration metric."""
        self.memory_search_duration.labels(
            query_type=query_type
        ).observe(duration)
    
    def record_context_update(
        self,
        conversation_id: str,
        agent_id: str
    ):
        """Record context update metric."""
        self.context_updates_total.labels(
            conversation_id=conversation_id,
            agent_id=agent_id
        ).inc()
    
    def set_context_size(self, conversation_id: str, size_bytes: int):
        """Set context size gauge."""
        self.context_size_bytes.labels(
            conversation_id=conversation_id
        ).set(size_bytes)
    
    def record_request_duration(
        self,
        duration: float,
        method: str,
        endpoint: str,
        status_code: int
    ):
        """Record HTTP request duration metric."""
        self.request_duration.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).observe(duration)
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int
    ):
        """Record HTTP request metric."""
        self.requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
    
    def set_active_connections(self, connection_type: str, count: int):
        """Set active connections gauge."""
        self.active_connections.labels(
            connection_type=connection_type
        ).set(count)
    
    def record_error(self, error_type: str, component: str):
        """Record error metric."""
        self.errors_total.labels(
            error_type=error_type,
            component=component
        ).inc()
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics as string."""
        return generate_latest()
    
    def get_metrics_content_type(self) -> str:
        """Get Prometheus metrics content type."""
        return CONTENT_TYPE_LATEST


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def setup_metrics() -> None:
    """Setup metrics collection."""
    global _metrics_collector
    
    if settings.monitoring.enable_metrics:
        _metrics_collector = MetricsCollector()
        print("Metrics collection enabled")
    else:
        print("Metrics collection disabled")


def get_metrics_collector() -> Optional[MetricsCollector]:
    """Get the metrics collector instance."""
    return _metrics_collector


# Convenience functions for metrics recording

def record_memory_route(
    source_agent_id: str,
    target_agent_id: str,
    status: str,
    priority: str = "normal"
):
    """Record memory route metric."""
    collector = get_metrics_collector()
    if collector:
        collector.record_memory_route(
            source_agent_id, target_agent_id, status, priority
        )


def record_memory_routing_duration(
    duration: float,
    source_agent_id: str,
    target_agent_id: str,
    priority: str = "normal"
):
    """Record memory routing duration metric."""
    collector = get_metrics_collector()
    if collector:
        collector.record_memory_routing_duration(
            duration, source_agent_id, target_agent_id, priority
        )


def record_agent_registration(status: str):
    """Record agent registration metric."""
    collector = get_metrics_collector()
    if collector:
        collector.record_agent_registration(status)


def record_agent_heartbeat(agent_id: str, status: str):
    """Record agent heartbeat metric."""
    collector = get_metrics_collector()
    if collector:
        collector.record_agent_heartbeat(agent_id, status)


def record_memory_stored(agent_id: str, memory_type: str):
    """Record memory stored metric."""
    collector = get_metrics_collector()
    if collector:
        collector.record_memory_stored(agent_id, memory_type)


def record_memory_storage_duration(
    duration: float,
    agent_id: str,
    memory_type: str
):
    """Record memory storage duration metric."""
    collector = get_metrics_collector()
    if collector:
        collector.record_memory_storage_duration(
            duration, agent_id, memory_type
        )


def record_memory_search_duration(duration: float, query_type: str):
    """Record memory search duration metric."""
    collector = get_metrics_collector()
    if collector:
        collector.record_memory_search_duration(duration, query_type)


def record_context_update(conversation_id: str, agent_id: str):
    """Record context update metric."""
    collector = get_metrics_collector()
    if collector:
        collector.record_context_update(conversation_id, agent_id)


def record_request_duration(
    duration: float,
    method: str,
    endpoint: str,
    status_code: int
):
    """Record HTTP request duration metric."""
    collector = get_metrics_collector()
    if collector:
        collector.record_request_duration(
            duration, method, endpoint, status_code
        )


def record_request(method: str, endpoint: str, status_code: int):
    """Record HTTP request metric."""
    collector = get_metrics_collector()
    if collector:
        collector.record_request(method, endpoint, status_code)


def record_error(error_type: str, component: str):
    """Record error metric."""
    collector = get_metrics_collector()
    if collector:
        collector.record_error(error_type, component)


# Metrics decorators

def track_memory_routing(priority: str = "normal"):
    """Decorator to track memory routing metrics."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Extract agent IDs from function arguments
                source_agent_id = kwargs.get('source_agent_id', 'unknown')
                target_agent_ids = kwargs.get('target_agent_ids', [])
                
                result = await func(*args, **kwargs)
                
                # Record success metrics
                duration = time.time() - start_time
                for target_agent_id in target_agent_ids:
                    record_memory_route(
                        source_agent_id, target_agent_id, 'success', priority
                    )
                    record_memory_routing_duration(
                        duration, source_agent_id, target_agent_id, priority
                    )
                
                return result
                
            except Exception as e:
                # Record failure metrics
                duration = time.time() - start_time
                source_agent_id = kwargs.get('source_agent_id', 'unknown')
                target_agent_ids = kwargs.get('target_agent_ids', [])
                
                for target_agent_id in target_agent_ids:
                    record_memory_route(
                        source_agent_id, target_agent_id, 'failure', priority
                    )
                    record_memory_routing_duration(
                        duration, source_agent_id, target_agent_id, priority
                    )
                
                record_error('memory_routing_failure', 'memory_router')
                raise
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Extract agent IDs from function arguments
                source_agent_id = kwargs.get('source_agent_id', 'unknown')
                target_agent_ids = kwargs.get('target_agent_ids', [])
                
                result = func(*args, **kwargs)
                
                # Record success metrics
                duration = time.time() - start_time
                for target_agent_id in target_agent_ids:
                    record_memory_route(
                        source_agent_id, target_agent_id, 'success', priority
                    )
                    record_memory_routing_duration(
                        duration, source_agent_id, target_agent_id, priority
                    )
                
                return result
                
            except Exception as e:
                # Record failure metrics
                duration = time.time() - start_time
                source_agent_id = kwargs.get('source_agent_id', 'unknown')
                target_agent_ids = kwargs.get('target_agent_ids', [])
                
                for target_agent_id in target_agent_ids:
                    record_memory_route(
                        source_agent_id, target_agent_id, 'failure', priority
                    )
                    record_memory_routing_duration(
                        duration, source_agent_id, target_agent_id, priority
                    )
                
                record_error('memory_routing_failure', 'memory_router')
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def track_memory_storage(memory_type: str = "general"):
    """Decorator to track memory storage metrics."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                agent_id = kwargs.get('agent_id', 'unknown')
                
                result = await func(*args, **kwargs)
                
                # Record success metrics
                duration = time.time() - start_time
                record_memory_stored(agent_id, memory_type)
                record_memory_storage_duration(duration, agent_id, memory_type)
                
                return result
                
            except Exception as e:
                # Record failure metrics
                duration = time.time() - start_time
                agent_id = kwargs.get('agent_id', 'unknown')
                
                record_memory_storage_duration(duration, agent_id, memory_type)
                record_error('memory_storage_failure', 'memory_service')
                raise
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                agent_id = kwargs.get('agent_id', 'unknown')
                
                result = func(*args, **kwargs)
                
                # Record success metrics
                duration = time.time() - start_time
                record_memory_stored(agent_id, memory_type)
                record_memory_storage_duration(duration, agent_id, memory_type)
                
                return result
                
            except Exception as e:
                # Record failure metrics
                duration = time.time() - start_time
                agent_id = kwargs.get('agent_id', 'unknown')
                
                record_memory_storage_duration(duration, agent_id, memory_type)
                record_error('memory_storage_failure', 'memory_service')
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Import asyncio for the decorators
import asyncio

"""
Routing Decision Models and Policies for Intelligent Memory Routing.

This module defines the data models, policies, and decision-making structures
for the intelligent memory routing system.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass

from .common import TimestampMixin, IDMixin


class RoutingBackend(str, Enum):
    """Available storage backends for routing decisions."""
    CIPHER = "cipher"
    WEAVIATE = "weaviate"
    SQLITE = "sqlite"
    HYBRID = "hybrid"


class RoutingPriority(int, Enum):
    """Routing priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class RoutingStatus(str, Enum):
    """Routing operation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class ContentType(str, Enum):
    """Content type classifications."""
    CONVERSATION = "conversation"
    KNOWLEDGE = "knowledge"
    EXPERIENCE = "experience"
    FACT = "fact"
    PROCEDURE = "procedure"
    CONTEXT = "context"
    CODE = "code"
    DOCUMENT = "document"
    MEDIA = "media"
    UNKNOWN = "unknown"


class RoutingStrategy(str, Enum):
    """Routing strategy types."""
    CONTENT_BASED = "content_based"
    PERFORMANCE_BASED = "performance_based"
    COST_BASED = "cost_based"
    HYBRID = "hybrid"
    MANUAL = "manual"


class RoutingRule(BaseModel):
    """Individual routing rule definition."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Rule name")
    description: str = Field(..., description="Rule description")
    condition: Dict[str, Any] = Field(..., description="Rule condition")
    action: Dict[str, Any] = Field(..., description="Rule action")
    priority: int = Field(default=1, ge=1, le=10, description="Rule priority")
    enabled: bool = Field(default=True, description="Whether rule is enabled")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('condition')
    def validate_condition(cls, v):
        """Validate rule condition structure."""
        required_fields = ['field', 'operator', 'value']
        if not all(field in v for field in required_fields):
            raise ValueError('Condition must contain field, operator, and value')
        return v
    
    @validator('action')
    def validate_action(cls, v):
        """Validate rule action structure."""
        if 'backend' not in v:
            raise ValueError('Action must specify backend')
        return v


class RoutingPolicy(BaseModel):
    """Routing policy containing multiple rules."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Policy name")
    description: str = Field(..., description="Policy description")
    strategy: RoutingStrategy = Field(..., description="Routing strategy")
    rules: List[RoutingRule] = Field(default_factory=list, description="Policy rules")
    default_backend: RoutingBackend = Field(default=RoutingBackend.SQLITE, description="Default backend")
    enabled: bool = Field(default=True, description="Whether policy is enabled")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def add_rule(self, rule: RoutingRule) -> None:
        """Add a rule to the policy."""
        self.rules.append(rule)
        self.updated_at = datetime.utcnow()
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from the policy."""
        for i, rule in enumerate(self.rules):
            if rule.id == rule_id:
                del self.rules[i]
                self.updated_at = datetime.utcnow()
                return True
        return False
    
    def get_enabled_rules(self) -> List[RoutingRule]:
        """Get all enabled rules sorted by priority."""
        enabled_rules = [rule for rule in self.rules if rule.enabled]
        return sorted(enabled_rules, key=lambda x: x.priority, reverse=True)


class RoutingDecision(BaseModel, TimestampMixin, IDMixin):
    """Routing decision model."""
    
    memory_id: str = Field(..., description="Memory identifier")
    backend: RoutingBackend = Field(..., description="Selected backend")
    reason: str = Field(..., description="Reason for routing decision")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Decision confidence")
    priority: RoutingPriority = Field(..., description="Routing priority")
    estimated_cost: float = Field(default=0.0, ge=0.0, description="Estimated cost")
    estimated_latency: float = Field(default=0.0, ge=0.0, description="Estimated latency in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    policy_id: Optional[str] = Field(None, description="Policy that made the decision")
    rule_id: Optional[str] = Field(None, description="Rule that triggered the decision")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate confidence score."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v


class RoutingContext(BaseModel):
    """Context information for routing decisions."""
    
    agent_id: str = Field(..., description="Agent identifier")
    project_id: str = Field(default="default", description="Project identifier")
    memory_type: str = Field(..., description="Memory type")
    content: str = Field(..., description="Memory content")
    priority: int = Field(default=2, ge=1, le=4, description="Memory priority")
    tags: List[str] = Field(default_factory=list, description="Memory tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    source: str = Field(default="unknown", description="Memory source")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    def get_content_size(self) -> int:
        """Get content size in bytes."""
        return len(self.content.encode('utf-8'))
    
    def get_word_count(self) -> int:
        """Get word count."""
        return len(self.content.split())


class RoutingRequest(BaseModel):
    """Request for routing decision."""
    
    context: RoutingContext = Field(..., description="Routing context")
    strategy: Optional[RoutingStrategy] = Field(None, description="Routing strategy override")
    policy_id: Optional[str] = Field(None, description="Specific policy to use")
    force_backend: Optional[RoutingBackend] = Field(None, description="Force specific backend")
    timeout: Optional[float] = Field(default=5.0, ge=0.1, le=30.0, description="Request timeout")
    
    @validator('timeout')
    def validate_timeout(cls, v):
        """Validate timeout value."""
        if v <= 0:
            raise ValueError('Timeout must be positive')
        return v


class RoutingResponse(BaseModel):
    """Response from routing decision."""
    
    decision: RoutingDecision = Field(..., description="Routing decision")
    success: bool = Field(..., description="Whether routing was successful")
    message: Optional[str] = Field(None, description="Response message")
    processing_time: float = Field(..., description="Processing time in seconds")
    alternatives: List[RoutingDecision] = Field(default_factory=list, description="Alternative decisions")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RoutingStatistics(BaseModel):
    """Routing statistics and metrics."""
    
    backend: RoutingBackend = Field(..., description="Backend name")
    total_requests: int = Field(default=0, ge=0, description="Total routing requests")
    successful_requests: int = Field(default=0, ge=0, description="Successful requests")
    failed_requests: int = Field(default=0, ge=0, description="Failed requests")
    average_latency: float = Field(default=0.0, ge=0.0, description="Average latency in seconds")
    average_cost: float = Field(default=0.0, ge=0.0, description="Average cost")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    def update_statistics(self, success: bool, latency: float, cost: float = 0.0) -> None:
        """Update statistics with new request data."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Update running averages
        if self.total_requests > 0:
            self.average_latency = ((self.average_latency * (self.total_requests - 1)) + latency) / self.total_requests
            self.average_cost = ((self.average_cost * (self.total_requests - 1)) + cost) / self.total_requests
            self.success_rate = self.successful_requests / self.total_requests
        
        self.last_updated = datetime.utcnow()


class RoutingPerformanceMetrics(BaseModel):
    """Performance metrics for routing operations."""
    
    operation_type: str = Field(..., description="Type of operation")
    backend: RoutingBackend = Field(..., description="Backend used")
    start_time: datetime = Field(..., description="Operation start time")
    end_time: Optional[datetime] = Field(None, description="Operation end time")
    duration: Optional[float] = Field(None, description="Operation duration in seconds")
    success: bool = Field(default=False, description="Whether operation was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metrics")
    
    def complete(self, success: bool = True, error_message: Optional[str] = None) -> None:
        """Mark operation as complete."""
        self.end_time = datetime.utcnow()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.success = success
        self.error_message = error_message


class DeduplicationResult(BaseModel):
    """Result of deduplication check."""
    
    is_duplicate: bool = Field(..., description="Whether content is duplicate")
    existing_memory_id: Optional[str] = Field(None, description="Existing memory ID if duplicate")
    existing_backend: Optional[RoutingBackend] = Field(None, description="Existing backend if duplicate")
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Similarity score")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Deduplication confidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional deduplication info")


class RoutingOptimization(BaseModel):
    """Routing optimization suggestions."""
    
    backend: RoutingBackend = Field(..., description="Backend to optimize")
    issue: str = Field(..., description="Issue description")
    current_metric: float = Field(..., description="Current metric value")
    suggested_metric: float = Field(..., description="Suggested metric value")
    recommendation: str = Field(..., description="Optimization recommendation")
    priority: RoutingPriority = Field(default=RoutingPriority.NORMAL, description="Optimization priority")
    estimated_impact: float = Field(default=0.0, description="Estimated impact score")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RoutingConfiguration(BaseModel):
    """Routing system configuration."""
    
    default_strategy: RoutingStrategy = Field(default=RoutingStrategy.HYBRID, description="Default routing strategy")
    default_backend: RoutingBackend = Field(default=RoutingBackend.SQLITE, description="Default backend")
    enable_caching: bool = Field(default=True, description="Enable routing decision caching")
    cache_ttl: int = Field(default=300, ge=60, le=3600, description="Cache TTL in seconds")
    enable_deduplication: bool = Field(default=True, description="Enable content deduplication")
    similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Similarity threshold for deduplication")
    max_alternatives: int = Field(default=3, ge=1, le=10, description="Maximum alternative decisions")
    timeout: float = Field(default=5.0, ge=0.1, le=30.0, description="Default routing timeout")
    retry_attempts: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    
    @validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        """Validate similarity threshold."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Similarity threshold must be between 0.0 and 1.0')
        return v


class RoutingHealthCheck(BaseModel):
    """Routing system health check result."""
    
    status: str = Field(..., description="Overall system status")
    backends: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Backend health status")
    policies: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Policy health status")
    cache_status: Dict[str, Any] = Field(default_factory=dict, description="Cache status")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    last_check: datetime = Field(default_factory=datetime.utcnow)
    issues: List[str] = Field(default_factory=list, description="Identified issues")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


class RoutingBatchRequest(BaseModel):
    """Batch routing request for multiple memories."""
    
    requests: List[RoutingRequest] = Field(..., description="List of routing requests")
    batch_id: str = Field(default_factory=lambda: str(uuid4()), description="Batch identifier")
    strategy: Optional[RoutingStrategy] = Field(None, description="Batch routing strategy")
    timeout: Optional[float] = Field(default=30.0, ge=1.0, le=300.0, description="Batch timeout")
    parallel: bool = Field(default=True, description="Process requests in parallel")
    continue_on_error: bool = Field(default=True, description="Continue processing on individual errors")
    
    @validator('requests')
    def validate_requests(cls, v):
        """Validate requests list."""
        if not v:
            raise ValueError('Requests list cannot be empty')
        if len(v) > 100:
            raise ValueError('Batch size cannot exceed 100 requests')
        return v


class RoutingBatchResponse(BaseModel):
    """Batch routing response."""
    
    batch_id: str = Field(..., description="Batch identifier")
    responses: List[RoutingResponse] = Field(..., description="Individual responses")
    total_requests: int = Field(..., description="Total number of requests")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    processing_time: float = Field(..., description="Total processing time")
    success_rate: float = Field(..., description="Overall success rate")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests


# Utility functions for routing models

def create_default_routing_policy() -> RoutingPolicy:
    """Create a default routing policy with common rules."""
    policy = RoutingPolicy(
        name="Default Routing Policy",
        description="Default policy for intelligent memory routing",
        strategy=RoutingStrategy.HYBRID
    )
    
    # Add content type rules
    content_rules = [
        RoutingRule(
            name="Conversation to Cipher",
            description="Route conversational content to Cipher",
            condition={"field": "content_type", "operator": "equals", "value": "conversation"},
            action={"backend": "cipher", "reason": "Conversational content optimized for Cipher"},
            priority=8
        ),
        RoutingRule(
            name="Knowledge to Weaviate",
            description="Route knowledge content to Weaviate for semantic search",
            condition={"field": "content_type", "operator": "equals", "value": "knowledge"},
            action={"backend": "weaviate", "reason": "Knowledge content benefits from semantic search"},
            priority=8
        ),
        RoutingRule(
            name="Code to SQLite",
            description="Route code content to SQLite for fast access",
            condition={"field": "content_type", "operator": "equals", "value": "code"},
            action={"backend": "sqlite", "reason": "Code content benefits from local storage"},
            priority=8
        ),
        RoutingRule(
            name="Critical Priority to SQLite",
            description="Route critical priority content to SQLite for fast access",
            condition={"field": "priority", "operator": "equals", "value": 4},
            action={"backend": "sqlite", "reason": "Critical content needs fast local access"},
            priority=9
        ),
        RoutingRule(
            name="Large Content to Weaviate",
            description="Route large content to Weaviate for distributed storage",
            condition={"field": "content_size", "operator": "greater_than", "value": 102400},
            action={"backend": "weaviate", "reason": "Large content optimized for distributed storage"},
            priority=6
        )
    ]
    
    for rule in content_rules:
        policy.add_rule(rule)
    
    return policy


def create_performance_routing_policy() -> RoutingPolicy:
    """Create a performance-optimized routing policy."""
    policy = RoutingPolicy(
        name="Performance Routing Policy",
        description="Policy optimized for performance and latency",
        strategy=RoutingStrategy.PERFORMANCE_BASED
    )
    
    # Add performance-focused rules
    performance_rules = [
        RoutingRule(
            name="High Performance to SQLite",
            description="Route high-performance content to SQLite",
            condition={"field": "priority", "operator": "greater_than_or_equal", "value": 3},
            action={"backend": "sqlite", "reason": "High priority content needs fast access"},
            priority=9
        ),
        RoutingRule(
            name="Frequent Access to SQLite",
            description="Route frequently accessed content to SQLite",
            condition={"field": "access_frequency", "operator": "greater_than", "value": 10},
            action={"backend": "sqlite", "reason": "Frequently accessed content cached locally"},
            priority=7
        )
    ]
    
    for rule in performance_rules:
        policy.add_rule(rule)
    
    return policy


def create_cost_optimized_routing_policy() -> RoutingPolicy:
    """Create a cost-optimized routing policy."""
    policy = RoutingPolicy(
        name="Cost Optimized Routing Policy",
        description="Policy optimized for cost efficiency",
        strategy=RoutingStrategy.COST_BASED
    )
    
    # Add cost-focused rules
    cost_rules = [
        RoutingRule(
            name="Low Priority to SQLite",
            description="Route low priority content to cost-effective SQLite",
            condition={"field": "priority", "operator": "less_than_or_equal", "value": 2},
            action={"backend": "sqlite", "reason": "Low priority content uses cost-effective storage"},
            priority=8
        ),
        RoutingRule(
            name="Archive to SQLite",
            description="Route archived content to SQLite",
            condition={"field": "age_days", "operator": "greater_than", "value": 30},
            action={"backend": "sqlite", "reason": "Archived content moved to cost-effective storage"},
            priority=6
        )
    ]
    
    for rule in cost_rules:
        policy.add_rule(rule)
    
    return policy

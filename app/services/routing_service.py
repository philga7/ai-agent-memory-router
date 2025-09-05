"""
Intelligent Memory Routing Service with SQLite Backend.

This service provides high-level routing operations that integrate the routing engine,
content classifiers, and storage backends to make intelligent routing decisions
for memory storage and retrieval.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import uuid4

from app.core.routing_engine import IntelligentRoutingEngine, RoutingContext, RoutingDecision
from app.core.classifiers import AdvancedContentClassifier, ContentAnalysis
from app.core.sqlite_storage import SQLiteConnectionPool
from app.models.routing import (
    RoutingRequest, RoutingResponse, RoutingPolicy, RoutingRule,
    RoutingBackend as ModelRoutingBackend, RoutingStrategy, RoutingPriority, RoutingStatistics,
    RoutingBatchRequest, RoutingBatchResponse, RoutingHealthCheck,
    RoutingConfiguration, DeduplicationResult, RoutingOptimization,
    RoutingContext as ModelRoutingContext, RoutingDecision as ModelRoutingDecision,
    create_default_routing_policy, create_performance_routing_policy,
    create_cost_optimized_routing_policy
)
from app.models.memory import MemoryItem, MemoryMetadata

logger = logging.getLogger(__name__)


class IntelligentRoutingService:
    """
    High-level routing service that orchestrates intelligent memory routing
    using the routing engine, content classifiers, and storage backends.
    """
    
    def __init__(self, connection_pool: SQLiteConnectionPool):
        self.connection_pool = connection_pool
        self.routing_engine = IntelligentRoutingEngine(connection_pool)
        self.content_classifier = AdvancedContentClassifier()
        self.policies: Dict[str, RoutingPolicy] = {}
        self.configuration = RoutingConfiguration()
        self.statistics: Dict[str, RoutingStatistics] = {}
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default routing policies."""
        try:
            # Create default policies
            default_policy = create_default_routing_policy()
            performance_policy = create_performance_routing_policy()
            cost_policy = create_cost_optimized_routing_policy()
            
            # Store policies
            self.policies[default_policy.id] = default_policy
            self.policies[performance_policy.id] = performance_policy
            self.policies[cost_policy.id] = cost_policy
            
            # Set default policy
            self.configuration.default_strategy = RoutingStrategy.HYBRID
            
            logger.info("Default routing policies initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize default policies: {e}")
    
    async def route_memory(self, request: RoutingRequest) -> RoutingResponse:
        """
        Route a memory item to the optimal storage backend.
        
        Args:
            request: Routing request with context and options
            
        Returns:
            RoutingResponse with routing decision and metadata
        """
        start_time = time.time()
        
        try:
            # Validate request
            if not request.context.content.strip():
                return RoutingResponse(
                    decision=self._create_fallback_decision(request.context),
                    success=False,
                    message="Empty content cannot be routed",
                    processing_time=time.time() - start_time
                )
            
            # Check for forced backend
            if request.force_backend:
                decision = self._create_forced_decision(request.context, request.force_backend)
                return RoutingResponse(
                    decision=decision,
                    success=True,
                    message=f"Backend forced to {request.force_backend.value}",
                    processing_time=time.time() - start_time
                )
            
            # Perform content analysis
            content_analysis = self.content_classifier.analyze_content(request.context.content)
            
            # Get routing recommendations
            recommendations = self.content_classifier.get_routing_recommendations(content_analysis)
            
            # Create routing context for engine
            memory_id = str(uuid4())  # Generate a unique memory ID
            routing_context = RoutingContext(
                memory_id=memory_id,
                agent_id=request.context.agent_id,
                project_id=request.context.project_id,
                memory_type=request.context.memory_type,
                content=request.context.content,
                priority=request.context.priority,
                tags=request.context.tags,
                metadata={
                    **request.context.metadata,
                    'content_analysis': content_analysis.__dict__ if hasattr(content_analysis, '__dict__') else str(content_analysis),
                    'recommendations': recommendations
                },
                source=request.context.source,
                timestamp=request.context.timestamp
            )
            
            # Make routing decision
            engine_decision = await self.routing_engine.make_routing_decision(routing_context)
            
            # Convert engine decision to Pydantic model
            from app.models.routing import RoutingDecision as ModelRoutingDecision, RoutingPriority as ModelRoutingPriority
            decision = ModelRoutingDecision(
                memory_id=engine_decision.memory_id,
                backend=ModelRoutingBackend(engine_decision.backend.value),
                reason=engine_decision.reason,
                confidence=engine_decision.confidence,
                priority=ModelRoutingPriority(engine_decision.priority.value),
                estimated_cost=engine_decision.estimated_cost,
                estimated_latency=engine_decision.estimated_latency,
                metadata=engine_decision.metadata,
                created_at=engine_decision.created_at
            )
            
            # Generate alternatives if requested
            alternatives = await self._generate_alternatives(routing_context, engine_decision)
            
            # Update statistics
            await self._update_routing_statistics(engine_decision, time.time() - start_time)
            
            return RoutingResponse(
                decision=decision,
                success=True,
                message="Routing decision completed successfully",
                processing_time=time.time() - start_time,
                alternatives=alternatives
            )
            
        except Exception as e:
            logger.error(f"Failed to route memory: {e}")
            return RoutingResponse(
                decision=self._create_fallback_decision(request.context),
                success=False,
                message=f"Routing failed: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    async def route_batch(self, batch_request: RoutingBatchRequest) -> RoutingBatchResponse:
        """
        Route multiple memory items in batch.
        
        Args:
            batch_request: Batch routing request
            
        Returns:
            RoutingBatchResponse with results for all requests
        """
        start_time = time.time()
        responses = []
        
        try:
            if batch_request.parallel:
                # Process requests in parallel
                tasks = [self.route_memory(req) for req in batch_request.requests]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle exceptions
                processed_responses = []
                for i, response in enumerate(responses):
                    if isinstance(response, Exception):
                        logger.error(f"Batch request {i} failed: {response}")
                        if batch_request.continue_on_error:
                            processed_responses.append(RoutingResponse(
                                decision=self._create_fallback_decision(batch_request.requests[i].context),
                                success=False,
                                message=f"Batch processing error: {str(response)}",
                                processing_time=0.0
                            ))
                        else:
                            raise response
                    else:
                        processed_responses.append(response)
                
                responses = processed_responses
            else:
                # Process requests sequentially
                for req in batch_request.requests:
                    try:
                        response = await self.route_memory(req)
                        responses.append(response)
                    except Exception as e:
                        logger.error(f"Sequential batch request failed: {e}")
                        if batch_request.continue_on_error:
                            responses.append(RoutingResponse(
                                decision=self._create_fallback_decision(req.context),
                                success=False,
                                message=f"Sequential processing error: {str(e)}",
                                processing_time=0.0
                            ))
                        else:
                            raise e
            
            # Calculate batch statistics
            total_requests = len(responses)
            successful_requests = sum(1 for r in responses if r.success)
            failed_requests = total_requests - successful_requests
            success_rate = successful_requests / total_requests if total_requests > 0 else 0.0
            
            return RoutingBatchResponse(
                batch_id=batch_request.batch_id,
                responses=responses,
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                processing_time=time.time() - start_time,
                success_rate=success_rate
            )
            
        except Exception as e:
            logger.error(f"Batch routing failed: {e}")
            return RoutingBatchResponse(
                batch_id=batch_request.batch_id,
                responses=responses,
                total_requests=len(batch_request.requests),
                successful_requests=0,
                failed_requests=len(batch_request.requests),
                processing_time=time.time() - start_time,
                success_rate=0.0
            )
    
    async def check_duplicates(self, content: str, threshold: float = None) -> DeduplicationResult:
        """
        Check for duplicate content across storage backends.
        
        Args:
            content: Content to check for duplicates
            threshold: Similarity threshold (uses default if None)
            
        Returns:
            DeduplicationResult with duplicate information
        """
        try:
            threshold = threshold or self.configuration.similarity_threshold
            
            # Use routing engine's duplicate checking
            duplicate_info = await self.routing_engine._check_duplicates(content)
            
            return DeduplicationResult(
                is_duplicate=duplicate_info['is_duplicate'],
                existing_memory_id=duplicate_info.get('existing_memory_id'),
                existing_backend=RoutingBackend(duplicate_info.get('existing_backend', 'sqlite')),
                similarity_score=duplicate_info.get('similarity_score', 0.0),
                confidence=0.9 if duplicate_info['is_duplicate'] else 0.1,
                metadata=duplicate_info
            )
            
        except Exception as e:
            logger.error(f"Failed to check duplicates: {e}")
            return DeduplicationResult(
                is_duplicate=False,
                similarity_score=0.0,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    async def get_routing_statistics(self, backend: Optional[ModelRoutingBackend] = None) -> Dict[str, Any]:
        """
        Get routing statistics and performance metrics.
        
        Args:
            backend: Optional backend filter
            
        Returns:
            Dictionary with routing statistics
        """
        try:
            # Get statistics from routing engine
            engine_stats = await self.routing_engine.get_routing_statistics()
            
            # Get cache statistics
            cache_stats = await self.routing_engine.get_cache_stats()
            
            # Combine statistics
            statistics = {
                'engine_statistics': engine_stats,
                'cache_statistics': cache_stats,
                'policy_count': len(self.policies),
                'configuration': self.configuration.dict(),
                'last_updated': datetime.utcnow().isoformat()
            }
            
            # Filter by backend if specified
            if backend:
                statistics = {
                    'backend': backend.value,
                    'statistics': engine_stats.get(backend.value, {}),
                    'cache_statistics': cache_stats,
                    'last_updated': datetime.utcnow().isoformat()
                }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Failed to get routing statistics: {e}")
            return {'error': str(e)}
    
    async def optimize_routing(self) -> List[RoutingOptimization]:
        """
        Analyze routing performance and suggest optimizations.
        
        Returns:
            List of routing optimization suggestions
        """
        try:
            # Get optimization suggestions from routing engine
            engine_optimizations = await self.routing_engine.optimize_routing_policies()
            
            optimizations = []
            for key, suggestion in engine_optimizations.items():
                optimization = RoutingOptimization(
                    backend=RoutingBackend.SQLITE,  # Default, could be parsed from key
                    issue=suggestion['issue'],
                    current_metric=suggestion.get('current_rate', suggestion.get('current_latency', 0.0)),
                    suggested_metric=suggestion.get('suggested_rate', suggestion.get('suggested_latency', 0.0)),
                    recommendation=suggestion['suggestion'],
                    priority=RoutingPriority.HIGH if 'critical' in suggestion['issue'].lower() else RoutingPriority.NORMAL,
                    estimated_impact=0.5  # Default impact score
                )
                optimizations.append(optimization)
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Failed to optimize routing: {e}")
            return []
    
    async def health_check(self) -> RoutingHealthCheck:
        """
        Perform comprehensive health check of routing system.
        
        Returns:
            RoutingHealthCheck with system status
        """
        try:
            health = RoutingHealthCheck(
                status="healthy",
                backends={},
                policies={},
                cache_status={},
                performance_metrics={},
                issues=[],
                recommendations=[]
            )
            
            # Check routing engine health
            try:
                engine_stats = await self.routing_engine.get_routing_statistics()
                health.performance_metrics['engine_statistics'] = engine_stats
            except Exception as e:
                health.issues.append(f"Routing engine health check failed: {e}")
                health.status = "degraded"
            
            # Check cache health
            try:
                cache_stats = await self.routing_engine.get_cache_stats()
                health.cache_status = cache_stats
            except Exception as e:
                health.issues.append(f"Cache health check failed: {e}")
                health.status = "degraded"
            
            # Check policies
            for policy_id, policy in self.policies.items():
                health.policies[policy_id] = {
                    'name': policy.name,
                    'enabled': policy.enabled,
                    'rule_count': len(policy.rules),
                    'enabled_rule_count': len(policy.get_enabled_rules())
                }
            
            # Check configuration
            health.performance_metrics['configuration'] = self.configuration.dict()
            
            # Generate recommendations
            if health.status == "healthy":
                health.recommendations.append("System is operating normally")
            else:
                health.recommendations.append("Review and resolve identified issues")
                health.recommendations.append("Consider running optimization analysis")
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return RoutingHealthCheck(
                status="unhealthy",
                issues=[f"Health check failed: {e}"],
                recommendations=["Restart routing service", "Check system logs"]
            )
    
    async def add_policy(self, policy: RoutingPolicy) -> bool:
        """
        Add a new routing policy.
        
        Args:
            policy: Routing policy to add
            
        Returns:
            True if policy was added successfully
        """
        try:
            self.policies[policy.id] = policy
            logger.info(f"Added routing policy: {policy.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add policy: {e}")
            return False
    
    async def remove_policy(self, policy_id: str) -> bool:
        """
        Remove a routing policy.
        
        Args:
            policy_id: ID of policy to remove
            
        Returns:
            True if policy was removed successfully
        """
        try:
            if policy_id in self.policies:
                del self.policies[policy_id]
                logger.info(f"Removed routing policy: {policy_id}")
                return True
            else:
                logger.warning(f"Policy not found: {policy_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove policy: {e}")
            return False
    
    async def update_configuration(self, config: RoutingConfiguration) -> bool:
        """
        Update routing configuration.
        
        Args:
            config: New configuration
            
        Returns:
            True if configuration was updated successfully
        """
        try:
            self.configuration = config
            logger.info("Routing configuration updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False
    
    async def clear_cache(self) -> bool:
        """
        Clear routing decision cache.
        
        Returns:
            True if cache was cleared successfully
        """
        try:
            await self.routing_engine.clear_cache()
            logger.info("Routing cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def _create_fallback_decision(self, context: ModelRoutingContext) -> ModelRoutingDecision:
        """Create a fallback routing decision."""
        from app.models.routing import RoutingDecision as ModelRoutingDecision, RoutingPriority as ModelRoutingPriority
        return ModelRoutingDecision(
            memory_id=str(uuid4()),
            backend=ModelRoutingBackend(self.configuration.default_backend),
            reason="Fallback routing due to error or empty content",
            confidence=0.1,
            priority=ModelRoutingPriority(context.priority),
            estimated_cost=0.0,
            estimated_latency=0.1,
            metadata={'fallback': True, 'context': context.model_dump()},
            created_at=datetime.utcnow()
        )
    
    def _create_forced_decision(self, context: ModelRoutingContext, backend: ModelRoutingBackend) -> ModelRoutingDecision:
        """Create a forced routing decision."""
        from app.models.routing import RoutingDecision as ModelRoutingDecision, RoutingPriority as ModelRoutingPriority
        return ModelRoutingDecision(
            memory_id=str(uuid4()),
            backend=backend,
            reason=f"Backend forced to {backend.value}",
            confidence=1.0,
            priority=ModelRoutingPriority(context.priority),
            estimated_cost=0.0,
            estimated_latency=0.01,
            metadata={'forced': True, 'context': context.model_dump()},
            created_at=datetime.utcnow()
        )
    
    async def _generate_alternatives(self, context: RoutingContext, primary_decision) -> List[ModelRoutingDecision]:
        """Generate alternative routing decisions."""
        try:
            from app.models.routing import RoutingDecision as ModelRoutingDecision, RoutingPriority as ModelRoutingPriority
            alternatives = []
            max_alternatives = self.configuration.max_alternatives
            
            # Get all available backends
            available_backends = [ModelRoutingBackend.SQLITE, ModelRoutingBackend.CIPHER, ModelRoutingBackend.WEAVIATE]
            
            # Remove primary backend from alternatives
            available_backends = [b for b in available_backends if b != ModelRoutingBackend(primary_decision.backend.value)]
            
            # Generate alternatives
            for i, backend in enumerate(available_backends[:max_alternatives]):
                alternative = ModelRoutingDecision(
                    memory_id=primary_decision.memory_id,
                    backend=backend,
                    reason=f"Alternative {i+1}: {backend.value} backend",
                    confidence=primary_decision.confidence * 0.8,  # Lower confidence for alternatives
                    priority=ModelRoutingPriority(primary_decision.priority.value),
                    estimated_cost=primary_decision.estimated_cost + 0.1,
                    estimated_latency=primary_decision.estimated_latency + 0.01,
                    metadata={'alternative': True, 'rank': i+1},
                    created_at=datetime.utcnow()
                )
                alternatives.append(alternative)
            
            return alternatives
            
        except Exception as e:
            logger.error(f"Failed to generate alternatives: {e}")
            return []
    
    async def _update_routing_statistics(self, decision: RoutingDecision, processing_time: float):
        """Update routing statistics with new decision data."""
        try:
            backend_key = decision.backend.value
            
            if backend_key not in self.statistics:
                self.statistics[backend_key] = RoutingStatistics(
                    backend=decision.backend,
                    total_requests=0,
                    successful_requests=0,
                    failed_requests=0,
                    average_latency=0.0,
                    average_cost=0.0,
                    success_rate=0.0
                )
            
            # Update statistics
            self.statistics[backend_key].update_statistics(
                success=True,  # Assume success if we got a decision
                latency=processing_time,
                cost=decision.estimated_cost
            )
            
        except Exception as e:
            logger.error(f"Failed to update routing statistics: {e}")


# Global routing service instance
_routing_service: Optional[IntelligentRoutingService] = None


async def get_routing_service(connection_pool: Optional[SQLiteConnectionPool] = None) -> IntelligentRoutingService:
    """
    Get the global routing service instance.
    
    Args:
        connection_pool: Optional connection pool (creates default if None)
        
    Returns:
        IntelligentRoutingService instance
    """
    global _routing_service
    
    if _routing_service is None:
        if connection_pool is None:
            # Create default connection pool
            from app.core.config import get_settings
            settings = get_settings()
            connection_pool = SQLiteConnectionPool(
                database_path=settings.database.url.replace("sqlite+aiosqlite:///", ""),
                max_connections=10
            )
            await connection_pool.initialize()
        
        _routing_service = IntelligentRoutingService(connection_pool)
        logger.info("Routing service initialized")
    
    return _routing_service


async def close_routing_service():
    """Close the global routing service."""
    global _routing_service
    
    if _routing_service is not None:
        try:
            await _routing_service.clear_cache()
            _routing_service = None
            logger.info("Routing service closed")
        except Exception as e:
            logger.error(f"Error closing routing service: {e}")

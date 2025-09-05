"""
Intelligent Memory Routing Engine with SQLite Backend.

This module implements the core routing logic that decides where to store memories
(Cipher vs Weaviate) and how to route retrieval requests based on content type,
project context, and intelligent algorithms.
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import uuid4
from dataclasses import dataclass
from enum import Enum

from app.core.sqlite_storage import SQLiteConnectionPool
from app.models.memory import MemoryItem, MemoryMetadata, MemoryRoute
from app.models.agent import Agent

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Available storage backends for routing decisions."""
    CIPHER = "cipher"
    WEAVIATE = "weaviate"
    SQLITE = "sqlite"
    HYBRID = "hybrid"


class ContentType(Enum):
    """Content type classifications for routing decisions."""
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


class RoutingPriority(Enum):
    """Routing priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class RoutingDecision:
    """Represents a routing decision with reasoning."""
    memory_id: str
    backend: StorageBackend
    reason: str
    confidence: float
    priority: RoutingPriority
    estimated_cost: float
    estimated_latency: float
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class RoutingContext:
    """Context information for routing decisions."""
    memory_id: str
    agent_id: str
    project_id: str
    memory_type: str
    content: str
    priority: int
    tags: List[str]
    metadata: Dict[str, Any]
    source: str
    timestamp: datetime


class ContentClassifier:
    """Classifies content type and characteristics for routing decisions."""
    
    def __init__(self):
        self.patterns = {
            ContentType.CONVERSATION: [
                r'\b(chat|conversation|discussion|talk|speak|say|tell|ask|question|answer)\b',
                r'\b(hi|hello|hey|thanks|thank you|please|sorry|excuse me)\b',
                r'\b(what|how|why|when|where|who|which)\b.*\?',
                r'\b(i think|i believe|i feel|in my opinion|from my perspective)\b'
            ],
            ContentType.KNOWLEDGE: [
                r'\b(knowledge|fact|information|data|research|study|analysis)\b',
                r'\b(according to|based on|research shows|studies indicate)\b',
                r'\b(definition|meaning|explanation|description|overview)\b',
                r'\b(learn|understand|comprehend|grasp|master)\b'
            ],
            ContentType.CODE: [
                r'```[\s\S]*?```',  # Code blocks
                r'\b(function|class|def|import|from|return|if|else|for|while)\b',
                r'\b(api|endpoint|database|sql|query|schema|table)\b',
                r'[{}();=<>!&|]',  # Code symbols
                r'\b(python|javascript|java|c\+\+|typescript|sql|html|css)\b'
            ],
            ContentType.PROCEDURE: [
                r'\b(step|steps|process|procedure|method|approach|technique)\b',
                r'\b(first|second|third|next|then|finally|lastly)\b',
                r'\b(how to|tutorial|guide|instructions|manual)\b',
                r'\b(1\.|2\.|3\.|4\.|5\.)',  # Numbered lists
                r'\b(do this|follow these|complete the following)\b'
            ],
            ContentType.DOCUMENT: [
                r'\b(document|file|report|paper|article|blog|post)\b',
                r'\b(title|heading|section|chapter|paragraph)\b',
                r'\b(author|published|created|modified|version)\b',
                r'\.(pdf|doc|docx|txt|md|html|xml|json)$'
            ]
        }
    
    def classify_content(self, content: str) -> Tuple[ContentType, float]:
        """
        Classify content type and return confidence score.
        
        Args:
            content: The content to classify
            
        Returns:
            Tuple of (content_type, confidence_score)
        """
        if not content or not content.strip():
            return ContentType.UNKNOWN, 0.0
        
        content_lower = content.lower()
        scores = {}
        
        # Calculate scores for each content type
        for content_type, patterns in self.patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
                score += matches * 0.1  # Each match adds 0.1 to the score
            
            # Normalize score based on content length
            normalized_score = min(score / max(len(content.split()), 1), 1.0)
            scores[content_type] = normalized_score
        
        # Find the best match
        if not scores or max(scores.values()) == 0:
            return ContentType.UNKNOWN, 0.0
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        # Boost confidence for obvious patterns
        if confidence > 0.3:
            confidence = min(confidence * 1.5, 1.0)
        
        return best_type, confidence
    
    def extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """Extract important keywords from content."""
        # Simple keyword extraction - in production, use NLP libraries
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Count word frequency
        word_counts = {}
        for word in words:
            if word not in stop_words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:max_keywords]]


class RoutingPolicy:
    """Defines routing policies and rules for intelligent routing decisions."""
    
    def __init__(self):
        self.policies = {
            # Content type based routing
            ContentType.CONVERSATION: {
                'preferred_backend': StorageBackend.CIPHER,
                'reason': 'Conversational content is better suited for Cipher',
                'confidence_boost': 0.2
            },
            ContentType.KNOWLEDGE: {
                'preferred_backend': StorageBackend.WEAVIATE,
                'reason': 'Knowledge content benefits from semantic search',
                'confidence_boost': 0.3
            },
            ContentType.CODE: {
                'preferred_backend': StorageBackend.SQLITE,
                'reason': 'Code content is structured and benefits from local storage',
                'confidence_boost': 0.4
            },
            ContentType.PROCEDURE: {
                'preferred_backend': StorageBackend.CIPHER,
                'reason': 'Procedural content is conversational in nature',
                'confidence_boost': 0.2
            },
            ContentType.DOCUMENT: {
                'preferred_backend': StorageBackend.WEAVIATE,
                'reason': 'Documents benefit from semantic search and retrieval',
                'confidence_boost': 0.3
            }
        }
        
        # Size-based routing rules
        self.size_thresholds = {
            'small': 1024,      # 1KB
            'medium': 10240,    # 10KB
            'large': 102400     # 100KB
        }
        
        # Priority-based routing rules
        self.priority_rules = {
            RoutingPriority.CRITICAL: {
                'preferred_backend': StorageBackend.SQLITE,
                'reason': 'Critical content needs fast local access',
                'latency_penalty': 0.1
            },
            RoutingPriority.HIGH: {
                'preferred_backend': StorageBackend.SQLITE,
                'reason': 'High priority content benefits from local storage',
                'latency_penalty': 0.05
            }
        }
    
    def get_routing_decision(
        self, 
        content_type: ContentType, 
        content_size: int, 
        priority: RoutingPriority,
        context: RoutingContext
    ) -> Dict[str, Any]:
        """
        Get routing decision based on policies.
        
        Args:
            content_type: Classified content type
            content_size: Size of content in bytes
            priority: Routing priority
            context: Routing context
            
        Returns:
            Dictionary with routing decision details
        """
        decision = {
            'backend': StorageBackend.SQLITE,  # Default fallback
            'reason': 'Default routing to SQLite',
            'confidence': 0.5,
            'estimated_cost': 0.0,
            'estimated_latency': 0.01
        }
        
        # Apply content type policy
        if content_type in self.policies:
            policy = self.policies[content_type]
            decision['backend'] = policy['preferred_backend']
            decision['reason'] = policy['reason']
            decision['confidence'] += policy['confidence_boost']
        
        # Apply priority rules
        if priority in self.priority_rules:
            priority_rule = self.priority_rules[priority]
            decision['backend'] = priority_rule['preferred_backend']
            decision['reason'] = f"{decision['reason']} + {priority_rule['reason']}"
            decision['estimated_latency'] -= priority_rule['latency_penalty']
        
        # Apply size-based adjustments
        if content_size > self.size_thresholds['large']:
            # Large content might benefit from distributed storage
            if decision['backend'] == StorageBackend.SQLITE:
                decision['backend'] = StorageBackend.WEAVIATE
                decision['reason'] += " (large content optimized for distributed storage)"
                decision['estimated_cost'] += 0.1
        
        # Ensure confidence is within bounds
        decision['confidence'] = min(max(decision['confidence'], 0.0), 1.0)
        
        return decision


class IntelligentRoutingEngine:
    """
    Core intelligent routing engine that makes routing decisions based on
    content analysis, context, and performance optimization.
    """
    
    def __init__(self, connection_pool: SQLiteConnectionPool):
        self.connection_pool = connection_pool
        self.classifier = ContentClassifier()
        self.policy = RoutingPolicy()
        self.routing_cache = {}
        self.performance_metrics = {}
        self._initialized = False
    
    async def _initialize_routing_tables(self):
        """Initialize routing-specific tables in SQLite."""
        try:
            conn = await self.connection_pool.get_connection()
            try:
                # Create routing decisions table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS routing_decisions (
                        id TEXT PRIMARY KEY,
                        memory_id TEXT NOT NULL,
                        backend TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        priority INTEGER NOT NULL,
                        estimated_cost REAL NOT NULL,
                        estimated_latency REAL NOT NULL,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (memory_id) REFERENCES memory_items(id) ON DELETE CASCADE
                    )
                """)
                
                # Create routing statistics table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS routing_statistics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        backend TEXT NOT NULL,
                        operation_type TEXT NOT NULL,
                        success_count INTEGER DEFAULT 0,
                        failure_count INTEGER DEFAULT 0,
                        total_latency REAL DEFAULT 0.0,
                        total_cost REAL DEFAULT 0.0,
                        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create deduplication table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_deduplication (
                        id TEXT PRIMARY KEY,
                        content_hash TEXT NOT NULL UNIQUE,
                        memory_id TEXT NOT NULL,
                        backend TEXT NOT NULL,
                        similarity_score REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (memory_id) REFERENCES memory_items(id) ON DELETE CASCADE
                    )
                """)
                
                # Create indexes for performance
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_routing_decisions_memory_id ON routing_decisions(memory_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_routing_decisions_backend ON routing_decisions(backend)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_routing_decisions_created_at ON routing_decisions(created_at)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_deduplication_content_hash ON memory_deduplication(content_hash)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_deduplication_similarity ON memory_deduplication(similarity_score)")
                
                await conn.commit()
                logger.info("Routing tables initialized successfully")
                
            finally:
                await self.connection_pool.return_connection(conn)
                
        except Exception as e:
            logger.error(f"Failed to initialize routing tables: {e}")
            raise
    
    async def make_routing_decision(self, context: RoutingContext) -> RoutingDecision:
        """Make an intelligent routing decision for a memory item."""
        # Initialize if not already done
        if not self._initialized:
            await self._initialize_routing_tables()
            self._initialized = True
        """
        Make an intelligent routing decision for a memory item.
        
        Args:
            context: Routing context with memory information
            
        Returns:
            RoutingDecision object with routing details
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(context)
            if cache_key in self.routing_cache:
                cached_decision = self.routing_cache[cache_key]
                if time.time() - cached_decision['timestamp'] < 300:  # 5 minute cache
                    logger.debug(f"Using cached routing decision for {context.memory_id}")
                    return RoutingDecision(**cached_decision['decision'])
            
            # Classify content
            content_type, classification_confidence = self.classifier.classify_content(context.content)
            
            # Get content size
            content_size = len(context.content.encode('utf-8'))
            
            # Determine priority
            priority = RoutingPriority(context.priority) if context.priority <= 4 else RoutingPriority.NORMAL
            
            # Check for duplicates
            duplicate_info = await self._check_duplicates(context.content)
            
            # Make routing decision based on policies
            policy_decision = self.policy.get_routing_decision(
                content_type, content_size, priority, context
            )
            
            # Adjust for duplicates
            if duplicate_info['is_duplicate']:
                policy_decision['backend'] = StorageBackend(duplicate_info['existing_backend'])
                policy_decision['reason'] = f"Duplicate detected: {policy_decision['reason']}"
                policy_decision['confidence'] += 0.2
            
            # Create routing decision
            decision = RoutingDecision(
                memory_id=context.memory_id,
                backend=policy_decision['backend'],
                reason=policy_decision['reason'],
                confidence=policy_decision['confidence'],
                priority=priority,
                estimated_cost=policy_decision['estimated_cost'],
                estimated_latency=policy_decision['estimated_latency'],
                metadata={
                    'content_type': content_type.value,
                    'classification_confidence': classification_confidence,
                    'content_size': content_size,
                    'duplicate_info': duplicate_info,
                    'keywords': self.classifier.extract_keywords(context.content),
                    'processing_time': time.time() - start_time
                },
                created_at=datetime.utcnow()
            )
            
            # Store decision in database
            await self._store_routing_decision(decision)
            
            # Cache the decision
            self.routing_cache[cache_key] = {
                'decision': decision.__dict__,
                'timestamp': time.time()
            }
            
            # Update performance metrics
            await self._update_performance_metrics(decision, time.time() - start_time)
            
            logger.info(f"Routing decision made for {context.memory_id}: {decision.backend.value} "
                       f"(confidence: {decision.confidence:.2f}, reason: {decision.reason})")
            
            return decision
            
        except Exception as e:
            logger.error(f"Failed to make routing decision: {e}")
            # Return fallback decision
            return RoutingDecision(
                memory_id=context.memory_id,
                backend=StorageBackend.SQLITE,
                reason=f"Fallback routing due to error: {str(e)}",
                confidence=0.1,
                priority=RoutingPriority.NORMAL,
                estimated_cost=0.0,
                estimated_latency=0.1,
                metadata={'error': str(e)},
                created_at=datetime.utcnow()
            )
    
    async def _check_duplicates(self, content: str) -> Dict[str, Any]:
        """Check for duplicate content across storage backends."""
        try:
            content_hash = self._generate_content_hash(content)
            
            conn = await self.connection_pool.get_connection()
            try:
                async with conn.execute("""
                    SELECT memory_id, backend, similarity_score, created_at
                    FROM memory_deduplication 
                    WHERE content_hash = ?
                    ORDER BY similarity_score DESC, created_at DESC
                    LIMIT 1
                """, (content_hash,)) as cursor:
                    row = await cursor.fetchone()
                
                if row:
                    return {
                        'is_duplicate': True,
                        'existing_memory_id': row[0],
                        'existing_backend': row[1],
                        'similarity_score': row[2],
                        'created_at': row[3]
                    }
                
                return {
                    'is_duplicate': False,
                    'similarity_score': 0.0
                }
                
            finally:
                await self.connection_pool.return_connection(conn)
                
        except Exception as e:
            logger.error(f"Failed to check duplicates: {e}")
            return {'is_duplicate': False, 'similarity_score': 0.0}
    
    async def _store_routing_decision(self, decision: RoutingDecision):
        """Store routing decision in database."""
        try:
            conn = await self.connection_pool.get_connection()
            try:
                await conn.execute("""
                    INSERT INTO routing_decisions 
                    (id, memory_id, backend, reason, confidence, priority, 
                     estimated_cost, estimated_latency, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid4()),
                    decision.memory_id,
                    decision.backend.value,
                    decision.reason,
                    decision.confidence,
                    decision.priority.value,
                    decision.estimated_cost,
                    decision.estimated_latency,
                    json.dumps(decision.metadata),
                    decision.created_at.isoformat()
                ))
                await conn.commit()
                
            finally:
                await self.connection_pool.return_connection(conn)
                
        except Exception as e:
            logger.error(f"Failed to store routing decision: {e}")
    
    async def _update_performance_metrics(self, decision: RoutingDecision, processing_time: float):
        """Update routing performance metrics."""
        try:
            conn = await self.connection_pool.get_connection()
            try:
                # Update or insert performance metrics
                await conn.execute("""
                    INSERT OR REPLACE INTO routing_statistics 
                    (backend, operation_type, success_count, total_latency, total_cost, recorded_at)
                    VALUES (?, ?, 1, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    decision.backend.value,
                    'routing_decision',
                    processing_time,
                    decision.estimated_cost
                ))
                await conn.commit()
                
            finally:
                await self.connection_pool.return_connection(conn)
                
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")
    
    def _generate_cache_key(self, context: RoutingContext) -> str:
        """Generate cache key for routing context."""
        key_data = {
            'agent_id': context.agent_id,
            'project_id': context.project_id,
            'memory_type': context.memory_type,
            'content_hash': self._generate_content_hash(context.content),
            'priority': context.priority
        }
        return json.dumps(key_data, sort_keys=True)
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        import hashlib
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def get_routing_statistics(self, backend: Optional[StorageBackend] = None) -> Dict[str, Any]:
        """Get routing statistics for analysis and optimization."""
        try:
            conn = await self.connection_pool.get_connection()
            try:
                if backend:
                    async with conn.execute("""
                        SELECT backend, operation_type, 
                               SUM(success_count) as total_success,
                               SUM(failure_count) as total_failure,
                               AVG(total_latency) as avg_latency,
                               AVG(total_cost) as avg_cost,
                               COUNT(*) as decision_count
                        FROM routing_statistics 
                        WHERE backend = ?
                        GROUP BY backend, operation_type
                    """, (backend.value,)) as cursor:
                        rows = await cursor.fetchall()
                else:
                    async with conn.execute("""
                        SELECT backend, operation_type, 
                               SUM(success_count) as total_success,
                               SUM(failure_count) as total_failure,
                               AVG(total_latency) as avg_latency,
                               AVG(total_cost) as avg_cost,
                               COUNT(*) as decision_count
                        FROM routing_statistics 
                        GROUP BY backend, operation_type
                    """) as cursor:
                        rows = await cursor.fetchall()
                
                statistics = {}
                for row in rows:
                    backend_name = row[0]
                    if backend_name not in statistics:
                        statistics[backend_name] = {}
                    
                    statistics[backend_name][row[1]] = {
                        'total_success': row[2] or 0,
                        'total_failure': row[3] or 0,
                        'avg_latency': row[4] or 0.0,
                        'avg_cost': row[5] or 0.0,
                        'decision_count': row[6] or 0
                    }
                
                return statistics
                
            finally:
                await self.connection_pool.return_connection(conn)
                
        except Exception as e:
            logger.error(f"Failed to get routing statistics: {e}")
            return {}
    
    async def optimize_routing_policies(self) -> Dict[str, Any]:
        """Analyze performance data and suggest routing policy optimizations."""
        try:
            statistics = await self.get_routing_statistics()
            optimizations = {}
            
            for backend, operations in statistics.items():
                for operation, metrics in operations.items():
                    if metrics['decision_count'] > 10:  # Only analyze if we have enough data
                        success_rate = metrics['total_success'] / (metrics['total_success'] + metrics['total_failure'])
                        
                        if success_rate < 0.8:  # Low success rate
                            optimizations[f"{backend}_{operation}"] = {
                                'issue': 'Low success rate',
                                'current_rate': success_rate,
                                'suggestion': 'Consider adjusting routing criteria or backend configuration'
                            }
                        
                        if metrics['avg_latency'] > 1.0:  # High latency
                            optimizations[f"{backend}_{operation}_latency"] = {
                                'issue': 'High latency',
                                'current_latency': metrics['avg_latency'],
                                'suggestion': 'Consider caching or backend optimization'
                            }
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Failed to optimize routing policies: {e}")
            return {}
    
    async def clear_cache(self):
        """Clear routing decision cache."""
        self.routing_cache.clear()
        logger.info("Routing cache cleared")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.routing_cache),
            'cache_keys': list(self.routing_cache.keys())[:10]  # First 10 keys
        }

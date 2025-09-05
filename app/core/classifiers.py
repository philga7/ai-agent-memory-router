"""
Content Type and Context Classifiers for Intelligent Memory Routing.

This module provides advanced content classification capabilities using local
metadata analysis, pattern matching, and machine learning techniques to
determine optimal storage routing for different types of content.
"""

import re
import json
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ContentComplexity(Enum):
    """Content complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class ContentDomain(Enum):
    """Content domain classifications."""
    TECHNICAL = "technical"
    BUSINESS = "business"
    CREATIVE = "creative"
    EDUCATIONAL = "educational"
    PERSONAL = "personal"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ContentUrgency(Enum):
    """Content urgency levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ContentAnalysis:
    """Results of content analysis."""
    content_type: str
    domain: ContentDomain
    complexity: ContentComplexity
    urgency: ContentUrgency
    confidence: float
    keywords: List[str]
    entities: List[str]
    sentiment: Optional[str]
    language: str
    size_category: str
    structure_score: float
    metadata: Dict[str, Any]


class PatternBasedClassifier:
    """Pattern-based content classifier using regex and rule-based analysis."""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.keywords = self._initialize_keywords()
        self.stop_words = self._load_stop_words()
    
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize classification patterns."""
        return {
            'conversation': [
                r'\b(hi|hello|hey|thanks|thank you|please|sorry|excuse me)\b',
                r'\b(what|how|why|when|where|who|which)\b.*\?',
                r'\b(i think|i believe|i feel|in my opinion|from my perspective)\b',
                r'\b(chat|conversation|discussion|talk|speak|say|tell|ask|answer)\b',
                r'\b(you|your|yours|we|us|our|ours)\b',
                r'\b(me|my|mine|i|i\'m|i\'ll|i\'ve|i\'d)\b'
            ],
            'knowledge': [
                r'\b(knowledge|fact|information|data|research|study|analysis)\b',
                r'\b(according to|based on|research shows|studies indicate)\b',
                r'\b(definition|meaning|explanation|description|overview)\b',
                r'\b(learn|understand|comprehend|grasp|master|teach)\b',
                r'\b(concept|principle|theory|hypothesis|theorem)\b',
                r'\b(evidence|proof|demonstration|example|illustration)\b'
            ],
            'code': [
                r'```[\s\S]*?```',  # Code blocks
                r'\b(function|class|def|import|from|return|if|else|for|while)\b',
                r'\b(api|endpoint|database|sql|query|schema|table)\b',
                r'[{}();=<>!&|]',  # Code symbols
                r'\b(python|javascript|java|c\+\+|typescript|sql|html|css)\b',
                r'\b(algorithm|data structure|programming|software|development)\b',
                r'\b(debug|test|unit test|integration|deployment)\b'
            ],
            'procedure': [
                r'\b(step|steps|process|procedure|method|approach|technique)\b',
                r'\b(first|second|third|next|then|finally|lastly|initially)\b',
                r'\b(how to|tutorial|guide|instructions|manual|walkthrough)\b',
                r'\b(1\.|2\.|3\.|4\.|5\.)',  # Numbered lists
                r'\b(do this|follow these|complete the following|perform)\b',
                r'\b(setup|install|configure|setup|initialize|prepare)\b'
            ],
            'document': [
                r'\b(document|file|report|paper|article|blog|post|memo)\b',
                r'\b(title|heading|section|chapter|paragraph|summary)\b',
                r'\b(author|published|created|modified|version|revision)\b',
                r'\.(pdf|doc|docx|txt|md|html|xml|json|yaml|yml)$',
                r'\b(abstract|introduction|conclusion|references|bibliography)\b',
                r'\b(table of contents|index|appendix|footnote)\b'
            ],
            'technical': [
                r'\b(architecture|design|system|component|module|interface)\b',
                r'\b(performance|optimization|scalability|reliability)\b',
                r'\b(security|authentication|authorization|encryption)\b',
                r'\b(monitoring|logging|metrics|analytics|dashboard)\b',
                r'\b(infrastructure|deployment|container|kubernetes|docker)\b',
                r'\b(api|rest|graphql|microservice|service mesh)\b'
            ],
            'business': [
                r'\b(business|strategy|plan|goal|objective|target)\b',
                r'\b(revenue|profit|cost|budget|investment|roi)\b',
                r'\b(market|customer|client|user|stakeholder)\b',
                r'\b(product|service|solution|offering|feature)\b',
                r'\b(team|organization|company|department|division)\b',
                r'\b(meeting|presentation|proposal|contract|agreement)\b'
            ],
            'creative': [
                r'\b(creative|artistic|design|visual|aesthetic|beautiful)\b',
                r'\b(idea|concept|inspiration|innovation|imagination)\b',
                r'\b(story|narrative|plot|character|theme|setting)\b',
                r'\b(music|sound|audio|video|image|photo|picture)\b',
                r'\b(color|style|mood|tone|atmosphere|feeling)\b',
                r'\b(poetry|poem|lyrics|song|verse|rhyme)\b'
            ]
        }
    
    def _initialize_keywords(self) -> Dict[str, List[str]]:
        """Initialize domain-specific keywords."""
        return {
            'technical': [
                'algorithm', 'database', 'server', 'client', 'protocol', 'framework',
                'library', 'dependency', 'configuration', 'environment', 'deployment',
                'testing', 'debugging', 'optimization', 'performance', 'security'
            ],
            'business': [
                'strategy', 'marketing', 'sales', 'finance', 'operations', 'management',
                'leadership', 'team', 'project', 'budget', 'revenue', 'profit',
                'customer', 'market', 'competition', 'growth', 'expansion'
            ],
            'educational': [
                'learning', 'teaching', 'education', 'course', 'lesson', 'tutorial',
                'student', 'teacher', 'curriculum', 'assessment', 'evaluation',
                'knowledge', 'skill', 'competency', 'certification', 'degree'
            ],
            'creative': [
                'design', 'art', 'creative', 'innovation', 'inspiration', 'aesthetic',
                'visual', 'artistic', 'imagination', 'originality', 'expression',
                'style', 'tone', 'mood', 'atmosphere', 'feeling'
            ]
        }
    
    def _load_stop_words(self) -> Set[str]:
        """Load common stop words for filtering."""
        return {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'a', 'an'
        }
    
    def classify_content_type(self, content: str) -> Tuple[str, float]:
        """
        Classify content type using pattern matching.
        
        Args:
            content: The content to classify
            
        Returns:
            Tuple of (content_type, confidence_score)
        """
        if not content or not content.strip():
            return 'unknown', 0.0
        
        content_lower = content.lower()
        scores = {}
        
        # Calculate scores for each content type
        for content_type, patterns in self.patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
                score += matches * 0.1  # Each match adds 0.1 to the score
            
            # Normalize score based on content length
            word_count = len(content.split())
            if word_count > 0:
                normalized_score = min(score / word_count, 1.0)
                scores[content_type] = normalized_score
        
        # Find the best match
        if not scores or max(scores.values()) == 0:
            return 'unknown', 0.0
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        # Boost confidence for obvious patterns
        if confidence > 0.3:
            confidence = min(confidence * 1.5, 1.0)
        
        return best_type, confidence
    
    def classify_domain(self, content: str) -> Tuple[ContentDomain, float]:
        """Classify content domain based on keywords and patterns."""
        content_lower = content.lower()
        domain_scores = {}
        
        for domain, keywords in self.keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in content_lower:
                    score += 1.0
            
            # Normalize by content length
            word_count = len(content.split())
            if word_count > 0:
                domain_scores[domain] = score / word_count
        
        if not domain_scores or max(domain_scores.values()) == 0:
            return ContentDomain.UNKNOWN, 0.0
        
        best_domain = max(domain_scores, key=domain_scores.get)
        confidence = domain_scores[best_domain]
        
        return ContentDomain(best_domain), confidence


class ContentStructureAnalyzer:
    """Analyzes content structure and organization."""
    
    def __init__(self):
        self.structure_patterns = {
            'list': r'^\s*[-*+]\s+',  # Bullet points
            'numbered_list': r'^\s*\d+\.\s+',  # Numbered lists
            'heading': r'^#{1,6}\s+',  # Markdown headings
            'code_block': r'```[\s\S]*?```',  # Code blocks
            'table': r'\|.*\|',  # Tables
            'link': r'\[.*?\]\(.*?\)',  # Links
            'emphasis': r'\*.*?\*|_.*?_',  # Emphasis
            'strong': r'\*\*.*?\*\*|__.*?__'  # Strong emphasis
        }
    
    def analyze_structure(self, content: str) -> Dict[str, Any]:
        """
        Analyze content structure and organization.
        
        Args:
            content: The content to analyze
            
        Returns:
            Dictionary with structure analysis results
        """
        lines = content.split('\n')
        structure_score = 0.0
        structure_elements = {}
        
        # Count different structure elements
        for element_type, pattern in self.structure_patterns.items():
            matches = len(re.findall(pattern, content, re.MULTILINE))
            structure_elements[element_type] = matches
            
            # Add to structure score based on element type
            if element_type in ['heading', 'list', 'numbered_list']:
                structure_score += matches * 0.1
            elif element_type in ['code_block', 'table']:
                structure_score += matches * 0.2
        
        # Analyze paragraph structure
        paragraphs = [line.strip() for line in lines if line.strip()]
        avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / max(len(paragraphs), 1)
        
        # Determine structure quality
        if structure_score > 0.5:
            structure_quality = 'well_structured'
        elif structure_score > 0.2:
            structure_quality = 'moderately_structured'
        else:
            structure_quality = 'unstructured'
        
        return {
            'structure_score': min(structure_score, 1.0),
            'structure_quality': structure_quality,
            'structure_elements': structure_elements,
            'paragraph_count': len(paragraphs),
            'avg_paragraph_length': avg_paragraph_length,
            'line_count': len(lines),
            'word_count': len(content.split())
        }


class ContentComplexityAnalyzer:
    """Analyzes content complexity and difficulty level."""
    
    def __init__(self):
        self.complexity_indicators = {
            'simple': [
                'basic', 'simple', 'easy', 'beginner', 'introductory', 'overview',
                'summary', 'brief', 'short', 'quick', 'fast', 'straightforward'
            ],
            'moderate': [
                'intermediate', 'moderate', 'standard', 'typical', 'common',
                'regular', 'normal', 'average', 'medium', 'balanced'
            ],
            'complex': [
                'advanced', 'complex', 'sophisticated', 'detailed', 'comprehensive',
                'thorough', 'in-depth', 'extensive', 'elaborate', 'intricate'
            ],
            'expert': [
                'expert', 'professional', 'specialist', 'technical', 'scientific',
                'research', 'academic', 'theoretical', 'cutting-edge', 'state-of-the-art'
            ]
        }
        
        self.technical_terms = [
            'algorithm', 'architecture', 'implementation', 'optimization',
            'configuration', 'deployment', 'infrastructure', 'scalability',
            'performance', 'security', 'authentication', 'authorization'
        ]
    
    def analyze_complexity(self, content: str) -> Tuple[ContentComplexity, float]:
        """
        Analyze content complexity level.
        
        Args:
            content: The content to analyze
            
        Returns:
            Tuple of (complexity_level, confidence_score)
        """
        content_lower = content.lower()
        complexity_scores = {}
        
        # Score based on complexity indicators
        for complexity, indicators in self.complexity_indicators.items():
            score = 0.0
            for indicator in indicators:
                if indicator in content_lower:
                    score += 1.0
            complexity_scores[complexity] = score
        
        # Adjust scores based on technical terms
        technical_term_count = sum(1 for term in self.technical_terms if term in content_lower)
        if technical_term_count > 5:
            complexity_scores['expert'] += 2.0
        elif technical_term_count > 2:
            complexity_scores['complex'] += 1.0
        
        # Adjust scores based on content length and structure
        word_count = len(content.split())
        if word_count > 1000:
            complexity_scores['complex'] += 1.0
            complexity_scores['expert'] += 0.5
        elif word_count > 500:
            complexity_scores['moderate'] += 1.0
            complexity_scores['complex'] += 0.5
        
        # Find the best match
        if not complexity_scores or max(complexity_scores.values()) == 0:
            return ContentComplexity.MODERATE, 0.5
        
        best_complexity = max(complexity_scores, key=complexity_scores.get)
        confidence = min(complexity_scores[best_complexity] / 5.0, 1.0)  # Normalize
        
        return ContentComplexity(best_complexity), confidence


class ContentUrgencyAnalyzer:
    """Analyzes content urgency and priority indicators."""
    
    def __init__(self):
        self.urgency_indicators = {
            'critical': [
                'urgent', 'critical', 'emergency', 'asap', 'immediately', 'now',
                'break', 'broken', 'error', 'failure', 'down', 'outage',
                'security', 'breach', 'vulnerability', 'threat'
            ],
            'high': [
                'important', 'priority', 'high', 'soon', 'quickly', 'fast',
                'deadline', 'due', 'required', 'necessary', 'essential'
            ],
            'normal': [
                'normal', 'standard', 'regular', 'routine', 'typical',
                'scheduled', 'planned', 'expected', 'usual'
            ],
            'low': [
                'low', 'minor', 'optional', 'nice to have', 'when possible',
                'eventually', 'later', 'someday', 'future', 'backlog'
            ]
        }
    
    def analyze_urgency(self, content: str) -> Tuple[ContentUrgency, float]:
        """
        Analyze content urgency level.
        
        Args:
            content: The content to analyze
            
        Returns:
            Tuple of (urgency_level, confidence_score)
        """
        content_lower = content.lower()
        urgency_scores = {}
        
        for urgency, indicators in self.urgency_indicators.items():
            score = 0.0
            for indicator in indicators:
                if indicator in content_lower:
                    score += 1.0
            urgency_scores[urgency] = score
        
        # Find the best match
        if not urgency_scores or max(urgency_scores.values()) == 0:
            return ContentUrgency.NORMAL, 0.5
        
        best_urgency = max(urgency_scores, key=urgency_scores.get)
        confidence = min(urgency_scores[best_urgency] / 3.0, 1.0)  # Normalize
        
        return ContentUrgency(best_urgency), confidence


class EntityExtractor:
    """Extracts named entities and important terms from content."""
    
    def __init__(self):
        self.entity_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'time': r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
            'version': r'\bv?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?\b',
            'file_path': r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*',
            'unix_path': r'/(?:[^/\s]+/)*[^/\s]*',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'hashtag': r'#\w+',
            'mention': r'@\w+'
        }
        
        self.technical_entities = [
            'API', 'REST', 'GraphQL', 'JSON', 'XML', 'YAML', 'SQL', 'NoSQL',
            'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP', 'Linux', 'Windows',
            'Python', 'JavaScript', 'Java', 'C++', 'TypeScript', 'React',
            'Node.js', 'Django', 'Flask', 'FastAPI', 'PostgreSQL', 'MongoDB',
            'Redis', 'Elasticsearch', 'Kafka', 'RabbitMQ'
        ]
    
    def extract_entities(self, content: str) -> Dict[str, List[str]]:
        """
        Extract named entities from content.
        
        Args:
            content: The content to analyze
            
        Returns:
            Dictionary with extracted entities by type
        """
        entities = {}
        
        # Extract using regex patterns
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))  # Remove duplicates
        
        # Extract technical entities
        content_upper = content.upper()
        technical_found = []
        for entity in self.technical_entities:
            if entity.upper() in content_upper:
                technical_found.append(entity)
        
        if technical_found:
            entities['technical'] = technical_found
        
        # Extract potential proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        if proper_nouns:
            # Filter out common words
            common_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An'}
            proper_nouns = [noun for noun in proper_nouns if noun not in common_words]
            if proper_nouns:
                entities['proper_nouns'] = list(set(proper_nouns))
        
        return entities


class KeywordExtractor:
    """Extracts important keywords and phrases from content."""
    
    def __init__(self):
        self.stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'a', 'an'
        }
    
    def extract_keywords(self, content: str, max_keywords: int = 20) -> List[str]:
        """
        Extract important keywords from content.
        
        Args:
            content: The content to analyze
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of important keywords
        """
        if not content:
            return []
        
        # Extract words (3+ characters, alphanumeric)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Count word frequency
        word_counts = {}
        for word in words:
            if word not in self.stop_words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:max_keywords]]


class AdvancedContentClassifier:
    """
    Advanced content classifier that combines multiple analysis techniques
    to provide comprehensive content classification.
    """
    
    def __init__(self):
        self.pattern_classifier = PatternBasedClassifier()
        self.structure_analyzer = ContentStructureAnalyzer()
        self.complexity_analyzer = ContentComplexityAnalyzer()
        self.urgency_analyzer = ContentUrgencyAnalyzer()
        self.entity_extractor = EntityExtractor()
        self.keyword_extractor = KeywordExtractor()
    
    def analyze_content(self, content: str) -> ContentAnalysis:
        """
        Perform comprehensive content analysis.
        
        Args:
            content: The content to analyze
            
        Returns:
            ContentAnalysis object with comprehensive results
        """
        if not content or not content.strip():
            return ContentAnalysis(
                content_type='unknown',
                domain=ContentDomain.UNKNOWN,
                complexity=ContentComplexity.SIMPLE,
                urgency=ContentUrgency.NORMAL,
                confidence=0.0,
                keywords=[],
                entities=[],
                sentiment=None,
                language='en',
                size_category='small',
                structure_score=0.0,
                metadata={}
            )
        
        # Classify content type
        content_type, type_confidence = self.pattern_classifier.classify_content_type(content)
        
        # Classify domain
        domain, domain_confidence = self.pattern_classifier.classify_domain(content)
        
        # Analyze complexity
        complexity, complexity_confidence = self.complexity_analyzer.analyze_complexity(content)
        
        # Analyze urgency
        urgency, urgency_confidence = self.urgency_analyzer.analyze_urgency(content)
        
        # Analyze structure
        structure_analysis = self.structure_analyzer.analyze_structure(content)
        
        # Extract entities
        entities_dict = self.entity_extractor.extract_entities(content)
        entities = []
        for entity_type, entity_list in entities_dict.items():
            entities.extend([f"{entity_type}:{entity}" for entity in entity_list])
        
        # Extract keywords
        keywords = self.keyword_extractor.extract_keywords(content)
        
        # Determine content size category
        content_size = len(content.encode('utf-8'))
        if content_size < 1024:
            size_category = 'small'
        elif content_size < 10240:
            size_category = 'medium'
        elif content_size < 102400:
            size_category = 'large'
        else:
            size_category = 'very_large'
        
        # Calculate overall confidence
        overall_confidence = (type_confidence + domain_confidence + 
                            complexity_confidence + urgency_confidence) / 4.0
        
        # Create metadata
        metadata = {
            'content_size_bytes': content_size,
            'word_count': len(content.split()),
            'line_count': len(content.split('\n')),
            'structure_analysis': structure_analysis,
            'entities_by_type': entities_dict,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
        
        return ContentAnalysis(
            content_type=content_type,
            domain=domain,
            complexity=complexity,
            urgency=urgency,
            confidence=overall_confidence,
            keywords=keywords,
            entities=entities,
            sentiment=None,  # Could be added with sentiment analysis
            language='en',  # Could be detected with language detection
            size_category=size_category,
            structure_score=structure_analysis['structure_score'],
            metadata=metadata
        )
    
    def get_routing_recommendations(self, analysis: ContentAnalysis) -> Dict[str, Any]:
        """
        Get routing recommendations based on content analysis.
        
        Args:
            analysis: Content analysis results
            
        Returns:
            Dictionary with routing recommendations
        """
        recommendations = {
            'preferred_backend': 'sqlite',  # Default
            'reason': 'Default routing',
            'confidence': 0.5,
            'estimated_cost': 0.0,
            'estimated_latency': 0.01,
            'optimization_suggestions': []
        }
        
        # Content type based recommendations
        if analysis.content_type == 'conversation':
            recommendations['preferred_backend'] = 'cipher'
            recommendations['reason'] = 'Conversational content optimized for Cipher'
            recommendations['confidence'] += 0.2
        elif analysis.content_type == 'knowledge':
            recommendations['preferred_backend'] = 'weaviate'
            recommendations['reason'] = 'Knowledge content benefits from semantic search'
            recommendations['confidence'] += 0.3
        elif analysis.content_type == 'code':
            recommendations['preferred_backend'] = 'sqlite'
            recommendations['reason'] = 'Code content benefits from local storage'
            recommendations['confidence'] += 0.4
        
        # Domain based recommendations
        if analysis.domain == ContentDomain.TECHNICAL:
            recommendations['preferred_backend'] = 'sqlite'
            recommendations['reason'] += ' (technical content)'
            recommendations['confidence'] += 0.1
        elif analysis.domain == ContentDomain.BUSINESS:
            recommendations['preferred_backend'] = 'cipher'
            recommendations['reason'] += ' (business content)'
            recommendations['confidence'] += 0.1
        
        # Complexity based recommendations
        if analysis.complexity == ContentComplexity.EXPERT:
            recommendations['preferred_backend'] = 'weaviate'
            recommendations['reason'] += ' (expert-level content)'
            recommendations['confidence'] += 0.1
        
        # Urgency based recommendations
        if analysis.urgency == ContentUrgency.CRITICAL:
            recommendations['preferred_backend'] = 'sqlite'
            recommendations['reason'] += ' (critical content needs fast access)'
            recommendations['estimated_latency'] = 0.005  # Faster access
        
        # Size based recommendations
        if analysis.metadata['content_size_bytes'] > 102400:  # > 100KB
            if recommendations['preferred_backend'] == 'sqlite':
                recommendations['preferred_backend'] = 'weaviate'
                recommendations['reason'] += ' (large content optimized for distributed storage)'
                recommendations['estimated_cost'] += 0.1
        
        # Structure based recommendations
        if analysis.structure_score > 0.7:
            recommendations['optimization_suggestions'].append(
                'Well-structured content - consider semantic indexing'
            )
        
        # Ensure confidence is within bounds
        recommendations['confidence'] = min(max(recommendations['confidence'], 0.0), 1.0)
        
        return recommendations

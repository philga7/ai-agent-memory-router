"""
Embeddings generation service for AI Agent Memory Router.

This module provides optional embedding generation capabilities for cases where
custom embeddings are needed. For most use cases, Weaviate's built-in vectorizers
should be used instead.
"""

import asyncio
import logging
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.settings = get_settings()
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the embedding model."""
        try:
            if not self._initialized:
                logger.info(f"Loading embedding model: {self.model_name}")
                
                # Load model in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    None, 
                    SentenceTransformer, 
                    self.model_name
                )
                
                self._initialized = True
                logger.info(f"Embedding model loaded successfully: {self.model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            return False
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of float values representing the embedding, or None if failed
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            if not self.model:
                logger.error("Embedding model not initialized")
                return None
            
            # Generate embedding in thread pool
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self.model.encode,
                text
            )
            
            # Convert numpy array to list
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings (or None for failed generations)
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            if not self.model:
                logger.error("Embedding model not initialized")
                return [None] * len(texts)
            
            # Generate embeddings in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self.model.encode,
                texts
            )
            
            # Convert numpy arrays to lists
            return [emb.tolist() for emb in embeddings]
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return [None] * len(texts)
    
    async def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    async def find_most_similar(
        self, 
        query_embedding: List[float], 
        candidate_embeddings: List[List[float]], 
        top_k: int = 5
    ) -> List[tuple]:
        """Find most similar embeddings to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            top_k: Number of top similar embeddings to return
            
        Returns:
            List of tuples (index, similarity_score) sorted by similarity
        """
        try:
            similarities = []
            
            for i, candidate in enumerate(candidate_embeddings):
                similarity = await self.calculate_similarity(query_embedding, candidate)
                similarities.append((i, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to find similar embeddings: {e}")
            return []
    
    def get_model_info(self) -> dict:
        """Get information about the current embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "initialized": self._initialized,
            "vector_dimension": self.settings.weaviate.vector_dimension,
            "model_loaded": self.model is not None
        }
    
    async def close(self):
        """Close the embedding service and free resources."""
        try:
            if self.model:
                # Clear model from memory
                del self.model
                self.model = None
            
            self._initialized = False
            logger.info("Embedding service closed")
            
        except Exception as e:
            logger.error(f"Error closing embedding service: {e}")


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


async def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
        await _embedding_service.initialize()
    
    return _embedding_service


async def close_embedding_service():
    """Close the global embedding service."""
    global _embedding_service
    
    if _embedding_service:
        await _embedding_service.close()
        _embedding_service = None

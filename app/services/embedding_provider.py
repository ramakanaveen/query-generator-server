# app/services/embedding_provider.py
import hashlib
import os
import time

from langchain_google_vertexai import VertexAIEmbeddings

from app.core.config import settings
from app.core.logging import logger


class EmbeddingProvider:
    """
    Service for generating embeddings using Google's text-embeddings model.
    Implements singleton pattern with caching for better performance.
    """

    _instance = None
    _embeddings = None
    _embedding_cache = {}  # Cache for embeddings
    _cache_ttl = 3600  # 1 hour cache TTL
    _cache_max_size = 1000  # Maximum cache size

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingProvider, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the embeddings model."""
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(settings.GOOGLE_CREDENTIALS_PATH)
        self._embeddings = VertexAIEmbeddings(
            model_name=settings.GOOGLE_EMBEDDING_MODEL_NAME,
            project=settings.GOOGLE_PROJECT_ID,
            location=settings.GOOGLE_LOCATION,
        )
        self._last_cleanup = time.time()

    def _get_cache_key(self, text):
        """Generate a cache key for the text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _cleanup_cache(self):
        """Clean up expired cache entries."""
        now = time.time()
        # Only clean up every 10 minutes
        if now - getattr(self, '_last_cleanup', 0) < 600:
            return

        expired_keys = []
        for key, (timestamp, _) in self._embedding_cache.items():
            if now - timestamp > self._cache_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self._embedding_cache[key]

        # If cache is still too large, remove oldest entries
        if len(self._embedding_cache) > self._cache_max_size:
            # Sort by timestamp (oldest first)
            sorted_items = sorted(
                self._embedding_cache.items(),
                key=lambda x: x[1][0]
            )
            # Remove oldest entries until cache is within max size
            for key, _ in sorted_items[:len(sorted_items) - self._cache_max_size]:
                del self._embedding_cache[key]

        self._last_cleanup = now

    async def get_embedding(self, text):
        """
        Get embedding for text, with caching.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not text:
            return None

        # Clean up expired cache entries periodically
        self._cleanup_cache()

        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self._embedding_cache:
            timestamp, embedding = self._embedding_cache[cache_key]
            # Update timestamp to show this entry was recently used
            self._embedding_cache[cache_key] = (time.time(), embedding)
            return embedding

        try:
            # Generate embedding
            embedding = await self._embeddings.aembed_query(text)

            # Cache the result
            self._embedding_cache[cache_key] = (time.time(), embedding)

            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
            return None
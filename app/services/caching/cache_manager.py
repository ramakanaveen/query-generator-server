# app/services/caching/cache_manager.py
import time
from typing import Optional, Dict, Any
import hashlib
import json
import os

from app.services.caching.caching_interface import CacheConfig, CacheInterface, CacheKeyType, CacheEntry
from app.services.caching.local_cache import LocalMemoryCache
from app.services.caching.redis_cache import RedisCache


class QueryCacheManager:
    """Main cache manager that abstracts local vs Redis implementation"""

    def __init__(self, config: Optional[CacheConfig] = None, use_redis: bool = False):
        self.config = config or CacheConfig()
        self.use_redis = use_redis
        self.cache: Optional[CacheInterface] = None
        self._initialize_cache()

    def _initialize_cache(self):
        """Initialize the appropriate cache implementation"""
        if self.use_redis:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.cache = RedisCache(self.config, redis_url)
        else:
            self.cache = LocalMemoryCache(self.config)

    async def switch_to_redis(self, redis_url: Optional[str] = None):
        """Switch from local cache to Redis cache"""
        if isinstance(self.cache, LocalMemoryCache):
            # Optional: Migrate existing cache data
            local_stats = await self.cache.get_stats()
            print(f"Switching to Redis. Local cache had {local_stats['current_size']} entries.")

        redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.cache = RedisCache(self.config, redis_url)
        self.use_redis = True

        print("Successfully switched to Redis cache")

    def generate_cache_key(
            self,
            query: str,
            context: Dict[str, Any],
            key_type: CacheKeyType = CacheKeyType.FULL_RESPONSE
    ) -> str:
        """Generate consistent cache key"""

        # Create context hash for consistent key generation
        context_items = [
            context.get('database_type', 'kdb'),
            context.get('user_id', 'anonymous'),
            str(sorted(context.get('directives', []))),
            context.get('model_version', 'default'),
            context.get('schema_version', 'latest')
        ]

        context_string = '|'.join(str(item) for item in context_items)
        context_hash = hashlib.md5(context_string.encode()).hexdigest()[:8]

        # Create query hash
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()[:12]

        # Combine into cache key
        return f"{key_type.value}:{context_hash}:{query_hash}"

    def calculate_ttl(self, query: str, key_type: CacheKeyType) -> int:
        """Calculate TTL based on query and key type"""
        base_ttl = self.config.ttl_by_type[key_type]

        # Reduce TTL for time-sensitive queries
        time_sensitive_keywords = ['today', 'now', 'current', 'latest', 'recent']
        if any(keyword in query.lower() for keyword in time_sensitive_keywords):
            return min(base_ttl, 300)  # Max 5 minutes for time-sensitive

        return base_ttl

    def should_cache_query(self, query: str, context: Dict[str, Any]) -> bool:
        """Determine if query should be cached"""

        # Don't cache very short queries
        if len(query.strip()) < 10:
            return False

        # Don't cache queries with user-specific data (for now)
        personal_indicators = ['my', 'i ', 'user:', 'personal']
        if any(indicator in query.lower() for indicator in personal_indicators):
            return False

        # Don't cache error-prone patterns
        error_patterns = ['test', 'debug', 'error']
        if any(pattern in query.lower() for pattern in error_patterns):
            return False

        return True

    async def get_cached_response(
            self,
            query: str,
            context: Dict[str, Any]
    ) -> Optional[Any]:
        """Get cached response for query"""

        if not self.should_cache_query(query, context):
            return None

        cache_key = self.generate_cache_key(query, context, CacheKeyType.FULL_RESPONSE)
        entry = await self.cache.get(cache_key)

        return entry.data if entry else None

    async def cache_response(
            self,
            query: str,
            response: Any,
            context: Dict[str, Any]
    ) -> bool:
        """Cache response for query"""

        if not self.should_cache_query(query, context):
            return False

        cache_key = self.generate_cache_key(query, context, CacheKeyType.FULL_RESPONSE)
        ttl = self.calculate_ttl(query, CacheKeyType.FULL_RESPONSE)

        entry = CacheEntry(
            data=response,
            timestamp=time.time(),
            ttl=ttl,
            query=query,
            context_hash=hashlib.md5(str(context).encode()).hexdigest()[:8],
            metadata={
                'key_type': CacheKeyType.FULL_RESPONSE.value,
                'context': context
            }
        )

        return await self.cache.set(cache_key, entry)

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        cache_stats = await self.cache.get_stats()

        return {
            'cache_type': 'redis' if self.use_redis else 'local',
            'config': self.config.__dict__,
            **cache_stats
        }

    async def clear_cache(self, pattern: Optional[str] = None) -> int:
        """Clear cache with optional pattern"""
        return await self.cache.clear(pattern)

# Singleton instance
_cache_manager: Optional[QueryCacheManager] = None

def get_cache_manager(use_redis: bool = False) -> QueryCacheManager:
    """Get or create cache manager singleton"""
    global _cache_manager

    if _cache_manager is None:
        config = CacheConfig(
            max_size=int(os.getenv('CACHE_MAX_SIZE', '1000')),
            default_ttl=int(os.getenv('CACHE_DEFAULT_TTL', '900')),
        )
        _cache_manager = QueryCacheManager(config, use_redis)

    return _cache_manager
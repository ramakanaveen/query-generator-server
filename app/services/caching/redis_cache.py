# app/services/caching/redis_cache.py

import redis.asyncio as redis
import json
import gzip
from typing import Optional, Dict, Any

from app.services.caching.caching_interface import CacheInterface, CacheConfig, CacheEntry


class RedisCache(CacheInterface):
    """Redis cache implementation (same interface as local cache)"""

    def __init__(self, config: CacheConfig, redis_url: str = "redis://localhost:6379"):
        self.config = config
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'connection_errors': 0
        }

    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection"""
        if self.redis is None:
            self.redis = redis.from_url(self.redis_url, decode_responses=False)
        return self.redis

    def _serialize_entry(self, entry: CacheEntry) -> bytes:
        """Serialize cache entry"""
        data = entry.to_dict()
        json_data = json.dumps(data, default=str).encode('utf-8')

        if self.config.enable_compression:
            return gzip.compress(json_data)
        return json_data

    def _deserialize_entry(self, data: bytes) -> CacheEntry:
        """Deserialize cache entry"""
        if self.config.enable_compression:
            data = gzip.decompress(data)

        json_data = json.loads(data.decode('utf-8'))
        return CacheEntry.from_dict(json_data)

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key"""
        try:
            redis_client = await self._get_redis()
            data = await redis_client.get(key)

            if data is None:
                self.stats['misses'] += 1
                return None

            entry = self._deserialize_entry(data)

            # Redis handles TTL automatically, but double-check
            if entry.is_expired():
                await self.delete(key)
                self.stats['misses'] += 1
                return None

            self.stats['hits'] += 1
            return entry

        except Exception as e:
            self.stats['connection_errors'] += 1
            print(f"Redis get error: {e}")
            return None

    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Set cache entry with TTL"""
        try:
            redis_client = await self._get_redis()
            serialized_data = self._serialize_entry(entry)

            success = await redis_client.setex(
                key,
                entry.ttl,
                serialized_data
            )

            if success:
                self.stats['sets'] += 1

            return bool(success)

        except Exception as e:
            self.stats['connection_errors'] += 1
            print(f"Redis set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete cache entry"""
        try:
            redis_client = await self._get_redis()
            result = await redis_client.delete(key)

            if result > 0:
                self.stats['deletes'] += 1
                return True
            return False

        except Exception as e:
            self.stats['connection_errors'] += 1
            print(f"Redis delete error: {e}")
            return False

    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries, optionally by pattern"""
        try:
            redis_client = await self._get_redis()

            if pattern is None:
                # Clear all keys (be careful!)
                return await redis_client.flushdb()

            # Pattern matching
            keys = await redis_client.keys(pattern)
            if keys:
                return await redis_client.delete(*keys)
            return 0

        except Exception as e:
            self.stats['connection_errors'] += 1
            print(f"Redis clear error: {e}")
            return 0

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            redis_client = await self._get_redis()
            return bool(await redis_client.exists(key))
        except Exception as e:
            self.stats['connection_errors'] += 1
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0

        # Try to get Redis info
        redis_info = {}
        try:
            redis_client = await self._get_redis()
            redis_info = await redis_client.info('memory')
        except Exception:
            pass

        return {
            **self.stats,
            'hit_rate': hit_rate,
            'redis_info': redis_info
        }

    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
# app/services/caching/local_cache.py
import threading
from collections import OrderedDict
from typing import Dict, Optional, Any

from app.services.caching.caching_interface import CacheInterface, CacheConfig, CacheEntry


class LocalMemoryCache(CacheInterface):
    """Local in-memory cache implementation with LRU eviction"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()  # Thread-safe for sync/async mixing

        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0,
            'expired_cleanups': 0
        }

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key"""
        with self.lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return None

            entry = self.cache[key]

            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                self.stats['expired_cleanups'] += 1
                self.stats['misses'] += 1
                return None

            # Move to end (LRU)
            self.cache.move_to_end(key)
            self.stats['hits'] += 1

            return entry

    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Set cache entry with LRU eviction"""
        with self.lock:
            try:
                # Remove existing entry if present
                if key in self.cache:
                    del self.cache[key]

                # Evict if at capacity
                while len(self.cache) >= self.config.max_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    self.stats['evictions'] += 1

                # Add new entry
                self.cache[key] = entry
                self.stats['sets'] += 1

                return True
            except Exception as e:
                print(f"Cache set error: {e}")
                return False

    async def delete(self, key: str) -> bool:
        """Delete cache entry"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats['deletes'] += 1
                return True
            return False

    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries, optionally by pattern"""
        with self.lock:
            if pattern is None:
                count = len(self.cache)
                self.cache.clear()
                return count

            # Pattern matching (simple prefix matching for now)
            keys_to_delete = [
                key for key in self.cache.keys()
                if key.startswith(pattern.replace('*', ''))
            ]

            for key in keys_to_delete:
                del self.cache[key]

            return len(keys_to_delete)

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        entry = await self.get(key)
        return entry is not None

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0

            return {
                **self.stats,
                'current_size': len(self.cache),
                'max_size': self.config.max_size,
                'hit_rate': hit_rate,
                'utilization': len(self.cache) / self.config.max_size
            }

    async def cleanup_expired(self) -> int:
        """Manual cleanup of expired entries"""
        with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                del self.cache[key]

            self.stats['expired_cleanups'] += len(expired_keys)
            return len(expired_keys)
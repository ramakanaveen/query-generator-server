from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
import time
import hashlib
import json
from dataclasses import dataclass, asdict
from enum import Enum

@dataclass
class CacheEntry:
    """Standard cache entry structure"""
    data: Any
    timestamp: float
    ttl: int  # Time to live in seconds
    query: str
    context_hash: str
    metadata: Dict[str, Any]

    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() - self.timestamp > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary"""
        return cls(**data)

class CacheKeyType(Enum):
    """Types of cache keys for different TTL strategies"""
    FULL_RESPONSE = "response"
    SCHEMA_DATA = "schema"
    INTENT_CLASSIFICATION = "intent"
    QUERY_ANALYSIS = "analysis"
    VALIDATION_RESULT = "validation"

@dataclass
class CacheConfig:
    """Cache configuration"""
    max_size: int = 1000
    default_ttl: int = 900  # 15 minutes
    ttl_by_type: Dict[CacheKeyType, int] = None
    enable_compression: bool = False

    def __post_init__(self):
        if self.ttl_by_type is None:
            self.ttl_by_type = {
                CacheKeyType.FULL_RESPONSE: 900,    # 15 minutes
                CacheKeyType.SCHEMA_DATA: 3600,     # 1 hour
                CacheKeyType.INTENT_CLASSIFICATION: 1800,  # 30 minutes
                CacheKeyType.QUERY_ANALYSIS: 1800,  # 30 minutes
                CacheKeyType.VALIDATION_RESULT: 600  # 10 minutes
            }

class CacheInterface(ABC):
    """Abstract interface for cache implementations"""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key"""
        pass

    @abstractmethod
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Set cache entry"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete cache entry"""
        pass

    @abstractmethod
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries, optionally by pattern"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass
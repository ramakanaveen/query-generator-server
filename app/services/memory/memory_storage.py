"""
Memory Module - Storage Layer

Handles all database operations for memory persistence.
"""

import json
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
import logging

from .memory_types import (
    Memory,
    MemoryType,
    SourceType,
    MemoryStats
)

logger = logging.getLogger(__name__)


class MemoryStorage:
    """
    Storage layer for memory persistence.

    Handles all direct database operations for the memory system.
    Uses PostgreSQL with pgvector for hybrid search.
    """

    def __init__(self, db_pool):
        """
        Initialize storage with database connection pool.

        Args:
            db_pool: asyncpg connection pool
        """
        self.db_pool = db_pool

    async def store(self, memory: Memory, embedding: Optional[List[float]] = None) -> UUID:
        """
        Store a memory in the database.

        Args:
            memory: Memory object to store
            embedding: Optional vector embedding

        Returns:
            UUID of the stored memory
        """
        async with self.db_pool.acquire() as conn:
            memory_id = await conn.fetchval("""
                INSERT INTO memory.entries (
                    memory_type,
                    user_id,
                    schema_group_id,
                    original_context,
                    learning_description,
                    corrected_version,
                    metadata,
                    embedding,
                    source_type,
                    source_conversation_id,
                    source_feedback_id,
                    confidence_score,
                    is_validated
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                RETURNING id
            """,
                memory.memory_type.value,
                memory.user_id,
                memory.schema_group_id,
                memory.original_context,
                memory.learning_description,
                memory.corrected_version,
                json.dumps(memory.metadata),
                embedding,
                memory.source_type.value,
                memory.source_conversation_id,
                memory.source_feedback_id,
                memory.confidence_score,
                memory.is_validated
            )

            # Store tags if provided
            if memory.tags:
                await self._store_tags(conn, memory_id, memory.tags)

            logger.info(
                f"Stored memory {memory_id}: type={memory.memory_type}, "
                f"user_specific={memory.is_user_specific}"
            )

            return memory_id

    async def _store_tags(self, conn, memory_id: UUID, tags: List[str]):
        """Store tags for a memory"""
        for tag in tags:
            await conn.execute("""
                INSERT INTO memory.tags (memory_id, tag)
                VALUES ($1, $2)
                ON CONFLICT (memory_id, tag) DO NOTHING
            """, memory_id, tag.lower().strip())

    async def retrieve_by_id(self, memory_id: UUID) -> Optional[Memory]:
        """
        Retrieve a specific memory by ID.

        Args:
            memory_id: UUID of the memory

        Returns:
            Memory object or None if not found
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                    id,
                    memory_type,
                    user_id,
                    schema_group_id,
                    original_context,
                    learning_description,
                    corrected_version,
                    metadata,
                    source_type,
                    source_conversation_id,
                    source_feedback_id,
                    confidence_score,
                    success_count,
                    failure_count,
                    created_at,
                    updated_at,
                    last_accessed_at,
                    access_count,
                    is_active,
                    is_validated
                FROM memory.entries
                WHERE id = $1
            """, memory_id)

            if not row:
                return None

            # Fetch tags
            tags = await conn.fetch("""
                SELECT tag FROM memory.tags WHERE memory_id = $1
            """, memory_id)

            return self._row_to_memory(row, [t['tag'] for t in tags])

    async def search_by_vector(
        self,
        embedding: List[float],
        user_id: Optional[str] = None,
        schema_group_id: Optional[UUID] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
        min_similarity: float = 0.7,
        include_global: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search memories using vector similarity.

        Args:
            embedding: Query embedding vector
            user_id: Optional user ID to filter
            schema_group_id: Optional schema group filter
            memory_types: Optional list of memory types to filter
            limit: Maximum results to return
            min_similarity: Minimum similarity threshold
            include_global: Include global memories (user_id IS NULL)

        Returns:
            List of dicts with memory data and similarity scores
        """
        # Build dynamic query
        conditions = ["is_active = true"]
        params = [embedding, limit, min_similarity]
        param_idx = 4

        # User filtering
        if user_id is not None:
            if include_global:
                conditions.append(f"(user_id = ${param_idx} OR user_id IS NULL)")
                params.append(user_id)
                param_idx += 1
            else:
                conditions.append(f"user_id = ${param_idx}")
                params.append(user_id)
                param_idx += 1
        elif not include_global:
            conditions.append("user_id IS NOT NULL")

        # Schema filtering
        if schema_group_id is not None:
            conditions.append(f"(schema_group_id = ${param_idx} OR schema_group_id IS NULL)")
            params.append(schema_group_id)
            param_idx += 1

        # Memory type filtering
        if memory_types:
            type_values = [mt.value for mt in memory_types]
            placeholders = ",".join([f"${i}" for i in range(param_idx, param_idx + len(type_values))])
            conditions.append(f"memory_type IN ({placeholders})")
            params.extend(type_values)

        where_clause = " AND ".join(conditions)

        query = f"""
            WITH ranked_memories AS (
                SELECT
                    id,
                    memory_type,
                    user_id,
                    schema_group_id,
                    original_context,
                    learning_description,
                    corrected_version,
                    metadata,
                    source_type,
                    source_conversation_id,
                    source_feedback_id,
                    confidence_score,
                    success_count,
                    failure_count,
                    created_at,
                    updated_at,
                    last_accessed_at,
                    access_count,
                    is_active,
                    is_validated,
                    -- Calculate similarity
                    1 - (embedding <=> $1) AS similarity_score,
                    -- Calculate age in days
                    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at)) / 86400.0 AS age_days,
                    -- Calculate success rate
                    CASE
                        WHEN (success_count + failure_count) > 0
                        THEN success_count::float / (success_count + failure_count)
                        ELSE 0.5
                    END AS success_rate
                FROM memory.entries
                WHERE {where_clause}
                  AND embedding IS NOT NULL
            )
            SELECT
                *,
                -- Combined score: quality * recency * similarity
                (
                    similarity_score * 0.5 +
                    (confidence_score * 0.4 + success_rate * 0.4 + 0.2) * 0.3 +
                    EXP(-age_days / 30.0) * 0.2
                ) AS combined_score
            FROM ranked_memories
            WHERE similarity_score >= $3
            ORDER BY combined_score DESC
            LIMIT $2
        """

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]

    async def update_access_stats(self, memory_id: UUID, was_helpful: Optional[bool] = None):
        """
        Update access statistics for a memory.

        Args:
            memory_id: Memory ID to update
            was_helpful: Whether the memory was helpful (None = just accessed)
        """
        async with self.db_pool.acquire() as conn:
            if was_helpful is None:
                # Just increment access count
                await conn.execute("""
                    UPDATE memory.entries
                    SET access_count = access_count + 1,
                        last_accessed_at = CURRENT_TIMESTAMP
                    WHERE id = $1
                """, memory_id)
            elif was_helpful:
                # Increment success count
                await conn.execute("""
                    UPDATE memory.entries
                    SET access_count = access_count + 1,
                        success_count = success_count + 1,
                        last_accessed_at = CURRENT_TIMESTAMP,
                        confidence_score = LEAST(confidence_score + 0.05, 1.0)
                    WHERE id = $1
                """, memory_id)
            else:
                # Increment failure count
                await conn.execute("""
                    UPDATE memory.entries
                    SET access_count = access_count + 1,
                        failure_count = failure_count + 1,
                        last_accessed_at = CURRENT_TIMESTAMP,
                        confidence_score = GREATEST(confidence_score - 0.05, 0.1)
                    WHERE id = $1
                """, memory_id)

    async def log_usage(
        self,
        memory_id: UUID,
        query_id: Optional[UUID] = None,
        conversation_id: Optional[UUID] = None,
        user_id: Optional[str] = None,
        was_helpful: Optional[bool] = None,
        applied_to_query: bool = False,
        similarity_score: Optional[float] = None
    ):
        """Log memory usage for analytics"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO memory.usage_log (
                    memory_id,
                    query_id,
                    conversation_id,
                    user_id,
                    was_helpful,
                    applied_to_query,
                    similarity_score
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
                memory_id,
                query_id,
                conversation_id,
                user_id,
                was_helpful,
                applied_to_query,
                similarity_score
            )

    async def get_stats(self, user_id: Optional[str] = None) -> MemoryStats:
        """
        Get statistics about stored memories.

        Args:
            user_id: Optional user ID to get user-specific stats

        Returns:
            MemoryStats object
        """
        async with self.db_pool.acquire() as conn:
            # Basic counts
            where_clause = "user_id = $1" if user_id else "1=1"
            param = [user_id] if user_id else []

            stats_row = await conn.fetchrow(f"""
                SELECT
                    COUNT(*) as total_memories,
                    COUNT(*) FILTER (WHERE is_active) as active_memories,
                    COUNT(*) FILTER (WHERE user_id IS NULL) as global_memories,
                    COUNT(*) FILTER (WHERE user_id IS NOT NULL) as user_specific_memories,
                    AVG(confidence_score) as avg_confidence,
                    AVG(
                        CASE
                            WHEN (success_count + failure_count) > 0
                            THEN success_count::float / (success_count + failure_count)
                            ELSE 0.5
                        END
                    ) as avg_success_rate,
                    SUM(access_count) as total_access_count,
                    SUM(success_count) as total_success_count,
                    SUM(failure_count) as total_failure_count,
                    MIN(created_at) as oldest_memory,
                    MAX(created_at) as newest_memory
                FROM memory.entries
                WHERE {where_clause}
            """, *param)

            # Counts by type
            type_rows = await conn.fetch(f"""
                SELECT memory_type, COUNT(*) as count
                FROM memory.entries
                WHERE {where_clause} AND is_active = true
                GROUP BY memory_type
            """, *param)

            memories_by_type = {
                MemoryType(row['memory_type']): row['count']
                for row in type_rows
            }

            return MemoryStats(
                total_memories=stats_row['total_memories'],
                active_memories=stats_row['active_memories'],
                global_memories=stats_row['global_memories'],
                user_specific_memories=stats_row['user_specific_memories'],
                memories_by_type=memories_by_type,
                avg_confidence=float(stats_row['avg_confidence'] or 0),
                avg_success_rate=float(stats_row['avg_success_rate'] or 0),
                total_access_count=stats_row['total_access_count'] or 0,
                total_success_count=stats_row['total_success_count'] or 0,
                total_failure_count=stats_row['total_failure_count'] or 0,
                oldest_memory=stats_row['oldest_memory'],
                newest_memory=stats_row['newest_memory']
            )

    async def deactivate(self, memory_id: UUID) -> bool:
        """
        Deactivate a memory (soft delete).

        Args:
            memory_id: Memory to deactivate

        Returns:
            True if deactivated, False if not found
        """
        async with self.db_pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE memory.entries
                SET is_active = false
                WHERE id = $1
            """, memory_id)

            return result == "UPDATE 1"

    async def apply_temporal_decay(self, decay_rate: float = 0.01) -> int:
        """
        Apply temporal decay to old memories.

        Args:
            decay_rate: Decay rate to apply

        Returns:
            Number of memories updated
        """
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT * FROM memory.apply_temporal_decay($1)
            """, decay_rate)

            logger.info(f"Applied temporal decay: {result} memories updated")
            return result

    async def archive_low_quality(self, min_quality: float = 0.3) -> int:
        """
        Archive low-quality memories.

        Args:
            min_quality: Minimum quality threshold

        Returns:
            Number of memories archived
        """
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT * FROM memory.archive_low_quality_memories($1)
            """, min_quality)

            logger.info(f"Archived low-quality memories: {result} memories archived")
            return result

    def _row_to_memory(self, row: Dict, tags: List[str] = None) -> Memory:
        """Convert database row to Memory object"""
        return Memory(
            id=row['id'],
            memory_type=MemoryType(row['memory_type']),
            user_id=row['user_id'],
            schema_group_id=row['schema_group_id'],
            original_context=row['original_context'],
            learning_description=row['learning_description'],
            corrected_version=row['corrected_version'],
            metadata=row['metadata'] if isinstance(row['metadata'], dict) else json.loads(row['metadata']),
            source_type=SourceType(row['source_type']),
            source_conversation_id=row['source_conversation_id'],
            source_feedback_id=row['source_feedback_id'],
            confidence_score=float(row['confidence_score']),
            success_count=row['success_count'],
            failure_count=row['failure_count'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            last_accessed_at=row['last_accessed_at'],
            access_count=row['access_count'],
            is_active=row['is_active'],
            is_validated=row['is_validated'],
            tags=tags or []
        )
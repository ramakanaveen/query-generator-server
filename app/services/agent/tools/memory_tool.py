"""
Memory tool — wraps the existing MemoryManager.
Activates the fully-built but previously disconnected memory system.
"""
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from app.services.memory.memory_manager import MemoryManager
from app.services.memory.memory_types import MemoryType

logger = logging.getLogger(__name__)

_manager: MemoryManager | None = None


def _get_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager


_TYPE_MAP = {t.value: t for t in MemoryType}


async def recall(
    context: str,
    user_id: str | None = None,
    memory_types: list[str] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant memories for the given context.
    Returns a list of dicts safe to embed in a system prompt.
    """
    try:
        types = [_TYPE_MAP[t] for t in (memory_types or []) if t in _TYPE_MAP] or None
        results = await _get_manager().retrieve(
            query=context,
            user_id=user_id,
            memory_types=types,
            limit=limit,
        )
        return [
            {
                "type": r.memory.memory_type.value,
                "learning": r.memory.learning_description,
                "corrected_version": r.memory.corrected_version,
                "score": round(r.combined_score, 3),
            }
            for r in results
        ]
    except Exception as exc:
        logger.warning(f"memory recall failed: {exc}")
        return []


async def store_pattern(
    original_query: str,
    generated_query: str,
    user_id: str | None = None,
) -> None:
    """Store a successful query pattern after generation."""
    try:
        await _get_manager().store(
            memory_type=MemoryType.QUERY_PATTERN,
            original_context=original_query,
            learning=f"Successfully generated: {generated_query}",
            corrected_version=generated_query,
            user_id=user_id,
            source_type="auto_detected",
        )
    except Exception as exc:
        logger.warning(f"memory store failed (non-fatal): {exc}")


async def store_clarification(
    vague_query: str,
    clarification_question: str,
    user_answer: str,
    user_id: str | None = None,
) -> None:
    """Store a clarification exchange so future similar queries don't need to ask again."""
    try:
        await _get_manager().store(
            memory_type=MemoryType.SCHEMA_CLARIFICATION,
            original_context=vague_query,
            learning=f"When user asks '{vague_query}', they mean: {user_answer}. "
                     f"(Clarification asked: {clarification_question})",
            user_id=user_id,
            source_type="auto_detected",
        )
    except Exception as exc:
        logger.warning(f"clarification store failed (non-fatal): {exc}")

"""
Long-Term Memory Module

A reusable, independent memory system for AI applications that learns from
user interactions across conversations.

Public API:
    - MemoryManager: Main interface for storing and retrieving memories
    - MemoryType: Enum of supported memory types
    - Memory: Data class representing a memory entry

Usage:
    from app.services.memory import MemoryManager, MemoryType

    # Initialize
    memory_manager = MemoryManager()

    # Store a memory
    memory_id = await memory_manager.store(
        memory_type=MemoryType.SYNTAX_CORRECTION,
        original_context="select from trades",
        learning="Table name should be 'trade' (singular)",
        corrected_version="select from trade"
    )

    # Retrieve relevant memories
    memories = await memory_manager.retrieve(
        query="Show me all trades",
        user_id="john@example.com",
        limit=5
    )
"""

from .memory_manager import MemoryManager
from .memory_types import (
    MemoryType,
    Memory,
    MemoryRetrievalResult,
    MemoryStats
)

__all__ = [
    'MemoryManager',
    'MemoryType',
    'Memory',
    'MemoryRetrievalResult',
    'MemoryStats'
]

__version__ = '1.0.0'
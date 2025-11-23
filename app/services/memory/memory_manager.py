"""
Memory Module - Main Manager

Public API for the long-term memory system.
This is the main interface that should be used by other parts of the system.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from app.core.db import get_db_pool
from ..embedding_provider import EmbeddingProvider
from ..llm_provider import LLMProvider

from .memory_types import (
    Memory,
    MemoryType,
    MemoryRetrievalResult,
    MemoryStats,
    MemoryExtractionRequest,
    MemoryConfig
)
from .memory_storage import MemoryStorage
from .memory_extractor import MemoryExtractor

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Main interface for the Long-Term Memory system.

    This class provides all functionality for storing, retrieving,
    and managing memories across conversations.

    Example Usage:
        ```python
        # Initialize
        memory_manager = MemoryManager()

        # Store a memory
        memory_id = await memory_manager.store(
            memory_type=MemoryType.SYNTAX_CORRECTION,
            original_context="select from trades",
            learning="Table should be 'trade' not 'trades'",
            corrected_version="select from trade"
        )

        # Retrieve relevant memories
        memories = await memory_manager.retrieve(
            query="Show me all trades",
            user_id="john@example.com",
            limit=5
        )

        # Auto-extract from feedback
        await memory_manager.extract_and_store_from_feedback(
            feedback_id=feedback_id,
            conversation_id=conv_id,
            user_id=user_id
        )
        ```
    """

    _instance = None  # Singleton instance

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        db_pool=None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_provider: Optional[LLMProvider] = None
    ):
        """
        Initialize Memory Manager.

        Args:
            config: Memory configuration (uses defaults if not provided)
            db_pool: Database connection pool (gets from app if not provided)
            embedding_provider: Embedding provider instance
            llm_provider: LLM provider instance
        """
        # Avoid re-initialization in singleton
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.config = config or MemoryConfig()
        self.config.validate()

        # Dependencies
        self.db_pool = db_pool or get_db_pool()
        self.embedding_provider = embedding_provider or EmbeddingProvider()
        self.llm_provider = llm_provider or LLMProvider()

        # Components
        self.storage = MemoryStorage(self.db_pool)
        self.extractor = MemoryExtractor(self.llm_provider)

        self._initialized = True

        logger.info("MemoryManager initialized")

    async def store(
        self,
        memory_type: MemoryType,
        original_context: str,
        learning: str,
        corrected_version: Optional[str] = None,
        user_id: Optional[str] = None,
        schema_group_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_type: str = "manual",
        source_conversation_id: Optional[UUID] = None,
        source_feedback_id: Optional[UUID] = None,
        confidence_score: float = 0.7,
        tags: Optional[List[str]] = None
    ) -> UUID:
        """
        Store a new memory.

        Args:
            memory_type: Type of memory to store
            original_context: The original context/query
            learning: Description of what was learned
            corrected_version: The corrected/improved version
            user_id: User ID (None for global memory)
            schema_group_id: Schema group ID
            metadata: Additional metadata
            source_type: Source of this memory
            source_conversation_id: Source conversation
            source_feedback_id: Source feedback
            confidence_score: Confidence in this memory (0-1)
            tags: Optional tags for categorization

        Returns:
            UUID of the stored memory
        """
        # Create Memory object
        memory = Memory(
            memory_type=memory_type,
            user_id=user_id,
            schema_group_id=schema_group_id,
            original_context=original_context,
            learning_description=learning,
            corrected_version=corrected_version,
            metadata=metadata or {},
            source_type=source_type,
            source_conversation_id=source_conversation_id,
            source_feedback_id=source_feedback_id,
            confidence_score=confidence_score,
            tags=tags or []
        )

        # Generate embedding
        embedding_text = self._build_embedding_text(memory)
        embedding = await self.embedding_provider.embed_query(embedding_text)

        # Store in database
        memory_id = await self.storage.store(memory, embedding.tolist() if hasattr(embedding, 'tolist') else embedding)

        logger.info(
            f"Stored memory {memory_id}: "
            f"type={memory_type.value}, "
            f"user_specific={user_id is not None}"
        )

        return memory_id

    async def retrieve(
        self,
        query: str,
        user_id: Optional[str] = None,
        schema_group_id: Optional[UUID] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: Optional[int] = None,
        min_similarity: Optional[float] = None,
        include_global: Optional[bool] = None
    ) -> List[MemoryRetrievalResult]:
        """
        Retrieve relevant memories for a query.

        Uses hybrid search combining:
        1. Vector similarity (semantic)
        2. Structured filtering (user, schema, type)
        3. Quality scoring (confidence, success rate)
        4. Temporal weighting (recency)

        Args:
            query: Query to find relevant memories for
            user_id: User ID to get personalized memories
            schema_group_id: Schema group for context
            memory_types: Filter by memory types
            limit: Max memories to return (uses config default if not specified)
            min_similarity: Minimum similarity threshold
            include_global: Include global memories

        Returns:
            List of MemoryRetrievalResult objects, ranked by relevance
        """
        # Use config defaults
        limit = limit or self.config.default_limit
        min_similarity = min_similarity or self.config.min_similarity_threshold
        include_global = include_global if include_global is not None else self.config.include_global_memories

        # Generate query embedding
        query_embedding = await self.embedding_provider.embed_query(query)

        # Search using storage layer
        results = await self.storage.search_by_vector(
            embedding=query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding,
            user_id=user_id,
            schema_group_id=schema_group_id,
            memory_types=memory_types,
            limit=limit,
            min_similarity=min_similarity,
            include_global=include_global
        )

        # Convert to MemoryRetrievalResult objects
        retrieval_results = []
        for row in results:
            memory = self.storage._row_to_memory(row)
            result = MemoryRetrievalResult(
                memory=memory,
                similarity_score=row['similarity_score'],
                combined_score=row['combined_score']
            )
            retrieval_results.append(result)

        # Update access stats (don't await, fire and forget)
        for result in retrieval_results:
            # Run in background
            try:
                await self.storage.update_access_stats(result.memory.id)
            except Exception as e:
                logger.warning(f"Failed to update access stats: {e}")

        logger.info(
            f"Retrieved {len(retrieval_results)} memories for query: '{query[:50]}...'"
        )

        return retrieval_results

    async def extract_and_store_from_feedback(
        self,
        feedback_id: UUID,
        conversation_id: UUID,
        original_query: str,
        corrected_query: Optional[str] = None,
        user_feedback: Optional[str] = None,
        user_id: Optional[str] = None,
        schema_group_id: Optional[UUID] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        auto_validate: bool = False
    ) -> List[UUID]:
        """
        Automatically extract and store memories from user feedback.

        This is the main entry point for auto-extraction.

        Args:
            feedback_id: Feedback record ID
            conversation_id: Conversation ID
            original_query: Original query
            corrected_query: Corrected query (if available)
            user_feedback: User's feedback text
            user_id: User ID
            schema_group_id: Schema group ID
            conversation_history: Recent conversation messages
            auto_validate: Auto-validate high-confidence memories

        Returns:
            List of created memory IDs
        """
        # Build extraction request
        request = MemoryExtractionRequest(
            feedback_id=feedback_id,
            conversation_id=conversation_id,
            user_id=user_id,
            schema_group_id=schema_group_id,
            original_query=original_query,
            corrected_query=corrected_query,
            user_feedback=user_feedback,
            conversation_history=conversation_history or [],
            auto_validate=auto_validate,
            min_confidence=self.config.extraction_min_confidence
        )

        # Extract memories
        memories = await self.extractor.extract_from_feedback(request)

        if not memories:
            logger.info("No memories extracted from feedback")
            return []

        # Store each extracted memory
        memory_ids = []
        for memory in memories:
            try:
                # Generate embedding
                embedding_text = self._build_embedding_text(memory)
                embedding = await self.embedding_provider.embed_query(embedding_text)

                # Store
                memory_id = await self.storage.store(
                    memory,
                    embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                )
                memory_ids.append(memory_id)

            except Exception as e:
                logger.error(f"Failed to store extracted memory: {e}", exc_info=True)

        logger.info(f"Auto-extracted and stored {len(memory_ids)} memories from feedback")
        return memory_ids

    async def mark_helpful(self, memory_id: UUID, helpful: bool = True):
        """
        Mark a memory as helpful or not helpful.

        This reinforces successful memories and weakens unsuccessful ones.

        Args:
            memory_id: Memory ID
            helpful: Whether the memory was helpful
        """
        await self.storage.update_access_stats(memory_id, was_helpful=helpful)

    async def log_usage(
        self,
        memory_id: UUID,
        query_id: Optional[UUID] = None,
        conversation_id: Optional[UUID] = None,
        user_id: Optional[str] = None,
        was_helpful: Optional[bool] = None,
        applied: bool = False,
        similarity_score: Optional[float] = None
    ):
        """
        Log memory usage for analytics.

        Args:
            memory_id: Memory that was used
            query_id: Query where it was used
            conversation_id: Conversation context
            user_id: User who used it
            was_helpful: Whether it helped
            applied: Whether it was actually applied
            similarity_score: Similarity score
        """
        await self.storage.log_usage(
            memory_id=memory_id,
            query_id=query_id,
            conversation_id=conversation_id,
            user_id=user_id,
            was_helpful=was_helpful,
            applied_to_query=applied,
            similarity_score=similarity_score
        )

    async def get_stats(self, user_id: Optional[str] = None) -> MemoryStats:
        """
        Get memory statistics.

        Args:
            user_id: Optional user ID for user-specific stats

        Returns:
            MemoryStats object
        """
        return await self.storage.get_stats(user_id)

    async def get_by_id(self, memory_id: UUID) -> Optional[Memory]:
        """
        Retrieve a specific memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            Memory object or None
        """
        return await self.storage.retrieve_by_id(memory_id)

    async def deactivate(self, memory_id: UUID) -> bool:
        """
        Deactivate a memory (soft delete).

        Args:
            memory_id: Memory to deactivate

        Returns:
            True if successful
        """
        return await self.storage.deactivate(memory_id)

    async def apply_maintenance(self):
        """
        Apply memory maintenance (decay, archiving).

        This should be run periodically (e.g., daily cron job).
        """
        logger.info("Running memory maintenance...")

        # Apply temporal decay
        decayed = await self.storage.apply_temporal_decay(
            self.config.temporal_decay_rate
        )

        # Archive low-quality memories
        archived = await self.storage.archive_low_quality(
            self.config.min_confidence_threshold
        )

        logger.info(
            f"Maintenance complete: {decayed} memories decayed, "
            f"{archived} memories archived"
        )

    def format_memories_for_prompt(
        self,
        memories: List[MemoryRetrievalResult],
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Format memories into a text block for LLM prompts.

        Args:
            memories: List of retrieved memories
            max_tokens: Optional token limit (truncates if exceeded)

        Returns:
            Formatted string for prompt inclusion
        """
        if not memories:
            return ""

        sections = ["## Relevant Learnings from Past Interactions\n"]

        for i, result in enumerate(memories, 1):
            memory = result.memory

            # Type header
            type_name = memory.memory_type.value.replace('_', ' ').title()
            sections.append(f"\n### {i}. {type_name}")

            # Learning description
            sections.append(f"**Learning**: {memory.learning_description}")

            # Show correction if available
            if memory.corrected_version:
                sections.append(f"\n**Example**:")
                sections.append(f"- Original: `{memory.original_context}`")
                sections.append(f"- Improved: `{memory.corrected_version}`")

            # Confidence indicator
            confidence_label = "High" if memory.confidence_score >= 0.8 else "Medium" if memory.confidence_score >= 0.6 else "Low"
            sections.append(f"\n_Confidence: {confidence_label} ({memory.confidence_score:.2f})_")

            # User-specific indicator
            if memory.is_user_specific:
                sections.append("_This is your personal preference_")

            sections.append("")  # Blank line

        formatted = "\n".join(sections)

        # TODO: Implement token limiting if needed
        # if max_tokens:
        #     formatted = truncate_to_tokens(formatted, max_tokens)

        return formatted

    def _build_embedding_text(self, memory: Memory) -> str:
        """Build text for embedding generation"""
        parts = [
            memory.original_context,
            memory.learning_description
        ]

        if memory.corrected_version:
            parts.append(memory.corrected_version)

        # Add metadata context
        if memory.metadata:
            metadata_str = " ".join([f"{k}:{v}" for k, v in memory.metadata.items()])
            parts.append(metadata_str)

        return " | ".join(parts)
"""
Memory Retrieval Node

LangGraph node that retrieves relevant long-term memories
and adds them to the query generation context.

Integration Point: After schema retrieval, before intelligent analyzer
"""

import logging
from typing import Dict, Any

from app.services.memory import MemoryManager, MemoryType
from app.services.query_generator_state import QueryGenerationState

logger = logging.getLogger(__name__)


async def memory_retrieval_node(state: QueryGenerationState) -> Dict[str, Any]:
    """
    Retrieve relevant long-term memories for the query.

    This node:
    1. Takes the user query
    2. Searches for relevant memories (corrections, patterns, definitions)
    3. Formats them for inclusion in prompts
    4. Adds them to the state

    Args:
        state: Current query generation state

    Returns:
        Updated state with memory information
    """
    try:
        # Initialize memory manager
        memory_manager = MemoryManager()

        # Define which memory types are most useful for query generation
        relevant_memory_types = [
            MemoryType.SYNTAX_CORRECTION,      # Syntax fixes
            MemoryType.USER_DEFINITION,         # User definitions
            MemoryType.APPROACH_RECOMMENDATION, # Approach suggestions
            MemoryType.SCHEMA_CLARIFICATION,    # Schema clarifications
            MemoryType.ERROR_CORRECTION         # Error fixes
        ]

        # Retrieve memories
        memory_results = await memory_manager.retrieve(
            query=state.user_query,
            user_id=getattr(state, 'user_id', None),  # User ID if available
            schema_group_id=state.schema_group_id,
            memory_types=relevant_memory_types,
            limit=5,  # Top 5 most relevant memories
            include_global=True  # Include both user-specific and global memories
        )

        if not memory_results:
            logger.info("No relevant memories found for query")
            return {
                "retrieved_memories": [],
                "memory_context": "",
                "memory_count": 0
            }

        # Format memories for prompt context
        memory_context = memory_manager.format_memories_for_prompt(memory_results)

        # Log memory IDs for tracking
        memory_ids = [str(result.memory.id) for result in memory_results]

        # Prepare memory summary for state
        memory_summary = []
        for result in memory_results:
            memory = result.memory
            memory_summary.append({
                "id": str(memory.id),
                "type": memory.memory_type.value,
                "learning": memory.learning_description,
                "confidence": memory.confidence_score,
                "similarity": result.similarity_score,
                "is_user_specific": memory.is_user_specific
            })

        logger.info(
            f"Retrieved {len(memory_results)} memories for query: "
            f"{[m['type'] for m in memory_summary]}"
        )

        # Return state updates
        return {
            "retrieved_memories": memory_summary,
            "memory_context": memory_context,
            "memory_count": len(memory_results),
            "memory_ids": memory_ids  # For tracking/logging
        }

    except Exception as e:
        logger.error(f"Error retrieving memories: {e}", exc_info=True)

        # Don't fail the pipeline if memory retrieval fails
        return {
            "retrieved_memories": [],
            "memory_context": "",
            "memory_count": 0,
            "memory_error": str(e)
        }


async def track_memory_usage(
    state: QueryGenerationState,
    was_helpful: bool,
    applied: bool = True
) -> None:
    """
    Track whether retrieved memories were helpful.

    Call this after query generation succeeds or fails to update
    memory effectiveness metrics.

    Args:
        state: Current state with memory information
        was_helpful: Whether the memories helped generate a good query
        applied: Whether the memories were actually used
    """
    try:
        if not hasattr(state, 'memory_ids') or not state.memory_ids:
            return

        memory_manager = MemoryManager()

        # Log usage for each memory
        for memory_id in state.memory_ids:
            await memory_manager.log_usage(
                memory_id=memory_id,
                query_id=getattr(state, 'query_id', None),
                conversation_id=getattr(state, 'conversation_id', None),
                user_id=getattr(state, 'user_id', None),
                was_helpful=was_helpful,
                applied=applied
            )

            # Also update the memory's success/failure counts
            await memory_manager.mark_helpful(memory_id, was_helpful)

        logger.info(f"Tracked usage for {len(state.memory_ids)} memories")

    except Exception as e:
        logger.error(f"Error tracking memory usage: {e}", exc_info=True)
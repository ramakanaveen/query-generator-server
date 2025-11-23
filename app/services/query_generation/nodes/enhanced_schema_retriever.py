# app/services/query_generation/nodes/enhanced_schema_retriever.py

import time
from typing import Dict, Any

from app.core.profiling import timeit
from app.core.logging import logger
from app.services.enhanced_schema_service import EnhancedSchemaService, SchemaRetrievalConfig

# Cache for schema service (singleton pattern)
_enhanced_schema_service = None

def get_enhanced_schema_service():
    """Get singleton instance of enhanced schema service."""
    global _enhanced_schema_service
    if _enhanced_schema_service is None:
        _enhanced_schema_service = EnhancedSchemaService()
    return _enhanced_schema_service

@timeit
async def retrieve_schema_with_examples(state):
    """
    Enhanced schema retrieval that consolidates schema + examples in a single operation.

    This replaces the old pattern of:
    - schema_retriever.py getting schema
    - query_generator_node.py getting examples separately

    With a single optimized operation that gets both.
    """
    try:
        # Enhanced thinking step
        if getattr(state, 'needs_schema_reselection', False):
            state.thinking.append("🔄 Re-selecting schema with examples based on feedback...")
        elif state.is_retry_request:
            state.thinking.append("🔍 Retrieving schema with examples using retry context...")
        else:
            state.thinking.append("📊 Retrieving schema with examples using enhanced service...")

        # Get enhanced schema service
        schema_service = get_enhanced_schema_service()

        # Determine configuration based on state
        config = SchemaRetrievalConfig(
            vector_similarity_threshold=0.4,
            max_tables=5,
            max_schema_examples=5,
            max_verified_examples=3,
            max_total_examples=8,
            include_user_examples=bool(state.user_id),
            include_schema_examples=True,
            similarity_boost_user_examples=0.1
        )

        # Enhanced configuration for retry requests
        if state.is_retry_request:
            # Be more lenient for retry to find more options
            config.vector_similarity_threshold = 0.35
            config.max_tables = 8
            config.max_total_examples = 10
            state.thinking.append("🔧 Using enhanced retry configuration")

        # Enhanced configuration for reanalysis
        if getattr(state, 'needs_reanalysis', False):
            config.max_tables = 6
            config.max_total_examples = 12
            state.thinking.append("🔍 Using enhanced reanalysis configuration")

        # Call the enhanced service
        result = await schema_service.retrieve_schema_with_examples(
            query_text=state.query,
            directives=state.directives,
            entities=state.entities,
            user_id=state.user_id,
            config=config
        )

        # Update state with comprehensive results
        state.query_schema = result.schema_structure
        state.few_shot_examples = result.examples  # Pre-ranked examples!
        state.schema_metadata = result.retrieval_metadata
        state.performance_metrics = result.performance_metrics

        # Enhanced logging
        tables_found = len(result.schema_structure.get("tables", {}))
        examples_found = len(result.examples)

        state.thinking.append(f"✅ Enhanced retrieval: {tables_found} tables, {examples_found} examples")

        if result.retrieval_metadata.get("cache_hit"):
            state.thinking.append("💾 Used cached results for better performance")

        if result.retrieval_metadata.get("embedding_reused"):
            state.thinking.append("⚡ Reused embeddings for optimal performance")

        # Enhanced error handling
        if tables_found == 0:
            state.thinking.append("❌ No relevant tables found with enhanced service")
            state.no_schema_found = True

        # Reset reselection flags
        if hasattr(state, 'needs_schema_reselection'):
            state.needs_schema_reselection = False

        return state

    except Exception as e:
        logger.error(f"Error in enhanced schema retrieval: {str(e)}", exc_info=True)
        state.thinking.append(f"❌ Enhanced schema retrieval failed: {str(e)}")

        # Fallback to setting no schema found
        state.query_schema = None
        state.few_shot_examples = []
        state.no_schema_found = True

        return state
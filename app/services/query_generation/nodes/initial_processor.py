import asyncio
import re
from typing import List
from app.core.profiling import timeit
from app.core.logging import logger
from app.services.query_generation.nodes import intent_classifier, enhanced_schema_retriever

def extract_directives(query: str) -> List[str]:
    """Extract directives from the query text using regex."""
    directives = []
    # Match @DIRECTIVE patterns
    directive_matches = re.findall(r'@([A-Z][A-Z0-9_]*)', query.upper())
    for directive in directive_matches:
        if directive not in directives:
            directives.append(directive)
    return directives

@timeit
async def process_initial_request(state):
    """
    Parallel processor for initial request handling.
    Runs intent classification and schema retrieval concurrently to reduce latency.
    """
    try:
        # 1. Pre-processing (fast, synchronous)
        if not state.directives:
            state.directives = extract_directives(state.query)
            
        state.thinking.append("🚀 Starting parallel execution: Intent Classification + Schema Retrieval")
        
        # 2. Parallel Execution
        # We pass the same state to both. Since they modify different fields (mostly),
        # and list appends (thinking) are safe in asyncio, this is acceptable.
        # However, to be safer and cleaner, we could copy state, but Pydantic deepcopy is slow.
        # We rely on the fact that they touch disjoint parts of the state.
        
        await asyncio.gather(
            intent_classifier.classify_intent(state),
            enhanced_schema_retriever.retrieve_schema_with_examples(state)
        )
        
        state.thinking.append("✅ Parallel execution completed")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in parallel initial processing: {str(e)}", exc_info=True)
        state.thinking.append(f"❌ Error in parallel processing: {str(e)}")
        return state

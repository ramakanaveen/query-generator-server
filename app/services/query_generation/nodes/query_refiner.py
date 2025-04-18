# app/services/query_generation/nodes/query_refiner.py
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate

from app.core.profiling import timeit
from app.services.query_generation.prompts.refiner_prompts import REFINER_PROMPT_TEMPLATE
from app.core.logging import logger
@timeit
async def refine_query(state):
    """
    Create improved guidance for query generation based on validation errors.
    
    Args:
        state: The current state of the workflow
        
    Returns:
        Updated state with refinement guidance
    """
    try:
        generated_query = state.generated_query
        validation_errors = state.validation_errors
        database_type = state.database_type
        llm = state.llm
        
        # Add thinking step
        state.thinking.append("Getting guidance based on validation errors...")
        state.refinement_count += 1
        # Use LLM to get refinement guidance
        prompt = ChatPromptTemplate.from_template(REFINER_PROMPT_TEMPLATE)
        chain = prompt | llm
        
        # Prepare the input for the LLM
        input_data = {
            "query": generated_query,
            "errors": validation_errors,
            "database_type": database_type
        }
        
        # Get the response from the LLM
        response = await chain.ainvoke(input_data)
        refinement_guidance = response.content.strip()
        
        # Store the original query and errors for reference
        state.original_query = generated_query
        state.original_errors = validation_errors
        
        # Add refinement guidance to state
        state.refinement_guidance = refinement_guidance
        state.thinking.append(f"Refinement guidance: {refinement_guidance}")
        
        return state
    
    except Exception as e:
        logger.error(f"Error in query refiner: {str(e)}")
        state.thinking.append(f"Error getting refinement guidance: {str(e)}")
        # Still return the state to continue the workflow
        return state
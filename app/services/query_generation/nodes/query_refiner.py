# app/services/query_generation/nodes/query_refiner.py
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate

from app.core.profiling import timeit
from app.core.logging import logger
from app.services.query_generation.prompts.refiner_prompts import ENHANCED_REFINER_PROMPT_TEMPLATE


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
        original_query = state.query
        schema = state.query_schema
        validation_errors = state.validation_errors
        detailed_feedback = state.detailed_feedback if hasattr(state, 'detailed_feedback') else []
        database_type = state.database_type
        llm = state.llm

        # Add thinking step
        state.thinking.append("Getting guidance based on validation errors...")
        state.refinement_count += 1

        # Format detailed feedback for the LLM
        formatted_feedback = "\n".join(detailed_feedback) if detailed_feedback else "\n".join(validation_errors)

        # Check if we have an LLM-suggested correction
        llm_correction_guidance = ""
        if hasattr(state, 'llm_corrected_query') and state.llm_corrected_query:
            llm_correction_guidance = f"""
            The LLM validator has suggested this correction as a starting point:
            ```
            {state.llm_corrected_query}
            ```
            You can use this as a reference, but make sure to carefully address all the issues mentioned in the validation feedback.
            """

        # Create an enhanced prompt that includes detailed feedback
        prompt = ChatPromptTemplate.from_template(ENHANCED_REFINER_PROMPT_TEMPLATE)
        chain = prompt | llm

        # Prepare the input for the LLM
        input_data = {
            "query": original_query,
            "generated_query": generated_query,
            "detailed_feedback": formatted_feedback,
            "database_type": database_type,
            "llm_correction_guidance": llm_correction_guidance,
            "schema": schema
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
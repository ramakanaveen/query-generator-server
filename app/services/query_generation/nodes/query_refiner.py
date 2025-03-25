# app/services/query_generation/nodes/query_refiner.py
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from app.services.query_generation.prompts.refiner_prompts import REFINER_PROMPT_TEMPLATE
from app.core.logging import logger

async def refine_query(state):
    """
    Refine the query if validation fails.
    
    Args:
        state: The current state of the workflow
        
    Returns:
        Updated state with refined query
    """
    try:
        generated_query = state.generated_query
        validation_errors = state.validation_errors
        database_type = state.database_type
        llm = state.llm
        
        # Add thinking step
        state.thinking.append("Refining query based on validation errors...")
        
        # Use LLM to refine the query
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
        refined_query = response.content.strip()
        
        # Update state with refined query
        state.generated_query = refined_query
        state.thinking.append(f"Refined query: {refined_query}")
        
        # Reset validation result to be checked again
        state.validation_result = None
        state.validation_errors = []
        
        return state
    
    except Exception as e:
        logger.error(f"Error in query refiner: {str(e)}")
        state.thinking.append(f"Error refining query: {str(e)}")
        # Still return the state to continue the workflow
        return state
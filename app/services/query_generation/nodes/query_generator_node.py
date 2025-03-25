# app/services/query_generation/nodes/query_generator_node.py (RENAMED NODE FILE)
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from app.services.query_generation.prompts.generator_prompts import GENERATOR_PROMPT_TEMPLATE
from app.core.logging import logger

async def generate_query(state):
    """
    Generate a database query based on the analysis and schema.
    
    Args:
        state: The current state of the workflow
        
    Returns:
        Updated state with generated query
    """
    try:
        query = state.query
        directives = state.directives
        entities = state.entities
        intent = state.intent
        schema = state.schema
        database_type = state.database_type
        llm = state.llm
        
        # Add thinking step
        state.thinking.append("Generating database query...")
        
        # Use LLM to generate the query
        prompt = ChatPromptTemplate.from_template(GENERATOR_PROMPT_TEMPLATE)
        chain = prompt | llm
        
        # Prepare the input for the LLM
        input_data = {
            "query": query,
            "directives": directives,
            "entities": entities,
            "intent": intent,
            "schema": schema,
            "database_type": database_type,
            "examples": schema.get("examples", [])
        }
        
        # Get the response from the LLM
        response = await chain.ainvoke(input_data)
        generated_query = response.content.strip()
        
        # Update state with generated query
        state.generated_query = generated_query
        state.thinking.append(f"Generated query: {generated_query}")
        
        return state
    
    except Exception as e:
        logger.error(f"Error in query generator: {str(e)}")
        state.thinking.append(f"Error generating query: {str(e)}")
        # Still return the state to continue the workflow
        state.generated_query = f"// Error generating query: {str(e)}"
        return state
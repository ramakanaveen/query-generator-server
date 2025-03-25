# app/services/query_generation/nodes/query_analyzer.py
from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate
from app.services.query_generation.prompts.anlyzer_prompts import ANALYZER_PROMPT_TEMPLATE
from app.core.logging import logger

# app/services/query_generation/nodes/query_analyzer.py
from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate
from app.services.query_generation.prompts.anlyzer_prompts import ANALYZER_PROMPT_TEMPLATE
from app.core.logging import logger

async def analyze_query(state):
    """
    Analyze the natural language query to extract directives, entities, and intent.
    
    Args:
        state: The current state of the workflow
        
    Returns:
        Updated state with extracted directives, entities, and intent
    """
    try:
        query = state.query  # Use attribute access instead of dictionary access
        llm = state.llm
        
        # Add thinking step
        state.thinking.append("Analyzing query to extract directives, entities, and intent...")
        
        # Extract directives from the query (simple regex approach)
        directives = []
        words = query.split()
        for word in words:
            if word.startswith('@'):
                directives.append(word.strip('@.,?!'))
        
        # Update state with directives
        state.directives = directives
        state.thinking.append(f"Extracted directives: {directives}")
        
        # Use LLM to analyze the query for entities and intent
        prompt = ChatPromptTemplate.from_template(ANALYZER_PROMPT_TEMPLATE)
        chain = prompt | llm
        
        # Prepare the input for the LLM
        input_data = {
            "query": query,
            "directives": directives,
            "database_type": state.database_type
        }
        
        # Get the response from the LLM
        response = await chain.ainvoke(input_data)
        response_text = response.content
        
        # Parse the response for entities and intent
        # Assuming the response has a format like:
        # Entities: entity1, entity2, ...
        # Intent: intent
        entities = []
        intent = None
        
        lines = response_text.strip().split('\n')
        for line in lines:
            if line.startswith('Entities:'):
                entities_str = line.replace('Entities:', '').strip()
                entities = [e.strip() for e in entities_str.split(',') if e.strip()]
            elif line.startswith('Intent:'):
                intent = line.replace('Intent:', '').strip()
        
        # Update state with entities and intent
        state.entities = entities
        state.intent = intent
        state.thinking.append(f"Extracted entities: {entities}")
        state.thinking.append(f"Identified intent: {intent}")
        
        return state
    
    except Exception as e:
        logger.error(f"Error in query analyzer: {str(e)}")
        state.thinking.append(f"Error analyzing query: {str(e)}")
        # Still return the state to continue the workflow
        return state
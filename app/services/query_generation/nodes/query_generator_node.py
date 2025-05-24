# app/services/query_generation/nodes/query_generator_node.py
from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate

from app.core.profiling import timeit
from app.services.query_generation.prompts.generator_prompts import GENERATOR_PROMPT_TEMPLATE, REFINED_PROMPT_TEMPLATE
from app.core.logging import logger
from app.services.feedback_manager import FeedbackManager

def format_conversation_history(history):
    """Format conversation history for the generator prompt with better follow-up handling."""
    if not history:
        return ""

    formatted = "Conversation Context:\n"

    # Add the last query and its response for better context
    last_user_query = None
    last_assistant_response = None

    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "user":
            last_user_query = content
        elif role == "assistant" and last_user_query:  # Only capture response to a known query
            last_assistant_response = content

            # Add this query-response pair to context
            formatted += f"User asked: {last_user_query}\n"
            formatted += f"System generated query: {content}\n\n"

            # Reset for next pair
            last_user_query = None

    # If we have a dangling user query with no response
    if last_user_query:
        formatted += f"User asked: {last_user_query}\n"

    return formatted

def format_few_shot_examples(examples):
    """Format few-shot examples for the generator prompt."""
    if not examples or len(examples) == 0:
        return ""
        
    formatted = "Few-Shot Examples:\n"
    for i, example in enumerate(examples, 1):
        formatted += f"Example {i}:\n"
        formatted += f"User query: {example.get('original_query', '')}\n"
        formatted += f"Generated query: {example.get('generated_query', '')}\n\n"
    return formatted

@timeit
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
        schema = state.query_schema
        database_type = state.database_type
        conversation_history = state.conversation_history
        user_id = state.user_id if hasattr(state, 'user_id') else None
        conversation_summary = state.conversation_summary if hasattr(state, 'conversation_summary') else None
        llm = state.llm
        
        # Check if this is a refined generation attempt
        is_refinement = hasattr(state, 'refinement_guidance') and state.refinement_guidance
        
        # Add thinking step
        if is_refinement:
            state.thinking.append("Generating improved query based on refinement guidance...")
        else:
            state.thinking.append("Generating database query...")
            
        # Check if schema is None
        if schema is None:
            state.thinking.append("Cannot generate query: No relevant schema found for this query")
            state.generated_query = "// I don't know how to generate a query for this request. " + \
                                   "I couldn't find any relevant tables in the available schemas."
            return state
        
        # Format conversation history
        conversation_context = format_conversation_history(conversation_history)
        
        # Add conversation summary if available
        if conversation_summary:
            summary_context = f"Previous Conversation Summary: {conversation_summary}\n\n"
            state.thinking.append(f"Using conversation summary: {conversation_summary[:100]}...")
        else:
            summary_context = ""
        
        # Find similar verified queries for few-shot learning
        similar_examples = []
        if not is_refinement:  # Only use few-shot for initial generation, not refinement
            try:
                feedback_manager = FeedbackManager()
                similar_examples = await feedback_manager.find_similar_verified_queries(
                    query_text=query,
                    user_id=user_id,
                    similarity_threshold=0.6,
                    limit=3
                )
                
                if similar_examples and len(similar_examples) > 0:
                    state.thinking.append(f"Found {len(similar_examples)} similar verified queries for few-shot learning")
                    for i, example in enumerate(similar_examples, 1):
                        note = "user's own example" if example.get('is_user_specific', False) else "shared example"
                        state.thinking.append(
                            f"Example {i} ({note}): '{example['original_query']}' → "
                            f"'{example['generated_query']}' (similarity: {example['similarity']:.2f})"
                        )
            except Exception as e:
                logger.error(f"Error finding few-shot examples: {str(e)}")
                state.thinking.append(f"Could not retrieve few-shot examples: {str(e)}")
        
        # Format few-shot examples
        few_shot_examples = format_few_shot_examples(similar_examples)
        
        # Use LLM to generate the query
        if is_refinement:
            # Use a specialized prompt that includes refinement guidance
            prompt = ChatPromptTemplate.from_template(REFINED_PROMPT_TEMPLATE)
            chain = prompt | llm
            
            # Prepare the input with refinement guidance
            input_data = {
                "query": query,
                "schema": schema,
                "database_type": database_type,
                "original_errors": state.original_errors,
                "refinement_guidance": state.refinement_guidance,
                # Add detailed feedback if available
                "detailed_feedback": "\n".join(state.detailed_feedback) if hasattr(state, 'detailed_feedback') else ""
            }
        else:
            # Use the standard prompt with few-shot examples
            # Construct the final prompt with all components
            final_prompt_template = f"{GENERATOR_PROMPT_TEMPLATE}\n\n{few_shot_examples}\n{summary_context}{conversation_context}"
            prompt = ChatPromptTemplate.from_template(final_prompt_template)
            chain = prompt | llm
            
            # Prepare the standard input
            input_data = {
                "query": query,
                "directives": directives,
                "entities": entities,
                "intent": intent,
                "schema": schema,
                "database_type": database_type,
                "examples": schema.get("examples", []),
                "conversation_context": conversation_context
            }
        
        # Get the response from the LLM
        response = await chain.ainvoke(input_data)
        generated_query = response.content.strip()
        
        # Update state with generated query
        state.generated_query = generated_query
        state.thinking.append(f"Generated query: {generated_query}")
        
        # Reset validation result since we have a new query
        state.validation_result = None
        state.validation_errors = []
        
        return state
    
    except Exception as e:
        logger.error(f"Error in query generator: {str(e)}", exc_info=True)
        state.thinking.append(f"Error generating query: {str(e)}")
        # Still return the state to continue the workflow
        state.generated_query = f"// Error generating query: {str(e)}"
        return state
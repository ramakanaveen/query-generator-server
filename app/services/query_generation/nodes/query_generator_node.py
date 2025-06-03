# app/services/query_generation/nodes/query_generator_node.py
from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate

from app.core.profiling import timeit
from app.services.query_generation.prompts.generator_prompts import GENERATOR_PROMPT_TEMPLATE, get_complexity_guidance
from app.services.query_generation.prompts.refiner_prompts import ENHANCED_REFINER_PROMPT_TEMPLATE
from app.services.query_generation.prompts.retry_prompts import RETRY_GENERATION_PROMPT
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

def format_feedback_trail(feedback_trail):
    """Format feedback trail for retry prompts."""
    if not feedback_trail:
        return "No previous corrections."

    formatted = "Previous Corrections Applied:\n"
    for i, feedback in enumerate(feedback_trail, 1):
        formatted += f"{i}. Issue: {feedback.get('issue_type', 'Unknown')}\n"
        formatted += f"   Feedback: {feedback.get('feedback', '')}\n"
        formatted += f"   Correction: {feedback.get('correction', '')}\n"
        formatted += f"   Learning: {feedback.get('learning_point', '')}\n\n"

    return formatted

def format_conversation_essence(essence):
    """Format conversation essence for prompts."""
    if not essence:
        return ""

    formatted = "Conversation Context:\n"

    if essence.get("original_intent"):
        formatted += f"Original Intent: {essence['original_intent']}\n"

    if essence.get("current_understanding"):
        formatted += f"Current Understanding: {essence['current_understanding']}\n"

    if essence.get("key_context"):
        formatted += f"Key Context: {', '.join(essence['key_context'])}\n"

    return formatted

@timeit
async def generate_query(state):
    """
    Enhanced query generation that handles both initial queries and retry requests.
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
        query_complexity = getattr(state, 'query_complexity', 'SINGLE_LINE')
        execution_plan = getattr(state, 'execution_plan', [])
        confidence = getattr(state, 'confidence', 'medium')
        reasoning = getattr(state, 'reasoning', '')

        # Check if this is a retry request
        is_retry = state.is_retry_request

        # Add thinking step
        if is_retry:
            state.thinking.append("🔄 Generating improved query based on feedback and conversation context...")
        else:
            if query_complexity != 'SINGLE_LINE' or execution_plan:
                state.thinking.append(f"🎯 Generating {query_complexity} query")
                if execution_plan:
                    state.thinking.append(f"📋 Plan: {' → '.join(execution_plan)}")
                if confidence != 'medium':
                    state.thinking.append(f"💭 Confidence: {confidence}")
                if reasoning:
                    state.thinking.append(f"🔍 Reasoning: {reasoning}")
            else:
                state.thinking.append("⚙️ Generating database query...")

        # Check if schema is None
        if schema is None:
            state.thinking.append("Cannot generate query: No relevant schema found for this query")
            state.generated_query = "// I don't know how to generate a query for this request. " + \
                                    "I couldn't find any relevant tables in the available schemas."
            return state

        # Check if this is a refinement attempt (validation failed)
        is_refinement = hasattr(state, 'refinement_guidance') and state.refinement_guidance

        # Generate query based on type (retry vs refinement vs initial)
        if is_retry:
            generated_query = await generate_retry_query(state, llm)
        elif is_refinement:
            generated_query = await generate_refinement_query(state, llm)
        else:
            generated_query = await generate_initial_query(state, llm)

        # Update state with generated query
        state.generated_query = generated_query
        state.thinking.append(f"Generated query: {generated_query}")

        # Reset validation result since we have a new query
        state.validation_result = None
        state.validation_errors = []

        return state

    except Exception as e:
        logger.error(f"Error in enhanced query generator: {str(e)}", exc_info=True)
        state.thinking.append(f"Error generating query: {str(e)}")
        # Still return the state to continue the workflow
        state.generated_query = f"// Error generating query: {str(e)}"
        return state

async def generate_retry_query(state, llm):
    """
    Generate query specifically for retry requests using enhanced context.
    """
    try:
        # Format all context for retry generation
        feedback_analysis = state.retry_context
        feedback_trail_text = format_feedback_trail(state.feedback_trail[:-1])  # Exclude current feedback
        key_context_text = ", ".join(state.key_context) if state.key_context else "None specified"

        # Get few-shot examples (same as initial flow for consistency)
        few_shot_examples = []
        if not state.is_retry_request:  # Only for initial, but we want them for retry too
            try:
                feedback_manager = FeedbackManager()
                few_shot_examples = await feedback_manager.find_similar_verified_queries(
                    query_text=state.query,
                    user_id=state.user_id,
                    similarity_threshold=0.6,
                    limit=3
                )

                if few_shot_examples:
                    state.thinking.append(f"📚 Using {len(few_shot_examples)} similar verified queries as examples")
            except Exception as e:
                logger.error(f"Error getting few-shot examples for retry: {str(e)}")

        few_shot_text = format_few_shot_examples(few_shot_examples)

        # Create retry-specific prompt
        prompt = ChatPromptTemplate.from_template(RETRY_GENERATION_PROMPT)
        chain = prompt | llm

        # Prepare input for retry generation
        input_data = {
            "original_query": state.query,
            "original_intent": state.original_intent or "Query refinement",
            "current_understanding": state.current_understanding or "Refining query based on feedback",
            "original_generated_query": state.original_generated_query,
            "user_feedback": state.user_feedback,
            "feedback_analysis": format_feedback_analysis(feedback_analysis),
            "feedback_trail": feedback_trail_text,
            "key_context": key_context_text,
            "schema": format_schema_for_retry(state.query_schema),
            "few_shot_examples": few_shot_text
        }

        # Get response from LLM
        response = await chain.ainvoke(input_data)
        generated_query = response.content.strip()

        state.thinking.append("✅ Generated retry query with full conversation context")
        return generated_query

    except Exception as e:
        logger.error(f"Error in retry query generation: {str(e)}", exc_info=True)
        state.thinking.append(f"Error in retry generation: {str(e)}")
        return f"// Error generating retry query: {str(e)}"

async def generate_initial_query(state, llm):
    """
    Generate query for initial requests (existing logic enhanced with conversation essence).
    """
    try:
        # Extract parameters
        query = state.query
        directives = state.directives
        entities = state.entities
        intent = state.intent
        schema = state.query_schema
        database_type = state.database_type
        conversation_history = state.conversation_history
        user_id = state.user_id
        query_complexity = getattr(state, 'query_complexity', 'SINGLE_LINE')
        execution_plan = getattr(state, 'execution_plan', [])

        # Format conversation history
        conversation_context = format_conversation_history(conversation_history)

        # Add conversation summary if available
        summary_context = ""
        if hasattr(state, 'conversation_summary') and state.conversation_summary:
            summary_context = f"Previous Conversation Summary: {state.conversation_summary}\n\n"
            state.thinking.append(f"Using conversation summary: {state.conversation_summary[:100]}...")

        # Add conversation essence context
        essence_context = ""
        if hasattr(state, 'conversation_essence') and state.conversation_essence:
            essence_context = format_conversation_essence(state.conversation_essence)
            state.thinking.append("📚 Using conversation essence for context")

        # Find similar verified queries for few-shot learning
        similar_examples = []
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

        # Get complexity-specific guidance and KDB notes
        complexity_guidance, kdb_notes = get_complexity_guidance(query_complexity, execution_plan)

        # Construct the final prompt with all components
        final_prompt_template = f"{GENERATOR_PROMPT_TEMPLATE}\n\n{few_shot_examples}\n{essence_context}\n{summary_context}{conversation_context}"
        prompt = ChatPromptTemplate.from_template(final_prompt_template)
        chain = prompt | llm

        # Prepare the input
        input_data = {
            "query": query,
            "directives": directives,
            "entities": entities,
            "intent": intent,
            "schema": schema,
            "database_type": database_type,
            "examples": schema.get("examples", []),
            "conversation_context": conversation_context,
            "complexity_guidance": complexity_guidance,
            "kdb_notes": kdb_notes,
            "query_complexity": query_complexity
        }

        # Get the response from the LLM
        response = await chain.ainvoke(input_data)
        generated_query = response.content.strip()

        return generated_query

    except Exception as e:
        logger.error(f"Error in initial query generation: {str(e)}", exc_info=True)
        return f"// Error generating initial query: {str(e)}"

async def generate_refinement_query(state, llm):
    """
    Generate query for refinement requests (when validation fails).
    This should actually be handled by the existing refiner node, not here.
    For now, we'll fallback to the original generation with a note about the errors.
    """
    try:
        # The refinement should actually be handled by query_refiner.py
        # This is a fallback in case refinement generation is called directly

        state.thinking.append("⚠️ Refinement should be handled by query_refiner node")

        # Use the initial query generation but with error context
        return await generate_initial_query(state, llm)

    except Exception as e:
        logger.error(f"Error in refinement query generation: {str(e)}", exc_info=True)
        return f"// Error generating refined query: {str(e)}"

def format_feedback_analysis(analysis):
    """Format feedback analysis for prompt inclusion."""
    if not analysis:
        return "No analysis available."

    formatted = f"""Analysis of Previous Attempt:
- Issue Type: {analysis.get('issue_type', 'Unknown')}
- Problem: {analysis.get('specific_problem', 'Not specified')}
- User's Actual Intent: {analysis.get('user_intent', 'Not clear')}
- Correction Strategy: {analysis.get('correction_strategy', 'General refinement')}
- Learning Point: {analysis.get('learning_point', 'Improve based on feedback')}"""

    return formatted

def format_schema_for_retry(schema):
    """Format schema information specifically for retry context."""
    if not schema:
        return "No schema information available."

    formatted = "Available Schema:\n"

    if isinstance(schema, dict) and "tables" in schema:
        for table_name, table_info in schema["tables"].items():
            formatted += f"\nTable: {table_name}\n"

            if isinstance(table_info, dict):
                if "description" in table_info:
                    formatted += f"Description: {table_info['description']}\n"

                if "columns" in table_info:
                    formatted += "Columns:\n"
                    for col in table_info["columns"][:10]:  # Limit to 10 columns
                        if isinstance(col, dict):
                            col_name = col.get("name", "unknown")
                            col_type = col.get("type", col.get("kdb_type", "unknown"))
                            col_desc = col.get("description", col.get("column_desc", ""))
                            formatted += f"  - {col_name} ({col_type}): {col_desc}\n"

    return formatted
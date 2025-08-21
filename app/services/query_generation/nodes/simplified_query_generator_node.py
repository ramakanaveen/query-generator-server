# app/services/query_generation/nodes/simplified_query_generator.py

from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate

from app.core.profiling import timeit
from app.services.query_generation.prompts.generator_prompts import GENERATOR_PROMPT_TEMPLATE, get_complexity_guidance
from app.services.query_generation.prompts.refiner_prompts import ENHANCED_REFINER_PROMPT_TEMPLATE
from app.services.query_generation.prompts.retry_prompts import RETRY_GENERATION_PROMPT
from app.core.logging import logger

def format_conversation_history(history):
    """Format conversation history for the generator prompt."""
    if not history:
        return ""

    formatted = "Conversation Context:\n"
    for msg in history[-3:]:  # Last 3 messages
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "user":
            formatted += f"User asked: {content}\n"
        elif role == "assistant":
            formatted += f"System generated: {content}\n"
    return formatted

def format_pre_fetched_examples(examples):
    """
    Format pre-fetched examples from enhanced schema service.
    These are already ranked by relevance!
    """
    if not examples or len(examples) == 0:
        return ""

    formatted = "Relevant Examples (pre-ranked by similarity):\n"
    for i, example in enumerate(examples, 1):
        # Enhanced schema service provides unified example format
        formatted += f"Example {i} (relevance: {example.relevance_score:.2f}):\n"
        formatted += f"User query: {example.natural_language}\n"
        formatted += f"Generated query: {example.generated_query}\n"

        # Add source info for debugging
        source_info = "user's example" if example.is_user_specific else f"{example.source_type}"
        formatted += f"Source: {source_info}\n\n"

    return formatted

def clean_generated_query(query: str) -> str:
    """Clean the generated query by removing markdown code blocks."""
    if not query:
        return query

    import re
    # Remove markdown code blocks
    query = re.sub(r'^```q\s*\n', '', query, flags=re.MULTILINE)
    query = re.sub(r'^```kdb\s*\n', '', query, flags=re.MULTILINE)
    query = re.sub(r'^```sql\s*\n', '', query, flags=re.MULTILINE)
    query = re.sub(r'^```\s*\n', '', query, flags=re.MULTILINE)
    query = re.sub(r'\n```\s*$', '', query)
    query = re.sub(r'^```\s*$', '', query, flags=re.MULTILINE)
    query = query.strip('`').strip()

    return query

@timeit
async def generate_query_with_prefetched_examples(state):
    """
    Simplified query generation that uses pre-fetched examples from enhanced schema service.

    This removes the duplicate example fetching that was happening in the old approach.
    """
    try:
        # Check if we have schema and examples from enhanced service
        if state.query_schema is None:
            state.thinking.append("❌ Cannot generate query: No schema from enhanced service")
            state.generated_query = "// No relevant schema found for this query"
            return state

        # Log what we received from enhanced schema service
        schema_tables = len(state.query_schema.get("tables", {}))
        pre_fetched_examples = len(getattr(state, 'few_shot_examples', []))

        state.thinking.append(f"📊 Using enhanced service results: {schema_tables} tables, {pre_fetched_examples} examples")

        # Determine generation approach
        if hasattr(state, 'generation_guidance') and state.generation_guidance:
            state.thinking.append("🎯 Using LLM guidance for targeted generation")
            generated_query = await generate_with_llm_guidance(state)
        elif state.is_retry_request:
            state.thinking.append("🔄 Generating retry query with enhanced context")
            generated_query = await generate_retry_query(state)
        else:
            state.thinking.append("⚙️ Generating initial query with pre-fetched examples")
            generated_query = await generate_initial_query_simplified(state)

        # Clean and set the query
        generated_query = clean_generated_query(generated_query)
        state.generated_query = generated_query
        state.thinking.append(f"✅ Generated query: {generated_query}")

        # Reset validation state for new query
        state.validation_result = None
        state.validation_errors = []

        return state

    except Exception as e:
        logger.error(f"Error in simplified query generator: {str(e)}", exc_info=True)
        state.thinking.append(f"❌ Error generating query: {str(e)}")
        state.generated_query = f"// Error generating query: {str(e)}"
        return state

async def generate_initial_query_simplified(state):
    """Generate initial query using pre-fetched examples (no duplicate fetching)."""
    try:
        # Extract parameters
        query = state.query
        directives = state.directives
        entities = state.entities
        intent = getattr(state, 'intent', 'query_generation')
        schema = state.query_schema
        database_type = state.database_type
        conversation_history = state.conversation_history
        query_complexity = getattr(state, 'query_complexity', 'SINGLE_LINE')
        execution_plan = getattr(state, 'execution_plan', [])

        # Format conversation context
        conversation_context = format_conversation_history(conversation_history)

        # ✅ KEY CHANGE: Use pre-fetched examples instead of fetching again
        pre_fetched_examples = getattr(state, 'few_shot_examples', [])
        few_shot_examples = format_pre_fetched_examples(pre_fetched_examples)

        state.thinking.append(f"📚 Using {len(pre_fetched_examples)} pre-ranked examples from enhanced service")

        # Get complexity-specific guidance
        complexity_guidance, kdb_notes = get_complexity_guidance(query_complexity, execution_plan)

        # Build enhanced prompt
        prompt = ChatPromptTemplate.from_template(GENERATOR_PROMPT_TEMPLATE)
        chain = prompt | state.llm

        # Prepare input with pre-fetched examples
        input_data = {
            "query": query,
            "directives": directives,
            "entities": entities,
            "intent": intent,
            "schema": format_schema_for_generation(schema),
            "database_type": database_type,
            "examples": schema.get("examples", []),  # Schema-level examples
            "conversation_context": conversation_context,
            "complexity_guidance": complexity_guidance,
            "kdb_notes": kdb_notes,
            "query_complexity": query_complexity
        }

        # Add pre-fetched examples to the template context
        enhanced_template = f"{GENERATOR_PROMPT_TEMPLATE}\n\n{few_shot_examples}\n{conversation_context}"
        enhanced_prompt = ChatPromptTemplate.from_template(enhanced_template)
        enhanced_chain = enhanced_prompt | state.llm

        # Generate the query
        response = await enhanced_chain.ainvoke(input_data)
        return response.content.strip()

    except Exception as e:
        logger.error(f"Error in simplified initial generation: {str(e)}")
        return f"// Error in simplified generation: {str(e)}"

async def generate_retry_query(state):
    """Generate retry query using enhanced context and pre-fetched examples."""
    try:
        # Format enhanced retry context
        feedback_analysis = getattr(state, 'retry_context', {})

        # ✅ KEY CHANGE: Use pre-fetched examples for retry too
        pre_fetched_examples = getattr(state, 'few_shot_examples', [])
        few_shot_text = format_pre_fetched_examples(pre_fetched_examples)

        state.thinking.append(f"🔄 Retry using {len(pre_fetched_examples)} enhanced examples")

        # Create retry prompt
        prompt = ChatPromptTemplate.from_template(RETRY_GENERATION_PROMPT)
        chain = prompt | state.llm

        input_data = {
            "original_query": state.query,
            "original_intent": getattr(state, 'original_intent', 'Query refinement'),
            "current_understanding": getattr(state, 'current_understanding', 'Improving query'),
            "original_generated_query": state.original_generated_query or 'Not available',
            "user_feedback": state.user_feedback or 'No feedback',
            "feedback_analysis": str(feedback_analysis),
            "feedback_trail": "Enhanced service context",
            "key_context": ", ".join(getattr(state, 'key_context', [])),
            "schema": format_schema_for_generation(state.query_schema),
            "few_shot_examples": few_shot_text
        }

        response = await chain.ainvoke(input_data)
        return response.content.strip()

    except Exception as e:
        logger.error(f"Error in simplified retry generation: {str(e)}")
        return f"// Error in retry generation: {str(e)}"

async def generate_with_llm_guidance(state):
    """Generate with LLM guidance using pre-fetched examples."""
    try:
        guidance = state.generation_guidance
        action_type = guidance.get("action_type", "general_generation")
        instructions = guidance.get("specific_instructions", "")

        state.thinking.append(f"🎯 LLM guidance: {action_type}")

        # Use pre-fetched examples in guidance generation too
        pre_fetched_examples = getattr(state, 'few_shot_examples', [])
        examples_context = format_pre_fetched_examples(pre_fetched_examples)

        # Create guidance-specific prompt
        guidance_prompt = f"""
You are applying specific LLM guidance for query generation.

Guidance Type: {action_type}
Instructions: {instructions}

Original Query: {state.query}
Available Schema: {format_schema_for_generation(state.query_schema)}

{examples_context}

Generate the query following the specific guidance above.
"""

        response = await state.llm.ainvoke(guidance_prompt)
        return response.content.strip()

    except Exception as e:
        logger.error(f"Error in LLM guidance generation: {str(e)}")
        return f"// Error in guidance generation: {str(e)}"

def format_schema_for_generation(schema):
    """Format schema information for generation prompts."""
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
                    for col in table_info["columns"][:10]:  # Limit for performance
                        if isinstance(col, dict):
                            col_name = col.get("name", "unknown")
                            col_type = col.get("type", col.get("kdb_type", "unknown"))
                            col_desc = col.get("description", col.get("column_desc", ""))
                            formatted += f"  - {col_name} ({col_type}): {col_desc}\n"

    return formatted
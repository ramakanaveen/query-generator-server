# app/services/query_generation/nodes/query_generator_node.py - Enhanced Version

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
    Enhanced query generation with LLM guidance integration.

    NEW FEATURES:
    - LLM guidance-based generation
    - Enhanced retry handling with specific feedback
    - Validator feedback integration
    - Context-aware refinement
    """
    try:
        query = state.query
        directives = state.directives
        entities = state.entities
        intent = getattr(state, 'intent', 'query_generation')
        schema = state.query_schema
        database_type = state.database_type
        conversation_history = state.conversation_history
        user_id = state.user_id if hasattr(state, 'user_id') else None
        conversation_summary = state.conversation_summary if hasattr(state, 'conversation_summary') else None
        llm = state.llm

        # Enhanced parameters from intelligent analyzer
        query_complexity = getattr(state, 'query_complexity', 'SINGLE_LINE')
        execution_plan = getattr(state, 'execution_plan', [])
        confidence = getattr(state, 'confidence', 'medium')
        reasoning = getattr(state, 'reasoning', '')

        # Check if schema is None
        if schema is None:
            state.thinking.append("❌ Cannot generate query: No relevant schema found for this query")
            state.generated_query = "// I don't know how to generate a query for this request. " + \
                                    "I couldn't find any relevant tables in the available schemas."
            return state

        # NEW: Check for LLM guidance from intelligent analyzer or validator feedback
        has_llm_guidance = hasattr(state, 'generation_guidance') and state.generation_guidance

        if has_llm_guidance:
            state.thinking.append("🎯 Applying LLM guidance for targeted regeneration...")
            generated_query = await generate_with_llm_guidance(state, llm)
        elif state.is_retry_request:
            state.thinking.append("🔄 Generating improved query based on user feedback...")
            generated_query = await generate_retry_query(state, llm)
        elif hasattr(state, 'refinement_guidance') and state.refinement_guidance:
            state.thinking.append("🔧 Generating refined query based on validation feedback...")
            generated_query = await generate_refinement_query(state, llm)
        else:
            # Standard generation with enhanced context
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

            generated_query = await generate_initial_query(state, llm)

        # Update state with generated query
        generated_query = clean_generated_query(generated_query)
        state.generated_query = generated_query
        state.thinking.append(f"✅ Generated query: {generated_query}")

        # Reset validation result since we have a new query
        state.validation_result = None
        state.validation_errors = []

        return state

    except Exception as e:
        logger.error(f"Error in enhanced query generator: {str(e)}", exc_info=True)
        state.thinking.append(f"❌ Error generating query: {str(e)}")
        # Still return the state to continue the workflow
        state.generated_query = f"// Error generating query: {str(e)}"
        return state
def clean_generated_query(query: str) -> str:
    """
    Clean the generated query by removing markdown code blocks and fixing formatting.

    Handles various LLM output formats:
    - ```q ... ```
    - ```kdb ... ```
    - ```sql ... ```
    - Extra whitespace and newlines
    """
    if not query:
        return query

    # Remove markdown code blocks
    query = remove_code_block_tags(query)

    return query

def remove_code_block_tags(query: str) -> str:
    """Remove various types of code block markdown tags."""
    import re

    # Remove ```q ... ``` blocks
    query = re.sub(r'^```q\s*\n', '', query, flags=re.MULTILINE)
    query = re.sub(r'^```kdb\s*\n', '', query, flags=re.MULTILINE)
    query = re.sub(r'^```sql\s*\n', '', query, flags=re.MULTILINE)
    query = re.sub(r'^```\s*\n', '', query, flags=re.MULTILINE)

    # Remove closing ```
    query = re.sub(r'\n```\s*$', '', query)
    query = re.sub(r'^```\s*$', '', query, flags=re.MULTILINE)

    # Remove any remaining ``` at start or end
    query = query.strip('`')
    query = query.strip()

    return query

async def generate_with_llm_guidance(state, llm):
    """
    Generate query with specific LLM guidance from intelligent analyzer.

    This handles targeted regeneration based on:
    - Validator feedback analysis
    - Schema correction guidance
    - Complexity escalation guidance
    """
    try:
        guidance = state.generation_guidance
        action_type = guidance.get("action_type", "general_generation")
        instructions = guidance.get("specific_instructions", "")
        complexity = guidance.get("complexity", state.query_complexity)
        reasoning = guidance.get("reasoning", "")

        state.thinking.append(f"🎯 LLM Guidance Type: {action_type}")
        state.thinking.append(f"📋 Instructions: {instructions}")

        # Select appropriate generation strategy based on guidance type
        if action_type == "fix_syntax_keep_complexity":
            return await generate_syntax_fix_query(state, llm, instructions, complexity)
        elif action_type == "fix_schema_references":
            return await generate_schema_fix_query(state, llm, instructions, complexity)
        elif action_type == "escalate_complexity":
            return await generate_complexity_escalated_query(state, llm, instructions, complexity)
        elif action_type == "revise_execution_plan":
            return await generate_revised_execution_query(state, llm, instructions, complexity)
        else:
            # General guidance - use enhanced initial generation
            return await generate_initial_query_with_guidance(state, llm, instructions, complexity)

    except Exception as e:
        logger.error(f"Error in LLM guidance generation: {str(e)}")
        # Fallback to initial generation
        return await generate_initial_query(state, llm)

async def generate_syntax_fix_query(state, llm, instructions, complexity):
    """Generate query focusing on syntax fixes while maintaining structure."""
    try:
        # Retrieve glossary
        glossary = {}
        schema = state.query_schema
        try:
            schema_group_name = schema.get("schema", "") if isinstance(schema, dict) else ""
            if schema_group_name:
                glossary = await get_glossary_for_schema_group(schema_group_name)
        except Exception as e:
            logger.error(f"Error loading glossary in syntax fix: {str(e)}")

        glossary_text = format_glossary_for_prompt(glossary)

        # Create syntax-focused prompt
        syntax_prompt = f"""
You are fixing KDB+/q syntax errors while maintaining the query's logic and complexity level.

Original Query: {state.query}
Previous Generated Query: {getattr(state, 'generated_query', 'Not available')}
Complexity Level: {complexity} (MAINTAIN THIS LEVEL)

Syntax Fix Instructions:
{instructions}

Available Schema:
{format_schema_for_generation(state.query_schema)}
{glossary_text}

Focus on:
1. Fixing syntax errors identified in the feedback
2. Maintaining the current query complexity ({complexity})
3. Preserving the original query logic and intent
4. Using correct KDB+/q syntax patterns

Generate the corrected query with proper syntax:
"""

        response = await llm.ainvoke(syntax_prompt)
        return response.content.strip()

    except Exception as e:
        logger.error(f"Error in syntax fix generation: {str(e)}")
        return f"// Error in syntax fix generation: {str(e)}"

async def generate_schema_fix_query(state, llm, instructions, complexity):
    """Generate query focusing on schema reference corrections."""
    try:
        # Retrieve glossary
        glossary = await get_glossary_from_schema(state.query_schema)
        glossary_text = format_glossary_for_prompt(glossary)

        # Create schema-focused prompt
        schema_prompt = f"""
You are fixing schema reference errors while maintaining query logic.

Original Query: {state.query}
Previous Generated Query: {getattr(state, 'generated_query', 'Not available')}
Complexity Level: {complexity} (MAINTAIN THIS LEVEL)

Schema Correction Instructions:
{instructions}

Available Schema (USE THESE EXACT REFERENCES):
{format_schema_for_generation(state.query_schema)}
{glossary_text}

Focus on:
1. Using correct table names from the available schema
2. Using correct column names from the available schema
3. Maintaining the current query complexity ({complexity})
4. Preserving the original query intent and logic

Generate the corrected query with proper schema references:
"""

        response = await llm.ainvoke(schema_prompt)
        return response.content.strip()

    except Exception as e:
        logger.error(f"Error in schema fix generation: {str(e)}")
        return f"// Error in schema fix generation: {str(e)}"

async def generate_complexity_escalated_query(state, llm, instructions, new_complexity):
    """Generate query with escalated complexity level."""
    try:
        # Get complexity guidance for the new level
        complexity_guidance, kdb_notes = get_complexity_guidance(new_complexity, state.execution_plan)

        # Retrieve glossary
        glossary = await get_glossary_from_schema(state.query_schema)
        glossary_text = format_glossary_for_prompt(glossary)

        # Create complexity escalation prompt
        escalation_prompt = f"""
You are escalating query complexity based on validation feedback.

Original Query: {state.query}
Previous Generated Query: {getattr(state, 'generated_query', 'Not available')}
Previous Complexity: {getattr(state, 'query_complexity', 'SINGLE_LINE')}
NEW Complexity Level: {new_complexity}

Escalation Instructions:
{instructions}

{complexity_guidance}

Available Schema:
{format_schema_for_generation(state.query_schema)}
{glossary_text}

{kdb_notes}

Focus on:
1. Escalating to {new_complexity} approach as instructed
2. Breaking down complex operations into logical steps
3. Using intermediate variables for clarity
4. Maintaining the original query intent

Generate the escalated complexity query:
"""

        response = await llm.ainvoke(escalation_prompt)
        return response.content.strip()

    except Exception as e:
        logger.error(f"Error in complexity escalation generation: {str(e)}")
        return f"// Error in complexity escalation generation: {str(e)}"

async def generate_revised_execution_query(state, llm, instructions, complexity):
    """Generate query with revised execution plan."""
    try:
        # Create execution revision prompt
        revision_prompt = f"""
You are revising the execution plan based on logic feedback while maintaining complexity.

Original Query: {state.query}
Previous Generated Query: {getattr(state, 'generated_query', 'Not available')}
Complexity Level: {complexity}

Execution Plan Revision Instructions:
{instructions}

Revised Execution Plan:
{' → '.join(state.execution_plan) if state.execution_plan else 'Standard execution'}

Available Schema:
{format_schema_for_generation(state.query_schema)}

Focus on:
1. Implementing the revised execution plan
2. Maintaining {complexity} complexity level
3. Ensuring the logic matches user intent better
4. Using proper KDB+/q patterns

Generate the query with revised execution approach:
"""

        response = await llm.ainvoke(revision_prompt)
        return response.content.strip()

    except Exception as e:
        logger.error(f"Error in execution revision generation: {str(e)}")
        return f"// Error in execution revision generation: {str(e)}"

async def generate_initial_query_with_guidance(state, llm, instructions, complexity):
    """Generate initial query with general LLM guidance."""
    try:
        # Enhance the standard generation with specific guidance
        guidance_text = f"""
Additional Generation Guidance:
{instructions}

Apply this guidance while following standard generation practices.
"""

        # Use the standard initial generation but with added guidance
        return await generate_initial_query(state, llm, additional_guidance=guidance_text)

    except Exception as e:
        logger.error(f"Error in guided initial generation: {str(e)}")
        return await generate_initial_query(state, llm)

async def generate_retry_query(state, llm):
    """
    Generate query specifically for retry requests using enhanced context.
    """
    try:
        # Format all context for retry generation
        feedback_analysis = getattr(state, 'retry_context', {})
        feedback_trail_text = format_feedback_trail(state.feedback_trail[:-1] if state.feedback_trail else [])
        key_context_text = ", ".join(state.key_context) if hasattr(state, 'key_context') and state.key_context else "None specified"

        # Get few-shot examples for consistency
        few_shot_examples = []
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

        # Retrieve glossary
        glossary = await get_glossary_from_schema(state.query_schema)
        glossary_text = format_glossary_for_prompt(glossary)

        # Create retry-specific prompt
        prompt = ChatPromptTemplate.from_template(RETRY_GENERATION_PROMPT)
        chain = prompt | llm

        # Prepare input for retry generation
        input_data = {
            "original_query": state.query,
            "original_intent": getattr(state, 'original_intent', 'Query refinement'),
            "current_understanding": getattr(state, 'current_understanding', 'Refining query based on feedback'),
            "original_generated_query": state.original_generated_query or 'Not available',
            "user_feedback": state.user_feedback or 'No specific feedback',
            "feedback_analysis": format_feedback_analysis(feedback_analysis),
            "feedback_trail": feedback_trail_text,
            "key_context": key_context_text,
            "schema": format_schema_for_retry(state.query_schema),
            "few_shot_examples": few_shot_text,
            "glossary": glossary_text
        }

        # Get response from LLM
        response = await chain.ainvoke(input_data)
        generated_query = response.content.strip()

        state.thinking.append("✅ Generated retry query with full conversation context")
        return generated_query

    except Exception as e:
        logger.error(f"Error in retry query generation: {str(e)}", exc_info=True)
        state.thinking.append(f"❌ Error in retry generation: {str(e)}")
        return f"// Error generating retry query: {str(e)}"

async def generate_initial_query(state, llm, additional_guidance=""):
    """
    Generate query for initial requests (enhanced version of existing logic).
    """
    try:
        # Extract parameters
        query = state.query
        directives = state.directives
        entities = state.entities
        intent = getattr(state, 'intent', 'query_generation')
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
            state.thinking.append(f"📝 Using conversation summary: {state.conversation_summary[:100]}...")

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
                state.thinking.append(f"📖 Found {len(similar_examples)} similar verified queries for few-shot learning")
                for i, example in enumerate(similar_examples, 1):
                    note = "user's own example" if example.get('is_user_specific', False) else "shared example"
                    state.thinking.append(
                        f"Example {i} ({note}): '{example['original_query']}' → "
                        f"'{example['generated_query']}' (similarity: {example['similarity']:.2f})"
                    )
        except Exception as e:
            logger.error(f"Error finding few-shot examples: {str(e)}")
            state.thinking.append(f"⚠️ Could not retrieve few-shot examples: {str(e)}")

        # Format few-shot examples
        few_shot_examples = format_few_shot_examples(similar_examples)

        # Get complexity-specific guidance and KDB notes
        complexity_guidance, kdb_notes = get_complexity_guidance(query_complexity, execution_plan)

        # Retrieve and format glossary
        glossary = {}
        try:
            schema_group_name = schema.get("schema", "") if isinstance(schema, dict) else ""
            if schema_group_name:
                glossary = await get_glossary_for_schema_group(schema_group_name)
                if glossary:
                    state.thinking.append(f"📖 Loaded business glossary with {len(glossary)} terms")
        except Exception as e:
            logger.error(f"Error loading glossary: {str(e)}")
            state.thinking.append(f"⚠️ Could not load glossary: {str(e)}")

        # Construct the final prompt with all components
        final_prompt_template = f"{GENERATOR_PROMPT_TEMPLATE}\n\n{few_shot_examples}\n{essence_context}\n{summary_context}{conversation_context}\n{additional_guidance}"
        prompt = ChatPromptTemplate.from_template(final_prompt_template)
        chain = prompt | llm

        # Prepare the input
        input_data = {
            "query": query,
            "directives": directives,
            "entities": entities,
            "intent": intent,
            "schema": format_schema_for_generation(schema),
            "database_type": database_type,
            "examples": schema.get("examples", []),
            "conversation_context": conversation_context,
            "complexity_guidance": complexity_guidance,
            "kdb_notes": kdb_notes,
            "query_complexity": query_complexity,
            "glossary": format_glossary_for_prompt(glossary)
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
    This is a legacy fallback - most refinement should be handled by intelligent_analyzer.
    """
    try:
        state.thinking.append("⚠️ Using legacy refinement generation - consider using intelligent analyzer instead")

        # Use the enhanced refinement prompt
        prompt = ChatPromptTemplate.from_template(ENHANCED_REFINER_PROMPT_TEMPLATE)
        chain = prompt | llm

        # Prepare input data
        input_data = {
            "query": state.query,
            "generated_query": getattr(state, 'generated_query', 'Not available'),
            "detailed_feedback": "\n".join(getattr(state, 'detailed_feedback', [])),
            "database_type": state.database_type,
            "llm_correction_guidance": getattr(state, 'llm_corrected_query', ''),
            "schema": format_schema_for_generation(state.query_schema)
        }

        # Get response from LLM
        response = await chain.ainvoke(input_data)
        return response.content.strip()

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
                    for col in table_info["columns"][:10]:  # Limit to 10 columns
                        if isinstance(col, dict):
                            col_name = col.get("name", "unknown")
                            col_type = col.get("type", col.get("kdb_type", "unknown"))
                            col_desc = col.get("description", col.get("column_desc", ""))
                            formatted += f"  - {col_name} ({col_type}): {col_desc}\n"

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

async def get_glossary_from_schema(schema: Dict[str, Any]) -> Dict[str, str]:
    """
    Helper function to retrieve glossary from schema object.

    Args:
        schema: The schema dictionary containing schema_group_name

    Returns:
        Dictionary of glossary terms, or empty dict if not available
    """
    try:
        schema_group_name = schema.get("schema", "") if isinstance(schema, dict) else ""
        if schema_group_name:
            return await get_glossary_for_schema_group(schema_group_name)
        return {}
    except Exception as e:
        logger.error(f"Error retrieving glossary from schema: {str(e)}")
        return {}

async def get_glossary_for_schema_group(schema_group_name: str) -> Dict[str, str]:
    """
    Retrieve business glossary for a given schema group.

    The glossary provides domain-specific term definitions to help the LLM
    understand business terminology when generating queries.

    Args:
        schema_group_name: The schema group identifier

    Returns:
        Dictionary of term definitions (term -> definition)
        Returns empty dict if no glossary exists for the schema group

    Note:
        Currently returns hardcoded sample data.
        TODO: Replace with actual database retrieval when DB method is implemented.
    """
    try:
        # TODO: Replace with actual database call when DB method is ready
        # Example: return await db.get_glossary(schema_group_name)

        # Hardcoded sample glossary for demonstration
        sample_glossary = {
            "TCA": "Transaction Cost Analysis - measurement of trading execution quality",
            "VWAP": "Volume Weighted Average Price - average price weighted by volume",
            "TWAP": "Time Weighted Average Price - average price weighted by time",
            "notional": "The total value of a position, calculated as price × quantity",
            "fill": "An executed order or a partial execution of an order",
            "slippage": "The difference between expected execution price and actual execution price",
            "spread": "The difference between the bid and ask price",
            "liquidity": "The ability to buy or sell an asset without causing significant price movement",
            "market impact": "The effect of a trade on the market price of the security",
            "alpha": "Excess return of an investment relative to a benchmark index"
        }

        logger.info(f"Retrieved glossary for schema group: {schema_group_name} ({len(sample_glossary)} terms)")
        return sample_glossary

    except Exception as e:
        logger.error(f"Error retrieving glossary for schema group {schema_group_name}: {str(e)}", exc_info=True)
        return {}

def format_glossary_for_prompt(glossary: Dict[str, str]) -> str:
    """
    Format glossary as text for inclusion in LLM prompts.

    Args:
        glossary: Dictionary of term definitions

    Returns:
        Formatted string for prompt inclusion, or empty string if no glossary
    """
    if not glossary:
        return ""

    formatted = "\n\nBusiness Glossary:\n"
    formatted += "The following terms have specific meanings in this domain. Use these definitions when interpreting the user's query:\n\n"

    for term, definition in glossary.items():
        formatted += f"  • {term}: {definition}\n"

    return formatted
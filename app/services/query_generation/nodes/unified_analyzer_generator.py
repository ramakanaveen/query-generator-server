"""
Unified Analyzer + Generator Node

Combines query analysis and code generation in a single LLM call using thinking mode.
Supports both KDB and SQL databases with appropriate prompts.
"""

import json
import re
from typing import Dict, Any, List, Optional
from langchain.prompts import ChatPromptTemplate

from app.core.profiling import timeit
from app.core.logging import logger
from app.services.query_generation.prompts.unified_prompts import (
    get_unified_prompt,
    get_database_specific_notes,
    RETRY_GUIDANCE_TEMPLATE
)


@timeit
async def unified_analyze_and_generate(state):
    """
    Unified node that performs analysis and generation in one LLM call.

    This node:
    1. Analyzes the query with schema context
    2. Determines complexity and execution plan
    3. Generates the query code
    4. Self-validates the generated code

    All in a single LLM call with thinking mode enabled.
    """
    try:
        query = state.query
        llm = state.llm
        database_type = state.database_type
        schema = state.query_schema

        # Log what we're doing
        if hasattr(state, 'retry_count') and state.retry_count > 0:
            state.thinking.append(f"🔄 Retry attempt {state.retry_count} with validation feedback...")
        else:
            state.thinking.append("🎯 Unified analyze + generate with schema context...")

        # Format all context for the prompt
        context = await prepare_unified_context(state)

        # Get the appropriate prompt template
        prompt_template = get_unified_prompt(database_type)

        # Build the full prompt
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm

        # Invoke with all context
        response = await chain.ainvoke(context)

        # Parse the JSON response
        result = parse_unified_response(response.content.strip())

        # Update state with all results
        update_state_from_unified_result(state, result)

        # Log results
        state.thinking.append(f"⚙️ Complexity: {state.query_complexity}")
        state.thinking.append(f"🔧 Confidence: {state.confidence_score}")
        state.thinking.append(f"📋 Execution plan: {' → '.join(state.execution_plan)}")

        # Add LLM thinking steps to state
        if 'thinking_steps' in result:
            for step in result['thinking_steps']:
                state.thinking.append(f"💭 {step}")

        return state

    except Exception as e:
        logger.error(f"Error in unified analyzer-generator: {str(e)}", exc_info=True)
        state.thinking.append(f"❌ Error in unified node: {str(e)}")

        # Set safe defaults on error
        state.generated_query = f"-- Error: {str(e)}"
        state.query_complexity = "SINGLE_LINE"
        state.execution_plan = ["Error in generation"]
        state.confidence_score = "low"

        return state


async def prepare_unified_context(state) -> Dict[str, Any]:
    """
    Prepare all context needed for the unified prompt.

    Args:
        state: Query generation state

    Returns:
        Dictionary with all context fields
    """
    # Format schema summary
    schema_summary = format_schema_summary(state.query_schema)

    # Format few-shot examples
    few_shot_examples = format_few_shot_examples(state)

    # Format conversation context
    conversation_context = format_conversation_context(state)

    # Get database-specific notes
    database_specific_notes = get_database_specific_notes(state.database_type)

    # Prepare retry guidance if this is a retry
    retry_guidance = ""
    if hasattr(state, 'retry_count') and state.retry_count > 0:
        retry_guidance = RETRY_GUIDANCE_TEMPLATE.format(
            validation_errors=state.validation_feedback or "Previous validation failed",
            previous_query=state.generated_query or "N/A",
            previous_complexity=state.query_complexity or "N/A",
            validation_feedback=state.validation_feedback or "Please address validation errors"
        )

    # Build context dictionary
    context = {
        "query": state.query,
        "database_type": state.database_type,
        "schema_summary": schema_summary,
        "few_shot_examples": few_shot_examples,
        "conversation_context": conversation_context,
        "business_glossary": "No business terms specified",  # Placeholder for future enhancement
        "database_specific_notes": database_specific_notes,
        "retry_guidance": retry_guidance
    }

    return context


def format_schema_summary(schema: Optional[Dict]) -> str:
    """
    Format schema information for the prompt.

    Args:
        schema: Schema dictionary

    Returns:
        Formatted schema string
    """
    if not schema or not isinstance(schema, dict):
        return "No schema information available"

    lines = []
    lines.append("Available Tables and Columns:\n")

    for table_name, table_info in schema.items():
        # Table name and description
        description = table_info.get('description', 'No description')
        lines.append(f"\n## Table: {table_name}")
        lines.append(f"Description: {description}")

        # Columns
        columns = table_info.get('columns', {})
        if columns:
            lines.append("Columns:")
            for col_name, col_info in columns.items():
                col_type = col_info.get('type', 'unknown')
                col_desc = col_info.get('description', '')
                lines.append(f"  - {col_name} ({col_type}): {col_desc}")
        else:
            lines.append("Columns: (schema not detailed)")

    return "\n".join(lines)


def format_few_shot_examples(state) -> str:
    """
    Format few-shot examples for the prompt.

    Args:
        state: Query generation state

    Returns:
        Formatted examples string
    """
    if not hasattr(state, 'few_shot_examples') or not state.few_shot_examples:
        return "No similar examples available"

    examples = state.few_shot_examples[:5]  # Limit to top 5
    lines = []
    lines.append("Similar Query Examples:\n")

    for i, example in enumerate(examples, 1):
        nl_query = example.get('original_query', example.get('natural_language_query', 'N/A'))
        db_query = example.get('generated_query', 'N/A')
        similarity = example.get('similarity_score', 0.0)

        lines.append(f"\nExample {i} (similarity: {similarity:.2f}):")
        lines.append(f"User asked: {nl_query}")
        lines.append(f"Generated: {db_query}")

    return "\n".join(lines)


def format_conversation_context(state) -> str:
    """
    Format conversation history and essence for the prompt.

    Args:
        state: Query generation state

    Returns:
        Formatted conversation context string
    """
    lines = []

    # Add conversation essence if available
    if hasattr(state, 'conversation_essence') and state.conversation_essence:
        lines.append("Conversation Context:")
        lines.append(f"Original Intent: {state.conversation_essence.get('original_intent', 'N/A')}")
        lines.append(f"Key Context: {state.conversation_essence.get('key_context', 'N/A')}")

    # Add recent conversation history
    if hasattr(state, 'conversation_history') and state.conversation_history:
        lines.append("\nRecent Messages:")
        for msg in state.conversation_history[-3:]:  # Last 3 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:100]  # First 100 chars
            lines.append(f"  {role}: {content}...")

    if not lines:
        return "No conversation context available"

    return "\n".join(lines)


def parse_unified_response(response_text: str) -> Dict[str, Any]:
    """
    Parse the JSON response from the unified LLM call.

    Args:
        response_text: Raw LLM response

    Returns:
        Parsed result dictionary

    Raises:
        ValueError: If parsing fails
    """
    try:
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON object found in response")

        # Parse JSON
        result = json.loads(json_str)

        # Validate required fields
        required_fields = ['query', 'complexity', 'execution_plan', 'confidence']
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        return result

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {str(e)}")
        logger.error(f"Response text: {response_text[:500]}...")

        # Fallback: Try to extract query manually
        query_match = re.search(r'"query":\s*"([^"]+)"', response_text)
        if query_match:
            return {
                'query': query_match.group(1),
                'complexity': 'SINGLE_LINE',
                'execution_plan': ['Parse error - extracted query only'],
                'confidence': 'low',
                'thinking_steps': [f'Parse error: {str(e)}'],
                'reasoning': 'Fallback parsing due to JSON error'
            }

        raise ValueError(f"Failed to parse unified response: {str(e)}")


def update_state_from_unified_result(state, result: Dict[str, Any]):
    """
    Update state with results from unified analysis-generation.

    Args:
        state: Query generation state
        result: Parsed result dictionary
    """
    # Core results
    state.generated_query = result.get('query', '').strip()
    state.query_complexity = result.get('complexity', 'SINGLE_LINE')
    state.execution_plan = result.get('execution_plan', [])
    state.confidence_score = result.get('confidence', 'medium')

    # Optional metadata
    if 'tables_used' in result:
        state.tables_used = result['tables_used']

    if 'columns_used' in result:
        state.columns_used = result['columns_used']

    if 'reasoning' in result:
        state.reasoning = result['reasoning']

    if 'query_type' in result:
        state.query_type = result['query_type']

    # Thinking steps are handled separately in main function

    logger.info(f"Generated query with complexity: {state.query_complexity}, confidence: {state.confidence_score}")

# app/services/query_generation/nodes/intelligent_analyzer.py
import re
from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate

from app.core.profiling import timeit
from app.core.logging import logger
from app.services.query_generation.prompts.intelligent_analyzer_prompts import (
    SCHEMA_AWARE_ANALYSIS_PROMPT,
    RETRY_ANALYSIS_PROMPT,
    VALIDATOR_FEEDBACK_ANALYSIS_PROMPT
)

@timeit
async def intelligent_analyze_query(state):
    """
    Enhanced query analyzer that makes decisions WITH schema context.

    This node handles:
    - Initial analysis with full schema context
    - Retry analysis with user feedback
    - Validator feedback analysis for re-analysis
    """
    try:
        query = state.query
        llm = state.llm
        database_type = state.database_type
        schema = state.query_schema

        # Determine the type of analysis needed
        if hasattr(state, 'needs_reanalysis') and state.needs_reanalysis:
            # This is validator feedback re-analysis
            state.thinking.append("🔍 Re-analyzing based on validator feedback...")
            await handle_validator_feedback_analysis(state, llm)

        elif state.is_retry_request:
            # This is user feedback retry analysis
            state.thinking.append("🔄 Analyzing retry request with full context...")
            await handle_retry_analysis(state, llm)

        else:
            # This is initial analysis with schema context
            state.thinking.append("🎯 Performing schema-aware analysis...")
            await handle_initial_analysis(state, llm)

        # Log the analysis results
        if hasattr(state, 'query_complexity'):
            state.thinking.append(f"⚙️ Complexity determined: {state.query_complexity}")
        if hasattr(state, 'query_type'):
            state.thinking.append(f"🔧 Query type: {state.query_type}")
        if hasattr(state, 'execution_plan') and state.execution_plan:
            state.thinking.append(f"📋 Execution plan: {' → '.join(state.execution_plan)}")

        return state

    except Exception as e:
        logger.error(f"Error in intelligent analyzer: {str(e)}", exc_info=True)
        state.thinking.append(f"❌ Error in intelligent analysis: {str(e)}")

        # Set safe defaults on error
        state.query_complexity = "SINGLE_LINE"
        state.query_type = "select_basic"
        state.execution_plan = ["Execute basic query"]
        state.confidence = "low"
        state.reasoning = f"Error in analysis: {str(e)}"

        return state

async def handle_initial_analysis(state, llm):
    """Handle initial schema-aware analysis for new requests."""
    try:
        # Format schema summary for the prompt
        schema_summary = format_schema_summary(state.query_schema)
        conversation_context = format_conversation_context(state)

        # Create the analysis prompt
        prompt = ChatPromptTemplate.from_template(SCHEMA_AWARE_ANALYSIS_PROMPT)
        chain = prompt | llm

        # Prepare input data
        input_data = {
            "query": state.query,
            "database_type": state.database_type,
            "directives": state.directives,
            "schema_summary": schema_summary,
            "conversation_context": conversation_context
        }

        # Get LLM analysis
        response = await chain.ainvoke(input_data)
        analysis_result = parse_schema_aware_analysis(response.content.strip())

        # Update state with analysis results
        update_state_from_analysis(state, analysis_result)

        # Log detailed analysis
        state.thinking.append(f"🎯 Confidence: {state.confidence}")
        if hasattr(state, 'schema_constraints') and state.schema_constraints:
            state.thinking.append(f"⚠️ Schema constraints: {state.schema_constraints}")

    except Exception as e:
        logger.error(f"Error in initial analysis: {str(e)}", exc_info=True)
        state.thinking.append(f"❌ Initial analysis error: {str(e)}")
        raise

async def handle_retry_analysis(state, llm):
    """Handle analysis for retry requests with user feedback."""
    try:
        # Format context for retry analysis
        schema_summary = format_schema_summary(state.query_schema)
        conversation_context = format_conversation_context(state)

        # Get previous execution plan if available
        previous_execution_plan = getattr(state, 'execution_plan', [])
        previous_complexity = getattr(state, 'query_complexity', 'SINGLE_LINE')

        # Create the retry analysis prompt
        prompt = ChatPromptTemplate.from_template(RETRY_ANALYSIS_PROMPT)
        chain = prompt | llm

        # Prepare input data
        input_data = {
            "original_query": state.query,
            "previous_query": state.original_generated_query or "Not available",
            "user_feedback": state.user_feedback or "",
            "previous_complexity": previous_complexity,
            "previous_execution_plan": previous_execution_plan,
            "schema_summary": schema_summary,
            "conversation_context": conversation_context
        }

        # Get retry analysis
        response = await chain.ainvoke(input_data)
        retry_result = parse_retry_analysis(response.content.strip())

        # Update state with retry analysis
        update_state_from_retry_analysis(state, retry_result)

        # Log retry analysis
        state.thinking.append(f"🔍 Feedback category: {getattr(state, 'feedback_category', 'Unknown')}")
        state.thinking.append(f"🔧 Improvement strategy: {getattr(state, 'improvement_strategy', 'Unknown')}")

    except Exception as e:
        logger.error(f"Error in retry analysis: {str(e)}", exc_info=True)
        state.thinking.append(f"❌ Retry analysis error: {str(e)}")
        raise

async def handle_validator_feedback_analysis(state, llm):
    """Handle analysis of validator feedback for re-analysis."""
    try:
        # Format context for validator feedback analysis
        schema_summary = format_schema_summary(state.query_schema)
        execution_plan = getattr(state, 'execution_plan', [])

        # Format validation errors
        validation_errors = format_validation_errors(state.validation_errors)

        # Create the validator feedback analysis prompt
        prompt = ChatPromptTemplate.from_template(VALIDATOR_FEEDBACK_ANALYSIS_PROMPT)
        chain = prompt | llm

        # Prepare input data
        input_data = {
            "original_query": state.query,
            "generated_query": state.generated_query or "Not available",
            "current_complexity": getattr(state, 'query_complexity', 'SINGLE_LINE'),
            "validation_errors": validation_errors,
            "schema_summary": schema_summary,
            "execution_plan": execution_plan
        }

        # Get validator feedback analysis
        response = await chain.ainvoke(input_data)
        validator_result = parse_validator_feedback_analysis(response.content.strip())

        # Update state with validator feedback analysis
        update_state_from_validator_analysis(state, validator_result)

        # Log validator analysis
        state.thinking.append(f"🔍 Issue type: {getattr(state, 'primary_issue_type', 'Unknown')}")
        state.thinking.append(f"🔧 Recommended action: {getattr(state, 'recommended_action', 'Unknown')}")

    except Exception as e:
        logger.error(f"Error in validator feedback analysis: {str(e)}", exc_info=True)
        state.thinking.append(f"❌ Validator analysis error: {str(e)}")
        raise

def format_schema_summary(schema):
    """Format schema information for prompts."""
    if not schema:
        return "No schema information available."

    summary_parts = []

    if isinstance(schema, dict) and "tables" in schema:
        summary_parts.append("Available Tables:")

        for table_name, table_info in schema["tables"].items():
            summary_parts.append(f"\nTable: {table_name}")

            if isinstance(table_info, dict):
                if "description" in table_info:
                    summary_parts.append(f"  Description: {table_info['description']}")

                if "columns" in table_info:
                    summary_parts.append("  Columns:")
                    for col in table_info["columns"][:10]:  # Limit to 10 columns
                        if isinstance(col, dict):
                            col_name = col.get("name", "unknown")
                            col_type = col.get("type", col.get("kdb_type", "unknown"))
                            col_desc = col.get("description", col.get("column_desc", ""))
                            summary_parts.append(f"    - {col_name} ({col_type}): {col_desc}")

    return "\n".join(summary_parts)

def format_conversation_context(state):
    """Format conversation context for prompts."""
    context_parts = []

    # Add conversation history if available
    if hasattr(state, 'conversation_history') and state.conversation_history:
        recent_messages = state.conversation_history[-3:]  # Last 3 messages
        if recent_messages:
            context_parts.append("Recent conversation:")
            for msg in recent_messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:100]  # Limit content length
                context_parts.append(f"  {role}: {content}...")

    # Add conversation essence if available
    if hasattr(state, 'conversation_essence') and state.conversation_essence:
        essence = state.conversation_essence
        if essence.get('original_intent'):
            context_parts.append(f"Original intent: {essence['original_intent']}")
        if essence.get('key_context'):
            context_parts.append(f"Key context: {', '.join(essence['key_context'][:5])}")

    return "\n".join(context_parts) if context_parts else "No conversation context"

def format_validation_errors(validation_errors):
    """Format validation errors for prompts."""
    if not validation_errors:
        return "No validation errors"

    if isinstance(validation_errors, list):
        return "\n".join([str(error) for error in validation_errors])
    else:
        return str(validation_errors)

def parse_schema_aware_analysis(response_text: str) -> Dict[str, Any]:
    """Parse the LLM response for schema-aware analysis."""
    result = {
        "entities": [],
        "complexity": "SINGLE_LINE",
        "execution_plan": [],
        "query_type": "select_basic",
        "confidence": "medium",
        "schema_constraints": "",
        "reasoning": ""
    }

    try:
        lines = response_text.split('\n')

        # ✅ FIXED: Better execution plan parsing
        execution_plan_section = False
        execution_plan_lines = []

        for i, line in enumerate(lines):
            line = line.strip()

            if line.startswith('ENTITIES:'):
                entities_str = line.replace('ENTITIES:', '').strip()
                if entities_str and entities_str != 'N/A':
                    entities = [e.strip() for e in entities_str.split(',') if e.strip()]
                    result["entities"] = entities

            elif line.startswith('COMPLEXITY:'):
                complexity = line.replace('COMPLEXITY:', '').strip()
                if complexity in ["SINGLE_LINE", "MULTI_LINE", "COMPLEX_ANALYSIS"]:
                    result["complexity"] = complexity

            elif line.startswith('EXECUTION_PLAN:'):
                # ✅ FIXED: Handle multiple execution plan formats
                execution_plan_section = True
                plan_str = line.replace('EXECUTION_PLAN:', '').strip()

                if plan_str and plan_str != 'N/A':
                    # Handle single-line format
                    result["execution_plan"] = parse_execution_plan_string(plan_str)
                else:
                    # Handle multi-line format (look for subsequent lines)
                    execution_plan_lines = []
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j].strip()
                        if next_line and not next_line.startswith(('QUERY_TYPE:', 'CONFIDENCE:', 'SCHEMA_CONSTRAINTS:', 'REASONING:')):
                            execution_plan_lines.append(next_line)
                        else:
                            break

                    if execution_plan_lines:
                        result["execution_plan"] = parse_execution_plan_lines(execution_plan_lines)

            elif line.startswith('QUERY_TYPE:'):
                execution_plan_section = False  # End of execution plan section
                query_type = line.replace('QUERY_TYPE:', '').strip()
                result["query_type"] = query_type

            elif line.startswith('CONFIDENCE:'):
                confidence = line.replace('CONFIDENCE:', '').strip()
                if confidence in ["high", "medium", "low"]:
                    result["confidence"] = confidence

            elif line.startswith('SCHEMA_CONSTRAINTS:'):
                result["schema_constraints"] = line.replace('SCHEMA_CONSTRAINTS:', '').strip()

            elif line.startswith('REASONING:'):
                result["reasoning"] = line.replace('REASONING:', '').strip()
    except Exception as e:
        logger.error(f"Error parsing schema-aware analysis: {str(e)}")
        logger.debug(f"Response text was: {response_text}")

        # ✅ FIXED: Provide meaningful fallback execution plan
        result["execution_plan"] = ["Execute query with error recovery"]

    return result

def parse_execution_plan_string(plan_str: str) -> List[str]:
    """Parse execution plan from a single string."""
    # Handle different formats
    if ',' in plan_str:
        # Comma-separated format
        return [step.strip() for step in plan_str.split(',') if step.strip()]
    elif ';' in plan_str:
        # Semicolon-separated format
        return [step.strip() for step in plan_str.split(';') if step.strip()]
    elif ' then ' in plan_str.lower():
        # "then" separated format
        return [step.strip() for step in plan_str.split(' then ') if step.strip()]
    elif ' → ' in plan_str:
        # Arrow separated format
        return [step.strip() for step in plan_str.split(' → ') if step.strip()]
    else:
        # Single step
        return [plan_str] if plan_str else []

def parse_execution_plan_lines(lines: List[str]) -> List[str]:
    """Parse execution plan from multiple lines."""
    steps = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove bullet points and numbering
        line = line.lstrip('- • * ')
        line = re.sub(r'^\d+\.\s*', '', line)  # Remove "1. ", "2. ", etc.

        if line:
            steps.append(line)

    return steps

def parse_retry_analysis(response_text: str) -> Dict[str, Any]:
    """Parse the LLM response for retry analysis."""
    result = {
        "feedback_category": "general",
        "root_cause": "",
        "improvement_strategy": "revise_execution_plan",
        "new_complexity": "keep_current",
        "new_execution_plan": [],
        "schema_changes": "",
        "specific_guidance": "",
        "reasoning": ""
    }

    try:
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()

            if line.startswith('FEEDBACK_CATEGORY:'):
                result["feedback_category"] = line.replace('FEEDBACK_CATEGORY:', '').strip()

            elif line.startswith('ROOT_CAUSE:'):
                result["root_cause"] = line.replace('ROOT_CAUSE:', '').strip()

            elif line.startswith('IMPROVEMENT_STRATEGY:'):
                result["improvement_strategy"] = line.replace('IMPROVEMENT_STRATEGY:', '').strip()

            elif line.startswith('NEW_COMPLEXITY:'):
                result["new_complexity"] = line.replace('NEW_COMPLEXITY:', '').strip()

            elif line.startswith('NEW_EXECUTION_PLAN:'):
                plan_str = line.replace('NEW_EXECUTION_PLAN:', '').strip()
                if plan_str and plan_str != 'N/A':
                    if ',' in plan_str:
                        result["new_execution_plan"] = [step.strip() for step in plan_str.split(',')]
                    else:
                        result["new_execution_plan"] = [plan_str]

            elif line.startswith('SCHEMA_CHANGES:'):
                result["schema_changes"] = line.replace('SCHEMA_CHANGES:', '').strip()

            elif line.startswith('SPECIFIC_GUIDANCE:'):
                result["specific_guidance"] = line.replace('SPECIFIC_GUIDANCE:', '').strip()

            elif line.startswith('REASONING:'):
                result["reasoning"] = line.replace('REASONING:', '').strip()

    except Exception as e:
        logger.error(f"Error parsing retry analysis: {str(e)}")
        logger.debug(f"Response text was: {response_text}")

    return result

def parse_validator_feedback_analysis(response_text: str) -> Dict[str, Any]:
    """Parse the LLM response for validator feedback analysis."""
    result = {
        "primary_issue_type": "general",
        "recommended_action": "revise_execution_plan",
        "specific_guidance": "",
        "complexity_recommendation": "keep_current",
        "schema_corrections": "",
        "reasoning": ""
    }

    try:
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()

            if line.startswith('PRIMARY_ISSUE_TYPE:'):
                result["primary_issue_type"] = line.replace('PRIMARY_ISSUE_TYPE:', '').strip()

            elif line.startswith('RECOMMENDED_ACTION:'):
                result["recommended_action"] = line.replace('RECOMMENDED_ACTION:', '').strip()

            elif line.startswith('SPECIFIC_GUIDANCE:'):
                result["specific_guidance"] = line.replace('SPECIFIC_GUIDANCE:', '').strip()

            elif line.startswith('COMPLEXITY_RECOMMENDATION:'):
                result["complexity_recommendation"] = line.replace('COMPLEXITY_RECOMMENDATION:', '').strip()

            elif line.startswith('SCHEMA_CORRECTIONS:'):
                result["schema_corrections"] = line.replace('SCHEMA_CORRECTIONS:', '').strip()

            elif line.startswith('REASONING:'):
                result["reasoning"] = line.replace('REASONING:', '').strip()

    except Exception as e:
        logger.error(f"Error parsing validator feedback analysis: {str(e)}")
        logger.debug(f"Response text was: {response_text}")

    return result

def update_state_from_analysis(state, analysis_result):
    """Update state with initial analysis results."""
    state.entities = analysis_result.get("entities", [])
    state.query_complexity = analysis_result.get("complexity", "SINGLE_LINE")
    state.execution_plan = analysis_result.get("execution_plan", [])
    state.query_type = analysis_result.get("query_type", "select_basic")
    state.confidence = analysis_result.get("confidence", "medium")
    state.schema_constraints = analysis_result.get("schema_constraints", "")
    state.reasoning = analysis_result.get("reasoning", "")

def update_state_from_retry_analysis(state, retry_result):
    """Update state with retry analysis results."""
    state.feedback_category = retry_result.get("feedback_category", "general")
    state.root_cause = retry_result.get("root_cause", "")
    state.improvement_strategy = retry_result.get("improvement_strategy", "revise_execution_plan")
    state.specific_guidance = retry_result.get("specific_guidance", "")
    state.reasoning = retry_result.get("reasoning", "")

    # Handle complexity changes
    new_complexity = retry_result.get("new_complexity", "keep_current")
    if new_complexity != "keep_current":
        state.query_complexity = new_complexity
        state.thinking.append(f"⚙️ Complexity updated to: {new_complexity}")

    # Handle execution plan changes
    new_plan = retry_result.get("new_execution_plan", [])
    if new_plan:
        state.execution_plan = new_plan
        state.thinking.append(f"📋 Execution plan updated")

    # Handle schema changes
    schema_changes = retry_result.get("schema_changes", "")
    if schema_changes:
        state.schema_changes_needed = schema_changes
        state.thinking.append(f"🔄 Schema changes needed: {schema_changes}")

def update_state_from_validator_analysis(state, validator_result):
    """Update state with validator feedback analysis results."""
    state.primary_issue_type = validator_result.get("primary_issue_type", "general")
    state.recommended_action = validator_result.get("recommended_action", "revise_execution_plan")
    state.specific_guidance = validator_result.get("specific_guidance", "")
    state.reasoning = validator_result.get("reasoning", "")

    # Handle complexity recommendations
    complexity_rec = validator_result.get("complexity_recommendation", "keep_current")
    if complexity_rec != "keep_current":
        if "escalate_to:" in complexity_rec:
            new_complexity = complexity_rec.split("escalate_to:")[-1].strip()
            state.query_complexity = new_complexity
            state.escalation_count = getattr(state, 'escalation_count', 0) + 1
            state.thinking.append(f"⬆️ Complexity escalated to: {new_complexity}")
        elif "simplify_to:" in complexity_rec:
            new_complexity = complexity_rec.split("simplify_to:")[-1].strip()
            state.query_complexity = new_complexity
            state.thinking.append(f"⬇️ Complexity simplified to: {new_complexity}")

    # Handle schema corrections
    schema_corrections = validator_result.get("schema_corrections", "")
    if schema_corrections:
        state.schema_corrections_needed = schema_corrections
        state.thinking.append(f"🔧 Schema corrections needed: {schema_corrections}")

    # Set generation guidance for the query generator
    state.generation_guidance = {
        "action_type": state.recommended_action,
        "specific_instructions": state.specific_guidance,
        "complexity": state.query_complexity,
        "reasoning": state.reasoning
    }
# app/services/query_generation/nodes/unified_query_analyzer.py

import re
from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate

from app.core.profiling import timeit
from app.core.logging import logger

from app.services.query_generation.prompts.unified_query_analyzer_prompts import UNIFIED_ANALYZER_PROMPT

@timeit
async def unified_analyze_query(state):
    """
    Unified LLM-based query analysis that handles intent classification and execution planning.
    This is the new analyzer that can be used instead of the original query_analyzer.
    """
    try:
        query = state.query
        llm = state.llm
        database_type = state.database_type

        # Extract directives using simple regex (still useful for context)
        directives = extract_directives(query)
        state.directives = directives

        # Add thinking step
        state.thinking.append("🔍 [UNIFIED] Performing comprehensive LLM analysis...")

        # Add context from conversation history if available
        previous_directives = []
        if hasattr(state, 'conversation_history') and state.conversation_history:
            for msg in state.conversation_history:
                if msg.get('role') == 'user':
                    directive_matches = re.findall(r'@([A-Z]+)', msg.get('content', ''))
                    for directive in directive_matches:
                        if directive not in previous_directives:
                            previous_directives.append(directive)

        # Use previous directives if current query has none
        if not directives and previous_directives:
            state.thinking.append(f"📝 Using previous directives: {previous_directives}")
            directives = previous_directives
            state.directives = directives

        # Prepare and execute unified analysis
        prompt = ChatPromptTemplate.from_template(UNIFIED_ANALYZER_PROMPT)
        chain = prompt | llm

        response = await chain.ainvoke({
            "query": query,
            "directives": directives,
            "database_type": database_type
        })

        # Parse the structured response
        analysis = parse_unified_response(response.content.strip())

        # Update state with comprehensive analysis
        state.intent_type = analysis.get('intent_type', 'query_generation')
        state.entities = analysis.get('entities', [])
        state.intent = analysis.get('query_type', 'select_basic')  # Backward compatibility

        # Enhanced analysis fields
        state.query_complexity = analysis.get('complexity', 'SINGLE_LINE')
        state.execution_plan = analysis.get('execution_plan', [])
        state.query_type = analysis.get('query_type', 'select_basic')
        state.confidence = analysis.get('confidence', 'medium')
        state.reasoning = analysis.get('reasoning', '')

        # Log comprehensive analysis
        state.thinking.append(f"🎯 Intent: {state.intent_type} (confidence: {state.confidence})")
        state.thinking.append(f"📊 Entities: {state.entities}")

        if state.intent_type == "query_generation":
            state.thinking.append(f"⚙️ Complexity: {state.query_complexity}")
            state.thinking.append(f"🔧 Type: {state.query_type}")
            state.thinking.append(f"📋 Plan: {state.execution_plan}")

        state.thinking.append(f"💭 Reasoning: {state.reasoning}")

        # Handle different intent types (reuse existing logic)
        if state.intent_type == "schema_description":
            await process_schema_description_intent(state, query, directives)
        elif state.intent_type == "help":
            await process_help_intent(state, query, directives)

        return state

    except Exception as e:
        logger.error(f"Error in unified query analyzer: {str(e)}", exc_info=True)
        state.thinking.append(f"❌ [UNIFIED] Analysis error: {str(e)}")

        # Fallback to original analyzer if available
        try:
            state.thinking.append("🔄 Falling back to original analyzer...")
            from app.services.query_generation.nodes.query_analyzer import analyze_query as original_analyze
            return await original_analyze(state)
        except Exception as fallback_error:
            logger.error(f"Fallback to original analyzer also failed: {str(fallback_error)}")

            # Set absolute safe defaults
            state.intent_type = "query_generation"
            state.entities = []
            state.intent = "select_basic"
            state.query_complexity = "SINGLE_LINE"
            state.execution_plan = ["Execute basic query"]
            state.query_type = "select_basic"
            state.confidence = "low"
            state.reasoning = "Error in analysis, using defaults"

            return state

def extract_directives(query):
    """Extract directives from the query text."""
    directives = []
    words = query.split()
    for word in words:
        if word.startswith('@'):
            directives.append(word.strip('@.,?!'))
    return directives

def parse_unified_response(response_text: str) -> Dict[str, Any]:
    """Parse the structured response from the unified analyzer."""
    analysis = {
        'intent_type': 'query_generation',
        'confidence': 'medium',
        'entities': [],
        'complexity': 'SINGLE_LINE',
        'execution_plan': [],
        'query_type': 'select_basic',
        'reasoning': ''
    }

    try:
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()

            if line.startswith('Intent_Type:'):
                intent_type = line.replace('Intent_Type:', '').strip()
                analysis['intent_type'] = intent_type

            elif line.startswith('Confidence:'):
                confidence = line.replace('Confidence:', '').strip()
                analysis['confidence'] = confidence

            elif line.startswith('Entities:'):
                entities_str = line.replace('Entities:', '').strip()
                if entities_str and entities_str != 'N/A':
                    entities = [e.strip() for e in entities_str.split(',') if e.strip()]
                    analysis['entities'] = entities

            elif line.startswith('Complexity:'):
                complexity = line.replace('Complexity:', '').strip()
                if complexity != 'N/A':
                    analysis['complexity'] = complexity

            elif line.startswith('Execution_Plan:'):
                plan_str = line.replace('Execution_Plan:', '').strip()
                if plan_str and plan_str != 'N/A':
                    if ',' in plan_str:
                        execution_plan = [step.strip() for step in plan_str.split(',') if step.strip()]
                    else:
                        execution_plan = [plan_str]
                    analysis['execution_plan'] = execution_plan

            elif line.startswith('Query_Type:'):
                query_type = line.replace('Query_Type:', '').strip()
                if query_type != 'N/A':
                    analysis['query_type'] = query_type

            elif line.startswith('Reasoning:'):
                reasoning = line.replace('Reasoning:', '').strip()
                analysis['reasoning'] = reasoning

        return analysis

    except Exception as e:
        logger.error(f"Error parsing unified response: {str(e)}")
        logger.debug(f"Response text was: {response_text}")
        return analysis

# Import existing helper functions to avoid duplication
async def process_schema_description_intent(state, query, directives):
    """Process schema description intent - reuse existing logic."""
    try:
        from app.services.query_generation.nodes.query_analyzer import (
            extract_table_targets, extract_column_targets, determine_detail_level
        )

        table_targets = extract_table_targets(query, directives)
        column_targets = extract_column_targets(query)
        detail_level = determine_detail_level(query)

        state.schema_targets = {
            "tables": table_targets,
            "columns": column_targets,
            "detail_level": detail_level
        }

        state.thinking.append(f"📋 Schema targets: tables={table_targets}, columns={column_targets}")
        state.thinking.append(f"📊 Detail level: {detail_level}")

    except Exception as e:
        logger.error(f"Error processing schema description intent: {str(e)}")
        state.schema_targets = {
            "tables": directives if directives else ["*ALL*"],
            "columns": [],
            "detail_level": "standard"
        }

async def process_help_intent(state, query, directives):
    """Process help intent - reuse existing logic."""
    try:
        from app.services.query_generation.nodes.query_analyzer import extract_help_topic

        help_topic = extract_help_topic(query)
        state.help_request = {
            "topic": help_topic,
            "directives": directives
        }

        state.thinking.append(f"❓ Help topic: {help_topic}")

    except Exception as e:
        logger.error(f"Error processing help intent: {str(e)}")
        state.help_request = {
            "topic": ["general"],
            "directives": directives
        }
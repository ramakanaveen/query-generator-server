# app/services/query_generation/nodes/intent_classifier.py

import re
from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate

from app.core.profiling import timeit
from app.core.logging import logger
from app.services.query_generation.prompts.intent_classifier_prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    RETRY_INTENT_ANALYSIS_PROMPT
)

@timeit
async def classify_intent(state):
    """
    LLM-based intent classification with conversation context and retry handling.

    This is the first node in the enhanced workflow that determines:
    - Primary intent type (query_generation, schema_description, help)
    - Whether this is a follow-up to previous queries
    - Retry request analysis if applicable
    """
    try:
        query = state.query
        llm = state.llm
        database_type = state.database_type

        # Extract directives using simple regex (still useful for context)
        directives = extract_directives(query)
        state.directives = directives

        # Add thinking step
        if state.is_retry_request:
            state.thinking.append("🔄 Analyzing retry request with user feedback...")
        else:
            state.thinking.append("🎯 Classifying user intent...")

        # Build conversation context
        conversation_context = build_conversation_context(state)

        # Handle retry requests with special analysis
        if state.is_retry_request:
            await handle_retry_intent_analysis(state, llm, conversation_context)
        else:
            await handle_initial_intent_classification(state, llm, directives, conversation_context)

        # Log the classification result
        state.thinking.append(f"📋 Intent classified as: {state.intent_type}")
        if hasattr(state, 'is_follow_up') and state.is_follow_up:
            state.thinking.append("🔗 Detected as follow-up to previous query")

        return state

    except Exception as e:
        logger.error(f"Error in intent classifier: {str(e)}", exc_info=True)
        state.thinking.append(f"❌ Error in intent classification: {str(e)}")

        # Set safe defaults on error
        state.intent_type = "query_generation"
        state.confidence = "low"
        state.is_follow_up = False

        return state

async def handle_initial_intent_classification(state, llm, directives, conversation_context):
    """Handle initial intent classification for new requests."""
    try:
        # Create the classification prompt
        prompt = ChatPromptTemplate.from_template(INTENT_CLASSIFICATION_PROMPT)
        chain = prompt | llm

        # Prepare input data
        input_data = {
            "query": state.query,
            "directives": directives,
            "database_type": state.database_type,
            "conversation_context": conversation_context,
            "is_retry": False,
            "user_feedback": ""
        }

        # Get LLM response
        response = await chain.ainvoke(input_data)
        classification_result = parse_intent_classification(response.content.strip())

        # Update state with classification results
        state.intent_type = classification_result.get("intent_type", "query_generation")
        state.confidence = classification_result.get("confidence", "medium")
        state.is_follow_up = classification_result.get("is_follow_up", False)
        state.classification_reasoning = classification_result.get("reasoning", "")
        state.conversation_context_summary = classification_result.get("conversation_context_summary", "")

        # Log detailed classification
        state.thinking.append(f"🎯 Confidence: {state.confidence}")
        if state.classification_reasoning:
            state.thinking.append(f"💭 Reasoning: {state.classification_reasoning}")

    except Exception as e:
        logger.error(f"Error in initial intent classification: {str(e)}", exc_info=True)
        state.thinking.append(f"❌ Classification error: {str(e)}")
        # Defaults are set in calling function

async def handle_retry_intent_analysis(state, llm, conversation_context):
    """Handle special analysis for retry requests."""
    try:
        # First do the standard intent classification
        await handle_initial_intent_classification(state, llm, state.directives, conversation_context)

        # Then do retry-specific analysis
        prompt = ChatPromptTemplate.from_template(RETRY_INTENT_ANALYSIS_PROMPT)
        chain = prompt | llm

        input_data = {
            "original_query": state.query,
            "previous_query": state.original_generated_query or "Not available",
            "user_feedback": state.user_feedback or "",
            "previous_intent": state.intent_type,
            "conversation_context": conversation_context
        }

        # Get retry analysis
        response = await chain.ainvoke(input_data)
        retry_analysis = parse_retry_analysis(response.content.strip())

        # Update state with retry analysis
        state.retry_intent_analysis = retry_analysis

        # Check if intent should change based on feedback
        if retry_analysis.get("intent_should_change", False):
            new_intent = retry_analysis.get("new_intent_type", state.intent_type)
            if new_intent != "same":
                state.intent_type = new_intent
                state.thinking.append(f"🔄 Intent changed to: {new_intent} based on feedback")

        # Store retry analysis details
        state.feedback_type = retry_analysis.get("feedback_type", "modification")
        state.preserve_context = retry_analysis.get("preserve_context", [])
        state.change_required = retry_analysis.get("change_required", "")

        state.thinking.append(f"🔍 Feedback type: {state.feedback_type}")
        if state.change_required:
            state.thinking.append(f"📝 Required changes: {state.change_required}")

    except Exception as e:
        logger.error(f"Error in retry intent analysis: {str(e)}", exc_info=True)
        state.thinking.append(f"❌ Retry analysis error: {str(e)}")

def extract_directives(query):
    """Extract directives from the query text using regex."""
    directives = []
    # Match @DIRECTIVE patterns
    directive_matches = re.findall(r'@([A-Z][A-Z0-9_]*)', query.upper())
    for directive in directive_matches:
        if directive not in directives:
            directives.append(directive)
    return directives

def build_conversation_context(state):
    """Build conversation context summary for intent classification."""
    context_parts = []

    # Add conversation history summary
    if hasattr(state, 'conversation_history') and state.conversation_history:
        recent_interactions = []
        for msg in state.conversation_history[-3:]:  # Last 3 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if role == 'user':
                recent_interactions.append(f"User: {content[:100]}...")
            elif role == 'assistant':
                recent_interactions.append(f"Assistant: {content[:100]}...")

        if recent_interactions:
            context_parts.append("Recent conversation:\n" + "\n".join(recent_interactions))

    # Add conversation essence if available
    if hasattr(state, 'conversation_essence') and state.conversation_essence:
        essence = state.conversation_essence
        if essence.get('original_intent'):
            context_parts.append(f"Original conversation intent: {essence['original_intent']}")
        if essence.get('key_context'):
            context_parts.append(f"Key context: {', '.join(essence['key_context'][:5])}")

    # Add user context if available
    if hasattr(state, 'user_id') and state.user_id:
        context_parts.append(f"User ID: {state.user_id}")

    return "\n\n".join(context_parts) if context_parts else "No previous conversation context"

def parse_intent_classification(response_text: str) -> Dict[str, Any]:
    """Parse the LLM response for intent classification."""
    result = {
        "intent_type": "query_generation",
        "confidence": "medium",
        "is_follow_up": False,
        "reasoning": "",
        "conversation_context_summary": ""
    }

    try:
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()

            if line.startswith('INTENT_TYPE:'):
                intent = line.replace('INTENT_TYPE:', '').strip()
                if intent in ["query_generation", "schema_description", "help"]:
                    result["intent_type"] = intent

            elif line.startswith('CONFIDENCE:'):
                confidence = line.replace('CONFIDENCE:', '').strip()
                if confidence in ["high", "medium", "low"]:
                    result["confidence"] = confidence

            elif line.startswith('IS_FOLLOW_UP:'):
                follow_up = line.replace('IS_FOLLOW_UP:', '').strip().lower()
                result["is_follow_up"] = follow_up in ["true", "yes", "1"]

            elif line.startswith('REASONING:'):
                result["reasoning"] = line.replace('REASONING:', '').strip()

            elif line.startswith('CONVERSATION_CONTEXT_SUMMARY:'):
                result["conversation_context_summary"] = line.replace('CONVERSATION_CONTEXT_SUMMARY:', '').strip()

    except Exception as e:
        logger.error(f"Error parsing intent classification: {str(e)}")
        logger.debug(f"Response text was: {response_text}")

    return result

def parse_retry_analysis(response_text: str) -> Dict[str, Any]:
    """Parse the LLM response for retry analysis."""
    result = {
        "intent_should_change": False,
        "new_intent_type": "same",
        "feedback_type": "modification",
        "preserve_context": [],
        "change_required": "",
        "reasoning": ""
    }

    try:
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()

            if line.startswith('INTENT_SHOULD_CHANGE:'):
                should_change = line.replace('INTENT_SHOULD_CHANGE:', '').strip().lower()
                result["intent_should_change"] = should_change in ["true", "yes", "1"]

            elif line.startswith('NEW_INTENT_TYPE:'):
                new_intent = line.replace('NEW_INTENT_TYPE:', '').strip()
                result["new_intent_type"] = new_intent

            elif line.startswith('FEEDBACK_TYPE:'):
                feedback_type = line.replace('FEEDBACK_TYPE:', '').strip()
                result["feedback_type"] = feedback_type

            elif line.startswith('PRESERVE_CONTEXT:'):
                context_str = line.replace('PRESERVE_CONTEXT:', '').strip()
                if context_str and context_str != "[]":
                    # Parse list-like string
                    context_items = [item.strip() for item in context_str.split(',') if item.strip()]
                    result["preserve_context"] = context_items

            elif line.startswith('CHANGE_REQUIRED:'):
                result["change_required"] = line.replace('CHANGE_REQUIRED:', '').strip()

            elif line.startswith('REASONING:'):
                result["reasoning"] = line.replace('REASONING:', '').strip()

    except Exception as e:
        logger.error(f"Error parsing retry analysis: {str(e)}")
        logger.debug(f"Response text was: {response_text}")

    return result
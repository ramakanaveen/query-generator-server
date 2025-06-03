# app/services/query_generation/nodes/query_analyzer.py
import re
from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate

from app.core.profiling import timeit
from app.services.query_generation.prompts.anlyzer_prompts import ANALYZER_PROMPT_TEMPLATE
from app.services.query_generation.prompts.intent_classifier_prompts import INTENT_CLASSIFIER_PROMPT_TEMPLATE
from app.services.query_generation.prompts.retry_prompts import FEEDBACK_ANALYSIS_PROMPT
from app.core.logging import logger

@timeit
async def analyze_query(state):
    """
    Enhanced analyze_query that handles both initial queries and retry requests with conversation essence.
    """
    try:
        query = state.query
        llm = state.llm

        # Load conversation essence if conversation_id is provided
        if state.conversation_id:
            await load_conversation_essence(state)

        # Add thinking step
        if state.is_retry_request:
            state.thinking.append("🔄 Analyzing retry request with conversation context...")
        else:
            state.thinking.append("🔍 Analyzing initial query to extract directives, entities, and intent...")

        # Extract directives
        directives = extract_directives(query)
        state.directives = directives
        state.thinking.append(f"Extracted directives: {directives}")

        # Handle retry vs initial request
        if state.is_retry_request:
            await process_retry_request(state, query, directives, llm)
        else:
            await process_initial_request(state, query, directives, llm)

        # Update conversation essence with new insights
        if state.conversation_id:
            await update_conversation_essence_from_analysis(state)

        return state

    except Exception as e:
        logger.error(f"Error in enhanced query analyzer: {str(e)}", exc_info=True)
        state.thinking.append(f"Error analyzing query: {str(e)}")
        # Default to query generation on error
        state.intent_type = "query_generation"
        return state

async def load_conversation_essence(state):
    """
    Load conversation essence from database into state.
    """
    try:
        from app.services.conversation_manager import ConversationManager
        conversation_manager = ConversationManager()

        # Get conversation essence
        essence = await conversation_manager.get_conversation_essence(state.conversation_id)

        if essence:
            state.conversation_essence = essence
            state.original_intent = essence.get("original_intent")
            state.current_understanding = essence.get("current_understanding")
            state.feedback_trail = essence.get("feedback_trail", [])
            state.key_context = essence.get("key_context", [])

            state.thinking.append(f"📚 Loaded conversation essence: {len(state.feedback_trail)} previous corrections")

            if state.original_intent:
                state.thinking.append(f"🎯 Original intent: {state.original_intent}")
            if state.current_understanding:
                state.thinking.append(f"💭 Current understanding: {state.current_understanding}")
        else:
            state.thinking.append("📝 No previous conversation essence found")

    except Exception as e:
        logger.error(f"Error loading conversation essence: {str(e)}", exc_info=True)
        state.thinking.append(f"Error loading conversation context: {str(e)}")

async def process_retry_request(state, query, directives, llm):
    """
    Process retry request with feedback analysis and context preservation.
    """
    try:
        state.thinking.append("🔍 Analyzing user feedback to understand what went wrong...")

        # Analyze the user feedback
        feedback_analysis = await analyze_user_feedback(
            original_query=query,
            generated_query=state.original_generated_query,
            user_feedback=state.user_feedback,
            original_intent=state.original_intent or "Unknown",
            current_understanding=state.current_understanding or "Unknown",
            previous_corrections=[f.get("correction", "") for f in state.feedback_trail],
            llm=llm
        )

        # Store feedback analysis in retry context
        state.retry_context = feedback_analysis
        state.thinking.append(f"📋 Feedback analysis: {feedback_analysis.get('issue_type', 'Unknown issue')}")

        # Extract entities from both original query and feedback
        entities = extract_entities_from_retry(query, state.user_feedback, state.key_context)
        state.entities = entities

        # Determine intent type (usually query_generation for retries)
        intent_type = await determine_intent_type(state, query, directives, llm)
        state.intent_type = intent_type

        # Set intent based on retry context
        state.intent = feedback_analysis.get("user_intent", "query_refinement")

        # Add current feedback to trail (will be saved later)
        new_feedback_entry = {
            "attempt": len(state.feedback_trail) + 1,
            "feedback": state.user_feedback,
            "correction": feedback_analysis.get("correction_strategy", "General refinement"),
            "issue_type": feedback_analysis.get("issue_type", "GENERAL"),
            "learning_point": feedback_analysis.get("learning_point", "")
        }
        state.feedback_trail.append(new_feedback_entry)

        state.thinking.append(f"🔧 Correction strategy: {feedback_analysis.get('correction_strategy', 'Not specified')}")

    except Exception as e:
        logger.error(f"Error processing retry request: {str(e)}", exc_info=True)
        state.thinking.append(f"Error analyzing retry feedback: {str(e)}")
        # Fallback for retry
        state.intent_type = "query_generation"
        state.intent = "query_refinement"

async def process_initial_request(state, query, directives, llm):
    """
    Process initial request with intent determination and context extraction.
    """
    try:
        # Add context from conversation history if available
        conversation_context = ""
        previous_directives = []

        if hasattr(state, 'conversation_history') and state.conversation_history:
            # Extract directives from previous messages
            for msg in state.conversation_history:
                if msg.get('role') == 'user':
                    directive_matches = re.findall(r'@([A-Z]+)', msg.get('content', ''))
                    for directive in directive_matches:
                        if directive not in previous_directives:
                            previous_directives.append(directive)

        # Use previous directives if current query has none
        if not directives and previous_directives:
            state.thinking.append(f"No directives in current query, using previous: {previous_directives}")
            state.directives = previous_directives
            directives = previous_directives

        # Determine intent type through multi-stage process
        intent_type = await determine_intent_type(state, query, directives, llm)
        state.intent_type = intent_type
        state.thinking.append(f"Determined intent type: {intent_type}")

        if intent_type == "query_generation":
            await process_query_generation_intent(state, query, directives, llm)
        elif intent_type == "schema_description":
            await process_schema_description_intent(state, query, directives)
        elif intent_type == "help":
            await process_help_intent(state, query, directives)
        else:
            # Handle unknown intent - default to query generation
            state.thinking.append(f"Unknown intent type '{intent_type}', defaulting to query generation")
            await process_query_generation_intent(state, query, directives, llm)

    except Exception as e:
        logger.error(f"Error processing initial request: {str(e)}", exc_info=True)
        state.thinking.append(f"Error in initial analysis: {str(e)}")
        state.intent_type = "query_generation"

async def analyze_user_feedback(original_query, generated_query, user_feedback,
                                original_intent, current_understanding, previous_corrections, llm):
    """
    Analyze user feedback to understand what went wrong and how to fix it.
    """
    try:
        # Format previous corrections for context
        corrections_text = "\n".join([f"- {corr}" for corr in previous_corrections]) if previous_corrections else "None"

        # Create prompt for feedback analysis
        prompt = ChatPromptTemplate.from_template(FEEDBACK_ANALYSIS_PROMPT)

        chain = prompt | llm
        response = await chain.ainvoke({
            "original_query": original_query,
            "generated_query": generated_query,
            "user_feedback": user_feedback,
            "original_intent": original_intent,
            "current_understanding": current_understanding,
            "previous_corrections": corrections_text
        })

        # Parse the structured response
        analysis = parse_feedback_analysis(response.content.strip())
        return analysis

    except Exception as e:
        logger.error(f"Error analyzing feedback: {str(e)}", exc_info=True)
        # Return fallback analysis
        return {
            "issue_type": "GENERAL",
            "specific_problem": "Analysis failed",
            "user_intent": "Refine the query based on feedback",
            "correction_strategy": "Address user feedback",
            "learning_point": "General refinement needed"
        }

def parse_feedback_analysis(response_text: str) -> Dict[str, str]:
    """
    Parse the structured feedback analysis response.
    """
    analysis = {
        "issue_type": "GENERAL",
        "specific_problem": "",
        "user_intent": "",
        "correction_strategy": "",
        "learning_point": ""
    }

    try:
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()

            if line.startswith('Issue_Type:'):
                analysis["issue_type"] = line.replace('Issue_Type:', '').strip()
            elif line.startswith('Specific_Problem:'):
                analysis["specific_problem"] = line.replace('Specific_Problem:', '').strip()
            elif line.startswith('User_Intent:'):
                analysis["user_intent"] = line.replace('User_Intent:', '').strip()
            elif line.startswith('Correction_Strategy:'):
                analysis["correction_strategy"] = line.replace('Correction_Strategy:', '').strip()
            elif line.startswith('Learning_Point:'):
                analysis["learning_point"] = line.replace('Learning_Point:', '').strip()

    except Exception as e:
        logger.error(f"Error parsing feedback analysis: {str(e)}")

    return analysis

def extract_entities_from_retry(original_query, user_feedback, existing_context):
    """
    Extract entities from both original query and user feedback for retry.
    """
    entities = []

    # Extract from original query
    original_entities = extract_entities_from_text(original_query)
    entities.extend(original_entities)

    # Extract from feedback
    feedback_entities = extract_entities_from_text(user_feedback)
    entities.extend(feedback_entities)

    # Include existing context entities
    if existing_context:
        entities.extend(existing_context)

    # Remove duplicates while preserving order
    unique_entities = []
    for entity in entities:
        if entity not in unique_entities:
            unique_entities.append(entity)

    return unique_entities

def extract_entities_from_text(text):
    """
    Extract entities (symbols, currencies, etc.) from text.
    """
    entities = []

    # Currency pairs (6 letters)
    currency_pairs = re.findall(r'\b[A-Z]{6}\b', text.upper())
    entities.extend(currency_pairs)

    # Stock symbols (3-5 letters)
    symbols = re.findall(r'\b[A-Z]{3,5}\b', text.upper())
    entities.extend([s for s in symbols if s not in currency_pairs])

    # Time references
    time_refs = re.findall(r'\b(?:today|yesterday|daily|hourly|weekly|monthly)\b', text.lower())
    entities.extend(time_refs)

    # Numerical values
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    entities.extend(numbers)

    return entities

async def update_conversation_essence_from_analysis(state):
    """
    Update conversation essence with insights from current analysis.
    """
    try:
        from app.services.conversation_manager import ConversationManager
        conversation_manager = ConversationManager()

        # Update original intent if this is the first query
        if not state.original_intent and not state.is_retry_request:
            intent = state.intent or "General database query"
            await conversation_manager.update_original_intent(state.conversation_id, intent)
            state.thinking.append(f"💾 Stored original intent: {intent}")

        # Update current understanding
        if state.intent_type == "query_generation":
            understanding = f"{state.intent} with entities: {', '.join(state.entities[:3])}"
            await conversation_manager.update_current_understanding(state.conversation_id, understanding)

        # Add key context (directives, entities)
        context_items = []
        context_items.extend([f"@{d}" for d in state.directives])
        context_items.extend(state.entities[:5])  # Limit to 5 entities

        if context_items:
            await conversation_manager.add_key_context(state.conversation_id, context_items)
            state.thinking.append(f"💾 Added key context: {context_items}")

        # For retry requests, append feedback to trail
        if state.is_retry_request and state.feedback_trail:
            latest_feedback = state.feedback_trail[-1]
            await conversation_manager.append_feedback_to_trail(state.conversation_id, latest_feedback)
            state.thinking.append("💾 Stored feedback in conversation trail")

    except Exception as e:
        logger.error(f"Error updating conversation essence: {str(e)}", exc_info=True)
        state.thinking.append(f"Warning: Could not update conversation context: {str(e)}")

def extract_directives(query):
    """Extract directives from the query text."""
    directives = []
    words = query.split()
    for word in words:
        if word.startswith('@'):
            directives.append(word.strip('@.,?!'))
    return directives

async def determine_intent_type(state, query, directives, llm):
    """
    Determine the intent type using multi-stage approach:
    1. Pattern matching
    2. Directive-based heuristics
    3. LLM classification for ambiguous cases
    """
    # First-pass intent detection using patterns
    intent_type = detect_intent_by_pattern(query, directives)

    # Refine intent detection using directives
    intent_type = refine_intent_with_directives(query, directives, intent_type)

    # For ambiguous cases, use LLM classification
    if is_intent_ambiguous(query, intent_type):
        intent_type = await classify_intent_with_llm(query, directives, llm)
        state.thinking.append("Used LLM to classify ambiguous intent")

    return intent_type

def detect_intent_by_pattern(query_text, directives):
    """
    Use regex pattern matching to identify clear intent signals.
    Returns an intent type string.
    """
    # Schema description patterns
    schema_patterns = [
        r'what tables',
        r'show tables',
        r'list tables',
        r'describe table',
        r'explain table',
        r'show columns',
        r'what columns',
        r'table schema',
        r'schema (of|for)',
        r'structure of'
    ]
    
    # Help patterns
    help_patterns = [
        r'how (do|to|can) I',
        r'help me',
        r'show me how',
        r'explain how',
        r'^help$'
    ]
    
    # Check if any schema patterns match
    for pattern in schema_patterns:
        if re.search(pattern, query_text, re.IGNORECASE):
            return "schema_description"
    
    # Check if any help patterns match
    for pattern in help_patterns:
        if re.search(pattern, query_text, re.IGNORECASE):
            return "help"
    
    # Default to query generation if no patterns match
    return "query_generation"

def refine_intent_with_directives(query_text, directives, initial_intent):
    """
    Use the presence of directives combined with query patterns to refine intent detection.
    """
    # If we have directives and question words, it's likely a schema question
    if directives and initial_intent == "query_generation":
        question_words = ["what", "how", "show", "list", "describe", "explain"]
        has_question_word = any(word in query_text.lower().split() for word in question_words)
        
        schema_words = ["table", "tables", "schema", "column", "columns", "structure"]
        has_schema_word = any(word in query_text.lower() for word in schema_words)
        
        if has_question_word and has_schema_word:
            return "schema_description"
    
    return initial_intent

def is_intent_ambiguous(query, current_intent):
    """
    Determine if the current intent determination is potentially ambiguous.
    """
    if current_intent != "query_generation":
        return False  # We already determined a specific intent
    
    # Potentially ambiguous words that could indicate either schema questions or data queries
    ambiguous_indicators = ["show", "what", "list", "describe", "explain", "display", "give me"]
    
    # Check if query has ambiguous indicators
    return any(indicator in query.lower() for indicator in ambiguous_indicators)

async def classify_intent_with_llm(query, directives, llm):
    """
    Use LLM to classify intent for ambiguous queries.
    """
    try:
        # Create a prompt for classification
        prompt = ChatPromptTemplate.from_template(INTENT_CLASSIFIER_PROMPT_TEMPLATE)
        
        # Get LLM classification
        chain = prompt | llm
        response = await chain.ainvoke({"query": query, "directives": directives})
        intent = response.content.strip()
        
        # Validate the response is one of our expected intents
        valid_intents = ["query_generation", "schema_description", "help"]
        if intent not in valid_intents:
            return "query_generation"  # Default to query generation if invalid response
        
        return intent
        
    except Exception as e:
        logger.error(f"Error in intent classification: {str(e)}")
        return "query_generation"  # Default to query generation on error

async def process_query_generation_intent(state, query, directives, llm):
    """
    Process a query generation intent by extracting entities and query intent.
    """
    # Add context from conversation history if available
    conversation_context = ""
    previous_directives = []
    
    if hasattr(state, 'conversation_history') and state.conversation_history:
        # Extract directives from previous messages
        for msg in state.conversation_history:
            if msg.get('role') == 'user':
                directive_matches = re.findall(r'@([A-Z]+)', msg.get('content', ''))
                for directive in directive_matches:
                    if directive not in previous_directives:
                        previous_directives.append(directive)
    
    # If current query has no directives but previous ones exist, use those
    if not directives and previous_directives:
        state.thinking.append(f"No directives in current query, using previous directives: {previous_directives}")
        state.directives = previous_directives
        directives = previous_directives
        
    # Use LLM to analyze the query for entities and query intent
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

async def process_schema_description_intent(state, query, directives):
    """
    Process a schema description intent by extracting targets and detail level.
    """
    try:
        # Extract which tables or columns the user is asking about
        table_targets = extract_table_targets(query, directives)
        column_targets = extract_column_targets(query)
        detail_level = determine_detail_level(query)
        
        # Store schema targets in state with debug output
        state.schema_targets = {
            "tables": table_targets,
            "columns": column_targets,
            "detail_level": detail_level
        }
        
        state.thinking.append(f"Schema description targets: tables={table_targets}, columns={column_targets}")
        state.thinking.append(f"Schema description detail level: {detail_level}")
        
        # For debugging, log the entire schema_targets dict
        logger.debug(f"Created schema_targets: {state.schema_targets}")
        
    except Exception as e:
        # Handle errors but ensure we still set a default schema_targets
        logger.error(f"Error processing schema description intent: {str(e)}", exc_info=True)
        state.thinking.append(f"Error processing intent: {str(e)}")
        
        # Set default values even on error
        state.schema_targets = {
            "tables": directives,
            "columns": [],
            "detail_level": "standard"
        }
    
    return state
async def process_help_intent(state, query, directives):
    """
    Process a help intent by identifying the help topic.
    """
    help_topic = extract_help_topic(query)
    
    state.help_request = {
        "topic": help_topic,
        "directives": directives
    }
    
    state.thinking.append(f"Help requested on topic: {help_topic}")

def extract_table_targets(query, directives):
    """Extract which tables the user is asking about."""
    # Case 1: If asking for ALL tables across ALL schemas
    all_tables_patterns = [
        r'all tables',
        r'every table',
        r'list all tables',
        r'show all tables',
        r'tables from all',
        r'tables in all'
    ]
    
    # Check for general "what tables" type questions
    general_table_questions = [
        r'what tables',
        r'which tables',
        r'list tables',
        r'show tables',
        r'display tables',
        r'tables (do|does|are|can) you',
    ]
    
    for pattern in all_tables_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return ["*ALL*"]  # Special marker for "all tables from all schemas"
    
    # Check for general table questions but with directives
    for pattern in general_table_questions:
        if re.search(pattern, query, re.IGNORECASE):
            if directives:
                return directives  # Return the schema group(s) specified by directives
            else:
                return ["*ALL*"]  # No directives specified, assume they want all tables
    
    # Case 2: If specific table names mentioned in query using more precise patterns
    # Look for table name in patterns like "table called X" or "X table"
    specific_table_patterns = [
        r'table\s+(?:named|called)?\s+[\'"]?([a-zA-Z0-9_]+)[\'"]?',  # table called X
        r'tables?\s+(?:named|called)?\s+[\'"]?([a-zA-Z0-9_]+)[\'"]?',  # tables called X
        r'([a-zA-Z0-9_]+)\s+table',  # X table
        r'information\s+(?:about|on|for)\s+[\'"]?([a-zA-Z0-9_]+)[\'"]?\s+table',  # information about X table
        r'describe\s+([a-zA-Z0-9_]+)'  # describe X
    ]
    
    table_names = []
    for pattern in specific_table_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        table_names.extend(matches)
    
    # Filter out common words that aren't likely to be table names
    common_words = ["me", "you", "the", "this", "that", "those", "these", "do", "does", "have", "has", "a", "an", "in", "on", "at", "by", "for", "with", "about"]
    table_names = [name for name in table_names if name.lower() not in common_words]
    
    if table_names:
        return table_names
    
    # If we have directives but no specific tables identified, use directives
    if directives:
        return directives
    
    return []  # No specific tables identified

def extract_column_targets(query):
    """Extract which columns the user is asking about."""
    column_pattern = r'(?:column|columns)\s+(?:named|called)?\s*[\'"]?([a-zA-Z0-9_]+)[\'"]?'
    matches = re.findall(column_pattern, query, re.IGNORECASE)
    return matches

def determine_detail_level(query):
    """Determine how detailed the response should be."""
    # Check for detail indicators
    if any(word in query.lower() for word in ["details", "detailed", "comprehensive", "all information"]):
        return "detailed"
    # Check for summary indicators
    elif any(word in query.lower() for word in ["brief", "summary", "overview", "list"]):
        return "summary"
    # Default
    return "standard"

def extract_help_topic(query):
    """Extract the help topic from the query."""
    # Pattern matching for common help topics
    topics = ["query", "schema", "directives", "kdb", "tables", "syntax"]
    found_topics = []
    
    for topic in topics:
        if topic in query.lower():
            found_topics.append(topic)
    
    if found_topics:
        return found_topics
    else:
        return ["general"]  # Default to general help
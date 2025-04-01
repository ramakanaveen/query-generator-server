# app/services/query_generation/nodes/query_analyzer.py
import re
from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate
from app.services.query_generation.prompts.anlyzer_prompts import ANALYZER_PROMPT_TEMPLATE
from app.services.query_generation.prompts.intent_classifier_prompts import INTENT_CLASSIFIER_PROMPT_TEMPLATE
from app.core.logging import logger

async def analyze_query(state):
    """
    Analyze the natural language query to extract directives, entities, intent, and intent type.
    Main orchestration function that delegates to specialized functions.
    """
    try:
        query = state.query
        llm = state.llm
        
        # Add thinking step
        state.thinking.append("Analyzing query to extract directives, entities, and determine intent type...")
        
        # Extract directives
        directives = extract_directives(query)
        state.directives = directives
        state.thinking.append(f"Extracted directives: {directives}")
        
        # Determine intent type through multi-stage process
        intent_type = await determine_intent_type(state, query, directives, llm)
        state.intent_type = intent_type
        state.thinking.append(f"Determined intent type: {intent_type}")
        logger.debug(f"Detected intent_type: {intent_type}")
        if intent_type == "schema_description":
            logger.debug("Processing schema description intent...")
            # Process specific intent type
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
        
        # Check state after processing
        logger.debug(f"State after processing: schema_targets={state.schema_targets}")
        
        return state
    
    except Exception as e:
        logger.error(f"Error in query analyzer: {str(e)}")
        state.thinking.append(f"Error analyzing query: {str(e)}")
        # Default to query generation on error
        state.intent_type = "query_generation"
        return state

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
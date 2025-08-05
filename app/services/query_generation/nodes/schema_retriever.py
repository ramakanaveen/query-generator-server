# app/services/query_generation/nodes/schema_retriever.py - Enhanced Version

import re
import time
from typing import Dict, Any, List

from app.core.profiling import timeit
from app.services.schema_management import SchemaManager
from app.core.logging import logger

# Schema cache (in-memory)
_schema_cache = {}
_schema_cache_ttl = 300  # 5 minutes
_last_cleanup = 0

def _get_cache_key(query_text, directives, database_type):
    """Generate a cache key for schema retrieval."""
    directive_str = ",".join(sorted(directives)) if directives else ""
    return f"{database_type}:{directive_str}:{query_text}"

def _cleanup_cache():
    """Clean up expired cache entries."""
    global _last_cleanup

    now = time.time()
    # Only cleanup every minute
    if now - _last_cleanup < 60:
        return

    expired_keys = []
    for key, (timestamp, _) in _schema_cache.items():
        if now - timestamp > _schema_cache_ttl:
            expired_keys.append(key)

    for key in expired_keys:
        del _schema_cache[key]

    _last_cleanup = now

@timeit
async def retrieve_schema(state):
    """
    Enhanced schema retrieval with intelligent context awareness.

    NEW FEATURES:
    - Retry-aware schema selection based on feedback
    - Schema reselection for validation failures
    - Better follow-up handling with conversation context
    - LLM-guided schema selection for complex scenarios
    """
    try:
        # Get query text, directives, and entities
        query_text = state.query
        directives = state.directives
        entities = state.entities
        database_type = state.database_type
        conversation_history = state.conversation_history if hasattr(state, 'conversation_history') else []

        # Enhanced thinking step
        if getattr(state, 'needs_schema_reselection', False):
            state.thinking.append("🔄 Re-selecting schema based on feedback...")
        elif state.is_retry_request:
            state.thinking.append("🔍 Retrieving schema with retry context...")
        else:
            state.thinking.append("📊 Retrieving schema information...")

        # Check cache first (unless reselection is needed)
        if not getattr(state, 'needs_schema_reselection', False):
            cache_key = _get_cache_key(query_text, directives, database_type)
            _cleanup_cache()

            if cache_key in _schema_cache:
                cached_timestamp, cached_schema = _schema_cache[cache_key]
                if time.time() - cached_timestamp <= _schema_cache_ttl:
                    state.thinking.append("💾 Using cached schema information")
                    state.query_schema = cached_schema
                    return state

        # Initialize schema manager
        schema_manager = SchemaManager()
        schemas_exist = await schema_manager.check_schemas_available()

        if not schemas_exist:
            state.thinking.append("❌ No schemas found in the database. Please upload schema information first.")
            state.query_schema = None
            state.no_schema_found = True
            return state

        # Enhanced schema selection logic
        if getattr(state, 'needs_schema_reselection', False):
            # Handle schema reselection based on feedback
            await handle_schema_reselection(state, schema_manager)
        elif state.is_retry_request:
            # Handle retry with schema context awareness
            await handle_retry_schema_selection(state, schema_manager, conversation_history)
        else:
            # Handle initial schema selection with enhanced logic
            await handle_initial_schema_selection(state, schema_manager, conversation_history)

        # Cache the result if we found something
        if state.query_schema and not state.no_schema_found:
            cache_key = _get_cache_key(query_text, directives, database_type)
            _schema_cache[cache_key] = (time.time(), state.query_schema)

        return state

    except Exception as e:
        logger.error(f"Error in enhanced schema retriever: {str(e)}", exc_info=True)
        state.thinking.append(f"❌ Error retrieving schema: {str(e)}")
        # Still return the state to continue the workflow
        state.query_schema = None
        state.no_schema_found = True
        return state

async def handle_schema_reselection(state, schema_manager):
    """Handle schema reselection based on validation feedback."""
    try:
        # Get guidance from intelligent analyzer
        schema_changes = getattr(state, 'schema_changes_needed', '')
        schema_corrections = getattr(state, 'schema_corrections_needed', '')

        state.thinking.append(f"🔧 Schema reselection guidance: {schema_changes}")

        # If we have specific corrections, try to apply them
        if schema_corrections:
            await apply_schema_corrections(state, schema_manager, schema_corrections)
        else:
            # Try alternative schema selection approach
            await try_alternative_schema_selection(state, schema_manager)

        # Reset the reselection flag
        state.needs_schema_reselection = False

    except Exception as e:
        logger.error(f"Error in schema reselection: {str(e)}")
        state.thinking.append(f"❌ Schema reselection failed: {str(e)}")
        state.no_schema_found = True

async def handle_retry_schema_selection(state, schema_manager, conversation_history):
    """Handle schema selection for retry requests with context awareness."""
    try:
        # Check if feedback indicates schema issues
        feedback_category = getattr(state, 'feedback_category', '')

        if feedback_category == 'schema_issues':
            state.thinking.append("🔍 Feedback indicates schema issues, trying different approach...")

            # Get schema guidance from retry analysis
            schema_changes = getattr(state, 'schema_changes_needed', '')
            if schema_changes:
                state.thinking.append(f"📋 Schema guidance: {schema_changes}")
                await apply_retry_schema_guidance(state, schema_manager, schema_changes)
            else:
                # Fallback to broader schema search
                await try_broader_schema_search(state, schema_manager)
        else:
            # Reuse previous schema context but potentially expand it
            await reuse_and_expand_schema_context(state, schema_manager, conversation_history)

    except Exception as e:
        logger.error(f"Error in retry schema selection: {str(e)}")
        state.thinking.append(f"❌ Retry schema selection failed: {str(e)}")
        await handle_initial_schema_selection(state, schema_manager, conversation_history)

async def handle_initial_schema_selection(state, schema_manager, conversation_history):
    """Enhanced initial schema selection with conversation awareness."""
    try:
        # Determine if this is likely a follow-up or a new question
        is_follow_up = getattr(state, 'is_follow_up', False)

        # ENHANCED: Handle follow-up questions by reusing previous schema context
        if is_follow_up and conversation_history:
            state.thinking.append("🔗 Processing as follow-up question...")

            # Extract successful schemas from conversation history
            previous_tables = await extract_successful_tables_from_history(conversation_history)
            previous_directives = await extract_directives_from_history(conversation_history)

            # Add previous directives to current state if we don't have our own
            if previous_directives and not state.directives:
                state.thinking.append(f"📝 Using directives from conversation history: {previous_directives}")
                state.directives = previous_directives

            if previous_tables:
                # Try to reuse successful table context
                relevant_tables = await get_tables_by_names(schema_manager, previous_tables)
                if relevant_tables:
                    state.query_schema = build_combined_schema(relevant_tables, "reused_context")
                    state.thinking.append(f"♻️ Reused {len(relevant_tables)} tables from successful previous queries")
                    return

            # If reuse failed, fall back to vector search
            state.thinking.append("🔍 Couldn't reuse previous context, falling back to vector search")

        # ENHANCED: Handle new questions with improved vector search
        state.thinking.append("🎯 Processing with enhanced vector search...")

        # Build enhanced search text
        search_text = build_enhanced_search_text(state)

        # Search for similar tables using enhanced multi-threshold search
        relevant_tables = await enhanced_multi_threshold_vector_search(
            schema_manager,
            search_text,
            max_results=5
        )

        # Process results with enhanced context building
        if relevant_tables:
            state.query_schema = build_enhanced_combined_schema(relevant_tables)
            state.thinking.append(f"✅ Built schema from {len(relevant_tables)} relevant tables")
        else:
            # No relevant schemas/tables found for this query
            state.thinking.append("❌ No relevant tables found for this query.")
            state.query_schema = None
            state.no_schema_found = True

    except Exception as e:
        logger.error(f"Error in initial schema selection: {str(e)}")
        state.thinking.append(f"❌ Initial schema selection failed: {str(e)}")
        state.no_schema_found = True

def build_enhanced_search_text(state):
    """Build enhanced search text with entities and context."""
    search_parts = [state.query]

    # Add directives
    if state.directives:
        directive_text = " ".join(state.directives)
        search_parts.append(directive_text)
        state.thinking.append(f"🎯 Including directives in search: {directive_text}")

    # Add entities from intelligent analyzer
    if state.entities:
        entity_text = " ".join(state.entities)
        search_parts.append(entity_text)
        state.thinking.append(f"🏷️ Including entities in search: {entity_text}")

    # Add context from retry feedback if available
    if state.is_retry_request and hasattr(state, 'preserve_context'):
        preserve_context = getattr(state, 'preserve_context', [])
        if preserve_context:
            context_text = " ".join(preserve_context)
            search_parts.append(context_text)
            state.thinking.append(f"🔄 Including preserved context: {context_text}")

    return " ".join(search_parts)

async def enhanced_multi_threshold_vector_search(schema_manager, search_text, max_results=5):
    """Enhanced multi-threshold vector search with adaptive thresholds."""
    # Progressive thresholds from strict to lenient
    thresholds = [0.65, 0.59, 0.55, 0.50, 0.45, 0.40, 0.35]

    logger.info(f"Starting enhanced multi-threshold search for: {search_text[:100]}...")

    for threshold in thresholds:
        try:
            logger.debug(f"Trying vector search with threshold {threshold}")

            results = await schema_manager.find_tables_by_vector_search(
                search_text,
                similarity_threshold=threshold,
                max_results=max_results
            )

            if results:
                # Found results at this threshold
                logger.info(f"Enhanced vector search found {len(results)} results at threshold {threshold}")

                # Add threshold info to results for debugging
                for result in results:
                    result['search_threshold_used'] = threshold

                return results

        except Exception as e:
            logger.warning(f"Vector search failed at threshold {threshold}: {str(e)}")
            continue

    # No results found at any threshold
    logger.info("Enhanced multi-threshold vector search found no results at any threshold")
    return []

async def extract_successful_tables_from_history(conversation_history):
    """Extract table names from successful queries in conversation history."""
    previous_tables = []

    # Go through messages in reverse to find most recent successful query
    for msg in reversed(conversation_history):
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')

            # Avoid processing if it looks like an error message
            if 'error' in content.lower():
                continue

            # Enhanced regex to detect SELECT query patterns and capture table names
            match = re.search(r'\bselect\s+.*?\s+from\s+(\S+)', content, re.IGNORECASE | re.DOTALL)

            if match:
                table_name = match.group(1)

                # Clean up potential backticks or quotes
                table_name = table_name.strip('`"')

                # Basic validation - ensure it looks like a table name
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
                    if table_name not in previous_tables:
                        previous_tables.append(table_name)
                    logger.info(f"Found previous successful query using table: {table_name}")
                    # Found the most recent valid query, we can stop or continue for more tables
                    if len(previous_tables) >= 3:  # Limit to 3 tables
                        break

    return previous_tables

async def extract_directives_from_history(conversation_history):
    """Extract directives from conversation history."""
    previous_directives = []

    for msg in conversation_history:
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            directive_matches = re.findall(r'@([A-Z]+)', content)
            for directive in directive_matches:
                if directive not in previous_directives:
                    previous_directives.append(directive)

    return previous_directives

async def get_tables_by_names(schema_manager, table_names):
    """Get table information by names from schema manager."""
    relevant_tables = []

    for table_name in table_names:
        table_info = await find_tables_by_name(schema_manager, table_name)
        if table_info and "tables" in table_info:
            # Convert table info to the format expected by the workflow
            for tn, table_data in table_info["tables"].items():
                relevant_tables.append({
                    "schema_name": "reused_schema",
                    "table_name": tn,
                    "content": table_data,
                    "description": table_data.get("description", f"Table {tn}"),
                    "similarity": 1.0  # Max similarity since we're reusing
                })

    return relevant_tables

def build_combined_schema(relevant_tables, schema_name="derived_schema"):
    """Build combined schema from relevant tables."""
    combined_schema = {
        "description": f"Schema for {schema_name}",
        "tables": {},
        "examples": []
    }

    # Add all tables to the schema
    for table in relevant_tables:
        table_name = table["table_name"]
        combined_schema["tables"][table_name] = table["content"]

    return combined_schema

def build_enhanced_combined_schema(relevant_tables):
    """Build enhanced combined schema with better organization."""
    # Group tables by schema
    tables_by_schema = {}
    for table in relevant_tables:
        schema_name = table["schema_name"]
        if schema_name not in tables_by_schema:
            tables_by_schema[schema_name] = {
                "description": f"Schema for {schema_name}",
                "tables": {}
            }

        # Add table to schema
        tables_by_schema[schema_name]["tables"][table["table_name"]] = table["content"]

    # Find the primary schema (most tables or highest similarity)
    primary_schema_name = max(
        tables_by_schema.keys(),
        key=lambda k: len(tables_by_schema[k]["tables"])
    )

    # Build the combined schema, prioritizing the primary schema
    combined_schema = {
        "description": tables_by_schema[primary_schema_name]["description"],
        "tables": {},
        "examples": []
    }

    # Add all tables, starting with primary schema
    combined_schema["tables"].update(tables_by_schema[primary_schema_name]["tables"])
    for schema_name, schema in tables_by_schema.items():
        if schema_name != primary_schema_name:
            combined_schema["tables"].update(schema["tables"])

    return combined_schema

async def apply_schema_corrections(state, schema_manager, schema_corrections):
    """Apply specific schema corrections based on feedback."""
    try:
        state.thinking.append(f"🔧 Applying schema corrections: {schema_corrections}")

        # Parse corrections and try to find better tables
        # This could involve regex parsing or LLM interpretation of corrections
        # For now, implement basic table name substitutions

        if "table" in schema_corrections.lower() and "not found" in schema_corrections.lower():
            # Extract suggested table names and try to find them
            suggested_tables = extract_suggested_tables(schema_corrections)
            if suggested_tables:
                relevant_tables = await get_tables_by_names(schema_manager, suggested_tables)
                if relevant_tables:
                    state.query_schema = build_combined_schema(relevant_tables, "corrected_schema")
                    state.thinking.append(f"✅ Applied schema corrections with {len(relevant_tables)} tables")
                    return

        # If specific corrections didn't work, fall back to broader search
        await try_alternative_schema_selection(state, schema_manager)

    except Exception as e:
        logger.error(f"Error applying schema corrections: {str(e)}")
        state.thinking.append(f"❌ Schema corrections failed: {str(e)}")

def extract_suggested_tables(corrections_text):
    """Extract suggested table names from correction text."""
    # Simple extraction - could be enhanced with LLM parsing
    table_patterns = [
        r"use table '([^']+)'",
        r"table '([^']+)' instead",
        r"available.*?'([^']+)'",
        r"try '([^']+)'"
    ]

    suggested_tables = []
    for pattern in table_patterns:
        matches = re.findall(pattern, corrections_text, re.IGNORECASE)
        suggested_tables.extend(matches)

    return list(set(suggested_tables))  # Remove duplicates

async def try_alternative_schema_selection(state, schema_manager):
    """Try alternative schema selection when corrections fail."""
    try:
        # Try with more lenient search criteria
        alternative_search_text = f"{state.query} {' '.join(state.entities[:3])}"

        # Use very lenient thresholds
        relevant_tables = await schema_manager.find_tables_by_vector_search(
            alternative_search_text,
            similarity_threshold=0.3,  # Very lenient
            max_results=10
        )

        if relevant_tables:
            state.query_schema = build_enhanced_combined_schema(relevant_tables)
            state.thinking.append(f"✅ Alternative selection found {len(relevant_tables)} tables")
        else:
            state.no_schema_found = True
            state.thinking.append("❌ Alternative schema selection found no results")

    except Exception as e:
        logger.error(f"Error in alternative schema selection: {str(e)}")
        state.no_schema_found = True

async def apply_retry_schema_guidance(state, schema_manager, schema_changes):
    """Apply schema guidance from retry analysis."""
    try:
        # Use LLM guidance to improve schema selection
        if "different table" in schema_changes.lower():
            # Try different tables with expanded search
            expanded_search = f"{state.query} {schema_changes}"
            relevant_tables = await enhanced_multi_threshold_vector_search(
                schema_manager, expanded_search, max_results=8
            )

            if relevant_tables:
                state.query_schema = build_enhanced_combined_schema(relevant_tables)
                state.thinking.append(f"✅ Retry guidance applied: {len(relevant_tables)} tables")
            else:
                await try_broader_schema_search(state, schema_manager)
        else:
            # General guidance - use broader search
            await try_broader_schema_search(state, schema_manager)

    except Exception as e:
        logger.error(f"Error applying retry schema guidance: {str(e)}")
        await try_broader_schema_search(state, schema_manager)

async def try_broader_schema_search(state, schema_manager):
    """Try broader schema search for difficult cases."""
    try:
        # Use just the basic query with very lenient thresholds
        broad_search_text = state.query

        relevant_tables = await schema_manager.find_tables_by_vector_search(
            broad_search_text,
            similarity_threshold=0.25,  # Very broad
            max_results=15
        )

        if relevant_tables:
            state.query_schema = build_enhanced_combined_schema(relevant_tables)
            state.thinking.append(f"✅ Broader search found {len(relevant_tables)} tables")
        else:
            state.no_schema_found = True
            state.thinking.append("❌ Even broader search found no relevant tables")

    except Exception as e:
        logger.error(f"Error in broader schema search: {str(e)}")
        state.no_schema_found = True

async def reuse_and_expand_schema_context(state, schema_manager, conversation_history):
    """Reuse previous schema context but potentially expand it."""
    try:
        # First try to reuse previous successful context
        await handle_initial_schema_selection(state, schema_manager, conversation_history)

        # If that didn't work or we want to expand, try additional tables
        if not state.no_schema_found and state.query_schema:
            current_table_count = len(state.query_schema.get("tables", {}))

            # If we have few tables, try to find more related ones
            if current_table_count < 3:
                additional_search = f"{state.query} related tables"
                additional_tables = await enhanced_multi_threshold_vector_search(
                    schema_manager, additional_search, max_results=3
                )

                if additional_tables:
                    # Merge additional tables into existing schema
                    for table in additional_tables:
                        table_name = table["table_name"]
                        if table_name not in state.query_schema["tables"]:
                            state.query_schema["tables"][table_name] = table["content"]

                    state.thinking.append(f"📈 Expanded schema with {len(additional_tables)} additional tables")

    except Exception as e:
        logger.error(f"Error in reuse and expand: {str(e)}")
        # If expansion fails, keep what we have

# Helper function from original code
async def find_tables_by_name(schema_manager, name):
    """Find tables by name, which could be a schema name or table name."""
    try:
        conn = await schema_manager._get_db_connection()
        try:
            # Try as table name first (more specific)
            query = """
                    SELECT
                        td.id as table_id,
                        td.name as table_name,
                        td.description as table_description,
                        td.content as table_content,
                        sd.name as schema_name
                    FROM
                        table_definitions td
                            JOIN
                        schema_versions sv ON td.schema_version_id = sv.id
                            JOIN
                        schema_definitions sd ON sv.schema_id = sd.id
                            JOIN
                        active_schemas a ON sv.id = a.current_version_id
                    WHERE
                        LOWER(td.name) = LOWER($1) \
                    """
            tables = await conn.fetch(query, name)

            # If no results, try as schema name
            if not tables:
                query = """
                        SELECT
                            td.id as table_id,
                            td.name as table_name,
                            td.description as table_description,
                            td.content as table_content,
                            sd.name as schema_name
                        FROM
                            table_definitions td
                                JOIN
                            schema_versions sv ON td.schema_version_id = sv.id
                                JOIN
                            schema_definitions sd ON sv.schema_id = sd.id
                                JOIN
                            active_schemas a ON sv.id = a.current_version_id
                        WHERE
                            LOWER(sd.name) = LOWER($1) \
                        """
                tables = await conn.fetch(query, name)

            # Process results
            if tables:
                result = {"tables": {}}

                for table in tables:
                    table_name = table["table_name"]
                    table_content = table["table_content"]

                    # Convert content to dict if it's a string
                    if isinstance(table_content, str):
                        import json
                        table_content = json.loads(table_content)

                    result["tables"][table_name] = {
                        "description": table["table_description"],
                        "columns": table_content.get("columns", [])
                    }

                return result

            return None

        finally:
            await conn.close()

    except Exception as e:
        logger.error(f"Error finding tables by name: {str(e)}", exc_info=True)
        return None
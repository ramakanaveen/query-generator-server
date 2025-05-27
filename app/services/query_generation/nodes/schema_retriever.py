# app/services/query_generation/nodes/schema_retriever.py
import re
import time

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
    Retrieve schema information based on directives and entities using vector similarity search.
    Also handles follow-up questions by reusing schema context from previous successful queries.

    Args:
        state: The current state of the workflow

    Returns:
        Updated state with schema information
    """
    try:
        # Get query text, directives, and entities
        query_text = state.query
        directives = state.directives
        entities = state.entities
        database_type = state.database_type
        conversation_history = state.conversation_history if hasattr(state, 'conversation_history') else []

        # Add thinking step
        state.thinking.append("Retrieving schema information...")

        # Check cache first
        cache_key = _get_cache_key(query_text, directives, database_type)
        _cleanup_cache()

        if cache_key in _schema_cache:
            cached_timestamp, cached_schema = _schema_cache[cache_key]

            # If the cache is valid, use it
            if time.time() - cached_timestamp <= _schema_cache_ttl:
                state.thinking.append("Using cached schema information")
                state.query_schema = cached_schema
                return state

        # Initialize schema manager
        schema_manager = SchemaManager()
        schemas_exist = await schema_manager.check_schemas_available()

        if not schemas_exist:
            state.thinking.append("No schemas found in the database. Please upload schema information first.")
            state.query_schema = None
            state.no_schema_found = True
            return state

        # First determine if this is likely a follow-up or a new question
        is_follow_up = False

        # Check for follow-up linguistic patterns
        follow_up_patterns = [
            "change", "modify", "update", "instead", "but", "rather",
            "change to", "switch to", "yesterday", "can you"
        ]
        is_follow_up = any(token in query_text.lower() for token in follow_up_patterns)

        # BRANCH 1: Handle follow-up questions by reusing previous schema context
        if is_follow_up and conversation_history:
            state.thinking.append("Processing as follow-up question...")
            
            # Find the most recent assistant message with a successful query
            previous_tables = []
            previous_directives = []
            
            # Extract directives from user messages
            for msg in conversation_history:
                if msg.get('role') == 'user':
                    directive_matches = re.findall(r'@([A-Z]+)', msg.get('content', ''))
                    for directive in directive_matches:
                        if directive not in previous_directives:
                            previous_directives.append(directive)
            
            # Add previous directives to current state if we don't have our own
            if previous_directives and not directives:
                state.thinking.append(f"Using directives from conversation history: {previous_directives}")
                state.directives = previous_directives
                directives = previous_directives

            # Go through messages in reverse to find most recent successful query
            for msg in reversed(conversation_history):
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')

                    # Avoid processing if it looks like an error message
                    if 'error' in content.lower():
                        continue

                    # --- MODIFIED SECTION START ---
                    # Regex to detect SELECT query patterns (KDB/SQL like) and capture the first table name after FROM
                    # \bselect\s+ : Matches "select" at a word boundary, followed by space(s)
                    # .*?         : Matches any characters non-greedily (columns, functions, etc.)
                    # \s+from\s+  : Matches " from " with space(s) around it
                    # (\S+)       : Captures the table name (one or more non-whitespace characters) into group 1
                    # re.IGNORECASE: Makes the match case-insensitive
                    # re.DOTALL    : Allows '.' to match newline characters if the query spans multiple lines
                    match = re.search(r'\bselect\s+.*?\s+from\s+(\S+)', content, re.IGNORECASE | re.DOTALL)

                    if match:
                        # Extract the captured table name (group 1)
                        table_name = match.group(1)

                        # Optional: Clean up potential backticks or quotes if the regex captures them
                        # table_name = table_name.strip('`"')

                        # Basic check to ensure it looks like a table name (avoids matching subqueries like 'from (select..)')
                        # You might adjust this check based on your expected table name patterns
                        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
                            if table_name not in previous_tables: # Avoid duplicates
                                previous_tables.append(table_name)
                            state.thinking.append(f"Found previous successful query using table: {table_name}")
                            # Found the most recent valid query, stop searching
                            break
                        else:
                            state.thinking.append(f"Detected SELECT pattern, but extracted name '{table_name}' doesn't look like a simple table name. Continuing search.")
                    # --- MODIFIED SECTION END ---

            if previous_tables:
                # Directly retrieve tables by name
                relevant_tables = []
                for table_name in previous_tables:
                    table_info = await find_tables_by_name(schema_manager, table_name)
                    if table_info and "tables" in table_info:
                        # Convert table info to the format expected by the next steps
                        for table_name, table_data in table_info["tables"].items():
                            relevant_tables.append({
                                "schema_name": "reused_schema",  # Placeholder, will be replaced
                                "table_name": table_name,
                                "content": table_data,
                                "description": table_data.get("description", f"Table {table_name}"),
                                "similarity": 1.0  # Max similarity since we're reusing
                            })
                
                if relevant_tables:
                    # Group tables by schema
                    tables_by_schema = {}
                    schema_name = "derived_schema"  # Default name since we might not know the original
                    
                    # Check if we have a directive that matches a schema name
                    if previous_directives:
                        schema_name = previous_directives[0]  # Use first directive as schema name
                    
                    tables_by_schema[schema_name] = {
                        "description": f"Schema for {schema_name}",
                        "tables": {}
                    }
                    
                    # Add tables to schema
                    for table in relevant_tables:
                        table["schema_name"] = schema_name
                        tables_by_schema[schema_name]["tables"][table["table_name"]] = table["content"]
                        
                        # Log the found table
                        state.thinking.append(
                            f"Reusing table from previous query: {schema_name}.{table['table_name']}"
                        )
                    
                    # Build the combined schema
                    combined_schema = {
                        "description": f"Schema for {schema_name}",
                        "tables": {},
                        "examples": []  # We'll fill this with examples next
                    }
                    
                    # Add all tables to the schema
                    combined_schema["tables"].update(tables_by_schema[schema_name]["tables"])

                    # Update state with combined schema
                    state.query_schema = combined_schema

                    # Update cache
                    _schema_cache[cache_key] = (time.time(), combined_schema)

                    state.thinking.append(
                        f"Built schema from {len(relevant_tables)} tables based on previous query context"
                    )
                    
                    return state
            
            # If we couldn't find previous tables, fall back to normal search
            state.thinking.append("Couldn't find suitable tables from previous queries, falling back to vector search")
        
        # BRANCH 2: Handle new questions with vector search
        state.thinking.append("Processing with vector search...")
        
        # Build search text combining query, directives, and entities
        search_text = query_text
        
        if directives:
            directive_text = " ".join(directives)
            search_text += f" {directive_text}"
            state.thinking.append(f"Including directives in search: {directive_text}")
            
        if entities:
            entity_text = " ".join(entities)
            search_text += f" {entity_text}"
            state.thinking.append(f"Including entities in search: {entity_text}")
        
        # Search for similar tables using vector search
        relevant_tables = await multi_threshold_vector_search(
            schema_manager,
            search_text,
            max_results=5
        )
        
        # Process results
        if relevant_tables:
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
                
                # Log the found table
                state.thinking.append(
                    f"Found relevant table: {table['schema_name']}.{table['table_name']} "
                    f"(similarity: {table['similarity']:.2f})"
                )
            
            # Find the primary schema (most tables or highest similarity)
            primary_schema_name = max(
                tables_by_schema.keys(),
                key=lambda k: len(tables_by_schema[k]["tables"])
            )
            
            # Build the combined schema, prioritizing the primary schema
            combined_schema = {
                "description": tables_by_schema[primary_schema_name]["description"],
                "tables": {},
                "examples": []  # We'll fill this with examples next
            }
            
            # Add all tables, starting with primary schema
            combined_schema["tables"].update(tables_by_schema[primary_schema_name]["tables"])
            for schema_name, schema in tables_by_schema.items():
                if schema_name != primary_schema_name:
                    combined_schema["tables"].update(schema["tables"])

            # Update state with combined schema
            state.query_schema = combined_schema

            # Update cache
            _schema_cache[cache_key] = (time.time(), combined_schema)

            state.thinking.append(
                f"Built schema from {len(relevant_tables)} tables across {len(tables_by_schema)} schemas"
            )
            
        else:
            # No relevant schemas/tables found for this query
            state.thinking.append("No relevant tables found for this query.")
            state.query_schema = None
            state.no_schema_found = True  # Add a flag to indicate no matching schemas were found
        
        return state
    
    except Exception as e:
        logger.error(f"Error in schema retriever: {str(e)}", exc_info=True)
        state.thinking.append(f"Error retrieving schema: {str(e)}")
        # Still return the state to continue the workflow
        state.query_schema = None
        state.no_schema_found = True
        return state

async def multi_threshold_vector_search(schema_manager, search_text, max_results=5):
    """
    Multi-threshold vector search that tries progressively lower thresholds
    until it finds results or exhausts all thresholds.

    Args:
        schema_manager: SchemaManager instance
        search_text: Text to search for
        max_results: Maximum number of results to return

    Returns:
        List of search results with same format as find_tables_by_vector_search
    """
    # Progressive thresholds from strict to lenient
    thresholds = [0.65, 0.59, 0.55, 0.50, 0.45, 0.40]

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
                logger.info(f"Vector search found {len(results)} results at threshold {threshold}")

                # Add threshold info to results for debugging
                for result in results:
                    result['search_threshold_used'] = threshold

                return results

        except Exception as e:
            logger.warning(f"Vector search failed at threshold {threshold}: {str(e)}")
            continue

    # No results found at any threshold
    logger.info("Multi-threshold vector search found no results at any threshold")
    return []

# Alternative version with configurable thresholds
async def multi_threshold_vector_search_configurable(schema_manager, search_text,
                                                     max_results=5,
                                                     thresholds=None,
                                                     min_results=1):
        """
        Configurable multi-threshold vector search.

        Args:
            schema_manager: SchemaManager instance
            search_text: Text to search for
            max_results: Maximum number of results to return
            thresholds: List of thresholds to try (defaults to [0.65, 0.59, 0.55, 0.50, 0.45, 0.40])
            min_results: Minimum number of results needed before stopping (default: 1)

        Returns:
            List of search results
        """
        if thresholds is None:
            thresholds = [0.65, 0.59, 0.55, 0.50, 0.45, 0.40]

        best_results = []
        threshold_used = None

        for threshold in thresholds:
            try:
                logger.debug(f"Trying vector search with threshold {threshold}")

                results = await schema_manager.find_tables_by_vector_search(
                    search_text,
                    similarity_threshold=threshold,
                    max_results=max_results
                )

                if results and len(results) >= min_results:
                    # Found sufficient results
                    logger.info(f"Vector search found {len(results)} results at threshold {threshold}")

                    # Add threshold info to results
                    for result in results:
                        result['search_threshold_used'] = threshold

                    return results
                elif results:
                    # Found some results but not enough, keep trying but save these as backup
                    if not best_results:
                        best_results = results
                        threshold_used = threshold

            except Exception as e:
                logger.warning(f"Vector search failed at threshold {threshold}: {str(e)}")
                continue

        # Return best results found, even if less than min_results
        if best_results:
            logger.info(f"Returning {len(best_results)} results from threshold {threshold_used}")
            for result in best_results:
                result['search_threshold_used'] = threshold_used
            return best_results

        # No results found at any threshold
        logger.info("Multi-threshold vector search found no results at any threshold")
        return []

# Helper function to find tables by name
async def find_tables_by_name(schema_manager, name):
    """
    Find tables by name, which could be a schema name or table name.
    
    Args:
        schema_manager: SchemaManager instance
        name: Name to search for (schema or table)
        
    Returns:
        Dict with tables information
    """
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
                LOWER(td.name) = LOWER($1)
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
                    LOWER(sd.name) = LOWER($1)
                """
                tables = await conn.fetch(query, name)
            
            # If still no results, try fuzzy match
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
                    LOWER(td.name) LIKE LOWER($1) OR
                    LOWER(sd.name) LIKE LOWER($1)
                """
                tables = await conn.fetch(query, f"%{name}%")
            
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
# app/services/query_generation/nodes/schema_retriever.py
import re
from app.services.schema_management import SchemaManager
from app.core.logging import logger

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
        
        # 1. Check for follow-up linguistic patterns
        follow_up_patterns = [
            "change", "modify", "update", "instead", "but", "rather", 
            "change to", "switch to", "yesterday", "show me", "can you"
        ]
        is_follow_up = any(token in query_text.lower() for token in follow_up_patterns)
        # has_follow_up_words = any(token in query_text.lower() for token in follow_up_patterns)
        
        # # 2. Check if it has its own directives (suggesting a new query)
        # has_own_directives = len(directives) > 0
        
        # # 3. Decide if it's a follow-up
        # # - If it has follow-up words and NO directives, likely a follow-up
        # # - If it has its own directives, likely a new query even with follow-up words
        # is_follow_up = has_follow_up_words and not has_own_directives
        
        # state.thinking.append(f"Query analysis: follow-up words: {has_follow_up_words}, " +
        #                      f"has directives: {has_own_directives}, " +
        #                      f"classified as follow-up: {is_follow_up}")
        
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
                    # Look for typical KDB query patterns that would indicate a successful query
                    if 'select from ' in content.lower() and not 'error' in content.lower():
                        # Extract table names from the query
                        table_matches = re.findall(r'from\s+(\w+)', content.lower())
                        if table_matches:
                            previous_tables.extend(table_matches)
                            state.thinking.append(f"Found previous successful query using tables: {previous_tables}")
                            break
            
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
                    
                    # Get examples for these tables if available
                    # This would need implementation based on your system
                    
                    # Update state with combined schema
                    state.query_schema = combined_schema
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
        relevant_tables = await schema_manager.find_tables_by_vector_search(
            search_text, 
            similarity_threshold=0.65,
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
            
            # Get examples for these tables
            table_ids = [table["id"] for table in relevant_tables if "id" in table]
            # Examples retrieval would be implemented based on your system
            
            # Update state with combined schema
            state.query_schema = combined_schema
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
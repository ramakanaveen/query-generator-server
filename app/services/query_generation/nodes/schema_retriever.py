# app/services/query_generation/nodes/schema_retriever.py
from app.services.schema_management import SchemaManager
from app.core.logging import logger

async def retrieve_schema(state):
    """
    Retrieve schema information based on directives and entities using vector similarity search.
    
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
        
        # Add thinking step
        state.thinking.append("Retrieving schema information using vector search...")
        
        # Initialize schema manager
        schema_manager = SchemaManager()
        schemas_exist = await schema_manager.check_schemas_available()

        if not schemas_exist:
            state.thinking.append("No schemas found in the database. Please upload schema information first.")
            state.query_schema = None
            state.no_schema_found = True  # Add a flag to indicate no schemas were found
            return state
        
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
            # This would be implemented in a more sophisticated way, but simplified here
            table_ids = [table["id"] for table in relevant_tables]
            # Get examples would go here
            
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
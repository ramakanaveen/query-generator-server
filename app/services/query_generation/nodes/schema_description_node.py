# app/services/query_generation/nodes/schema_description_node.py
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from app.core.logging import logger
from app.services.query_generation.prompts.schema_description_prompts import SCHEMA_DESCRIPTION_PROMPT_TEMPLATE

async def generate_schema_description(state):
    """
    Generate a natural language description of schema information.
    
    Args:
        state: The current state with schema_targets and directives
        
    Returns:
        Updated state with schema description
    """
    try:
        # Extract relevant information from state
        schema_targets = state.schema_targets
        directives = state.directives
        llm = state.llm
        
        # Add thinking step
        state.thinking.append("Generating schema description...")
        
        # Get schema information
        schema_info = await retrieve_schema_information(state, schema_targets, directives)
        if not schema_info:
            state.thinking.append("No schema information found")
            state.generated_content = "I couldn't find any schema information matching your request."
            return state
            
        # Format schema information based on detail level
        formatted_schema = format_schema_for_prompt(schema_info, schema_targets["detail_level"])
        
        # Generate natural language description using LLM
        prompt = ChatPromptTemplate.from_template(SCHEMA_DESCRIPTION_PROMPT_TEMPLATE)
        
        chain = prompt | llm
        response = await chain.ainvoke({
            "schema_info": formatted_schema,
            "detail_level": schema_targets["detail_level"],
            "database_type": state.database_type,
            "query": state.query
        })
        
        # Update state with the generated description
        state.generated_content = response.content.strip()
        state.thinking.append("Generated schema description")
        
        return state
    
    except Exception as e:
        logger.error(f"Error in schema description generator: {str(e)}", exc_info=True)
        state.thinking.append(f"Error generating schema description: {str(e)}")
        state.generated_content = f"I encountered an error while retrieving schema information: {str(e)}"
        return state

async def retrieve_schema_information(state, schema_targets, directives):
    """
    Retrieve schema information based on targets and directives.
    """
    try:
        from app.services.schema_management import SchemaManager
        schema_manager = SchemaManager()
        
        # Extract table targets
        tables = schema_targets.get("tables", [])
        columns = schema_targets.get("columns", [])
        
        # Check if we need to get ALL tables from ALL schemas
        get_all_tables = False
        if not tables or "*ALL*" in tables:
            get_all_tables = True
        
        # Get schema information from database
        schema_info = {}
        
        if get_all_tables:
            # Fetch all active schemas and their tables
            conn = await schema_manager._get_db_connection()
            try:
                # Get all active schemas and their tables
                query = """
                SELECT 
                    sd.id as schema_id,
                    sd.name as schema_name,
                    td.id as table_id,
                    td.name as table_name,
                    td.description as table_description,
                    td.content as table_content
                FROM 
                    schema_definitions sd
                JOIN 
                    active_schemas a ON sd.id = a.schema_id
                JOIN 
                    schema_versions sv ON a.current_version_id = sv.id
                JOIN 
                    table_definitions td ON sv.id = td.schema_version_id
                ORDER BY
                    sd.name, td.name
                """
                all_tables = await conn.fetch(query)
                
                # Organize results by schema
                for table in all_tables:
                    schema_name = table["schema_name"]
                    table_name = table["table_name"]
                    table_content = table["table_content"]
                    
                    # Convert content to dict if it's a string
                    if isinstance(table_content, str):
                        import json
                        table_content = json.loads(table_content)
                    
                    # Initialize schema entry if not exists
                    if schema_name not in schema_info:
                        schema_info[schema_name] = {"tables": {}}
                    
                    # Add table information
                    schema_info[schema_name]["tables"][table_name] = {
                        "description": table["table_description"],
                        "columns": table_content.get("columns", [])
                    }
            finally:
                await conn.close()
        else:
            # Process specific tables or schemas
            for table_name in tables:
                # Try to find tables by name
                table_info = await find_tables_by_name(schema_manager, table_name)
                if table_info:
                    schema_info[table_name] = table_info
        
        # If we have specific columns to look for, filter the results
        if columns and schema_info:
            for schema_name, schema_data in schema_info.items():
                for table_name, table_data in schema_data.get("tables", {}).items():
                    # Filter columns
                    if "columns" in table_data and columns:
                        filtered_columns = []
                        for column in table_data["columns"]:
                            if column.get("name") in columns:
                                filtered_columns.append(column)
                        # Replace with filtered columns
                        if filtered_columns:
                            table_data["columns"] = filtered_columns
        
        return schema_info
    
    except Exception as e:
        logger.error(f"Error retrieving schema information: {str(e)}", exc_info=True)
        return None

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
            # Try as schema name first
            query = """
            SELECT 
                sd.id as schema_id,
                sd.name as schema_name,
                td.id as table_id,
                td.name as table_name,
                td.description as table_description,
                td.content as table_content
            FROM 
                schema_definitions sd
            JOIN 
                active_schemas a ON sd.id = a.schema_id
            JOIN 
                schema_versions sv ON a.current_version_id = sv.id
            JOIN 
                table_definitions td ON sv.id = td.schema_version_id
            WHERE 
                LOWER(sd.name) = LOWER($1)
            """
            tables = await conn.fetch(query, name)
            
            # If no results, try as table name
            if not tables:
                query = """
                SELECT 
                    sd.id as schema_id,
                    sd.name as schema_name,
                    td.id as table_id,
                    td.name as table_name,
                    td.description as table_description,
                    td.content as table_content
                FROM 
                    table_definitions td
                JOIN 
                    schema_versions sv ON td.schema_version_id = sv.id
                JOIN 
                    schema_definitions sd ON sv.schema_id = sd.id
                JOIN 
                    active_schemas a ON sd.id = a.schema_id
                WHERE 
                    LOWER(td.name) = LOWER($1)
                """
                tables = await conn.fetch(query, name)
            
            # If still no results, try fuzzy match
            if not tables:
                query = """
                SELECT 
                    sd.id as schema_id,
                    sd.name as schema_name,
                    td.id as table_id,
                    td.name as table_name,
                    td.description as table_description,
                    td.content as table_content
                FROM 
                    table_definitions td
                JOIN 
                    schema_versions sv ON td.schema_version_id = sv.id
                JOIN 
                    schema_definitions sd ON sv.schema_id = sd.id
                JOIN 
                    active_schemas a ON sd.id = a.schema_id
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

def format_schema_for_prompt(schema_info, detail_level):
    """
    Format schema information for the LLM prompt using proper markdown formatting.
    """
    if not schema_info:
        return "No schema information available."
    
    result = []
    
    # Count total schemas and tables for summary
    total_schemas = len(schema_info)
    total_tables = sum(len(schema_data.get("tables", {})) 
                      for schema_data in schema_info.values())
    
    # Add summary header if showing multiple schemas
    if total_schemas > 1:
        result.append(f"# Schema Information\n")
        result.append(f"Found {total_tables} tables across {total_schemas} schemas.\n")
    
    # Process each schema
    for schema_name, schema_data in schema_info.items():
        # Add schema header
        result.append(f"## Schema: {schema_name}\n")
        
        # Process tables in this schema
        tables = schema_data.get("tables", {})
        
        # For a large number of tables with summary detail level, just list them
        if len(tables) > 5 and detail_level == "summary":
            table_names = list(tables.keys())
            result.append("**Tables**: " + ", ".join(table_names) + "\n")
            continue
        
        # Otherwise process each table in detail
        for table_name, table_data in tables.items():
            result.append(f"### Table: `{table_name}`\n")
            
            if "description" in table_data:
                result.append(f"**Description**: {table_data['description']}\n")
            
            if "columns" in table_data:
                result.append(f"#### Columns:\n")
                
                # Create a properly formatted markdown table
                if detail_level != "summary":
                    # Add table header
                    result.append("| Column Name | Data Type | Description |")
                    result.append("|------------|-----------|-------------|")
                    
                    # Add table rows
                    for column in table_data["columns"]:
                        col_name = column.get("name", "Unknown")
                        
                        if detail_level == "standard" or detail_level == "detailed":
                            col_type = column.get("type", column.get("kdb_type", "Unknown"))
                            col_desc = column.get("column_desc", column.get("description", ""))
                            
                            # Escape pipe characters in description to maintain table formatting
                            col_desc = col_desc.replace("|", "\\|")
                            
                            # Add row to table
                            result.append(f"| `{col_name}` | {col_type} | {col_desc} |")
                else:
                    # Simple list for summary mode
                    for column in table_data["columns"]:
                        col_name = column.get("name", "Unknown")
                        result.append(f"- `{col_name}`")
                
                result.append("")  # Add empty line after columns
            
            # Add example queries if any
            if "examples" in table_data and len(table_data["examples"]) > 0:
                result.append("#### Example Queries:\n")
                
                for example in table_data["examples"]:
                    nl_query = example.get("natural_language", "")
                    query = example.get("query", "")
                    
                    if nl_query and query:
                        result.append(f"**{nl_query}**\n")
                        result.append("```q")
                        result.append(query)
                        result.append("```\n")
    
    return "\n".join(result)
# app/services/query_generation/nodes/schema_retriever.py
from typing import Dict, Any
import json
import os
from pathlib import Path
from app.core.logging import logger
from app.core.config import settings

async def retrieve_schema(state):
    """
    Retrieve schema information based on directives and entities.
    
    Args:
        state: The current state of the workflow
        
    Returns:
        Updated state with schema information
    """
    try:
        directives = state.directives
        entities = state.entities
        database_type = state.database_type
        
        # Add thinking step
        state.thinking.append("Retrieving schema information...")
        
        # Get schema based on directives
        schema = {}
        
        # For now, we'll use a simple approach of loading from JSON files
        schemas_dir = Path(settings.SCHEMAS_DIRECTORY)
        logger.info(f"Schemas directory: {schemas_dir}")
        # Try to find schema files based on directives
        schema_files = []
        for directive in directives:
            directive_path = schemas_dir / f"{directive.lower()}.json"
            if directive_path.exists():
                schema_files.append(directive_path)
        
        # If no schema files found by directives, try to find by entities
        if not schema_files:
            state.thinking.append("No schemas found for directives, searching by entities...")
            for entity in entities:
                entity_path = schemas_dir / f"{entity.lower()}.json"
                if entity_path.exists():
                    schema_files.append(entity_path)
        
        # Load and combine schemas
        for schema_file in schema_files:
            try:
                with open(schema_file, 'r') as f:
                    file_schema = json.load(f)
                    # Combine schemas (simple approach, just update dictionaries)
                    schema.update(file_schema)
            except Exception as e:
                state.thinking.append(f"Error loading schema file {schema_file}: {str(e)}")
        
        # If no schemas found, use a default schema
        if not schema:
            state.thinking.append("No specific schema found, using default schema...")
            default_schema_path = schemas_dir / "default.json"
            if default_schema_path.exists():
                try:
                    with open(default_schema_path, 'r') as f:
                        schema = json.load(f)
                except Exception as e:
                    state.thinking.append(f"Error loading default schema: {str(e)}")
            else:
                state.thinking.append("No default schema found, proceeding with minimal schema...")
                schema = {
                    "description": "Generic database with tables and columns.",
                    "tables": {
                        "table": {
                            "description": "Generic table",
                            "columns": ["column1", "column2"]
                        }
                    }
                }
        
        # Update state with schema
        state.schema = schema
        state.thinking.append(f"Retrieved schema for {len(schema_files)} resources" if schema_files else "Using default schema")
        
        return state
    
    except Exception as e:
        logger.error(f"Error in schema retriever: {str(e)}")
        state["thinking"].append(f"Error retrieving schema: {str(e)}")
        # Still return the state to continue the workflow
        return state
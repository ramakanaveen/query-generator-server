# app/routes/schema_management.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import Optional
import json
import tempfile
import os

from app.services.schema_management import SchemaManager
from app.core.logging import logger
# from app.dependencies import get_current_user  # Uncomment when you add auth

router = APIRouter()

@router.post("/schemas/upload")
async def upload_schema(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    # user = Depends(get_current_user)  # Uncomment when you add auth
):
    """
    Upload a schema file to import into the database.
    """
    try:
        # For now, hardcode user_id until auth is implemented
        user_id = 1  # Replace with user.id when auth is added
        
        # Create temporary file to store uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            # Write uploaded file content to temp file
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Initialize schema manager
            schema_manager = SchemaManager()
            
            # Read file to get schema data for validation and default values
            with open(temp_path, 'r') as f:
                schema_data = json.load(f)
            
            # Use filename as schema name if not provided
            if name is None:
                name = os.path.splitext(file.filename)[0].upper()
            
            # Use schema description if not provided
            if description is None:
                description = schema_data.get("description", f"Schema for {name}")
            
            # Create schema definition
            schema_id = await schema_manager.create_schema_definition(
                name=name,
                description=description
            )
            
            if not schema_id:
                raise HTTPException(status_code=500, detail="Failed to create schema definition")
            
            # Create schema version
            version_id = await schema_manager.create_schema_version(
                schema_id=schema_id,
                version=1,
                created_by=user_id,
                notes=f"Uploaded via API: {file.filename}"
            )
            
            if not version_id:
                raise HTTPException(status_code=500, detail="Failed to create schema version")
            
            # Process tables
            tables = schema_data.get("tables", {})
            table_ids = {}
            
            for table_name, table_data in tables.items():
                # Extract description from table data
                table_description = table_data.get("description", f"Table {table_name}")
                
                # Add table definition
                table_id = await schema_manager.add_table_definition(
                    schema_version_id=version_id,
                    table_name=table_name,
                    table_content=table_data,
                    description=table_description
                )
                
                if table_id:
                    table_ids[table_name] = table_id
                else:
                    logger.warning(f"Failed to add table {table_name}")
            
            # Process examples
            examples = schema_data.get("examples", [])
            example_count = 0
            
            for example in examples:
                nl_query = example.get("natural_language", "")
                generated_query = example.get("query", "")
                
                if not nl_query or not generated_query:
                    continue
                
                # Determine related tables
                example_tables = []
                
                if "table" in example:
                    # If example explicitly specifies a table
                    table_name = example["table"]
                    if table_name in table_ids:
                        example_tables.append(table_ids[table_name])
                else:
                    # Try to infer from the query content
                    query = example.get("query", "").lower()
                    for table_name, tid in table_ids.items():
                        if table_name.lower() in query:
                            example_tables.append(tid)
                
                # Use the first match as primary table, or None if no matches
                primary_table_id = example_tables[0] if example_tables else None
                
                # Add example
                example_id = await schema_manager.add_schema_example(
                    schema_version_id=version_id,
                    nl_query=nl_query,
                    generated_query=generated_query,
                    table_id=primary_table_id,
                    description=example.get("description", "")
                )
                
                if example_id:
                    example_count += 1
                    # If example relates to multiple tables, create mappings
                    if len(example_tables) > 1:
                        await schema_manager.map_example_to_tables(
                            example_id=example_id,
                            table_ids=example_tables
                        )
            
            # Activate the schema version
            activation_success = await schema_manager.activate_schema_version(schema_id, version_id)
            
            return {
                "success": True,
                "schema_id": schema_id,
                "version_id": version_id,
                "name": name,
                "table_count": len(table_ids),
                "example_count": example_count,
                "activated": activation_success
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in schema file")
    except Exception as e:
        logger.error(f"Error uploading schema: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing schema: {str(e)}")

@router.get("/schemas")
async def list_schemas():
    """Get all available schemas."""
    try:
        conn = await SchemaManager()._get_db_connection()
        try:
            # Fetch active schemas with their versions
            query = """
            SELECT 
                sd.id, 
                sd.name, 
                sd.description,
                sv.version,
                sv.id as version_id,
                (SELECT COUNT(*) FROM table_definitions td WHERE td.schema_version_id = sv.id) as table_count
            FROM 
                schema_definitions sd
            JOIN 
                active_schemas a ON sd.id = a.schema_id
            JOIN 
                schema_versions sv ON a.current_version_id = sv.id
            ORDER BY 
                sd.name
            """
            
            results = await conn.fetch(query)
            
            schemas = []
            for row in results:
                schemas.append({
                    "id": row["id"],
                    "name": row["name"],
                    "description": row["description"],
                    "current_version": row["version"],
                    "version_id": row["version_id"],
                    "table_count": row["table_count"]
                })
            
            return {"schemas": schemas}
        finally:
            await conn.close()
    except Exception as e:
        logger.error(f"Error listing schemas: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching schemas: {str(e)}")
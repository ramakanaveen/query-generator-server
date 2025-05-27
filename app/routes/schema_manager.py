# app/routes/schema_manager.py

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Body
from typing import Optional, List, Dict, Any
import json
import uuid
import csv
import io
import pandas as pd
import re
from datetime import datetime

from app.core.db import db_pool
from app.core.logging import logger
from app.services.ai_description_service import AISchemaService
from app.services.embedding_provider import EmbeddingProvider
from app.services.llm_provider import LLMProvider
from app.services.schema_editor import SchemaEditorService

router = APIRouter(prefix="/schema-manager", tags=["schema-manager"])

# Initialize services
embedding_provider = EmbeddingProvider()
schema_editor = SchemaEditorService()
ai_schema_service = AISchemaService()
llm_provider = LLMProvider()

@router.get("/groups")
async def list_schema_groups():
    """List all schema groups."""
    try:
        conn = await db_pool.get_connection()
        try:
            query = """
                    SELECT
                        id,
                        name,
                        description,
                        created_at,
                        updated_at,
                        (SELECT COUNT(*) FROM schema_definitions WHERE group_id = schema_groups.id) as schema_count
                    FROM
                        schema_groups
                    ORDER BY
                        name \
                    """

            results = await conn.fetch(query)

            groups = [dict(row) for row in results]

            return {"groups": groups}
        finally:
            await db_pool.release_connection(conn)
    except Exception as e:
        logger.error(f"Error listing schema groups: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching schema groups: {str(e)}")

@router.post("/groups")
async def create_schema_group(
        data: Dict[str, Any] = Body(...)
):
    """Create a new schema group."""
    try:
        name = data.get("name")
        description = data.get("description", f"Group for {name}")

        if not name:
            raise HTTPException(status_code=400, detail="Group name is required")

        conn = await db_pool.get_connection()
        try:
            # Check if group with same name already exists
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM schema_groups WHERE name = $1)",
                name
            )

            if exists:
                raise HTTPException(status_code=409, detail=f"Group with name '{name}' already exists")

            # Create new group
            group_id = await conn.fetchval(
                """
                INSERT INTO schema_groups (name, description, created_at, updated_at)
                VALUES ($1, $2, $3, $3)
                    RETURNING id
                """,
                name, description, datetime.now()
            )

            # Get the created group
            group = await conn.fetchrow(
                "SELECT id, name, description, created_at, updated_at FROM schema_groups WHERE id = $1",
                group_id
            )

            return dict(group)
        finally:
            await db_pool.release_connection(conn)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating schema group: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating schema group: {str(e)}")

@router.get("/groups/{group_id}")
async def get_schema_group(group_id: int):
    """Get a specific schema group by ID."""
    try:
        conn = await db_pool.get_connection()
        try:
            query = """
                    SELECT
                        id,
                        name,
                        description,
                        created_at,
                        updated_at
                    FROM
                        schema_groups
                    WHERE
                        id = $1 \
                    """

            group = await conn.fetchrow(query, group_id)

            if not group:
                raise HTTPException(status_code=404, detail=f"Group with ID {group_id} not found")

            # Get schemas in this group
            schemas_query = """
                            SELECT
                                id,
                                name,
                                description
                            FROM
                                schema_definitions
                            WHERE
                                group_id = $1
                            ORDER BY
                                name \
                            """

            schemas = await conn.fetch(schemas_query, group_id)

            return {
                **dict(group),
                "schemas": [dict(schema) for schema in schemas]
            }
        finally:
            await db_pool.release_connection(conn)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching schema group: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching schema group: {str(e)}")

@router.put("/groups/{group_id}")
async def update_schema_group(
        group_id: int,
        data: Dict[str, Any] = Body(...)
):
    """Update a schema group."""
    try:
        conn = await db_pool.get_connection()
        try:
            # Check if group exists
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM schema_groups WHERE id = $1)",
                group_id
            )

            if not exists:
                raise HTTPException(status_code=404, detail=f"Group with ID {group_id} not found")

            # Extract updatable fields
            name = data.get("name")
            description = data.get("description")

            # Build update query
            update_fields = []
            params = [group_id]

            if name:
                update_fields.append(f"name = ${len(params) + 1}")
                params.append(name)

            if description:
                update_fields.append(f"description = ${len(params) + 1}")
                params.append(description)

            if not update_fields:
                raise HTTPException(status_code=400, detail="No fields to update")

            # Add updated_at
            update_fields.append(f"updated_at = ${len(params) + 1}")
            params.append(datetime.now())

            # Execute update
            query = f"""
            UPDATE schema_groups
            SET {", ".join(update_fields)}
            WHERE id = $1
            RETURNING id, name, description, created_at, updated_at
            """

            updated_group = await conn.fetchrow(query, *params)

            return dict(updated_group)
        finally:
            await db_pool.release_connection(conn)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating schema group: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error updating schema group: {str(e)}")

@router.delete("/groups/{group_id}")
async def delete_schema_group(group_id: int):
    """Delete a schema group."""
    try:
        conn = await db_pool.get_connection()
        try:
            # Check if group exists
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM schema_groups WHERE id = $1)",
                group_id
            )

            if not exists:
                raise HTTPException(status_code=404, detail=f"Group with ID {group_id} not found")

            # Check if group has schemas
            has_schemas = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM schema_definitions WHERE group_id = $1)",
                group_id
            )

            if has_schemas:
                raise HTTPException(
                    status_code=409,
                    detail="Cannot delete group that contains schemas. Delete all schemas first."
                )

            # Delete group
            await conn.execute(
                "DELETE FROM schema_groups WHERE id = $1",
                group_id
            )

            return {"success": True, "message": f"Group with ID {group_id} deleted"}
        finally:
            await db_pool.release_connection(conn)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting schema group: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting schema group: {str(e)}")

# Now let's implement the schema endpoints
@router.get("/schemas")
async def list_schemas(group_id: Optional[int] = None):
    """List all schemas, optionally filtered by group."""
    try:
        conn = await db_pool.get_connection()
        try:
            # Base query
            query = """
                    SELECT
                        sd.id,
                        sd.name,
                        sd.description,
                        sd.group_id,
                        sg.name as group_name,
                        sv.id as current_version_id,
                        sv.version as current_version,
                        sv.status,
                        sv.created_at,
                        sd.updated_at,
                        (SELECT COUNT(*) FROM table_definitions td WHERE td.schema_version_id = sv.id) as table_count
                    FROM
                        schema_definitions sd
                            JOIN
                        schema_groups sg ON sd.group_id = sg.id
                            LEFT JOIN
                        active_schemas a ON sd.id = a.schema_id
                            LEFT JOIN
                        schema_versions sv ON a.current_version_id = sv.id \
                    """

            # Add group filter if provided
            if group_id:
                query += " WHERE sd.group_id = $1"
                query += " ORDER BY sd.name"

                results = await conn.fetch(query, group_id)
            else:
                query += " ORDER BY sg.name, sd.name"

                results = await conn.fetch(query)

            schemas = []
            for row in results:
                schemas.append({
                    "id": row["id"],
                    "name": row["name"],
                    "description": row["description"],
                    "group_id": row["group_id"],
                    "group_name": row["group_name"],
                    "current_version_id": row["current_version_id"],
                    "current_version": row["current_version"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "table_count": row["table_count"]
                })

            return {"schemas": schemas}
        finally:
            await db_pool.release_connection(conn)
    except Exception as e:
        logger.error(f"Error listing schemas: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching schemas: {str(e)}")

@router.post("/schemas")
async def create_schema(
        data: Dict[str, Any] = Body(...)
):
    """Create a new schema."""
    try:
        # Extract required fields
        name = data.get("name")
        group_id = data.get("group_id")
        description = data.get("description", f"Schema for {name}")

        if not name:
            raise HTTPException(status_code=400, detail="Schema name is required")

        if not group_id:
            raise HTTPException(status_code=400, detail="Group ID is required")

        conn = await db_pool.get_connection()
        try:
            # Check if group exists
            group_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM schema_groups WHERE id = $1)",
                group_id
            )

            if not group_exists:
                raise HTTPException(status_code=404, detail=f"Group with ID {group_id} not found")

            # Check if schema with same name exists in group
            schema_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM schema_definitions WHERE group_id = $1 AND name = $2)",
                group_id, name
            )

            if schema_exists:
                raise HTTPException(
                    status_code=409,
                    detail=f"Schema with name '{name}' already exists in this group"
                )

            # Create schema
            async with conn.transaction():
                # Create schema definition
                schema_id = await conn.fetchval(
                    """
                    INSERT INTO schema_definitions (group_id, name, description, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $4)
                        RETURNING id
                    """,
                    group_id, name, description, datetime.now()
                )

                # Create initial version
                version_id = await conn.fetchval(
                    """
                    INSERT INTO schema_versions (schema_id, version, status, created_at)
                    VALUES ($1, 1, 'draft', $2)
                        RETURNING id
                    """,
                    schema_id, datetime.now()
                )

                # Create empty raw JSON in the version
                await conn.execute(
                    """
                    UPDATE schema_versions
                    SET raw_json = $1
                    WHERE id = $2
                    """,
                    json.dumps({
                        "name": name,
                        "description": description,
                        "tables": {},
                        "examples": []
                    }),
                    version_id
                )

                # Get the created schema
                schema = await conn.fetchrow(
                    """
                    SELECT
                        sd.id,
                        sd.name,
                        sd.description,
                        sd.group_id,
                        sg.name as group_name,
                        sv.id as version_id,
                        sv.version,
                        sv.status
                    FROM
                        schema_definitions sd
                            JOIN
                        schema_groups sg ON sd.group_id = sg.id
                            JOIN
                        schema_versions sv ON sd.id = sv.schema_id
                    WHERE
                        sd.id = $1
                    """,
                    schema_id
                )

                return dict(schema)
        finally:
            await db_pool.release_connection(conn)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating schema: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating schema: {str(e)}")

@router.get("/schemas/{schema_id}")
async def get_schema(schema_id: int):
    """Get a specific schema by ID."""
    try:
        conn = await db_pool.get_connection()
        try:
            # Get schema details
            schema_query = """
                           SELECT
                               sd.id,
                               sd.name,
                               sd.description,
                               sd.group_id,
                               sg.name as group_name,
                               sv.id as current_version_id,
                               sv.version as current_version,
                               sv.status,
                               sv.created_at,
                               sd.updated_at
                           FROM
                               schema_definitions sd
                                   JOIN
                               schema_groups sg ON sd.group_id = sg.id
                                   LEFT JOIN
                               active_schemas a ON sd.id = a.schema_id
                                   LEFT JOIN
                               schema_versions sv ON a.current_version_id = sv.id
                           WHERE
                               sd.id = $1 \
                           """

            schema = await conn.fetchrow(schema_query, schema_id)

            if not schema:
                raise HTTPException(status_code=404, detail=f"Schema with ID {schema_id} not found")

            result = dict(schema)

            # Get tables in this schema's active version
            if result.get("current_version_id"):
                tables_query = """
                               SELECT
                                   id,
                                   name,
                                   description,
                                   content
                               FROM
                                   table_definitions
                               WHERE
                                   schema_version_id = $1
                               ORDER BY
                                   name \
                               """

                tables = await conn.fetch(tables_query, result["current_version_id"])

                result["tables"] = [
                    {
                        "id": table["id"],
                        "name": table["name"],
                        "description": table["description"],
                        "content": table["content"] if isinstance(table["content"], dict) else json.loads(table["content"])
                    }
                    for table in tables
                ]
            else:
                result["tables"] = []

            # Get versions history
            versions_query = """
                             SELECT
                                 id,
                                 version,
                                 status,
                                 created_at
                             FROM
                                 schema_versions
                             WHERE
                                 schema_id = $1
                             ORDER BY
                                 version DESC \
                             """

            versions = await conn.fetch(versions_query, schema_id)

            result["versions"] = [dict(version) for version in versions]

            return result
        finally:
            await db_pool.release_connection(conn)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching schema: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching schema: {str(e)}")

@router.get("/schemas/{schema_id}/json")
async def get_schema_json(schema_id: int, version_id: Optional[int] = None):
    """Get the raw JSON for a schema."""
    try:
        conn = await db_pool.get_connection()
        try:
            # Determine which version to get
            if version_id:
                # Check if specific version exists
                version_exists = await conn.fetchval(
                    """
                    SELECT EXISTS(
                        SELECT 1 FROM schema_versions
                        WHERE id = $1 AND schema_id = $2
                    )
                    """,
                    version_id, schema_id
                )

                if not version_exists:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Version with ID {version_id} not found for schema {schema_id}"
                    )
            else:
                # Get active version
                version_id = await conn.fetchval(
                    """
                    SELECT current_version_id
                    FROM active_schemas
                    WHERE schema_id = $1
                    """,
                    schema_id
                )

                if not version_id:
                    # If no active version, get latest version
                    version_id = await conn.fetchval(
                        """
                        SELECT id
                        FROM schema_versions
                        WHERE schema_id = $1
                        ORDER BY version DESC
                            LIMIT 1
                        """,
                        schema_id
                    )

                    if not version_id:
                        raise HTTPException(
                            status_code=404,
                            detail=f"No versions found for schema {schema_id}"
                        )

            # Get raw JSON from version
            raw_json = await conn.fetchval(
                """
                SELECT raw_json
                FROM schema_versions
                WHERE id = $1
                """,
                version_id
            )

            # Parse JSON
            if raw_json:
                try:
                    parsed_json = json.loads(raw_json)
                    if 'schema_json' in parsed_json:
                        return parsed_json['schema_json']
                    else:
                        return parsed_json
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in schema version {version_id}")
                    # Return empty schema as fallback
                    return {"tables": {}, "examples": []}
            else:
                # If no raw JSON, generate it from tables and examples
                schema_name = await conn.fetchval(
                    "SELECT name FROM schema_definitions WHERE id = $1",
                    schema_id
                )

                # Get tables
                tables_query = """
                               SELECT
                                   name,
                                   content
                               FROM
                                   table_definitions
                               WHERE
                                   schema_version_id = $1 \
                               """

                tables = await conn.fetch(tables_query, version_id)

                # Get examples
                examples_query = """
                                 SELECT
                                     natural_language_query,
                                     generated_query,
                                     description
                                 FROM
                                     schema_examples
                                 WHERE
                                     schema_version_id = $1 \
                                 """

                examples = await conn.fetch(examples_query, version_id)

                # Build JSON structure
                schema_json = {
                    "name": schema_name,
                    "tables": {},
                    "examples": []
                }

                for table in tables:
                    table_content = table["content"]
                    if isinstance(table_content, str):
                        try:
                            table_content = json.loads(table_content)
                        except json.JSONDecodeError:
                            table_content = {}

                    schema_json["tables"][table["name"]] = table_content

                for example in examples:
                    schema_json["examples"].append({
                        "natural_language": example["natural_language_query"],
                        "query": example["generated_query"],
                        "description": example["description"]
                    })

                return schema_json
        finally:
            await db_pool.release_connection(conn)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching schema JSON: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching schema JSON: {str(e)}")

@router.put("/schemas/{schema_id}/json")
async def update_schema_json(
        schema_id: int,
        schema_json: Dict[str, Any] = Body(...),
        create_version: bool = False,
        version_notes: Optional[str] = None
):
    """
    Update the schema JSON, optionally creating a new version.

    Args:
        schema_id: ID of the schema to update
        schema_json: Complete schema JSON
        create_version: Whether to create a new version (default: false)
        version_notes: Optional notes for the new version
    """
    try:
        conn = await db_pool.get_connection()
        try:
            # Check if schema exists
            schema_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM schema_definitions WHERE id = $1)",
                schema_id
            )

            if not schema_exists:
                raise HTTPException(status_code=404, detail=f"Schema with ID {schema_id} not found")

            # Get active or most recent version ID
            active_version_id = await conn.fetchval(
                """
                SELECT current_version_id
                FROM active_schemas
                WHERE schema_id = $1
                """,
                schema_id
            )

            if not active_version_id:
                # Get latest version
                active_version_id = await conn.fetchval(
                    """
                    SELECT id
                    FROM schema_versions
                    WHERE schema_id = $1
                    ORDER BY version DESC
                        LIMIT 1
                    """,
                    schema_id
                )

            target_version_id = None

            async with conn.transaction():
                if create_version and active_version_id:
                    # Get current version number
                    current_version = await conn.fetchval(
                        "SELECT version FROM schema_versions WHERE id = $1",
                        active_version_id
                    )

                    # Create new version
                    target_version_id = await conn.fetchval(
                        """
                        INSERT INTO schema_versions
                            (schema_id, version, status, notes, created_at)
                        VALUES ($1, $2, 'draft', $3, $4)
                            RETURNING id
                        """,
                        schema_id, current_version + 1, version_notes, datetime.now()
                    )

                    logger.info(f"Created new schema version: {target_version_id}")
                else:
                    # Use existing version
                    target_version_id = active_version_id

                if not target_version_id:
                    # If still no version, create first version
                    target_version_id = await conn.fetchval(
                        """
                        INSERT INTO schema_versions
                            (schema_id, version, status, notes, created_at)
                        VALUES ($1, 1, 'draft', $2, $3)
                            RETURNING id
                        """,
                        schema_id, version_notes, datetime.now()
                    )

                # Convert schema_json to string
                schema_json_str = json.dumps(schema_json)

                # Update raw JSON in version
                await conn.execute(
                    """
                    UPDATE schema_versions
                    SET raw_json = $1, updated_at = $2
                    WHERE id = $3
                    """,
                    schema_json_str, datetime.now(), target_version_id
                )

                # Process the schema to update tables and examples
                await schema_editor.process_schema_json(target_version_id, schema_json)

                # Update embedding for tables
                table_ids = await conn.fetch(
                    "SELECT id, name FROM table_definitions WHERE schema_version_id = $1",
                    target_version_id
                )

                for table in table_ids:
                    await schema_editor.update_table_embedding(table["id"], table["name"])

                return {
                    "success": True,
                    "schema_id": schema_id,
                    "version_id": target_version_id,
                    "is_new_version": create_version
                }
        finally:
            await db_pool.release_connection(conn)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating schema JSON: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error updating schema JSON: {str(e)}")

@router.post("/schemas/{schema_id}/activate")
async def activate_schema_version(
        schema_id: int,
        version_id: int
):
    """Activate a specific schema version."""
    try:
        conn = await db_pool.get_connection()
        try:
            # Check if schema exists
            schema_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM schema_definitions WHERE id = $1)",
                schema_id
            )

            if not schema_exists:
                raise HTTPException(status_code=404, detail=f"Schema with ID {schema_id} not found")

            # Check if version exists for this schema
            version_exists = await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM schema_versions
                    WHERE id = $1 AND schema_id = $2
                )
                """,
                version_id, schema_id
            )

            if not version_exists:
                raise HTTPException(
                    status_code=404,
                    detail=f"Version with ID {version_id} not found for schema {schema_id}"
                )

            async with conn.transaction():
                # Update currently active version to deprecated
                await conn.execute(
                    """
                    UPDATE schema_versions
                    SET status = 'deprecated', deprecated_at = $1
                    WHERE schema_id = $2 AND status = 'active'
                    """,
                    datetime.now(), schema_id
                )

                # Update new version to active
                await conn.execute(
                    """
                    UPDATE schema_versions
                    SET status = 'active', activated_at = $1
                    WHERE id = $2
                    """,
                    datetime.now(), version_id
                )

                # Update active_schemas table
                await conn.execute(
                    """
                    INSERT INTO active_schemas (schema_id, current_version_id, last_updated)
                    VALUES ($1, $2, $3)
                        ON CONFLICT (schema_id) 
                    DO UPDATE SET current_version_id = $2, last_updated = $3
                    """,
                    schema_id, version_id, datetime.now()
                )

                return {
                    "success": True,
                    "schema_id": schema_id,
                    "version_id": version_id,
                    "activated_at": datetime.now().isoformat()
                }
        finally:
            await db_pool.release_connection(conn)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating schema version: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error activating schema version: {str(e)}")

# AI-assisted endpoints
@router.post("/ai/describe")
async def generate_descriptions(
        data: Dict[str, Any] = Body(...)
):
    """Generate descriptions for tables or columns using AI."""
    try:
        # Extract fields
        column_name = data.get("column_name")
        table_name = data.get("table_name")
        schema_name = data.get("schema_name")
        table_context = data.get("table_context")
        count = data.get("count", 3)  # Number of suggestions to generate

        if not column_name and not table_name:
            raise HTTPException(
                status_code=400,
                detail="Either column_name or table_name is required"
            )

        # Generate descriptions using the service
        descriptions = await ai_schema_service.generate_descriptions(
            column_name=column_name,
            table_name=table_name,
            schema_name=schema_name,
            table_context=table_context,
            count=count
        )

        return {"descriptions": descriptions}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating descriptions: {str(e)}", exc_info=True)

        # Provide fallback descriptions if possible
        try:
            if column_name and table_name:
                fallback_descriptions = ai_schema_service.generate_fallback_descriptions(column_name, table_name)
                if fallback_descriptions:
                    return {"descriptions": fallback_descriptions[:count]}
        except:
            pass

        raise HTTPException(status_code=500, detail=f"Error generating descriptions: {str(e)}")

@router.post("/ai/examples")
async def generate_examples(
        data: Dict[str, Any] = Body(...)
):
    """Generate example queries for a table using AI."""
    try:
        # Extract fields
        table_name = data.get("table_name")
        table_schema = data.get("table_schema")
        count = data.get("count", 3)  # Number of examples to generate

        if not table_name or not table_schema:
            raise HTTPException(
                status_code=400,
                detail="Table name and schema are required"
            )

        # Generate examples using the service
        examples = await ai_schema_service.generate_examples(
            table_name=table_name,
            table_schema=table_schema,
            count=count
        )

        return {"examples": examples}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating examples: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating examples: {str(e)}")

@router.post("/ai/generate-schema")
async def generate_table_schema(
        data: Dict[str, Any] = Body(...)
):
    """Generate table schema from CSV or JSON content."""
    try:
        # Extract fields
        table_name = data.get("tableName")
        description = data.get("description", "")
        content = data.get("content")
        content_type = data.get("contentType", "csv")  # 'csv' or 'json'

        if not table_name or not content:
            raise HTTPException(status_code=400, detail="Table name and content are required")

        # Process content based on type
        if content_type == "csv":
            # Parse CSV content
            return await parse_csv_schema(table_name, description, content)
        elif content_type == "json":
            # Parse JSON content
            return await parse_json_schema(table_name, description, content)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported content type: {content_type}")

    except Exception as e:
        logger.error(f"Error generating table schema: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating table schema: {str(e)}")

async def parse_csv_schema(table_name: str, description: str, content: str):
    """Parse CSV content and generate schema using AI."""
    try:
        # Read CSV into pandas
        df = pd.read_csv(io.StringIO(content), nrows=10)  # Only need a few rows for schema detection

        # Get column information
        column_info = []
        for col in df.columns:
            # Get sample values
            non_null_values = df[col].dropna().head(5).tolist()
            sample_values = [str(val) for val in non_null_values]

            column_info.append({
                "name": col,
                "sample_values": sample_values,
                "dtype": str(df[col].dtype)
            })

        # Prepare prompt for LLM
        prompt = f"""
        Analyze this CSV table data and generate a database schema:
        
        Table Name: {table_name}
        Table Description: {description}
        
        Sample data columns:
        {column_info}
        
        For each column, determine:
        1. The appropriate KDB+ data type (symbol, char, string, boolean, byte, short, int, long, real, float, time, minute, second, timestamp, month, date, datetime, timespan)
        2. A descriptive name (lowercase with underscores)
        3. A short but informative description
        4. Whether it should be a key column (pick the most likely primary key)
        5. Whether it should be a required column (usually true for key columns)
        
        Provide the schema in JSON format with this structure:
        {{
            "columns": [
                {{
                    "name": "column_name",
                    "type": "kdb_type",
                    "description": "Short description",
                    "required": true/false,
                    "key": true/false
                }}
            ]
        }}
        
        Return ONLY valid JSON without any explanations.
        """

        # Get response from LLM
        response = await llm_provider.get_completion(prompt)

        # Parse JSON from response
        try:
            # Find JSON content in the response
            import re
            import json

            # Look for JSON content
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                schema = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                schema = json.loads(response)

            # Validate schema structure
            if "columns" not in schema or not isinstance(schema["columns"], list):
                raise ValueError("Invalid schema structure: missing 'columns' array")

            return schema

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            logger.error(f"LLM response: {response}")

            # Fallback: Generate basic schema from DataFrame
            return generate_fallback_schema(table_name, description, df)

    except Exception as e:
        logger.error(f"Error parsing CSV: {str(e)}", exc_info=True)
        # Use fallback method
        return generate_fallback_schema_from_str(table_name, description, content)

async def parse_json_schema(table_name: str, description: str, content: str):
    """Parse JSON content and generate schema using AI."""
    try:
        # Parse JSON content
        import json
        json_data = json.loads(content)

        # Check if it's an array of objects
        if not isinstance(json_data, list) or len(json_data) == 0 or not isinstance(json_data[0], dict):
            sample_obj = json_data
            is_array = False
        else:
            sample_obj = json_data[0]  # Take first object as sample
            is_array = True

        # Prepare column information
        column_info = []
        for key, value in sample_obj.items():
            column_info.append({
                "name": key,
                "sample_value": str(value),
                "type": type(value).__name__
            })

        # Prepare prompt for LLM
        prompt = f"""
        Analyze this JSON data and generate a database schema:
        
        Table Name: {table_name}
        Table Description: {description}
        Data Structure: {"Array of objects" if is_array else "Single object"}
        
        Sample data fields:
        {column_info}
        
        For each field, determine:
        1. The appropriate KDB+ data type (symbol, char, string, boolean, byte, short, int, long, real, float, time, minute, second, timestamp, month, date, datetime, timespan)
        2. A descriptive name (lowercase with underscores)
        3. A short but informative description
        4. Whether it should be a key column (pick the most likely primary key)
        5. Whether it should be a required column (usually true for key columns)
        
        Provide the schema in JSON format with this structure:
        {{
            "columns": [
                {{
                    "name": "column_name",
                    "type": "kdb_type",
                    "description": "Short description",
                    "required": true/false,
                    "key": true/false
                }}
            ]
        }}
        
        Return ONLY valid JSON without any explanations.
        """

        # Get response from LLM
        response = await llm_provider.get_completion(prompt)

        # Parse JSON from response
        try:
            # Find JSON content in the response
            import re

            # Look for JSON content
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                schema = json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                schema = json.loads(response)

            # Validate schema structure
            if "columns" not in schema or not isinstance(schema["columns"], list):
                raise ValueError("Invalid schema structure: missing 'columns' array")

            return schema

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            logger.error(f"LLM response: {response}")

            # Fallback: Generate basic schema from JSON
            return generate_fallback_schema_from_json(table_name, description, json_data)

    except Exception as e:
        logger.error(f"Error parsing JSON: {str(e)}", exc_info=True)
        # Use fallback method
        return generate_fallback_schema_from_str(table_name, description, content)

def generate_fallback_schema_from_str(table_name: str, description: str, content: str):
    """Generate a basic schema from CSV string when AI fails."""
    try:
        # Try to parse as CSV
        csv_reader = csv.reader(io.StringIO(content))
        headers = next(csv_reader)
        data_row = next(csv_reader, [])

        columns = []
        for i, header in enumerate(headers):
            # Clean column name
            clean_name = header.strip().lower().replace(' ', '_').replace('-', '_')
            clean_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in clean_name)

            # Get sample value
            sample = data_row[i].strip() if i < len(data_row) else ""

            # Infer type
            col_type = infer_kdb_type(sample)

            columns.append({
                "name": clean_name,
                "type": col_type,
                "description": f"Column for {header}",
                "required": i == 0,  # First column is required by default
                "key": i == 0  # First column is key by default
            })

        return {
            "columns": columns
        }

    except Exception as e:
        logger.error(f"Fallback schema generation failed: {str(e)}", exc_info=True)

        # Return minimal schema as last resort
        return {
            "columns": [
                {
                    "name": "id",
                    "type": "symbol",
                    "description": "Primary key",
                    "required": True,
                    "key": True
                },
                {
                    "name": "value",
                    "type": "symbol",
                    "description": "Data value",
                    "required": False,
                    "key": False
                }
            ]
        }

def generate_fallback_schema_from_json(table_name: str, description: str, json_data):
    """Generate a basic schema from JSON data when AI fails."""
    try:
        # Determine if it's an array of objects
        if isinstance(json_data, list) and len(json_data) > 0 and isinstance(json_data[0], dict):
            sample_obj = json_data[0]
        elif isinstance(json_data, dict):
            sample_obj = json_data
        else:
            # Can't determine schema, return minimal
            return {
                "columns": [
                    {
                        "name": "id",
                        "type": "symbol",
                        "description": "Primary key",
                        "required": True,
                        "key": True
                    },
                    {
                        "name": "value",
                        "type": "symbol",
                        "description": "Data value",
                        "required": False,
                        "key": False
                    }
                ]
            }

        columns = []
        for i, (key, value) in enumerate(sample_obj.items()):
            # Clean column name
            clean_name = key.strip().lower().replace(' ', '_').replace('-', '_')
            clean_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in clean_name)

            # Infer type
            col_type = infer_kdb_type_from_json(value)

            columns.append({
                "name": clean_name,
                "type": col_type,
                "description": f"Column for {key}",
                "required": key.lower() == "id" or i == 0,  # ID or first column is required
                "key": key.lower() == "id" or i == 0  # ID or first column is key
            })

        return {
            "columns": columns
        }

    except Exception as e:
        logger.error(f"Fallback JSON schema generation failed: {str(e)}", exc_info=True)

        # Return minimal schema as last resort
        return {
            "columns": [
                {
                    "name": "id",
                    "type": "symbol",
                    "description": "Primary key",
                    "required": True,
                    "key": True
                },
                {
                    "name": "value",
                    "type": "symbol",
                    "description": "Data value",
                    "required": False,
                    "key": False
                }
            ]
        }

def generate_fallback_schema(table_name: str, description: str, df):
    """Generate a basic schema from pandas DataFrame when AI fails."""
    columns = []

    # Find a good key column (id, name, etc.)
    key_column_candidates = ['id', 'ID', 'key', 'KEY', 'name', 'NAME', 'code', 'CODE']
    key_column = None

    for candidate in key_column_candidates:
        if candidate in df.columns:
            key_column = candidate
            break

    # If no key found, use first column
    if key_column is None and len(df.columns) > 0:
        key_column = df.columns[0]

    # Build schema
    for i, col in enumerate(df.columns):
        # Clean column name
        clean_name = col.strip().lower().replace(' ', '_').replace('-', '_')
        clean_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in clean_name)

        # Infer type from pandas dtype
        col_type = infer_kdb_type_from_pandas(df[col].dtype)

        # Determine if this is a key column
        is_key = col == key_column

        columns.append({
            "name": clean_name,
            "type": col_type,
            "description": f"Column for {col}",
            "required": is_key,
            "key": is_key
        })

    return {
        "columns": columns
    }

def infer_kdb_type(value):
    """Infer KDB type from a string value."""
    if not value:
        return "symbol"

    # Check if value is a number
    try:
        float_value = float(value)
        if float_value.is_integer():
            if -32768 <= float_value <= 32767:
                return "short"
            elif -2147483648 <= float_value <= 2147483647:
                return "int"
            else:
                return "long"
        else:
            return "float"
    except ValueError:
        pass

    # Check if value is a boolean
    if value.lower() in ["true", "false", "yes", "no", "y", "n", "t", "f"]:
        return "boolean"

    # Check if value is a date
    import re
    date_pattern = re.compile(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$')
    if date_pattern.match(value):
        return "date"

    # Check if value is a timestamp
    timestamp_pattern = re.compile(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\s\d{1,2}:\d{1,2}(:\d{1,2})?$')
    if timestamp_pattern.match(value):
        return "timestamp"

    # Default to symbol
    return "symbol"

def infer_kdb_type_from_json(value):
    """Infer KDB type from a JSON value."""
    if value is None:
        return "symbol"

    if isinstance(value, bool):
        return "boolean"

    if isinstance(value, int):
        if -32768 <= value <= 32767:
            return "short"
        elif -2147483648 <= value <= 2147483647:
            return "int"
        else:
            return "long"

    if isinstance(value, float):
        return "float"

    if isinstance(value, str):
        # Check if it's a date or timestamp
        import re
        date_pattern = re.compile(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$')
        if date_pattern.match(value):
            return "date"

        timestamp_pattern = re.compile(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\s\d{1,2}:\d{1,2}(:\d{1,2})?$')
        if timestamp_pattern.match(value):
            return "timestamp"

    # Default to symbol for strings and other types
    return "symbol"

def infer_kdb_type_from_pandas(dtype):
    """Infer KDB type from pandas dtype."""
    dtype_str = str(dtype).lower()

    if 'int' in dtype_str:
        if '8' in dtype_str or '16' in dtype_str:
            return "short"
        elif '32' in dtype_str:
            return "int"
        else:
            return "long"

    if 'float' in dtype_str:
        return "float"

    if 'bool' in dtype_str:
        return "boolean"

    if 'datetime' in dtype_str:
        return "timestamp"

    if 'date' in dtype_str:
        return "date"

    if 'time' in dtype_str:
        return "time"

    # Default to symbol for object and other types
    return "symbol"
# app/services/schema_management.py
import json
import os
import asyncpg
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

from app.core.config import settings
from app.core.logging import logger
from app.services.embedding_provider import EmbeddingProvider
from app.core.db import db_pool

class SchemaManager:
    """
    Service for managing schema documents in the database.
    Handles the hierarchical structure of group > schemas > tables.
    """
    
    def __init__(self):
        self.embedding_provider = EmbeddingProvider()
        # self.db_url = settings.DATABASE_URL
    
    async def _get_db_connection(self):
        """Get a database connection from the pool."""
        return await db_pool.get_connection()
    
    async def create_schema_group(self, name: str, description: Optional[str] = None) -> Optional[int]:
        """
        Create a new schema group or get existing one.
        
        Args:
            name: Group name
            description: Optional group description
            
        Returns:
            Group ID if successful, None otherwise
        """
        try:
            conn = await self._get_db_connection()
            try:
                # Use default description if not provided
                if not description:
                    description = f"Group {name}"
                
                group_id = await conn.fetchval(
                    """
                    INSERT INTO schema_groups (name, description)
                    VALUES ($1, $2)
                    ON CONFLICT (name) DO UPDATE 
                    SET updated_at = NOW()
                    RETURNING id
                    """,
                    name, description
                )
                
                return group_id
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error creating schema group: {str(e)}", exc_info=True)
            return None
    
    async def create_schema_definition(self, group_id: int, name: str, 
                                      description: Optional[str] = None) -> Optional[int]:
        """
        Create a new schema definition.
        
        Args:
            group_id: Parent group ID
            name: Schema name
            description: Optional schema description
            
        Returns:
            Schema ID if successful, None otherwise
        """
        try:
            conn = await self._get_db_connection()
            try:
                # Use default description if not provided
                if not description:
                    description = f"Schema {name}"
                
                schema_id = await conn.fetchval(
                    """
                    INSERT INTO schema_definitions (group_id, name, description)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (group_id, name) DO UPDATE 
                    SET description = $3, updated_at = NOW()
                    RETURNING id
                    """,
                    group_id, name, description
                )
                
                return schema_id
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error creating schema definition: {str(e)}", exc_info=True)
            return None
    
    async def create_schema_version(self, schema_id: int, version: Optional[int] = None,
                                   created_by: Optional[int] = None, 
                                   notes: Optional[str] = None,
                                   metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """
        Create a new version of a schema.
        
        Args:
            schema_id: Parent schema ID
            version: Optional version number (auto-incremented if not provided)
            created_by: Optional user ID who created this version
            notes: Optional notes about this version
            metadata: Optional metadata
            
        Returns:
            Version ID if successful, None otherwise
        """
        try:
            conn = await self._get_db_connection()
            try:
                # Determine version number if not provided
                if version is None:
                    latest_version = await conn.fetchval(
                        "SELECT MAX(version) FROM schema_versions WHERE schema_id = $1",
                        schema_id
                    )
                    version = 1 if latest_version is None else latest_version + 1
                
                version_id = await conn.fetchval(
                    """
                    INSERT INTO schema_versions 
                    (schema_id, version, created_by, notes, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                    """,
                    schema_id, version, created_by, notes, json.dumps(metadata or {})
                )
                
                return version_id
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error creating schema version: {str(e)}", exc_info=True)
            return None
    
    async def add_table_definition(self, schema_version_id: int, table_name: str, 
                                  table_content: Dict[str, Any], 
                                  description: Optional[str] = None,
                                  metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """
        Add a table definition to a schema version.
        
        Args:
            schema_version_id: Parent schema version ID
            table_name: Table name
            table_content: Table content including columns
            description: Optional table description
            metadata: Optional metadata
            
        Returns:
            Table ID if successful, None otherwise
        """
        try:
            # Use table description from content if not provided separately
            if not description and isinstance(table_content, dict):
                description = table_content.get("description", f"Table {table_name}")
            
            # Generate embedding for the table
            table_text = f"{table_name} {description or ''}"
            
            # Include important table content in the embedding text
            if isinstance(table_content, dict):
                # Add column information
                if "columns" in table_content:
                    for column in table_content["columns"]:
                        if isinstance(column, dict):
                            col_name = column.get("name", "")
                            col_desc = column.get("column_desc", "")
                            col_type = column.get("type", "")
                            col_kdb_type = column.get("kdb_type", "")
                            table_text += f" {col_name} {col_desc} {col_type} {col_kdb_type}"
            
            # Get embedding
            embedding = await self.embedding_provider.get_embedding(table_text)
                    
            if embedding is None:
                logger.error(f"Failed to generate embedding for table {table_name}")
                return None
            
            conn = await self._get_db_connection()
            try:
                # Format the embedding as a PostgreSQL array string
                # This is the key change: convert Python list to PG array syntax
                if isinstance(embedding, list):
                    embedding_str = '[' + ','.join(str(float(x)) for x in embedding) + ']'
                else:
                    logger.error(f"Invalid embedding format: {type(embedding)}")
                    return None
                
                query = """
                INSERT INTO table_definitions 
                (schema_version_id, name, description, content, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5::vector, $6)
                RETURNING id
                """
                
                # Convert content to JSON string if it's a dict
                content_json = json.dumps(table_content) if isinstance(table_content, dict) else table_content
                metadata_json = json.dumps(metadata or {})
                
                table_id = await conn.fetchval(
                    query, 
                    schema_version_id, 
                    table_name, 
                    description, 
                    content_json,
                    embedding_str,  # Use the JSON-style array with brackets
                    metadata_json
                )
                
                return table_id
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error adding table definition: {str(e)}", exc_info=True)
            return None
    async def add_table_relationship(self, schema_version_id: int, source_table_id: int,
                                    target_table_id: int, relationship_type: str,
                                    join_column: Optional[str] = None,
                                    description: Optional[str] = None,
                                    metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """
        Add a relationship between two tables.
        
        Args:
            schema_version_id: Parent schema version ID
            source_table_id: Source table ID
            target_table_id: Target table ID
            relationship_type: Type of relationship (e.g., "foreign_key")
            join_column: Optional column used for joining
            description: Optional relationship description
            metadata: Optional metadata
            
        Returns:
            Relationship ID if successful, None otherwise
        """
        try:
            conn = await self._get_db_connection()
            try:
                query = """
                INSERT INTO table_relationships 
                (schema_version_id, source_table_id, target_table_id, 
                 relationship_type, join_column, description, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """
                
                relationship_id = await conn.fetchval(
                    query, 
                    schema_version_id, 
                    source_table_id, 
                    target_table_id,
                    relationship_type,
                    join_column,
                    description,
                    json.dumps(metadata or {})
                )
                
                return relationship_id
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error adding table relationship: {str(e)}", exc_info=True)
            return None
    
    async def add_schema_example(self, schema_version_id: int, nl_query: str, 
                                generated_query: str, table_id: Optional[int] = None,
                                description: Optional[str] = None,
                                is_cross_schema: bool = False,
                                metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """
        Add an example query to a schema.
        
        Args:
            schema_version_id: Parent schema version ID
            nl_query: Natural language query
            generated_query: Generated database query
            table_id: Optional table ID this example applies to
            description: Optional example description
            is_cross_schema: Whether this example spans multiple schemas
            metadata: Optional metadata
            
        Returns:
            Example ID if successful, None otherwise
        """
        try:
            conn = await self._get_db_connection()
            try:
                query = """
                INSERT INTO schema_examples 
                (schema_version_id, table_id, natural_language_query, generated_query, 
                 description, is_cross_schema, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """
                
                example_id = await conn.fetchval(
                    query, 
                    schema_version_id, 
                    table_id,
                    nl_query,
                    generated_query,
                    description,
                    is_cross_schema,
                    json.dumps(metadata or {})
                )
                
                return example_id
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error adding schema example: {str(e)}", exc_info=True)
            return None
    
    async def map_example_to_tables(self, example_id: int, table_ids: List[int], 
                                   relevance_scores: Optional[List[float]] = None) -> bool:
        """
        Map an example to multiple tables with optional relevance scores.
        
        Args:
            example_id: Example ID
            table_ids: List of table IDs
            relevance_scores: Optional list of relevance scores (one per table)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not table_ids:
                return False
                
            # Use default relevance score if not provided
            if relevance_scores is None:
                relevance_scores = [1.0] * len(table_ids)
            
            # Ensure same length
            assert len(table_ids) == len(relevance_scores), "table_ids and relevance_scores must have same length"
                
            conn = await self._get_db_connection()
            try:
                async with conn.transaction():
                    for i, table_id in enumerate(table_ids):
                        await conn.execute(
                            """
                            INSERT INTO example_table_mappings (example_id, table_id, relevance_score)
                            VALUES ($1, $2, $3)
                            ON CONFLICT (example_id, table_id) DO UPDATE
                            SET relevance_score = $3
                            """,
                            example_id, table_id, relevance_scores[i]
                        )
                    
                    # Mark example as cross-schema if tables are from different schemas
                    if len(table_ids) > 1:
                        # Check if tables belong to different schemas
                        schema_count = await conn.fetchval("""
                            SELECT COUNT(DISTINCT sv.schema_id)
                            FROM table_definitions td
                            JOIN schema_versions sv ON td.schema_version_id = sv.id
                            WHERE td.id = ANY($1)
                        """, table_ids)
                        
                        if schema_count > 1:
                            await conn.execute(
                                "UPDATE schema_examples SET is_cross_schema = TRUE WHERE id = $1",
                                example_id
                            )
                
                return True
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error mapping example to tables: {str(e)}", exc_info=True)
            return False
    
    async def activate_schema_version(self, schema_id: int, version_id: int) -> bool:
        """
        Activate a specific schema version.
        
        Args:
            schema_id: Schema ID
            version_id: Version ID to activate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = await self._get_db_connection()
            try:
                async with conn.transaction():
                    # Update version status to active
                    await conn.execute(
                        """
                        UPDATE schema_versions 
                        SET status = 'active', activated_at = NOW() 
                        WHERE id = $1
                        """,
                        version_id
                    )
                    
                    # Update any previously active versions to deprecated
                    await conn.execute(
                        """
                        UPDATE schema_versions 
                        SET status = 'deprecated', deprecated_at = NOW()
                        WHERE schema_id = $1 AND status = 'active' AND id != $2
                        """,
                        schema_id, version_id
                    )
                    
                    # Update or insert active schema record
                    await conn.execute(
                        """
                        INSERT INTO active_schemas (schema_id, current_version_id, last_updated)
                        VALUES ($1, $2, NOW())
                        ON CONFLICT (schema_id) DO UPDATE
                        SET current_version_id = $2, last_updated = NOW()
                        """,
                        schema_id, version_id
                    )
                    
                    return True
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error activating schema version: {str(e)}", exc_info=True)
            return False
    
    async def check_schemas_available(self):
        """Check if any schemas are available in the database."""
        try:
            conn = await self._get_db_connection()
            try:
                # Check if any schemas exist
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM schema_definitions"
                )
                return count > 0
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error checking if schemas are available: {str(e)}", exc_info=True)
            return False
    
    async def find_tables_by_vector_search(self, query_text: str, 
                                      group_name: Optional[str] = None,
                                      similarity_threshold: float = 0.65, 
                                      max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find tables similar to the query text using vector search.
        
        Args:
            query_text: Text to search for
            group_name: Optional group name to filter results
            similarity_threshold: Minimum similarity threshold
            max_results: Maximum number of results to return
            
        Returns:
            List of matching tables with similarity scores
        """
        try:
            # Get embedding for query text
            query_embedding = await self.embedding_provider.get_embedding(query_text)
            if not query_embedding:
                logger.error("Failed to generate embedding for query text")
                return []
            
            # Format the embedding as expected by pgvector: "[0.1,0.2,0.3,...]"
            if isinstance(query_embedding, list):
                query_embedding_str = '[' + ','.join(str(float(x)) for x in query_embedding) + ']'
            else:
                logger.error(f"Invalid embedding format: {type(query_embedding)}")
                return []
            
            conn = await self._get_db_connection()
            
            try:
                # Base query
                base_query = """
                SELECT 
                    td.id,
                    td.name AS table_name,
                    td.description,
                    td.content,
                    sd.id AS schema_id,
                    sd.name AS schema_name,
                    sg.id AS group_id,
                    sg.name AS group_name,
                    sv.id AS schema_version_id,
                    sv.version AS schema_version,
                    1 - (td.embedding <=> $1::vector) AS similarity
                FROM 
                    table_definitions td
                JOIN 
                    schema_versions sv ON td.schema_version_id = sv.id
                JOIN 
                    schema_definitions sd ON sv.schema_id = sd.id
                JOIN
                    schema_groups sg ON sd.group_id = sg.id
                JOIN 
                    active_schemas a ON sv.id = a.current_version_id
                WHERE 
                    1 - (td.embedding <=> $1::vector) > $2
                """
                
                # Add group filter if specified
                params = [query_embedding_str, similarity_threshold]  # Use the string representation
                if group_name:
                    base_query += " AND sg.name = $3"
                    params.append(group_name)
                
                # Add order and limit
                base_query += " ORDER BY similarity DESC LIMIT $" + str(len(params) + 1)
                params.append(max_results)
                
                # Execute query
                results = await conn.fetch(base_query, *params)
                
                # Process results
                tables = []
                for row in results:
                    content = json.loads(row["content"]) if isinstance(row["content"], str) else row["content"]
                    tables.append({
                        "id": row["id"],
                        "table_name": row["table_name"],
                        "schema_id": row["schema_id"],
                        "schema_name": row["schema_name"],
                        "group_id": row["group_id"],
                        "group_name": row["group_name"],
                        "schema_version_id": row["schema_version_id"],
                        "schema_version": row["schema_version"],
                        "description": row["description"],
                        "content": content,
                        "similarity": row["similarity"]
                    })
                
                return tables
                    
            finally:
                await conn.close()
                    
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}", exc_info=True)
            return []
      
    async def get_examples_for_tables(self, table_ids: List[int], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get examples relevant to the specified tables.
        
        Args:
            table_ids: List of table IDs
            limit: Maximum number of examples to return
            
        Returns:
            List of examples
        """
        try:
            if not table_ids:
                return []
                
            conn = await self._get_db_connection()
            
            try:
                # First try to get examples directly linked to tables
                query = """
                SELECT 
                    se.id,
                    se.natural_language_query,
                    se.generated_query,
                    se.description
                FROM 
                    schema_examples se
                WHERE 
                    se.table_id = ANY($1)
                
                UNION ALL
                
                SELECT 
                    se.id,
                    se.natural_language_query,
                    se.generated_query,
                    se.description
                FROM 
                    schema_examples se
                JOIN
                    example_table_mappings etm ON se.id = etm.example_id
                WHERE 
                    etm.table_id = ANY($1)
                    AND se.table_id IS NULL
                
                ORDER BY id DESC
                LIMIT $2
                """
                
                results = await conn.fetch(query, table_ids, limit)
                
                examples = []
                for row in results:
                    examples.append({
                        "id": row["id"],
                        "natural_language": row["natural_language_query"],
                        "query": row["generated_query"],
                        "description": row["description"]
                    })
                
                return examples
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error getting examples for tables: {str(e)}", exc_info=True)
            return []
    
    async def process_table_relationships(self, schema_version_id: int, 
                                         tables_by_name: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process relationships between tables based on references.
        
        Args:
            schema_version_id: Schema version ID
            tables_by_name: Dictionary of tables by name with their IDs and content
            
        Returns:
            List of created relationships
        """
        try:
            relationships = []
            
            # For each table
            for table_name, table_data in tables_by_name.items():
                source_table_id = table_data.get("id")
                if not source_table_id:
                    continue
                    
                # For each column in the table
                if "columns" in table_data.get("content", {}):
                    for column in table_data["content"]["columns"]:
                        # Check if column has references
                        references = column.get("references")
                        if references and references != "null":
                            # Parse the reference target
                            target_table = references
                            
                            # Find target table ID
                            if target_table in tables_by_name:
                                target_table_id = tables_by_name[target_table].get("id")
                                
                                if target_table_id:
                                    # Add relationship
                                    relationship_id = await self.add_table_relationship(
                                        schema_version_id=schema_version_id,
                                        source_table_id=source_table_id,
                                        target_table_id=target_table_id,
                                        relationship_type="foreign_key",
                                        join_column=column.get("name"),
                                        description=f"Foreign key from {table_name}.{column.get('name')} to {target_table}"
                                    )
                                    
                                    if relationship_id:
                                        relationships.append({
                                            "id": relationship_id,
                                            "source_table": table_name,
                                            "target_table": target_table,
                                            "join_column": column.get("name")
                                        })
            
            return relationships
                                
        except Exception as e:
            logger.error(f"Error processing table relationships: {str(e)}", exc_info=True)
            return []
    
    async def process_examples(self, schema_version_id: int, examples: List[Dict[str, Any]], 
                              table_ids: Dict[str, int]) -> int:
        """
        Process example queries for a schema.
        
        Args:
            schema_version_id: Schema version ID
            examples: List of example queries
            table_ids: Dictionary of table names to table IDs
            
        Returns:
            Number of examples processed
        """
        try:
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
                    query = generated_query.lower()
                    for table_name, tid in table_ids.items():
                        if table_name.lower() in query:
                            example_tables.append(tid)
                
                # Use the first match as primary table, or None if no matches
                primary_table_id = example_tables[0] if example_tables else None
                
                # Add example
                example_id = await self.add_schema_example(
                    schema_version_id=schema_version_id,
                    nl_query=nl_query,
                    generated_query=generated_query,
                    table_id=primary_table_id,
                    description=example.get("description", "")
                )
                
                if example_id:
                    example_count += 1
                    # If example relates to multiple tables, create mappings
                    if len(example_tables) > 1:
                        await self.map_example_to_tables(
                            example_id=example_id,
                            table_ids=example_tables
                        )
            
            return example_count
            
        except Exception as e:
            logger.error(f"Error processing examples: {str(e)}", exc_info=True)
            return 0
    
    async def import_schema(self, file_path: str, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Import a schema file with group and schemas structure.
        
        Args:
            file_path: Path to the schema JSON file
            user_id: User ID of the importer
            
        Returns:
            Dictionary with import results
        """
        try:
            # Load schema file
            with open(file_path, 'r') as f:
                file_data = json.load(f)
            
            # Extract group information
            group_name = file_data.get("group", "default")
            # Just use group name for description
            group_description = f"Group {group_name}"
            
            # Create or get group
            group_id = await self.create_schema_group(group_name, group_description)
            if not group_id:
                return {
                    "success": False,
                    "error": f"Failed to create group {group_name}"
                }
            
            # Process schemas - note the additional nesting in your structure
            schemas_array = file_data.get("schemas", [])
            imported_schemas = []
            
            for schemas_container in schemas_array:
                # Handle the additional 'schema' array inside each item of 'schemas'
                schema_items = schemas_container.get("schema", [])
                
                for schema_data in schema_items:
                    # Extract schema information
                    schema_name = schema_data.get("name", "unknown")
                    schema_description = schema_data.get("description", f"Schema {schema_name}")
                    
                    # Create schema definition
                    schema_id = await self.create_schema_definition(group_id, schema_name, schema_description)
                    if not schema_id:
                        logger.error(f"Failed to create schema definition for {schema_name}")
                        continue
                    
                    # Create schema version
                    version_id = await self.create_schema_version(
                        schema_id=schema_id,
                        version=1,
                        created_by=user_id,
                        notes=f"Initial import from {os.path.basename(file_path)}"
                    )
                    
                    if not version_id:
                        logger.error(f"Failed to create version for schema {schema_name}")
                        continue
                    
                    # Process tables - in your structure, tables is an array, not an object
                    tables_array = schema_data.get("tables", [])
                    table_ids = {}
                    total_examples = 0
                    
                    for table_obj in tables_array:
                        # Extract table information
                        table_name = table_obj.get("kdb_table_name", "unknown")
                        table_description = table_obj.get("description", f"Table {table_name}")
                        
                        # Add table definition
                        table_id = await self.add_table_definition(
                            schema_version_id=version_id,
                            table_name=table_name,
                            table_content=table_obj,
                            description=table_description
                        )
                        
                        if table_id:
                            table_ids[table_name] = table_id
                        else:
                            logger.warning(f"Failed to add table {table_name}")
                        
                        # Process examples for this table if present (optional)
                        examples = table_obj.get("examples", [])
                        example_count = 0
                        
                        # Process examples at the table level directly
                        if examples:
                            for example in examples:
                                nl_query = example.get("natural_language", "")
                                generated_query = example.get("query", "")
                                
                                if not nl_query or not generated_query:
                                    continue
                                
                                # Add example
                                example_id = await self.add_schema_example(
                                    schema_version_id=version_id,
                                    nl_query=nl_query,
                                    generated_query=generated_query,
                                    table_id=table_id,  # Link directly to this table
                                    description=f"Example for {table_name}"
                                )
                                
                                if example_id:
                                    example_count += 1
                        
                        total_examples += example_count
                        logger.info(f"Imported table {table_name} with {example_count} examples")
                    
                    # Activate the schema version
                    activation_success = await self.activate_schema_version(schema_id, version_id)
                    
                    imported_schemas.append({
                        "schema_id": schema_id,
                        "name": schema_name,
                        "version_id": version_id,
                        "table_count": len(table_ids),
                        "example_count": total_examples,
                        "activated": activation_success
                    })
            
            return {
                "success": True,
                "group_id": group_id,
                "group_name": group_name,
                "schema_count": len(imported_schemas),
                "schemas": imported_schemas
            }
            
        except Exception as e:
            logger.error(f"Error importing schema file: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
      
    async def batch_import_schemas(self, directory_path: str, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Import all schema files from a directory.
        
        Args:
            directory_path: Directory containing schema JSON files
            user_id: User ID of the importer
            
        Returns:
            Dictionary with import results
        """
        try:
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                return {
                    "success": False,
                    "error": f"Directory not found: {directory_path}"
                }
            
            # Get all JSON files in the directory
            schema_files = list(directory.glob("*.json"))
            if not schema_files:
                return {
                    "success": False,
                    "error": f"No JSON files found in {directory_path}"
                }
            
            logger.info(f"Found {len(schema_files)} schema files in {directory_path}")
            
            # Process each file
            results = []
            success_count = 0
            
            for schema_file in schema_files:
                logger.info(f"Processing {schema_file.name}...")
                
                # Import the schema
                result = await self.import_schema(
                    file_path=str(schema_file),
                    user_id=user_id
                )
                
                if result.get("success", False):
                    success_count += 1
                    logger.info(f"Successfully imported {schema_file.name}")
                else:
                    logger.error(f"Failed to import {schema_file.name}: {result.get('error', 'Unknown error')}")
                
                results.append({
                    "file": schema_file.name,
                    "result": result
                })
            
            return {
                "success": success_count > 0,
                "total_files": len(schema_files),
                "success_count": success_count,
                "failure_count": len(schema_files) - success_count,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error in batch import: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
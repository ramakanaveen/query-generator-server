# app/services/schema_editor.py

import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.core.db import db_pool
from app.core.logging import logger
from app.services.embedding_provider import EmbeddingProvider

class SchemaEditorService:
    """
    Service for editing and processing schema JSON.
    """

    def __init__(self):
        self.embedding_provider = EmbeddingProvider()

    async def process_schema_json(self, version_id: int, schema_json: Dict[str, Any]) -> bool:
        """
        Process a schema JSON and update tables and examples.

        Args:
            version_id: The schema version ID
            schema_json: The complete schema JSON

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = await db_pool.get_connection()
            try:
                # Begin transaction
                async with conn.transaction():
                    # Process tables
                    tables = schema_json.get("tables", {})

                    # Get existing tables for this version
                    existing_tables = await conn.fetch(
                        "SELECT id, name FROM table_definitions WHERE schema_version_id = $1",
                        version_id
                    )

                    existing_table_names = {row["name"]: row["id"] for row in existing_tables}

                    # Track processed tables
                    processed_tables = set()

                    # For each table in the schema JSON
                    for table_name, table_data in tables.items():
                        processed_tables.add(table_name)

                        # Get description from table data or set default
                        description = table_data.get("description", f"Table for {table_name}")

                        # Ensure table_data is a dict
                        if not isinstance(table_data, dict):
                            table_data = {"content": table_data}

                        if table_name in existing_table_names:
                            # Update existing table
                            await conn.execute(
                                """
                                UPDATE table_definitions
                                SET description = $1, content = $2, updated_at = $3
                                WHERE id = $4
                                """,
                                description,
                                json.dumps(table_data),
                                datetime.now(),
                                existing_table_names[table_name]
                            )

                            logger.info(f"Updated table: {table_name} (ID: {existing_table_names[table_name]})")
                        else:
                            # Insert new table
                            table_id = await conn.fetchval(
                                """
                                INSERT INTO table_definitions
                                    (schema_version_id, name, description, content, created_at, updated_at)
                                VALUES ($1, $2, $3, $4, $5, $5)
                                    RETURNING id
                                """,
                                version_id,
                                table_name,
                                description,
                                json.dumps(table_data),
                                datetime.now()
                            )

                            logger.info(f"Created new table: {table_name} (ID: {table_id})")

                            # Update the existing table names dict with the new ID
                            existing_table_names[table_name] = table_id

                    # Delete tables that are no longer in the schema
                    tables_to_delete = set(existing_table_names.keys()) - processed_tables

                    for table_name in tables_to_delete:
                        await conn.execute(
                            "DELETE FROM table_definitions WHERE id = $1",
                            existing_table_names[table_name]
                        )

                        logger.info(f"Deleted table: {table_name} (ID: {existing_table_names[table_name]})")

                    # Process examples
                    examples = schema_json.get("examples", [])

                    # Delete existing examples for this version
                    await conn.execute(
                        "DELETE FROM schema_examples WHERE schema_version_id = $1",
                        version_id
                    )

                    # Insert new examples
                    for example in examples:
                        # Get values with default fallbacks
                        nl_query = example.get("natural_language", "")
                        query = example.get("query", "")
                        description = example.get("description", "")

                        # Skip empty examples
                        if not nl_query or not query:
                            continue

                        # Determine which table this example is for
                        table_name = example.get("table", "")
                        table_id = None

                        if table_name and table_name in existing_table_names:
                            table_id = existing_table_names[table_name]
                        else:
                            # Try to infer from the query
                            for name, tid in existing_table_names.items():
                                if name.lower() in query.lower():
                                    table_id = tid
                                    break

                        # Insert example
                        example_id = await conn.fetchval(
                            """
                            INSERT INTO schema_examples
                            (schema_version_id, table_id, natural_language_query, generated_query, description, is_cross_schema)
                            VALUES ($1, $2, $3, $4, $5, $6)
                                RETURNING id
                            """,
                            version_id,
                            table_id,
                            nl_query,
                            query,
                            description,
                            False  # is_cross_schema
                        )

                        logger.info(f"Added example: {nl_query[:30]}... (ID: {example_id})")

                return True
            finally:
                await db_pool.release_connection(conn)
        except Exception as e:
            logger.error(f"Error processing schema JSON: {str(e)}", exc_info=True)
            return False

    async def update_table_embedding(self, table_id: int, table_name: str) -> bool:
        """
        Update the embedding for a table.

        Args:
            table_id: The table ID
            table_name: The name of the table

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = await db_pool.get_connection()
            try:
                # Get table content for embedding context
                table_data = await conn.fetchrow(
                    "SELECT description, content FROM table_definitions WHERE id = $1",
                    table_id
                )

                if not table_data:
                    logger.error(f"Table not found: {table_id}")
                    return False

                # Extract valuable text for embedding
                embedding_text = table_name + " "

                if table_data["description"]:
                    embedding_text += table_data["description"] + " "

                # Parse content if needed
                content = table_data["content"]
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except:
                        content = {}

                # Add columns to embedding text
                if isinstance(content, dict) and "columns" in content:
                    for column in content["columns"]:
                        if isinstance(column, dict):
                            col_name = column.get("name", "")
                            col_type = column.get("type", column.get("kdb_type", ""))
                            col_desc = column.get("description", column.get("column_desc", ""))

                            if col_name:
                                embedding_text += f"{col_name} {col_type} {col_desc} "

                # Generate embedding
                embedding = await self.embedding_provider.get_embedding(embedding_text)

                if not embedding:
                    logger.error(f"Failed to generate embedding for table {table_name}")
                    return False

                # Format embedding for database
                if isinstance(embedding, list):
                    embedding_str = '[' + ','.join(str(float(x)) for x in embedding) + ']'
                else:
                    logger.error(f"Invalid embedding format: {type(embedding)}")
                    return False

                # Update embedding in database
                await conn.execute(
                    """
                    UPDATE table_definitions
                    SET embedding = $1::vector, updated_at = $2
                    WHERE id = $3
                    """,
                    embedding_str,
                    datetime.now(),
                    table_id
                )

                logger.info(f"Updated embedding for table: {table_name} (ID: {table_id})")

                return True
            finally:
                await db_pool.release_connection(conn)
        except Exception as e:
            logger.error(f"Error updating table embedding: {str(e)}", exc_info=True)
            return False
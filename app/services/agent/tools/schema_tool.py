"""
Schema tool — wraps EnhancedSchemaService.
Interface is stable: swapping to GraphRAG later only requires changing the implementation here.
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.enhanced_schema_service import (
    EnhancedSchemaService,
    SchemaRetrievalConfig,
)
from app.core.db import db_pool

logger = logging.getLogger(__name__)

_schema_service: EnhancedSchemaService | None = None


def _get_service() -> EnhancedSchemaService:
    global _schema_service
    if _schema_service is None:
        _schema_service = EnhancedSchemaService()
    return _schema_service


async def search_schema(
    query: str,
    database_type: str,
    limit: int = 5,
    user_id: str | None = None,
    groups: list[str] | None = None,
) -> dict[str, Any]:
    """
    Search for relevant tables/columns for a natural language query.
    Returns a dict suitable for injecting into Claude tool_result.
    Each table entry includes group and schema labels for agent reasoning.
    """
    try:
        config = SchemaRetrievalConfig(max_tables=limit)
        result = await _get_service().retrieve_schema_with_examples(
            query_text=query,
            user_id=user_id,
            config=config,
            groups=groups or None,
        )

        tables = []
        schema = result.schema_structure or {}
        tables_dict = schema.get("tables") or {}
        for table_name, table_info in tables_dict.items():
            if not isinstance(table_info, dict):
                continue
            meta = table_info.get("metadata", {})
            columns = [
                {
                    "name": col.get("name", col) if isinstance(col, dict) else col,
                    "type": col.get("type", col.get("kdb_type", "unknown")) if isinstance(col, dict) else "unknown",
                }
                for col in (table_info.get("columns") or [])
            ]
            tables.append({
                "table": table_name,
                "group": meta.get("group_name", ""),
                "schema": meta.get("schema_name", ""),
                "columns": columns,
            })

        examples = [
            {
                "nl": ex.natural_language,
                "query": ex.generated_query,
                "tables": ex.table_names,
            }
            for ex in (result.examples or [])[:3]
        ]

        return {
            "tables": tables,
            "examples": examples,
            "source": "vector",
            "tables_found": len(tables),
        }

    except Exception as exc:
        logger.warning(f"schema search failed: {exc}")
        return {"tables": [], "examples": [], "source": "vector", "error": str(exc)}


async def list_schema_groups(database_type: str) -> dict[str, Any]:
    """
    Return all active schema groups with descriptions and table counts.
    Used by the agent to understand available business domains and route
    ambiguous queries without requiring user-provided directives.
    """
    try:
        conn = await db_pool.get_connection()
        try:
            rows = await conn.fetch("""
                SELECT
                    sg.name,
                    sg.description,
                    COUNT(DISTINCT td.id) AS table_count
                FROM schema_groups sg
                JOIN schema_definitions sd ON sd.group_id = sg.id
                JOIN schema_versions sv ON sv.schema_id = sd.id
                JOIN active_schemas a ON a.current_version_id = sv.id
                JOIN table_definitions td ON td.schema_version_id = sv.id
                GROUP BY sg.name, sg.description
                ORDER BY sg.name
            """)
            groups = [
                {
                    "name": row["name"],
                    "description": row["description"] or "",
                    "table_count": row["table_count"],
                }
                for row in rows
            ]
            return {"groups": groups, "count": len(groups)}
        finally:
            await db_pool.release_connection(conn)
    except Exception as exc:
        logger.warning(f"list_schema_groups failed: {exc}")
        return {"groups": [], "count": 0, "error": str(exc)}


async def get_table_details(
    table_name: str,
    database_type: str,
) -> dict[str, Any]:
    """
    Get full column definitions and examples for a specific table.
    """
    try:
        result = await _get_service().retrieve_schema_with_examples(
            query_text=f"details for table {table_name}",
            entities=[table_name],
            config=SchemaRetrievalConfig(max_tables=1),
        )

        schema = result.schema_structure or {}
        tables_dict = schema.get("tables") or {}
        for tname, table_info in tables_dict.items():
            if not isinstance(table_info, dict):
                continue
            if tname.lower() == table_name.lower():
                meta = table_info.get("metadata", {})
                return {
                    "table": tname,
                    "group": meta.get("group_name", ""),
                    "schema": meta.get("schema_name", ""),
                    "columns": table_info.get("columns", []),
                    "description": table_info.get("description", ""),
                    "examples": [
                        {"nl": ex.natural_language, "query": ex.generated_query}
                        for ex in (result.examples or [])[:5]
                        if table_name.lower() in [t.lower() for t in ex.table_names]
                    ],
                }

        return {"table": table_name, "columns": [], "error": "Table not found in schema"}

    except Exception as exc:
        logger.warning(f"get_table_details failed for {table_name}: {exc}")
        return {"table": table_name, "columns": [], "error": str(exc)}

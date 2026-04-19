"""
Schema tool — wraps EnhancedSchemaService.
Interface is stable: swapping to GraphRAG later only requires changing the implementation here.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.services.enhanced_schema_service import (
    EnhancedSchemaService,
    SchemaRetrievalConfig,
)

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
) -> dict[str, Any]:
    """
    Search for relevant tables/columns for a natural language query.
    Returns a dict suitable for injecting into Claude tool_result.
    """
    try:
        config = SchemaRetrievalConfig(max_tables=limit)
        result = await _get_service().retrieve_schema_with_examples(
            query_text=query,
            user_id=user_id,
            config=config,
        )

        tables = []
        schema = result.schema_structure or {}
        # schema_structure shape: {"description": str, "tables": {table_name: table_info}}
        tables_dict = schema.get("tables") or {}
        for table_name, table_info in tables_dict.items():
            if not isinstance(table_info, dict):
                continue
            columns = [
                {
                    "name": col.get("name", col) if isinstance(col, dict) else col,
                    "type": col.get("type", "unknown") if isinstance(col, dict) else "unknown",
                }
                for col in (table_info.get("columns") or [])
            ]
            tables.append({"table": table_name, "columns": columns})

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
                return {
                    "table": tname,
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

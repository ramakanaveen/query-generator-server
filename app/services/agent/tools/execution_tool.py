"""
Execution tool — wraps the existing connector factory (KDB + Starburst).
Stores results so the download endpoint can retrieve them.
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.connectors.connector_factory import get_connector

logger = logging.getLogger(__name__)

# In-memory result store keyed by execution_id.
# Simple for now; can be backed by Redis or DB later.
_result_store: dict[str, dict] = {}


async def execute_query(
    query: str,
    database_type: str,
    execution_id: str,
    page: int = 0,
    page_size: int = 100,
    query_complexity: str = "MULTI_LINE",
) -> dict[str, Any]:
    """
    Execute a query and return paginated results.
    Results are also stored internally for the download endpoint.
    """
    try:
        connector = get_connector(database_type=database_type)

        kwargs: dict[str, Any] = {}
        if database_type.lower() == "kdb":
            kwargs["query_complexity"] = query_complexity

        results, metadata, total_count = await connector.execute_paginated(
            query=query,
            page=page,
            page_size=page_size,
            params={},
            **kwargs,
        )

        import math
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1

        payload = {
            "rows": results,
            "metadata": metadata,
            "pagination": {
                "currentPage": page + 1,
                "totalPages": total_pages,
                "totalRows": total_count,
                "pageSize": page_size,
                "returnedRows": len(results),
            },
            "execution_id": execution_id,
        }

        # Store for download (keep all rows, not just this page)
        if execution_id not in _result_store:
            _result_store[execution_id] = {
                "query": query,
                "database_type": database_type,
                "rows": results,
                "metadata": metadata,
                "total_count": total_count,
            }

        return payload

    except Exception as exc:
        logger.error(f"execution_tool.execute_query failed: {exc}")
        return {"error": str(exc), "rows": [], "execution_id": execution_id}


def get_stored_result(execution_id: str) -> dict | None:
    return _result_store.get(execution_id)

"""
Haiku-based description generator for schema groups and schemas.
Generates short 1-sentence summaries that are stored in:
  schema_groups.description
  schema_definitions.description

Called:
  - One-time via POST /api/admin/schema-groups/generate-descriptions
  - After every schema import (at end of schema_management.py import flow)
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.core.db import db_pool
from app.core.config import settings

logger = logging.getLogger(__name__)


async def _haiku_describe(prompt: str) -> str:
    """Single Haiku call returning a plain-text 1-sentence description."""
    from app.services.agent.llm_client import get_llm_client
    client = get_llm_client()
    model = settings.CLAUDE_FAST_MODEL
    response = await client.create(
        model=model,
        system="You write concise 1-sentence descriptions of financial database schema objects. Output only the description — no preamble, no quotes.",
        messages=[{"role": "user", "content": prompt}],
        tools=[],
        max_tokens=120,
    )
    for block in response.content:
        if hasattr(block, "text"):
            return block.text.strip()
    return ""


async def generate_group_description(group_id: int, group_name: str, tables: list[str]) -> str:
    """
    Generate and persist a 1-sentence description for a schema group.
    Returns the generated description.
    """
    table_sample = ", ".join(tables[:15]) if tables else "(no tables)"
    prompt = (
        f"Schema group name: {group_name}\n"
        f"Tables in this group (sample): {table_sample}\n\n"
        "Write a 1-sentence description of what business domain or data this group covers."
    )
    description = await _haiku_describe(prompt)
    if not description:
        return ""

    try:
        conn = await db_pool.get_connection()
        try:
            await conn.execute(
                "UPDATE schema_groups SET description = $1 WHERE id = $2",
                description, group_id,
            )
        finally:
            await db_pool.release_connection(conn)
    except Exception as exc:
        logger.warning(f"Failed to persist group description for {group_name}: {exc}")

    return description


async def generate_schema_description(schema_id: int, schema_name: str, tables: list[str]) -> str:
    """
    Generate and persist a 1-sentence description for a schema (KDB namespace).
    Returns the generated description.
    """
    table_sample = ", ".join(tables[:15]) if tables else "(no tables)"
    prompt = (
        f"KDB namespace (schema) name: {schema_name}\n"
        f"Tables in this namespace (sample): {table_sample}\n\n"
        "Write a 1-sentence description of what this KDB namespace stores."
    )
    description = await _haiku_describe(prompt)
    if not description:
        return ""

    try:
        conn = await db_pool.get_connection()
        try:
            await conn.execute(
                "UPDATE schema_definitions SET description = $1 WHERE id = $2",
                description, schema_id,
            )
        finally:
            await db_pool.release_connection(conn)
    except Exception as exc:
        logger.warning(f"Failed to persist schema description for {schema_name}: {exc}")

    return description


async def generate_all_descriptions(overwrite: bool = False) -> dict[str, Any]:
    """
    Generate descriptions for all groups and schemas that currently have empty descriptions.
    Set overwrite=True to regenerate even existing ones.

    Returns a summary dict with counts.
    """
    conn = await db_pool.get_connection()
    try:
        # Fetch groups
        filter_clause = "" if overwrite else "WHERE sg.description IS NULL OR sg.description = ''"
        group_rows = await conn.fetch(f"""
            SELECT sg.id, sg.name,
                   ARRAY_AGG(DISTINCT td.name ORDER BY td.name) AS table_names
            FROM schema_groups sg
            LEFT JOIN schema_definitions sd ON sd.group_id = sg.id
            LEFT JOIN schema_versions sv ON sv.schema_id = sd.id
            LEFT JOIN active_schemas a ON a.current_version_id = sv.id
            LEFT JOIN table_definitions td ON td.schema_version_id = sv.id
            {filter_clause}
            GROUP BY sg.id, sg.name
        """)

        # Fetch schemas
        schema_filter = "" if overwrite else "WHERE sd.description IS NULL OR sd.description = ''"
        schema_rows = await conn.fetch(f"""
            SELECT sd.id, sd.name,
                   ARRAY_AGG(DISTINCT td.name ORDER BY td.name) AS table_names
            FROM schema_definitions sd
            LEFT JOIN schema_versions sv ON sv.schema_id = sd.id
            LEFT JOIN active_schemas a ON a.current_version_id = sv.id
            LEFT JOIN table_definitions td ON td.schema_version_id = sv.id
            {schema_filter}
            GROUP BY sd.id, sd.name
        """)
    finally:
        await db_pool.release_connection(conn)

    groups_updated = 0
    schemas_updated = 0
    errors = []

    for row in group_rows:
        tables = [t for t in (row["table_names"] or []) if t]
        try:
            desc = await generate_group_description(row["id"], row["name"], tables)
            if desc:
                groups_updated += 1
        except Exception as exc:
            errors.append(f"group {row['name']}: {exc}")
            logger.warning(f"generate_group_description failed for {row['name']}: {exc}")

    for row in schema_rows:
        tables = [t for t in (row["table_names"] or []) if t]
        try:
            desc = await generate_schema_description(row["id"], row["name"], tables)
            if desc:
                schemas_updated += 1
        except Exception as exc:
            errors.append(f"schema {row['name']}: {exc}")
            logger.warning(f"generate_schema_description failed for {row['name']}: {exc}")

    return {
        "groups_updated": groups_updated,
        "schemas_updated": schemas_updated,
        "errors": errors,
    }

"""
System prompt builder for the v2 agent.
Injects DB-specific syntax rules, retrieved memories, and conversation context.
"""
from __future__ import annotations

from app.services.query_generation.prompts.shared_constants import (
    KDB_SYNTAX_CORRECTIONS,
    KDB_NOTES_SINGLE,
    KDB_NOTES_MULTI,
)

_SQL_NOTES = """
### Guidelines for Starburst/Trino SQL Query Generation:

1. Use standard ANSI SQL syntax — Trino is largely ANSI-compatible.
2. Table references: use catalog.schema.table format when fully qualified.
3. Date/time: use `CURRENT_DATE`, `CURRENT_TIMESTAMP`, `DATE_ADD`, `DATE_DIFF`.
4. String functions: `LOWER()`, `UPPER()`, `REGEXP_LIKE()`, `SPLIT_PART()`.
5. Aggregations: `SUM`, `AVG`, `MIN`, `MAX`, `COUNT`, `APPROX_DISTINCT`.
6. Window functions: `ROW_NUMBER()`, `RANK()`, `LEAD()`, `LAG()` with `OVER (PARTITION BY ... ORDER BY ...)`.
7. Pagination: use `LIMIT N OFFSET M`.
8. Arrays: `UNNEST()`, `ARRAY_AGG()`, `CARDINALITY()`.
9. JSON: `JSON_EXTRACT_SCALAR()`, `JSON_ARRAY_LENGTH()`.
10. Avoid MySQL-specific syntax (`IFNULL` → use `COALESCE`, backtick identifiers → use double quotes).
"""

_BASE_SYSTEM = """You are an expert database query generation assistant with deep knowledge of time-series and financial data.

Your job is to help users generate accurate database queries, understand schema, execute queries, and analyse results.

## Capabilities
- Generate KDB+/q queries and Starburst/Trino SQL queries
- Answer schema and table description questions
- Execute queries and return results
- Analyse result data for patterns, anomalies, and insights
- Ask for clarification when a request is genuinely ambiguous

## Output Format
When you have generated a query, wrap your final answer in <result> tags:
<result>
{{
  "generated_query": "<the exact query>",
  "complexity": "SINGLE_LINE" or "MULTI_LINE",
  "confidence": 0.0-1.0,
  "explanation": "<one sentence explaining what the query does>"
}}
</result>

For schema descriptions, explanations, or analysis — respond in plain text. No <result> tags needed.

## General Rules
- Always call search_schema or get_table_details before generating a query — do not guess table/column names.
- If the request is ambiguous and context does not resolve it, call clarify() with a precise question.
- Do not call clarify() for things you can infer from schema or reasonable defaults (e.g., assume today's date, assume the configured database type).
- Prefer the simplest query that correctly answers the question.

## Schema Groups and Namespace Routing
- Schema groups are business domain labels (e.g. stirt, spot, titan). Each group contains one or more schemas (KDB namespaces).
- Table names are namespace-qualified: `.namespace.tablename` (e.g. `.stirtimtm.bookedtrades`, `.spot.trade`). Always use the full `.namespace.table` form in generated queries — never bare table names.
- User directives like `@stirt` are optional routing hints. If present, pass them as the `groups` filter in search_schema. If absent, search without a filter or call list_schema_groups first when the business domain is genuinely unclear.
- For ambiguous queries (e.g. "my trades" — could be STIRT or SPOT), call list_schema_groups to survey available domains before calling search_schema.
- Multi-group queries (e.g. Titan trades that span SPOT and STIRT data) are valid — search each relevant group separately and join/union in the generated query.
"""


def build(
    database_type: str,
    memories: list[dict] | None = None,
    conversation_history: list[dict] | None = None,
) -> str:
    parts = [_BASE_SYSTEM]

    # DB-specific syntax rules
    if database_type == "kdb":
        parts.append("## Database: KDB+/q")
        parts.append(KDB_NOTES_MULTI)
        parts.append(KDB_SYNTAX_CORRECTIONS)
    else:
        parts.append(f"## Database: {database_type.upper()} (Starburst/Trino SQL)")
        parts.append(_SQL_NOTES)

    # Inject relevant memories
    if memories:
        lines = ["## Relevant Past Learnings (from memory)"]
        for m in memories:
            line = f"- [{m['type']}] {m['learning']}"
            if m.get("corrected_version"):
                line += f" → Use: `{m['corrected_version']}`"
            lines.append(line)
        parts.append("\n".join(lines))

    # Inject conversation context
    if conversation_history:
        lines = ["## Recent Conversation Context"]
        for msg in conversation_history[-5:]:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            if isinstance(content, str):
                lines.append(f"{role}: {content[:300]}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)

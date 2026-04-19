"""
Post-execution analyst — single Claude call that produces:
  summary, anomalies, suggested_followups, chart_hint
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.services.agent.llm_client import get_llm_client
from app.core.config import settings

logger = logging.getLogger(__name__)

_SYSTEM = """You are a data analyst. Given a database query and its results, produce a concise analysis.

Respond with valid JSON only — no markdown, no explanation outside the JSON:
{
  "summary": "<2-3 sentence plain-English narrative of what the data shows>",
  "anomalies": ["<anomaly 1>", "<anomaly 2>"],
  "suggested_followups": ["<followup question 1>", "<followup question 2>", "<followup question 3>"],
  "chart_hint": {
    "type": "bar" | "line" | "scatter" | "pie" | "table",
    "x": "<column name for x-axis>",
    "y": "<column name for y-axis>",
    "color_by": "<column name or null>"
  }
}

Rules:
- anomalies: list only genuinely unusual patterns; empty list if none
- suggested_followups: exactly 3 natural follow-up questions the user might ask next
- chart_hint: pick the most appropriate chart type for this data shape
- Keep summary factual and concise"""


async def analyze(
    nl_query: str,
    generated_query: str,
    results: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Run post-execution analysis. Returns analysis dict (never raises)."""
    if not results:
        return {
            "summary": "Query returned no results.",
            "anomalies": [],
            "suggested_followups": [],
            "chart_hint": {"type": "table", "x": None, "y": None, "color_by": None},
        }

    try:
        llm = get_llm_client()
        sample = results[:50]  # cap prompt size
        col_names = list(sample[0].keys()) if sample else []

        user_content = (
            f"User asked: {nl_query}\n\n"
            f"Query executed: {generated_query}\n\n"
            f"Columns: {', '.join(col_names)}\n"
            f"Total rows: {metadata.get('row_count', len(results))}\n\n"
            f"Sample data (up to 50 rows):\n{json.dumps(sample, default=str)}"
        )

        response = await llm.create(
            model=settings.CLAUDE_FAST_MODEL,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            tools=[],
            max_tokens=1024,
        )

        text = "".join(
            b.text for b in response.content if hasattr(b, "type") and b.type == "text"
        )
        return _parse_analysis(text)

    except Exception as exc:
        logger.warning(f"analyst.analyze failed (non-fatal): {exc}")
        return {
            "summary": "Analysis unavailable.",
            "anomalies": [],
            "suggested_followups": [],
            "chart_hint": {"type": "table", "x": None, "y": None, "color_by": None},
        }


def _parse_analysis(text: str) -> dict:
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {
        "summary": text[:500] if text else "Analysis unavailable.",
        "anomalies": [],
        "suggested_followups": [],
        "chart_hint": {"type": "table", "x": None, "y": None, "color_by": None},
    }

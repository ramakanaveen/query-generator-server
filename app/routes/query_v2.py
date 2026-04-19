"""
v2 Query API — Claude tool-use agent with SSE streaming.
  POST /api/v2/query        → SSE stream (Accept: text/event-stream) or JSON fallback
  GET  /api/v2/download/{id} → CSV / JSON / Excel download
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import time
import uuid
from typing import Optional, AsyncGenerator

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from app.services.agent.query_agent import QueryAgent, AgentRequest
from app.services.agent.tools.execution_tool import get_stored_result

logger = logging.getLogger(__name__)
router = APIRouter()
_agent = QueryAgent()


# ── Request / Response models ─────────────────────────────────────────────────

class QueryV2Request(BaseModel):
    query: str
    model: Optional[str] = None
    database_type: str = "kdb"
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    response_mode: str = "query"
    auto_execute: bool = False
    analysis: bool = False
    extended_thinking: bool = False
    clarification_id: Optional[str] = None
    clarification_answer: Optional[str] = None

    class Config:
        extra = "allow"


class QueryV2Response(BaseModel):
    execution_id: str
    generated_query: Optional[str] = None
    generated_content: Optional[str] = None
    response_type: str = "query"
    complexity: str = "SINGLE_LINE"
    confidence: float = 0.0
    explanation: str = ""
    thinking: list[str] = Field(default_factory=list)
    execution_result: Optional[dict] = None
    analysis_result: Optional[dict] = None
    clarification: Optional[dict] = None
    error: Optional[str] = None
    duration_ms: int = 0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/query")
async def generate_query_v2(
    request: QueryV2Request,
    accept: str = Header(default="application/json"),
):
    """
    Generate (and optionally execute + analyse) a database query via the Claude agent.
    Send Accept: text/event-stream for real-time SSE streaming.
    """
    execution_id = str(uuid.uuid4())
    agent_request = AgentRequest(
        query=request.query,
        model=request.model or "",
        database_type=request.database_type,
        conversation_id=request.conversation_id,
        user_id=request.user_id,
        response_mode=request.response_mode,
        auto_execute=request.auto_execute,
        analysis=request.analysis,
        extended_thinking=request.extended_thinking,
        clarification_id=request.clarification_id,
        clarification_answer=request.clarification_answer,
        execution_id=execution_id,
    )

    if "text/event-stream" in accept:
        return _sse_response(agent_request)

    # Blocking JSON fallback
    t0 = time.perf_counter()
    result = await _agent.run(agent_request)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(f"[v2] POST /query completed in {elapsed_ms}ms | model={agent_request.model} | query={agent_request.query[:60]!r}")

    if result.error:
        raise HTTPException(status_code=500, detail=result.error)

    return QueryV2Response(
        execution_id=result.execution_id,
        generated_query=result.generated_query,
        generated_content=result.generated_content,
        response_type=result.response_type,
        complexity=result.complexity,
        confidence=result.confidence,
        explanation=result.explanation,
        thinking=result.thinking,
        execution_result=result.execution_result,
        analysis_result=result.analysis_result,
        clarification=result.clarification,
        error=result.error,
        duration_ms=result.duration_ms,
    )


def _sse_response(agent_request: AgentRequest) -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue()

    async def producer() -> None:
        try:
            await _agent.run_streaming(agent_request, queue)
        except Exception as exc:
            logger.exception(f"SSE producer error: {exc}")
            from app.services.agent.streaming import SSEEvent, EventType
            await queue.put(SSEEvent(EventType.ERROR, {"message": str(exc)}))
            await queue.put(None)

    async def generator() -> AsyncGenerator[bytes, None]:
        task = asyncio.create_task(producer())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event.to_sse_bytes()
        finally:
            task.cancel()

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


@router.get("/download/{execution_id}")
async def download_results(
    execution_id: str,
    format: str = "csv",
):
    """
    Download execution results as CSV, JSON, or Excel.
    execution_id comes from the query_generated or results SSE event.
    """
    stored = get_stored_result(execution_id)
    if not stored:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for execution_id={execution_id}. "
                   "Results may have expired or the query was not executed.",
        )

    rows: list[dict] = stored.get("rows", [])
    filename_base = f"query_results_{execution_id[:8]}"

    if format == "json":
        content = json.dumps(rows, default=str, indent=2).encode()
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.json"'},
        )

    if format == "excel":
        try:
            import openpyxl
            wb = openpyxl.Workbook()
            ws = wb.active
            if rows:
                ws.append(list(rows[0].keys()))
                for row in rows:
                    ws.append([str(v) if v is not None else "" for v in row.values()])
            buf = io.BytesIO()
            wb.save(buf)
            buf.seek(0)
            return StreamingResponse(
                iter([buf.read()]),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f'attachment; filename="{filename_base}.xlsx"'},
            )
        except ImportError:
            raise HTTPException(status_code=501, detail="openpyxl not installed; use csv or json format")

    # Default: CSV
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    content = buf.getvalue().encode()
    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename_base}.csv"'},
    )


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "v2"}

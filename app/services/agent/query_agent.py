"""
Core v2 Query Agent — Claude tool-use orchestrator.
Supports both blocking (JSON) and streaming (SSE via asyncio.Queue) modes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.services.agent.llm_client import get_llm_client, LLMClient
from app.services.agent import system_prompt as sp
from app.services.agent.streaming import SSEEvent, EventType
from app.services.agent.rule_checker import sanity_check
from app.services.agent.tools import schema_tool, memory_tool
from app.services.agent.tools import execution_tool, clarification_tool
from app.services.agent import analyst

logger = logging.getLogger(__name__)

TOOLS: list[dict] = [
    {
        "name": "search_schema",
        "description": (
            "Search for relevant database tables and columns given a natural language description. "
            "Always call this before generating any query to ground yourself in the actual schema."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "database_type": {"type": "string", "enum": ["kdb", "starburst", "postgres"]},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query", "database_type"],
        },
    },
    {
        "name": "get_table_details",
        "description": "Get full column definitions, data types, and example queries for a specific table.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string"},
                "database_type": {"type": "string"},
            },
            "required": ["table_name", "database_type"],
        },
    },
    {
        "name": "recall_memory",
        "description": (
            "Retrieve relevant past learnings and successful query patterns. "
            "Call this early to benefit from previous corrections."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "context": {"type": "string"},
                "memory_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "syntax_correction", "user_definition", "approach_recommendation",
                            "query_pattern", "error_correction", "schema_clarification",
                        ],
                    },
                },
            },
            "required": ["context"],
        },
    },
    {
        "name": "execute_query",
        "description": "Execute a generated query and return results. Only call when auto_execute is enabled or explicitly requested.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "database_type": {"type": "string"},
                "page": {"type": "integer", "default": 0},
                "page_size": {"type": "integer", "default": 100},
            },
            "required": ["query", "database_type"],
        },
    },
    {
        "name": "clarify",
        "description": (
            "Ask the user a clarification question when genuinely ambiguous. "
            "Only use when you cannot proceed without more information."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question"],
        },
    },
]


@dataclass
class AgentRequest:
    query: str
    model: str = ""
    database_type: str = "kdb"
    conversation_id: str | None = None
    user_id: str | None = None
    response_mode: str = "query"
    auto_execute: bool = False
    analysis: bool = False
    extended_thinking: bool = False
    clarification_id: str | None = None
    clarification_answer: str | None = None
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if not self.model:
            self.model = settings.CLAUDE_DEFAULT_MODEL


@dataclass
class AgentResult:
    execution_id: str
    generated_query: str | None = None
    generated_content: str | None = None
    response_type: str = "query"
    thinking: list[str] = field(default_factory=list)
    complexity: str = "SINGLE_LINE"
    confidence: float = 0.0
    explanation: str = ""
    execution_result: dict | None = None
    analysis_result: dict | None = None
    paused: bool = False
    clarification: dict | None = None
    error: str | None = None
    events: list[dict] = field(default_factory=list)
    duration_ms: int = 0


# Models that are too small for reliable query generation — upgrade to Sonnet
_WEAK_MODELS = {"claude-haiku", "claude-haiku-4-5", "claude-haiku-4-5-20251001"}
_SONNET_FALLBACK = "claude-sonnet"

_DB_LABELS = {
    "kdb": "KDB+/q",
    "starburst": "Starburst/Trino SQL",
    "postgres": "PostgreSQL SQL",
}


def _build_user_message(request: AgentRequest) -> str:
    """Prepend a hard database-type instruction so weaker models can't miss it."""
    db_label = _DB_LABELS.get(request.database_type.lower(), request.database_type.upper())
    prefix = f"[DATABASE: {db_label} — generate {db_label} syntax ONLY, never SQL for other databases]\n\n"
    return prefix + request.query


def _effective_model(request: AgentRequest) -> str:
    """Upgrade Haiku to Sonnet for query generation — Haiku makes syntax errors."""
    if request.model.lower() in _WEAK_MODELS:
        logger.info(f"Upgrading model {request.model!r} → {_SONNET_FALLBACK!r} for query generation")
        return _SONNET_FALLBACK
    return request.model


class QueryAgent:
    def __init__(self) -> None:
        self._llm: LLMClient | None = None

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = get_llm_client()
        return self._llm

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(self, request: AgentRequest) -> AgentResult:
        """Blocking mode: collects all events internally, returns completed AgentResult."""
        queue: asyncio.Queue[SSEEvent | None] = asyncio.Queue()
        result_holder: list[AgentResult] = []

        async def _collect(r: AgentResult) -> None:
            result_holder.append(r)

        await self._run_core(request, queue, _collect)
        return result_holder[0] if result_holder else AgentResult(
            execution_id=request.execution_id, error="Agent returned no result"
        )

    async def run_streaming(
        self,
        request: AgentRequest,
        event_queue: asyncio.Queue,
    ) -> None:
        """
        Streaming mode: puts SSEEvent objects into event_queue.
        Puts None as sentinel when done.
        """
        async def _finish(r: AgentResult) -> None:
            # Emit final structured events
            if r.paused and r.clarification:
                await event_queue.put(SSEEvent(EventType.CLARIFICATION_NEEDED, r.clarification))
            elif r.error:
                await event_queue.put(SSEEvent(EventType.ERROR, {"message": r.error}))
            else:
                if r.generated_query:
                    await event_queue.put(SSEEvent(EventType.QUERY_GENERATED, {
                        "query": r.generated_query,
                        "complexity": r.complexity,
                        "confidence": r.confidence,
                        "explanation": r.explanation,
                    }))
                elif r.generated_content:
                    await event_queue.put(SSEEvent(EventType.SCHEMA_INFO, {"text": r.generated_content}))

                if r.execution_result:
                    await event_queue.put(SSEEvent(EventType.RESULTS, r.execution_result))

                if r.analysis_result:
                    await event_queue.put(SSEEvent(EventType.ANALYSIS, r.analysis_result))

            await event_queue.put(SSEEvent(EventType.DONE, {"execution_id": r.execution_id, "duration_ms": r.duration_ms}))
            await event_queue.put(None)  # sentinel

        await self._run_core(request, event_queue, _finish)

    # ── Core implementation ────────────────────────────────────────────────────

    async def _run_core(
        self,
        request: AgentRequest,
        event_queue: asyncio.Queue,
        finish_cb,
    ) -> None:
        result = AgentResult(execution_id=request.execution_id)
        _t0 = time.perf_counter()

        try:
            # Resume from clarification if provided
            if request.clarification_id and request.clarification_answer:
                messages, system = await self._resume_from_clarification(request)
            else:
                memories = await memory_tool.recall(request.query, user_id=request.user_id)
                history = await self._load_conversation(request.conversation_id)
                system = sp.build(request.database_type, memories=memories, conversation_history=history)
                messages: list[dict] = [{"role": "user", "content": _build_user_message(request)}]

            schema_tables: list[dict] = []
            thinking_chunks: list[str] = []

            for attempt in range(2):
                messages, thinking_chunks, schema_tables, clarification = await self._agent_loop(
                    request, system, messages, event_queue
                )

                if clarification:
                    cid = clarification_tool.save_context(messages, {
                        "model": request.model,
                        "database_type": request.database_type,
                        "user_id": request.user_id,
                    })
                    clarification["clarification_id"] = cid
                    result.paused = True
                    result.clarification = clarification
                    break

                final_text = self._extract_last_text(messages)
                parsed = self._parse_result(final_text)

                if not parsed.get("generated_query"):
                    result.generated_content = final_text
                    result.response_type = "content"
                    break

                check = sanity_check(parsed["generated_query"], schema_tables, request.database_type)
                if check.ok or attempt == 1:
                    result.generated_query = parsed["generated_query"]
                    result.complexity = parsed.get("complexity", "SINGLE_LINE")
                    result.confidence = float(parsed.get("confidence", 0.8))
                    result.explanation = parsed.get("explanation", "")
                    break

                issue_text = "; ".join(check.issues)
                messages.append({"role": "user", "content": (
                    f"Your query has issues: {issue_text}. "
                    "Please fix and return the corrected query in <result> tags."
                )})

            result.thinking = thinking_chunks

            # Execute if requested
            if result.generated_query and request.auto_execute and not result.paused:
                await event_queue.put(SSEEvent(EventType.EXECUTING, {"status": "running", "execution_id": request.execution_id}))
                exec_result = await execution_tool.execute_query(
                    query=result.generated_query,
                    database_type=request.database_type,
                    execution_id=request.execution_id,
                    query_complexity=result.complexity,
                )
                result.execution_result = exec_result

                # Analyse if requested
                if request.analysis and not exec_result.get("error"):
                    analysis = await analyst.analyze(
                        nl_query=request.query,
                        generated_query=result.generated_query,
                        results=exec_result.get("rows", []),
                        metadata=exec_result.get("metadata", {}),
                    )
                    result.analysis_result = analysis

            # Store successful pattern to memory
            if result.generated_query and not result.paused:
                await memory_tool.store_pattern(
                    original_query=request.query,
                    generated_query=result.generated_query,
                    user_id=request.user_id,
                )

        except Exception as exc:
            logger.exception(f"QueryAgent._run_core failed: {exc}")
            result.error = str(exc)

        elapsed_ms = int((time.perf_counter() - _t0) * 1000)
        result.duration_ms = elapsed_ms
        logger.info(
            f"[v2] agent done in {elapsed_ms}ms | "
            f"model={request.model} | "
            f"query={request.query[:60]!r} | "
            f"has_query={bool(result.generated_query)}"
        )
        await finish_cb(result)

    async def _agent_loop(
        self,
        request: AgentRequest,
        system: str,
        messages: list[dict],
        event_queue: asyncio.Queue,
    ) -> tuple[list[dict], list[str], list[dict], dict | None]:
        thinking_chunks: list[str] = []
        schema_tables: list[dict] = []

        thinking_param = (
            {"type": "enabled", "budget_tokens": settings.CLAUDE_THINKING_BUDGET}
            if request.extended_thinking else None
        )

        while True:
            # ── Streaming call ──────────────────────────────────────────────
            text_buffer = ""
            tool_use_blocks: list[dict] = []
            _current_tool: dict | None = None

            async with self.llm.stream(
                model=_effective_model(request),
                system=system,
                messages=messages,
                tools=TOOLS,
                max_tokens=settings.CLAUDE_V2_MAX_TOKENS,
                thinking=thinking_param,
            ) as stream:
                async for event in stream:
                    etype = getattr(event, "type", None)

                    if etype == "content_block_start":
                        cb = event.content_block
                        if cb.type == "tool_use":
                            _current_tool = {"id": cb.id, "name": cb.name, "input_raw": ""}

                    elif etype == "content_block_delta":
                        delta = event.delta
                        dtype = getattr(delta, "type", None)
                        if dtype == "thinking_delta":
                            chunk = delta.thinking
                            thinking_chunks.append(chunk)
                            await event_queue.put(SSEEvent(EventType.THINKING, {"text": chunk}))
                        elif dtype == "text_delta":
                            text_buffer += delta.text
                        elif dtype == "input_json_delta" and _current_tool is not None:
                            _current_tool["input_raw"] += delta.partial_json

                    elif etype == "content_block_stop":
                        if _current_tool is not None:
                            try:
                                _current_tool["input"] = json.loads(_current_tool["input_raw"] or "{}")
                            except json.JSONDecodeError:
                                _current_tool["input"] = {}
                            tool_use_blocks.append(_current_tool)
                            _current_tool = None

                final_msg = await stream.get_final_message()

            stop_reason = final_msg.stop_reason

            if stop_reason == "end_turn":
                messages.append({"role": "assistant", "content": final_msg.content})
                break

            if stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": final_msg.content})

                tool_results: list[dict] = []
                for tool in tool_use_blocks:
                    tool_result, clarification = await self._dispatch_tool(tool, request, event_queue)

                    if clarification:
                        return messages, thinking_chunks, schema_tables, clarification

                    if tool["name"] in ("search_schema", "get_table_details"):
                        tables = tool_result.get("tables") or []
                        if isinstance(tables, list):
                            schema_tables.extend(tables)
                        elif "table" in tool_result:
                            schema_tables.append(tool_result)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool["id"],
                        "content": json.dumps(tool_result, default=str),
                    })

                messages.append({"role": "user", "content": tool_results})
            else:
                messages.append({"role": "assistant", "content": final_msg.content})
                break

        return messages, thinking_chunks, schema_tables, None

    async def _dispatch_tool(
        self,
        tool: dict,
        request: AgentRequest,
        event_queue: asyncio.Queue,
    ) -> tuple[dict, dict | None]:
        name = tool["name"]
        inputs = tool.get("input") or {}

        if name == "search_schema":
            result = await schema_tool.search_schema(
                query=inputs.get("query", request.query),
                database_type=inputs.get("database_type", request.database_type),
                limit=inputs.get("limit", 5),
                user_id=request.user_id,
            )
            await event_queue.put(SSEEvent(EventType.SCHEMA_LOOKUP, {
                "tables": [t["table"] for t in result.get("tables", [])],
                "source": result.get("source", "vector"),
            }))
            return result, None

        if name == "get_table_details":
            result = await schema_tool.get_table_details(
                table_name=inputs.get("table_name", ""),
                database_type=inputs.get("database_type", request.database_type),
            )
            return result, None

        if name == "recall_memory":
            memories = await memory_tool.recall(
                context=inputs.get("context", request.query),
                user_id=request.user_id,
                memory_types=inputs.get("memory_types"),
            )
            return {"memories": memories}, None

        if name == "execute_query":
            await event_queue.put(SSEEvent(EventType.EXECUTING, {
                "status": "running", "execution_id": request.execution_id
            }))
            result = await execution_tool.execute_query(
                query=inputs.get("query", ""),
                database_type=inputs.get("database_type", request.database_type),
                execution_id=request.execution_id,
                page=inputs.get("page", 0),
                page_size=inputs.get("page_size", 100),
            )
            await event_queue.put(SSEEvent(EventType.RESULTS, result))
            return result, None

        if name == "clarify":
            clarification = {
                "question": inputs.get("question", ""),
                "options": inputs.get("options", []),
            }
            return clarification, clarification

        logger.warning(f"Unknown tool: {name}")
        return {"error": f"Unknown tool: {name}"}, None

    # ── Helpers ────────────────────────────────────────────────────────────────

    async def _resume_from_clarification(
        self, request: AgentRequest
    ) -> tuple[list[dict], str]:
        ctx = clarification_tool.load_context(request.clarification_id)
        clarification_tool.delete_context(request.clarification_id)

        if not ctx:
            # Context expired — start fresh with the answer embedded
            memories = await memory_tool.recall(request.query, user_id=request.user_id)
            history = await self._load_conversation(request.conversation_id)
            system = sp.build(request.database_type, memories=memories, conversation_history=history)
            messages: list[dict] = [{"role": "user", "content": (
                f"{request.query}\n\n(Clarification provided: {request.clarification_answer})"
            )}]
            return messages, system

        snap = ctx.get("request_snapshot", {})
        system = sp.build(snap.get("database_type", request.database_type))
        messages = ctx["messages"]

        # Store clarification exchange to memory
        if messages:
            original_query = next(
                (m["content"] for m in messages if m.get("role") == "user"), ""
            )
            if isinstance(original_query, str):
                await memory_tool.store_clarification(
                    vague_query=original_query,
                    clarification_question="(from prior session)",
                    user_answer=request.clarification_answer or "",
                    user_id=request.user_id,
                )

        # Append the user's answer and ask Claude to continue
        messages.append({
            "role": "user",
            "content": f"Clarification: {request.clarification_answer}. Please proceed.",
        })
        return messages, system

    def _extract_last_text(self, messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        b.text for b in content
                        if hasattr(b, "type") and b.type == "text"
                    )
        return ""

    def _parse_result(self, text: str) -> dict:
        match = re.search(r"<result>\s*(.*?)\s*</result>", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        code_match = re.search(r"```(?:q|sql|kdb)?\s*(.*?)```", text, re.DOTALL)
        if code_match:
            q = code_match.group(1).strip()
            return {
                "generated_query": q,
                "complexity": "MULTI_LINE" if "\n" in q else "SINGLE_LINE",
                "confidence": 0.7,
            }
        return {}

    async def _load_conversation(self, conversation_id: str | None) -> list[dict]:
        if not conversation_id:
            return []
        try:
            from app.services.conversation_manager import ConversationManager
            mgr = ConversationManager()
            return await mgr.get_conversation_history(conversation_id) or []
        except Exception as exc:
            logger.warning(f"Failed to load conversation {conversation_id}: {exc}")
            return []

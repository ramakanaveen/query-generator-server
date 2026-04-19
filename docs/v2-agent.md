# Query Generator v2 — Agent Implementation

**Branch:** `feature-query-agent-v2`
**Status:** In progress — core agent live, UI changes pending

---

## What Changed

### Architecture

The v1 system runs a fixed LangGraph pipeline with 3–5 sequential LLM calls per query (intent → schema → generate → validate → retry). On PAYG this costs up to 4 minutes.

The v2 system replaces that with a **Claude tool-use agent** running in a single streaming session. Claude decides which tools to call, how many times, and when it has enough context to generate. This collapses the pipeline into one LLM session + cheap tool calls.

```
v1:  intent classify → schema retrieve → unified_analyzer → validator → [retry ×2]
     3–5 serial LLM calls → 4 min on PAYG

v2:  Claude agent (streaming)
       ├─ tool: search_schema    (embedding + DB, ~0.5s)
       ├─ tool: get_table_details (DB, ~0.2s)
       ├─ tool: recall_memory    (embedding + DB, ~0.5s)
       ├─ tool: execute_query    (KDB/Starburst connector)
       └─ tool: clarify          (pause stream, ask user)
     1 LLM session → ~5s typical
```

**Production v1 endpoints are untouched.** v2 runs on `/api/v2/` prefix.

---

## New Files

### `app/services/agent/`

| File | Purpose |
|------|---------|
| `llm_client.py` | LLM backend abstraction — `AnthropicDirectClient` and `AnthropicVertexClient` with identical `create()` / `stream()` interface. Backend selected via `LLM_BACKEND` env var (`direct` or `vertex`). Auto-converts model aliases and direct API model IDs to Vertex format. |
| `query_agent.py` | Core agent orchestrator. Runs Claude tool-use loop, handles streaming, clarification pause/resume, rule-check retry, memory store-after-generation. Exposes `run()` (blocking JSON) and `run_streaming()` (SSE via asyncio.Queue). |
| `streaming.py` | `EventType` enum and `SSEEvent` dataclass. Serialises events to `text/event-stream` bytes. |
| `system_prompt.py` | Builds Claude system prompt: role definition, DB-specific syntax rules (KDB+/q or Starburst/Trino SQL), injected memories, conversation history. |
| `analyst.py` | Post-execution analysis. Single Claude Haiku call → NL summary, anomaly list, follow-up suggestions, chart hint. Called when `analysis=true` in request. |
| `rule_checker.py` | Stub — always returns `ok=True`. Placeholder for custom validation rules; does not call an LLM. |

### `app/services/agent/tools/`

| File | Purpose |
|------|---------|
| `schema_tool.py` | Wraps `EnhancedSchemaService`. Exposes `search_schema()` and `get_table_details()`. Interface is GraphRAG-ready — swap the implementation without touching the agent. |
| `memory_tool.py` | Wraps `MemoryManager` (previously built but completely unused). Activates `recall()` before generation and `store_pattern()` after success. Also `store_clarification()` when a vague-query exchange resolves. |
| `execution_tool.py` | Wraps KDB and Starburst connectors via `get_connector()`. Stores results by `execution_id` for the download endpoint. |
| `clarification_tool.py` | In-memory pause/resume state. Saves message history keyed by `clarification_id`; restores it on resume request. |

### `app/routes/query_v2.py`

Two endpoints:

**`POST /api/v2/query`**
- `Accept: text/event-stream` → SSE streaming (real-time thinking, schema lookup, query, results, analysis events)
- `Accept: application/json` → blocking JSON (backwards-compatible for existing clients)
- Returns `duration_ms` in both modes

**`GET /api/v2/download/{execution_id}`**
- `?format=csv` (default), `json`, or `excel`
- Streams file directly; no in-memory buffering

---

## Modified Files

| File | Change |
|------|--------|
| `app/core/config.py` | Added v2 settings: `LLM_BACKEND`, `CLAUDE_DEFAULT_MODEL`, `CLAUDE_FAST_MODEL`, `CLAUDE_POWERFUL_MODEL`, `CLAUDE_V2_MAX_TOKENS`, `CLAUDE_THINKING_BUDGET`. `CLAUDE_DEFAULT_MODEL` falls back to `CLAUDE_MODEL_NAME` for backwards compat. |
| `app/core/db.py` | Added `get_db_pool()` function and `acquire()` async context manager to `DatabasePool` (was missing; blocked memory module). |
| `app/main.py` | Registered v2 router on `/api/v2` prefix. |
| `app/routes/feedback.py` | Positive feedback now fires `memory_tool.store_pattern()` as a background task — activates the memory learning loop. |
| `run.py` | Enabled `access_log=True` for HTTP-level request timing in logs. |
| `app/services/query_generation/prompts/intent_classifier_prompts.py` | Added `INTENT_CLASSIFICATION_PROMPT` backwards-compat alias (was missing, blocked server startup). |
| `app/services/memory/memory_manager.py` | Fixed `embed_query` → `get_embedding` (wrong method name; blocked memory recall). |

---

## API Reference

### `POST /api/v2/query`

```json
{
  "query": "string",
  "model": "claude-sonnet-4-20250514 | claude-haiku-4-5 | claude-opus-4-7",
  "database_type": "kdb | starburst | postgres",
  "conversation_id": "uuid (optional)",
  "user_id": "string (optional)",
  "response_mode": "query | execute | analyze | download | schema_info | explain | agent",
  "auto_execute": false,
  "analysis": false,
  "extended_thinking": false,
  "clarification_id": "uuid (optional — resume a paused clarification)",
  "clarification_answer": "string (optional — answer to clarification question)"
}
```

### SSE Event Stream

| Event | Payload | When |
|-------|---------|------|
| `thinking` | `{"text": "..."}` | `extended_thinking: true` |
| `schema_lookup` | `{"tables": [...], "source": "vector"}` | After `search_schema` tool call |
| `clarification_needed` | `{"question": "...", "options": [...], "clarification_id": "..."}` | Vague query |
| `query_generated` | `{"query": "...", "complexity": "...", "confidence": 0.0–1.0, "explanation": "..."}` | Query ready |
| `executing` | `{"status": "running", "execution_id": "..."}` | `auto_execute: true` |
| `results` | `{"rows": [...], "metadata": {...}, "pagination": {...}}` | After execution |
| `analysis` | `{"summary": "...", "anomalies": [...], "suggested_followups": [...], "chart_hint": {...}}` | `analysis: true` |
| `schema_info` | `{"text": "..."}` | Schema description intent |
| `download_ready` | `{"download_url": "...", "formats": [...]}` | `response_mode: download` |
| `error` | `{"message": "..."}` | On failure |
| `done` | `{"execution_id": "...", "duration_ms": N}` | Always last |

### JSON Response (non-streaming)

Same fields as above, plus `duration_ms` at the top level.

---

## 10 Supported Scenarios

| # | Scenario | `response_mode` |
|---|----------|----------------|
| 1 | Generate query + optionally execute | `query` |
| 2 | Schema / table description | `schema_info` |
| 3 | Execute + full analytics | `analyze` |
| 4 | Long agentic / multi-step prompt | `agent` |
| 5 | Execute → paginated table | `execute` |
| 6 | Execute → file download | `download` |
| 7 | Conversational follow-up | `query` (with `conversation_id`) |
| 8 | Explain or debug a query | `explain` |
| 9 | Compare two time periods | `agent` (multi-query) |
| 10 | Vague query → clarification → resume | any |

---

## Configuration

Key `.env` variables for v2:

```bash
LLM_BACKEND=direct              # "direct" (Anthropic API key) or "vertex" (Google Cloud)
CLAUDE_DEFAULT_MODEL=claude-sonnet-4-20250514
CLAUDE_FAST_MODEL=claude-haiku-4-5        # used for post-execution analysis only
CLAUDE_POWERFUL_MODEL=claude-opus-4-7
CLAUDE_V2_MAX_TOKENS=16000
CLAUDE_THINKING_BUDGET=8000               # token budget when extended_thinking=true
```

Note: `CLAUDE_DEFAULT_MODEL` falls back to `CLAUDE_MODEL_NAME` if not explicitly set.

---

## Model Routing

| Situation | Behaviour |
|-----------|-----------|
| `model: "claude-haiku-*"` in request | Auto-upgraded to `claude-sonnet` for query generation (Haiku produces syntax errors) |
| `analysis: true` | Always uses `CLAUDE_FAST_MODEL` (Haiku) — analysis does not require KDB/SQL precision |
| `extended_thinking: true` | Enables Claude's native thinking tokens; requires Sonnet or Opus |

---

## Known Limitations / Future Work

| Item | Status |
|------|--------|
| GraphRAG schema retrieval | Planned — `schema_tool.py` interface is ready to swap |
| `rule_checker.py` validation rules | Stub — fill in when rules are decided |
| Excel download | Requires `openpyxl` (not in `requirements.txt`) |
| Memory feedback wiring (negative feedback) | Positive feedback wired; negative feedback learning not yet wired |
| UI changes | Deferred — server API contracts are stable, see SSE event table above |
| Clarification context storage | In-memory only — lost on server restart; move to Redis for production |
| Execution result storage | In-memory only — move to DB/Redis for production |

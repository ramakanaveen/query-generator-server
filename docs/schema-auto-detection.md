# Schema Auto-Detection & Directive Removal — 3-Phase Plan

**Status:** Planning  
**Branch:** `feature-query-agent-v2` (build here; `feature-html-analysis-dashboard` parked)  
**Replaces:** Mandatory `@GROUP` directives in user queries

---

## Discussion Summary (How We Got Here)

**Problem:** 30+ schema groups exist (STIRT, SPOT, TITAN, …). Users must type `@stirt` or `@spot` to scope queries. This fails because: (a) users don't know which directive to use, (b) multi-group queries like Titan trades spanning SPOT + STIRT namespaces are impossible.

**KDB namespace model (confirmed):** Single KDB instance, multiple namespaces. Hierarchy is `schema_group` (business domain label, user-defined) → `schema` (KDB namespace, e.g. `stirtimtm`) → `table` (`.stirtimtm.bookedtrades`) → `column`. Group name ≠ namespace. One group can have many schemas/namespaces.

**Why directives exist today:** `enhanced_schema_service.py` retrieves `group_name`/`schema_name` via DB JOINs but `schema_tool.py` strips that metadata before the LLM sees it. The LLM sees bare table names with no group identity and cannot self-route. Directives compensate for this stripped context.

**Two-layer ambiguity problem:**
- Layer A (naming collision): same table name in different groups — resolved by namespace-qualified names (`.spot.trade` vs `.stirtimtm.bookedtrades`) which are unambiguous strings
- Layer B (semantic overlap): same concept, different names across groups — resolved by showing group/schema labels in LLM context so the agent can reason

**Schema context format decided:** Preamble + short labels (not inline repetition). Group/schema descriptions appear once at the top of the schema context block; per-table entries carry only a short `[group/schema]` label. ~220–280 tokens for 5 tables vs 500–700 for naive inline repetition.

**Knowledge graph decision:** Not a separate graph database. The existing infrastructure already contains what's needed:
- `table_relationships` table in DB (defined, empty, no indexes) → FK/join edges
- `memory.entries` with `schema_group_id`, `QUERY_PATTERN` type → concept→table mappings
- `memory_manager.store(schema_group_id=...)` parameter already exists, just never called
- Knowledge builds passively from every conversation, not as a one-time project

**Ongoing learning:** Three real-time wiring points after each interaction (fire-and-forget, zero latency impact): (1) after successful query — extract concept→group mappings async via Haiku, (2) after negative feedback — `MemoryExtractor.extract_and_store_from_feedback()` (built, never wired), (3) after positive feedback — increment confidence on existing QUERY_PATTERN memory.

**V1 scope:** Only one V1 node changes — `simplified_query_generator_node.py` `format_schema_for_generation()` switches to the new preamble formatter. All other V1 pipeline nodes untouched.

---

## What Exists Already (Do Not Rebuild)

| Asset | Location | Status |
|-------|----------|--------|
| DB query retrieving group_name + schema_name | `enhanced_schema_service.py` lines 341–368 | ✅ Works, data is there |
| `_build_schema_structure()` — metadata includes group_name, schema_name | `enhanced_schema_service.py` lines 811–837 | ✅ Fields present |
| `memory_manager.store(schema_group_id=...)` | `memory_manager.py` lines 110–174 | ✅ Param exists, not called |
| `table_relationships` DB table | `scripts/db_scripts/01_schema_tables.sql` lines 85–94 | ✅ Defined, empty, no indexes |
| `MemoryExtractor.extract_and_store_from_feedback()` | `app/services/memory/memory_extractor.py` | ✅ Built, never called |

---

## Phase 1 — Directive-Free Routing

**Goal:** Directives become optional. Agent auto-detects group from query. V1 + V2 both benefit.  
**Prerequisite:** Schema manager updated to store table names as `.namespace.table`.

### 1a. Auto-generate group descriptions

Group descriptions are empty today. Generate 1-sentence summaries by aggregating table names and descriptions per group.

**New file:** `app/services/schema_description_generator.py`
- `async def generate_group_description(group_id, group_name, tables) -> str` — Haiku call
- `async def generate_schema_description(schema_id, schema_name, tables) -> str` — Haiku call
- Writes to `schema_groups.description` and `schema_definitions.description`

**New admin endpoint** (new file `app/routes/admin_schema.py`, registered in `main.py`):
- `POST /api/admin/schema-groups/generate-descriptions` — generates for all groups missing descriptions
- Also call `generate_group_description()` at end of schema import in `app/services/schema_management.py`

### 1b. Schema context formatter — preamble + short labels (V1 + V2)

**File:** `app/services/enhanced_schema_service.py`

Add `format_schema_context_for_llm(schema_structure: dict) -> str`:

```
Groups referenced:
- stirt: STIRT derivatives — FX forwards, swaps, NDFs
  - stirtimtm: Intraday MTM trade bookings and positions

Tables (3):
.stirtimtm.bookedtrades [stirt/stirtimtm]
All booked STIRT trade economics
Columns: sym(sym), tradedate(date), notional(float), ccypair(sym), trader(sym)

.spot.trade [spot/spot]
Spot FX cash transactions
Columns: sym(sym), price(float), size(long), side(sym), date(date)
```

Rules:
- Group + schema descriptions appear **once** in the preamble, keyed by name
- Only groups/schemas present in the retrieved table set are listed
- Per-table: `[group/schema]` short label only — no descriptions repeated
- If `schema_groups.description` is empty, omit that line gracefully

**V1 wire-up:** In `app/services/query_generation/nodes/simplified_query_generator_node.py`, replace `format_schema_for_generation()` with `format_schema_context_for_llm()`. This is the **only V1 node change**.

### 1c. schema_tool.py changes (V2)

**File:** `app/services/agent/tools/schema_tool.py`

Three changes:

1. **Expose group/schema in search_schema() result** — replace `{"table": table_name, "columns": columns}` with `{"table": table_name, "group": group_name, "schema": schema_name, "columns": columns}`. Data already exists in `schema_structure[table]["metadata"]`.

2. **Add `groups` filter** — add `groups: list[str] | None = None` to `search_schema()`. Pass to `EnhancedSchemaService` which adds `AND sg.name = ANY($groups::text[])` to the WHERE clause in `_find_relevant_tables_optimized()`.

3. **New `list_schema_groups()` function:**
```python
async def list_schema_groups(database_type: str) -> list[dict]:
    # SELECT sg.name, sg.description, COUNT(td.id) as table_count
    # FROM schema_groups sg
    # JOIN schema_definitions sd ON sd.group_id = sg.id
    # JOIN schema_versions sv ON sv.schema_id = sd.id
    # JOIN active_schemas a ON a.current_version_id = sv.id
    # JOIN table_definitions td ON td.schema_version_id = sv.id
    # GROUP BY sg.name, sg.description ORDER BY sg.name
    # Returns: [{name, description, table_count}]
```

### 1d. Agent tool registry + system prompt (V2)

**File:** `app/services/agent/query_agent.py`
- Add `list_schema_groups` to TOOLS: "List all available schema groups with descriptions. Call this when the query is ambiguous about which business domain to search, or when the user asks about available data sources."
- Add `groups` parameter to the `search_schema` tool definition.

**File:** `app/services/agent/system_prompt.py`
- Add: "Table names are namespace-qualified (e.g. `.stirtimtm.bookedtrades`). Always use the full `.namespace.table` form in generated queries — never bare table names."
- Add: "Directives like `@stirt` are optional hints. If present, use them as `groups` filter in `search_schema`. If absent, either search all groups or call `list_schema_groups` first for ambiguous queries."

### 1e. Backwards compatibility

Directives continue to work unchanged. If `@GROUP` tokens are present, extract them and pass as the `groups` filter. V1 `initial_processor.py` already does this extraction — no change needed.

---

## Phase 2 — KG-Lite: Relationships + Memory Enrichment

**Goal:** Agent knows how to join tables across namespaces. User affinity steers group selection for implicit queries.

### 2a. table_relationships indexes + join-path tool

**New migration:** `scripts/db_scripts/04_table_relationships_indexes.sql`
```sql
CREATE INDEX idx_table_rel_source ON table_relationships(source_table_id);
CREATE INDEX idx_table_rel_target ON table_relationships(target_table_id);
```

**Admin endpoint** (in `app/routes/admin_schema.py`):
- `POST /api/admin/table-relationships` — bulk import FK edges
- Body: `[{source_table, target_table, relationship_type, join_column, description}]`
- Looks up `table_definitions.id` by name, inserts rows

**New tool** in `schema_tool.py`: `get_join_path(table_a: str, table_b: str, database_type: str) -> dict`
- Queries `table_relationships` for direct edge between two tables
- Returns `{join_column, relationship_type, description}` or empty if none
- Register in `query_agent.py` TOOLS: "Find how two tables can be joined. Call this when generating a query that needs to join tables from different schemas."

### 2b. QUERY_PATTERN memory enrichment

**File:** `app/services/agent/tools/memory_tool.py`

Update `store_pattern()`:
```python
async def store_pattern(
    original_query: str,
    generated_query: str,
    tables_used: list[str] = [],       # e.g. [".stirtimtm.bookedtrades", ".spot.trade"]
    user_id: str | None = None,
) -> None
```
- Extract group IDs from `tables_used` by querying `schema_groups` (via table_definitions join)
- Pass primary `schema_group_id` to `memory_manager.store()`
- Store `tables_used` in metadata JSONB: `{"tables": tables_used}`

**Wire-up:** In `query_agent.py` `_run_core()`, after successful generation, pass `tables_used` from the schema tool result to `store_pattern()`.

### 2c. User group affinity signal

**File:** `app/services/conversation_manager.py`

New method:
```python
async def get_user_group_affinity(user_id: str) -> list[dict]:
    # Query memory.entries WHERE user_id = $1 AND memory_type = 'query_pattern'
    # GROUP BY schema_group_id, ORDER BY count DESC LIMIT 5
    # JOIN schema_groups to get group name
    # Returns: [{group_name: "stirt", query_count: 47}, ...]
```

**File:** `app/services/agent/system_prompt.py`
- Call `get_user_group_affinity(user_id)` in the system prompt builder
- Inject when affinity exists: "User context: this user primarily queries stirt (47 queries), spot (12). When group is ambiguous, prefer stirt tables."
- Omit entirely for new users with no history

---

## Phase 3 — Learning from Conversations

**Goal:** Bootstrap concept→table mappings from existing conversations. Passive real-time learning going forward.

### 3a. Conversation mining backfill (one-time)

**New file:** `scripts/mine_conversation_patterns.py`
- Fetch all conversations from DB with successful queries
- For each: extract (nl_query, generated_query, directive/group from message metadata)
- Call `memory_manager.store(memory_type=QUERY_PATTERN, ...)` for each
- Run once: `python scripts/mine_conversation_patterns.py`

### 3b. Real-time learning after every interaction (ongoing)

Three wiring points that fire immediately, never blocking the response:

**After every successful query** (`query_agent.py` `_run_core()`):
- Extend to call `MemoryExtractor.extract_concept_mappings(nl_query, tables_used, group_names)` — lightweight Haiku call extracting implicit concept→group mappings
- Store as `SCHEMA_CLARIFICATION` memories with `user_id` + `schema_group_id`
- Runs via `asyncio.create_task()` — fire-and-forget, zero latency impact

**After negative feedback/correction** (`app/routes/feedback.py`):
- `MemoryExtractor.extract_and_store_from_feedback()` is built but **never called** — wire it here
- Stores `SYNTAX_CORRECTION` or `ERROR_CORRECTION` memories immediately on correction submission

**After positive feedback** (`app/routes/feedback.py`):
- Extend existing `memory_tool.store_pattern` call to also increment `success_count` on the matching QUERY_PATTERN memory
- Promotes patterns that are repeatedly confirmed vs. ones used only once

**New method** in `app/services/memory/memory_extractor.py`:
```python
async def extract_concept_mappings(
    nl_query: str,
    tables_used: list[str],
    group_names: list[str],
    user_id: str | None,
) -> None
    # Haiku call: extract concept phrases → table/group mappings
    # Store each as SCHEMA_CLARIFICATION memory
```

### 3c. Passive SCHEMA_CLARIFICATION on disambiguation

**File:** `app/services/agent/query_agent.py`

On clarification resume (user answers "Did you mean STIRT or SPOT?"):
- Store `SCHEMA_CLARIFICATION` memory: `original_context=ambiguous_phrase`, `learning=resolved_table`, `schema_group_id`, `user_id`
- Wire `MemoryExtractor.extract_and_store_from_feedback()` here (currently built, never called)

### 3d. Group-level embeddings (scaling optimization)

**New migration:** `scripts/db_scripts/05_schema_group_embeddings.sql`
```sql
ALTER TABLE schema_groups ADD COLUMN embedding vector(768);
CREATE INDEX idx_schema_groups_embedding ON schema_groups
  USING ivfflat (embedding vector_cosine_ops);
```

**File:** `app/services/schema_description_generator.py` (Phase 1)
- After generating group description, compute embedding via `EmbeddingProvider.get_embedding()`
- Store in `schema_groups.embedding`

**File:** `app/services/enhanced_schema_service.py`
- Optional pre-filter in `_find_relevant_tables_optimized()`: if group embeddings exist, rank groups by cosine similarity first (top-K=3), then restrict table search to those groups
- Falls back to full search when column is not populated
- Meaningful optimization once group count exceeds ~30

---

## Critical Files Map

| File | Phase | Change |
|------|-------|--------|
| `app/services/enhanced_schema_service.py` | 1, 3 | `format_schema_context_for_llm()`, `groups` filter, group-embedding pre-filter |
| `app/services/agent/tools/schema_tool.py` | 1, 2 | Expose group/schema in result, `list_schema_groups()`, `groups` param, `get_join_path()` |
| `app/services/agent/query_agent.py` | 1, 2, 3 | Add tools to TOOLS list, pass `tables_used` to `store_pattern`, SCHEMA_CLARIFICATION on resume |
| `app/services/agent/system_prompt.py` | 1, 2 | Namespace instruction, directive-optional guidance, user affinity injection |
| `app/services/agent/tools/memory_tool.py` | 2, 3 | `store_pattern()` with `tables_used` + `schema_group_id` |
| `app/services/conversation_manager.py` | 2 | `get_user_group_affinity()` |
| `app/services/schema_management.py` | 1 | Call `generate_group_description()` at end of schema import |
| `app/services/query_generation/nodes/simplified_query_generator_node.py` | 1 | Use `format_schema_context_for_llm()` — **only V1 node change** |
| `app/routes/feedback.py` | 3 | Wire `MemoryExtractor.extract_and_store_from_feedback()` on correction; increment confidence on positive feedback |
| `app/services/memory/memory_extractor.py` | 3 | Add `extract_concept_mappings()` — async fire-and-forget after every successful query |
| **New:** `app/services/schema_description_generator.py` | 1, 3 | LLM description + embedding generation for groups/schemas |
| **New:** `app/routes/admin_schema.py` | 1, 2 | `generate-descriptions`, `table-relationships` admin endpoints |
| **New:** `scripts/db_scripts/04_table_relationships_indexes.sql` | 2 | Indexes on `table_relationships` |
| **New:** `scripts/db_scripts/05_schema_group_embeddings.sql` | 3 | `embedding` column on `schema_groups` |
| **New:** `scripts/mine_conversation_patterns.py` | 3 | One-time backfill from existing conversations |

**Not touched:** `app/routes/query.py`, all LangGraph nodes except `simplified_query_generator_node.py`, `query_generator.py`, all connectors.

---

## Verification

### Phase 1
1. Run `POST /api/admin/schema-groups/generate-descriptions` — verify `schema_groups.description` is populated in DB
2. Send `POST /api/v2/query` with `{"query": "give me EURUSD top 10 bid ask"}` (no `@` directive) — SSE `schema_lookup` event should show tables with `[group/schema]` labels
3. Same query via V1 `POST /api/v1/query` — generated KDB query should use `.namespace.table` form
4. Multi-group query: `"give me titan trades across spot and stirt today"` — agent generates union/join spanning both groups
5. `@stirt give me trades` — old directive style still routes correctly to STIRT

### Phase 2
1. Bulk insert test FK edges via `POST /api/admin/table-relationships`
2. Join query: `"show me titan clients and their STIRT bookings"` — agent calls `get_join_path`, uses correct join column
3. Run 5 queries as user X in STIRT → `get_user_group_affinity(X)` returns STIRT with count 5
4. 6th query `"give me my trades today"` (no directive) — system prompt injects STIRT preference; query targets `.stirtimtm.*`

### Phase 3
1. Run backfill: `python scripts/mine_conversation_patterns.py` — `memory.entries` has new QUERY_PATTERN rows with `schema_group_id` set
2. Ask a query matching conversation history — `recall_memory` returns the matched pattern
3. Trigger disambiguation clarification, answer it — `memory.entries` has SCHEMA_CLARIFICATION row
4. **Ongoing (realtime):** Send any v2 query, wait 2s — `SELECT * FROM memory.entries ORDER BY created_at DESC LIMIT 5` shows new SCHEMA_CLARIFICATION rows from async extraction
5. Submit negative feedback correction — SYNTAX_CORRECTION or ERROR_CORRECTION row appears immediately
6. Submit positive feedback — `success_count` incremented on matching QUERY_PATTERN row
7. After Phase 3d migration + re-running description generation — `schema_groups.embedding` is populated

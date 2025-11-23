# Human-in-the-Loop (HITL) Query Clarification System

**Version:** 2.0
**Last Updated:** 2025-11-23
**Status:** Design Phase - Ready for Review
**Branch:** `feature-human-in-the-loop-clarification`

---

## Document Context & Purpose

This document provides **complete context** for implementing Human-in-the-Loop (HITL) clarification in the query generation system. When you return to this project, this document will give you full context on:

1. **The problem** we're solving
2. **Background research** (DeepAgents architecture)
3. **Three implementation approaches** with pros/cons
4. **Detailed code examples** for each approach
5. **Recommended path forward**

**Background Reading:**
- **DeepAgents Framework:** https://github.com/langchain-ai/deepagents
  - Modern agent architecture with built-in HITL support
  - Key features: Planning middleware, sub-agent delegation, interrupt_on capability, persistent storage
  - Uses LangGraph but adds sophisticated middleware layers

---

## Current System Architecture

**Tech Stack:**
- FastAPI backend
- LangGraph for query generation pipeline
- Google Gemini and Anthropic Claude LLMs
- PostgreSQL with pgvector for schema embeddings
- Conversation management and shared conversations features

**Current Query Generation Pipeline:**
```
┌──────────────────┐
│  User NL Query   │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    LangGraph Pipeline                            │
│                                                                  │
│  ┌────────────────┐    ┌──────────────────┐                    │
│  │ Intent         │───▶│ Schema           │                    │
│  │ Classifier     │    │ Retriever        │                    │
│  └────────────────┘    └──────────┬───────┘                    │
│                                    │                            │
│                                    ▼                            │
│                        ┌──────────────────┐                    │
│                        │ Query            │                    │
│                        │ Analyzer         │                    │
│                        └──────────┬───────┘                    │
│                                    │                            │
│                                    ▼                            │
│                        ┌──────────────────┐                    │
│                        │ Query            │                    │
│                        │ Generator        │                    │
│                        └──────────┬───────┘                    │
│                                    │                            │
│                                    ▼                            │
│                        ┌──────────────────┐                    │
│                        │ Validator        │                    │
│                        └──────────────────┘                    │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  Generated Query │
└──────────────────┘
```

**Key Files:**
- `app/services/query_generator.py` - Main pipeline orchestrator
- `app/services/query_generation/nodes/` - Individual pipeline nodes
- `app/routes/query.py` - API endpoints
- `app/services/query_generator_state.py` - State management

---

## Problem Statement

### Current Behavior (No HITL)

```
User: "Show me high volume trades"
         ↓
System processes all nodes
         ↓
Generates query (guessing values):
  "select from trades where volume > 1000000"  // Arbitrary threshold!
         ↓
User: "That's wrong, I meant volume > 100k for AAPL today"
         ↓
Retry...
```

**Issues:**
1. System wastes processing on ambiguous queries
2. User wastes time guessing what went wrong
3. Multiple failed attempts consume LLM tokens ($$$)
4. Poor UX - reactive failure instead of proactive guidance
5. Lower accuracy - system makes assumptions

### Desired Behavior (With HITL)

```
User: "Show me high volume trades"
         ↓
System detects ambiguity after schema retrieval
         ↓
System: "I need clarification:
  1. What date range? (Today / This week / Other)
  2. Which symbol? (e.g., AAPL)
  3. What volume threshold? (e.g., 100000)"
         ↓
User: "Today, AAPL, 100000"
         ↓
Generates accurate query first time:
  "select from trades where date=.z.d, sym=`AAPL, volume > 100000"
```

**Benefits:**
- ✅ Higher accuracy - no guessing
- ✅ Fewer retries - save time and money
- ✅ Better UX - proactive guidance
- ✅ Learning opportunity - can store patterns

---

## Requirements Summary

Based on brainstorming discussion:

### 1. Where to Ask for Clarification
- **Primary:** After schema retrieval (better context for meaningful questions)
- **Secondary:** After intent classification if intent itself is ambiguous

### 2. Latency Tolerance
- **1 extra round trip is acceptable** (better than multiple failed attempts)

### 3. Types of Ambiguities
- **Open-ended:** System should detect ANY ambiguity using LLM intelligence
- Don't hard-code specific patterns - let LLM decide
- Examples: date ranges, symbols, column choices, thresholds, vague qualifiers

### 4. Clarification Mode
- **Hybrid approach:** Automatic by default, with opt-out option for power users
- `enable_clarification: true` (default) or `false` (skip)

### 5. Learning & Pattern Storage
- **Yes - store successful clarification patterns**
- Auto-resolve common ambiguities for repeat users
- Example: User always means "today" when saying "recent"

### 6. Question Generation
- **LLM generates questions directly** (not manual templates)
- More natural, contextual, flexible
- Simpler code - no manual mapping logic

---

## Three Implementation Approaches

### Overview Comparison

| Aspect | Option 1: Minimal | Option 2: Medium | Option 3: Full DeepAgents |
|--------|-------------------|------------------|---------------------------|
| **Complexity** | Low | Medium | High |
| **Code Changes** | ~100 lines | ~300 lines | ~2000+ lines |
| **Risk** | Low | Medium | High |
| **Time to Implement** | 1-2 days | 1 week | 3-4 weeks |
| **Architecture Change** | None | Minimal | Complete redesign |
| **Recommendation** | ⭐ **RECOMMENDED** | Future upgrade | Long-term vision |

---

## OPTION 1: Minimal Change - Add HITL to Existing Pipeline ⭐ RECOMMENDED

### Overview

Keep your current LangGraph pipeline intact. Add ONE new node for ambiguity checking with LangGraph's built-in checkpointing and interrupts.

**Philosophy:** "Don't fix what ain't broke" - minimal changes, minimal risk.

### Architecture Changes

```
BEFORE:
Intent → Schema → Analyzer → Generator → Validator

AFTER:
Intent → Schema → [NEW: Ambiguity Check] → Analyzer → Generator → Validator
                            ↓
                    If ambiguous: INTERRUPT
                    Return to user with questions
                    User responds → Resume pipeline
```

### Key Concept

LangGraph already has everything we need:
- `MemorySaver` checkpointer - saves state at each node
- Conditional edges - can route to END (interrupt)
- State resumption - continue from where we left off

We just add:
1. One ambiguity detection node
2. Conditional routing based on clarity
3. Simple resume logic

### Implementation

#### Step 1: Extend State Model

```python
# app/services/query_generator_state.py

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class ClarificationQuestion(BaseModel):
    """Single clarification question."""
    question: str
    type: str  # "multiple_choice", "text", "number"
    options: Optional[List[str]] = None
    context: Optional[str] = None

class QueryGeneratorState(BaseModel):
    """Existing state with HITL fields added."""

    # ... all your existing fields ...
    query: str
    user_id: str
    conversation_id: Optional[str]
    intent_type: str
    query_schema: Dict[str, Any]
    generated_query: Optional[str]
    # ... etc ...

    # NEW: HITL fields
    enable_clarification: bool = True  # User can opt-out
    requires_clarification: bool = False
    clarification_questions: List[ClarificationQuestion] = Field(default_factory=list)
    clarification_responses: Optional[Dict[str, Any]] = None
    clarification_round: int = 0
    max_clarification_rounds: int = 2  # Prevent loops
```

#### Step 2: Create Ambiguity Detection Node

```python
# app/services/query_generation/nodes/ambiguity_detector.py

import json
from typing import Dict, Any
from app.services.query_generator_state import QueryGeneratorState, ClarificationQuestion
from app.core.logging import logger

async def check_ambiguity_node(state: QueryGeneratorState) -> QueryGeneratorState:
    """
    Detect ambiguities and generate clarification questions.

    This node:
    1. Checks if clarification is enabled
    2. Uses LLM to detect ambiguities (let LLM decide!)
    3. LLM generates natural language questions
    4. Sets requires_clarification flag if needed

    If requires_clarification=True, the pipeline will interrupt.
    """

    # Skip if clarification disabled
    if not state.enable_clarification:
        state.thinking.append("⏭️  Clarification disabled by user")
        return state

    # Skip if already have clarification responses
    if state.clarification_responses:
        state.thinking.append("✓ Clarification responses received, applying...")
        return state

    # Skip if already exceeded max rounds
    if state.clarification_round >= state.max_clarification_rounds:
        state.thinking.append("⚠️  Max clarification rounds reached, proceeding anyway")
        return state

    try:
        llm = state.llm

        # Format schema for prompt
        schema_text = format_schema_for_clarity_check(state.query_schema)

        # LLM prompt - let LLM do the heavy lifting!
        prompt = f"""
You are checking if a natural language query has sufficient information to generate an accurate database query.

User Query: {state.query}
Intent Type: {state.intent_type}
Available Schema:
{schema_text}

Conversation Context:
{format_conversation_context(state.conversation_history)}

Analyze if this query is clear and complete. Check for:
1. **Missing Filters**: Date ranges, symbols, identifiers
2. **Ambiguous References**: Multiple columns match description (e.g., "price" → bid_price, ask_price, last_price)
3. **Vague Qualifiers**: "high", "low", "recent", "best" without specifics
4. **Undefined Thresholds**: Comparisons without values

Be conservative - only ask for clarification if truly necessary for accuracy.

Return JSON:
{{{{
  "is_clear": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation",
  "questions": [
    {{{{
      "question": "Natural language question for user",
      "type": "multiple_choice" | "text" | "number",
      "options": ["opt1", "opt2"],  // only for multiple_choice
      "context": "Why you're asking this"
    }}}}
  ]
}}}}

Example good questions:
- "What date range do you want? (Today / Last 7 days / Custom)"
- "I see 3 price columns: bid_price, ask_price, last_price. Which one?"
- "What threshold for 'high volume'? Please provide a number (e.g., 100000)."
"""

        response = await llm.ainvoke(prompt)
        result = json.loads(response.content)

        if not result["is_clear"]:
            # Ambiguity detected - prepare for interruption
            questions = [
                ClarificationQuestion(**q) for q in result.get("questions", [])
            ]

            state.requires_clarification = True
            state.clarification_questions = questions
            state.thinking.append(
                f"❓ Ambiguity detected (confidence: {result['confidence']:.2f}): "
                f"{result['reasoning']}"
            )
            state.thinking.append(f"🔄 Requesting {len(questions)} clarifications from user")
        else:
            state.thinking.append(
                f"✓ Query is clear (confidence: {result['confidence']:.2f})"
            )

        return state

    except Exception as e:
        logger.error(f"Error in ambiguity detection: {str(e)}", exc_info=True)
        state.thinking.append(f"⚠️  Ambiguity check failed: {str(e)}, proceeding anyway")
        # Don't block on error - continue pipeline
        return state


def format_schema_for_clarity_check(schema: Dict[str, Any]) -> str:
    """Format schema for LLM prompt."""
    formatted = ""
    for table in schema.get("tables", []):
        formatted += f"\n**Table: {table['name']}**\n"
        for col in table.get("columns", []):
            formatted += f"  - {col['name']} ({col['type']}): {col.get('description', '')}\n"
    return formatted


def format_conversation_context(history: List[Dict[str, Any]]) -> str:
    """Format conversation history for context."""
    if not history:
        return "No previous context"

    formatted = "Previous conversation:\n"
    for msg in history[-3:]:  # Last 3 messages
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        formatted += f"- {role}: {content[:100]}...\n"
    return formatted
```

#### Step 3: Update Pipeline Workflow

```python
# app/services/query_generator.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.services.query_generation.nodes.ambiguity_detector import check_ambiguity_node
# ... your existing imports ...

def create_query_generation_workflow():
    """
    Create the query generation workflow with HITL support.

    ONLY CHANGE: Added ambiguity_detector node and conditional routing.
    """

    workflow = StateGraph(QueryGeneratorState)

    # Add all existing nodes
    workflow.add_node("intent_classifier", classify_intent)
    workflow.add_node("schema_retriever", retrieve_schema)
    workflow.add_node("ambiguity_detector", check_ambiguity_node)  # NEW NODE
    workflow.add_node("query_analyzer", analyze_query)
    workflow.add_node("query_generator", generate_query_node)
    workflow.add_node("validator", validate_query)

    # Set entry point
    workflow.set_entry_point("intent_classifier")

    # Edges - mostly unchanged
    workflow.add_edge("intent_classifier", "schema_retriever")
    workflow.add_edge("schema_retriever", "ambiguity_detector")  # NEW: Route to ambiguity check

    # NEW: Conditional routing based on clarity
    workflow.add_conditional_edges(
        "ambiguity_detector",
        lambda state: "interrupt" if state.requires_clarification else "continue",
        {
            "interrupt": END,  # Stop here, return to user
            "continue": "query_analyzer"  # Proceed normally
        }
    )

    # Rest of pipeline unchanged
    workflow.add_edge("query_analyzer", "query_generator")
    workflow.add_edge("query_generator", "validator")
    workflow.add_edge("validator", END)

    # ADD CHECKPOINTER - this enables state saving/resumption
    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


# Create compiled graph
query_graph = create_query_generation_workflow()


# Main API functions
async def generate_query_with_hitl(
    query: str,
    user_id: str,
    conversation_id: Optional[str] = None,
    enable_clarification: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate query with HITL support.

    Returns either:
    - Success: {"status": "success", "generated_query": "..."}
    - Needs clarification: {"status": "needs_clarification", "questions": [...], "thread_id": "..."}
    """

    # Generate thread_id for state management
    thread_id = conversation_id or f"thread_{user_id}_{uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    # Initial state
    initial_state = QueryGeneratorState(
        query=query,
        user_id=user_id,
        conversation_id=conversation_id,
        enable_clarification=enable_clarification,
        thinking=[],
        # ... other initial fields ...
    )

    # Run pipeline
    final_state = await query_graph.ainvoke(initial_state.dict(), config)

    # Check if clarification needed
    if final_state.get("requires_clarification"):
        return {
            "status": "needs_clarification",
            "thread_id": thread_id,
            "questions": [q.dict() for q in final_state["clarification_questions"]],
            "thinking": final_state.get("thinking", [])
        }

    # Success
    return {
        "status": "success",
        "generated_query": final_state.get("generated_query"),
        "execution_id": final_state.get("execution_id"),
        "thinking": final_state.get("thinking", []),
        "thread_id": thread_id
    }


async def continue_after_clarification(
    thread_id: str,
    clarification_responses: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Resume pipeline after user provides clarification.

    LangGraph checkpointer automatically loads the saved state!
    """

    config = {"configurable": {"thread_id": thread_id}}

    # Get current state from checkpoint
    state_snapshot = await query_graph.aget_state(config)
    current_state = state_snapshot.values

    if not current_state:
        raise ValueError(f"No saved state found for thread_id: {thread_id}")

    # Apply clarification responses
    current_state["clarification_responses"] = clarification_responses
    current_state["requires_clarification"] = False
    current_state["clarification_round"] += 1

    # Resume pipeline from current node
    final_state = await query_graph.ainvoke(current_state, config)

    # Check if another round of clarification needed
    if final_state.get("requires_clarification"):
        return {
            "status": "needs_clarification",
            "thread_id": thread_id,
            "questions": [q.dict() for q in final_state["clarification_questions"]],
            "round": final_state.get("clarification_round")
        }

    # Success
    return {
        "status": "success",
        "generated_query": final_state.get("generated_query"),
        "execution_id": final_state.get("execution_id"),
        "thinking": final_state.get("thinking", [])
    }
```

#### Step 4: Update API Endpoints

```python
# app/routes/query.py

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.services.query_generator import generate_query_with_hitl, continue_after_clarification
from app.schemas.query import QueryRequest, ClarificationRequest

router = APIRouter()

@router.post("/query")
async def generate_query(request: QueryRequest):
    """
    Generate database query from natural language.

    Returns:
    - 200: Success with generated query
    - 202: Needs clarification (with questions)
    - 500: Error
    """
    try:
        result = await generate_query_with_hitl(
            query=request.query,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            enable_clarification=request.enable_clarification,
            model=request.model,
            database_type=request.database_type
        )

        if result["status"] == "needs_clarification":
            return JSONResponse(
                status_code=202,  # Accepted - awaiting more info
                content={
                    "status": "needs_clarification",
                    "thread_id": result["thread_id"],
                    "questions": result["questions"],
                    "thinking": result.get("thinking", [])
                }
            )

        return {
            "status": "success",
            "generated_query": result["generated_query"],
            "execution_id": result["execution_id"],
            "thinking": result["thinking"],
            "thread_id": result["thread_id"]
        }

    except Exception as e:
        logger.error(f"Error generating query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/clarify")
async def submit_clarification(request: ClarificationRequest):
    """
    Submit clarification responses and continue query generation.

    Request body:
    {
        "thread_id": "thread_abc123",
        "responses": {
            "date_range": "Today",
            "symbol": "AAPL",
            "volume_threshold": 100000
        }
    }
    """
    try:
        result = await continue_after_clarification(
            thread_id=request.thread_id,
            clarification_responses=request.responses
        )

        if result["status"] == "needs_clarification":
            # Another round of clarification needed
            return JSONResponse(
                status_code=202,
                content={
                    "status": "needs_clarification",
                    "thread_id": result["thread_id"],
                    "questions": result["questions"],
                    "round": result.get("round", 2)
                }
            )

        return {
            "status": "success",
            "generated_query": result["generated_query"],
            "execution_id": result["execution_id"],
            "thinking": result["thinking"]
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing clarification: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
```

#### Step 5: Update Pydantic Schemas

```python
# app/schemas/query.py

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class QueryRequest(BaseModel):
    """Request to generate a query."""
    query: str
    user_id: str
    conversation_id: Optional[str] = None
    model: str = "gemini"
    database_type: str = "kdb"
    enable_clarification: bool = True  # NEW: Allow opt-out

class ClarificationRequest(BaseModel):
    """Request to submit clarification responses."""
    thread_id: str
    responses: Dict[str, Any]  # Flexible - any JSON structure

class ClarificationQuestion(BaseModel):
    """Single clarification question."""
    question: str
    type: str
    options: Optional[List[str]] = None
    context: Optional[str] = None
```

### Implementation Checklist

```
Phase 1: Core HITL (Day 1)
[ ] Update QueryGeneratorState with HITL fields
[ ] Create ambiguity_detector.py node
[ ] Update query_generator.py workflow
[ ] Update API routes
[ ] Update Pydantic schemas
[ ] Basic testing

Phase 2: Polish (Day 2)
[ ] Add pattern learning (optional)
[ ] Improve LLM prompts
[ ] Add metrics/logging
[ ] Documentation
[ ] Integration testing
```

### Pros & Cons

**Pros:**
- ✅ Minimal code changes (~100 lines)
- ✅ Low risk - doesn't touch existing nodes
- ✅ Fast implementation (1-2 days)
- ✅ Uses LangGraph's built-in features
- ✅ Easy to test incrementally
- ✅ Can be enhanced later

**Cons:**
- ❌ State management still in memory (MemorySaver)
- ❌ No persistent learning yet (can add later)
- ❌ Less sophisticated than DeepAgents

**When to Use:**
- You want HITL quickly
- You want to validate the concept first
- You prefer incremental changes
- You want minimal risk

---

## OPTION 2: Medium - Add DeepAgents Concepts

### Overview

Keep your LangGraph pipeline but adopt DeepAgents patterns:
- Add planning step (task list creation)
- Add sub-agent concept for specialized tasks
- Add persistent storage backend
- Still use your existing nodes

**Philosophy:** "Modernize gradually" - adopt patterns without full rewrite.

### Architecture Changes

```
[Planning Agent]
    ↓
Creates task list: ["classify intent", "check clarity", "retrieve schema", ...]
    ↓
[Task Executor] iterates through tasks
    ↓
For each task, routes to appropriate node/sub-agent
    ↓
Sub-agents:
- Clarification Sub-agent (asks user questions)
- Schema Sub-agent (retrieves schemas)
- Generator Sub-agent (generates queries)
```

### Key Additions

1. **Planning Layer:** LLM creates execution plan before starting
2. **Sub-agents:** Wrap existing nodes as isolated sub-agents
3. **Backend Strategy:** Use CompositeBackend for different storage needs
4. **Task Tool:** Add `task()` tool to delegate to sub-agents

### Implementation Sketch

```python
# app/services/agents/planning_agent.py

async def create_plan(state: QueryGeneratorState) -> QueryGeneratorState:
    """
    Planning agent creates a task list.
    Inspired by DeepAgents' TodoListMiddleware.
    """

    llm = state.llm

    prompt = f"""
Create a task plan for generating a database query:

User Query: {state.query}
Current Info: {get_current_info(state)}

Create a task list with these possible actions:
- classify_intent: Determine user's intent
- check_clarity: Check if query is clear (IMPORTANT: use this if ambiguous!)
- retrieve_schema: Get relevant database schema
- analyze_query: Analyze query requirements
- generate_query: Generate the database query
- validate_query: Validate generated query

Return JSON task list:
{{
  "tasks": [
    {{"action": "classify_intent", "reason": "..."}},
    {{"action": "check_clarity", "reason": "...", "critical": true}},
    ...
  ]
}}
"""

    result = await llm.ainvoke(prompt)
    plan = json.loads(result.content)

    state.task_plan = plan["tasks"]
    state.thinking.append(f"📋 Created plan with {len(plan['tasks'])} tasks")

    return state


# app/services/agents/task_executor.py

async def execute_task_plan(state: QueryGeneratorState) -> QueryGeneratorState:
    """
    Execute tasks from plan sequentially.
    Route to sub-agents as needed.
    """

    for task in state.task_plan:
        action = task["action"]

        if action == "classify_intent":
            state = await classify_intent(state)

        elif action == "check_clarity":
            state = await check_clarity_subagent(state)

            # If clarification needed, interrupt
            if state.requires_clarification:
                return state  # Halt execution

        elif action == "retrieve_schema":
            state = await retrieve_schema(state)

        # ... other actions ...

    return state


# Sub-agents as wrappers
async def check_clarity_subagent(state: QueryGeneratorState) -> QueryGeneratorState:
    """
    Sub-agent for clarity checking.
    Isolated context - can't affect main state directly.
    """

    # Create sub-state
    sub_state = {
        "query": state.query,
        "intent": state.intent_type,
        "schema": state.query_schema
    }

    # Run clarity check in isolation
    result = await clarity_checker_tool(sub_state)

    # Apply results back to main state
    if not result["is_clear"]:
        state.requires_clarification = True
        state.clarification_questions = result["questions"]

    return state
```

### Pros & Cons

**Pros:**
- ✅ More sophisticated architecture
- ✅ Better for complex workflows
- ✅ Easier to add new capabilities
- ✅ Learning patterns from modern agents

**Cons:**
- ❌ More code changes (~300 lines)
- ❌ More complex to understand
- ❌ Takes longer (1 week)
- ❌ Still not "real" DeepAgents

**When to Use:**
- Option 1 is working but you want to evolve
- You're planning major features later
- You like the DeepAgents patterns

---

## OPTION 3: Full DeepAgents - Complete Redesign

### Overview

Complete rewrite using the DeepAgents framework. Your existing nodes become tools, and DeepAgents middleware handles everything.

**Philosophy:** "Modern architecture from scratch" - leverage best practices.

### Architecture

```
┌─────────────────────────────────────────────┐
│          DeepAgents Framework               │
│                                             │
│  ┌────────────────────────────────────┐   │
│  │  Built-in Middleware (automatic)   │   │
│  │  - TodoListMiddleware              │   │
│  │  - HumanInTheLoopMiddleware        │   │
│  │  - FilesystemMiddleware            │   │
│  │  - SubAgentMiddleware              │   │
│  │  - SummarizationMiddleware         │   │
│  └────────────────────────────────────┘   │
│                                             │
│  Your Custom Tools:                         │
│  - classify_intent()                        │
│  - retrieve_schema()                        │
│  - check_query_clarity()  ← HITL trigger   │
│  - generate_database_query()                │
│  - validate_query()                         │
│                                             │
│  Sub-agents:                                │
│  - clarification-agent                      │
│  - schema-analyst                           │
│  - query-generator                          │
└─────────────────────────────────────────────┘
```

### Implementation Overview

```python
# app/services/deep_agents_query_generator.py

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

# Convert existing nodes to tools
def classify_intent(query: str, context: str = "") -> dict:
    """Tool: Classify query intent."""
    # Your existing logic
    pass

def check_query_clarity(query: str, intent: dict, schema: dict) -> dict:
    """
    Tool: Check if query is clear.

    Returns: {"is_clear": bool, "questions": [...]}

    This is the HITL trigger - DeepAgents will interrupt when is_clear=false.
    """
    # Your LLM-based clarity check
    pass

# Create agent
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[classify_intent, retrieve_schema, check_query_clarity, generate_query, validate_query],

    # Sub-agents for specialized tasks
    subagents=[
        {
            "name": "clarification-agent",
            "description": "Ask users for clarification",
            "tools": [check_query_clarity]
        }
    ],

    # HITL: Interrupt on clarity check failure
    interrupt_on={
        "check_query_clarity": {
            "allowed_decisions": ["approve", "provide_clarification"],
            "condition": lambda result: not result.get("is_clear", True)
        }
    },

    # Persistent storage
    backend=CompositeBackend(
        default=StateBackend(),
        routes={"/patterns/": StoreBackend(store=InMemoryStore())}
    ),

    system_prompt="""
You are a query generation assistant.

Workflow:
1. write_todos to create plan
2. classify_intent
3. check_query_clarity (ALWAYS do this!)
4. If not clear → ask user
5. generate_query
6. validate_query
"""
)

# Usage
async def generate_with_deepagents(query: str, user_id: str):
    config = {"configurable": {"thread_id": f"user_{user_id}"}}

    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": query}]},
        config=config
    ):
        # Check for interrupts
        if "interrupt" in chunk:
            return {
                "status": "needs_clarification",
                "questions": chunk["interrupt"]["result"]["questions"]
            }

    return {"status": "success", "query": extract_query(chunk)}
```

### Pros & Cons

**Pros:**
- ✅ State-of-the-art architecture
- ✅ Built-in HITL, planning, summarization
- ✅ Highly extensible
- ✅ Persistent storage
- ✅ Best long-term

**Cons:**
- ❌ Complete rewrite (~2000+ lines)
- ❌ High risk - everything changes
- ❌ Long implementation (3-4 weeks)
- ❌ Learning curve for team
- ❌ May be overkill

**When to Use:**
- Long-term strategic architecture change
- Building many agent features
- Team is comfortable with major rewrites

---

## Recommendation & Decision Matrix

### Quick Decision Guide

**Choose Option 1 if:**
- ✅ You want results quickly (1-2 days)
- ✅ You want to validate HITL concept first
- ✅ You prefer low-risk changes
- ✅ Your current pipeline works well
- ⭐ **RECOMMENDED FOR MOST CASES**

**Choose Option 2 if:**
- You've validated Option 1 and want to evolve
- You're planning more agent features
- You have 1 week to implement
- You like DeepAgents patterns but not full rewrite

**Choose Option 3 if:**
- This is a strategic long-term initiative
- You're building many AI features
- You have 3-4 weeks
- You're comfortable with major architecture changes

### Suggested Path

**Phase 1 (Now):** Implement Option 1
- Quick win, low risk
- Validate HITL improves accuracy
- Gather user feedback

**Phase 2 (3 months later):** Evaluate evolution
- If HITL is successful, consider Option 2
- Add planning, sub-agents gradually

**Phase 3 (6+ months later):** Consider Option 3
- If building many agent features, full DeepAgents might make sense
- By then, DeepAgents may have matured further

---

## Pattern Learning (Optional Enhancement)

All three options can add pattern learning for auto-resolution:

### Database Schema

```sql
-- Store successful clarification patterns
CREATE TABLE clarification_patterns (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    query_pattern TEXT NOT NULL,
    intent_type VARCHAR(50),
    ambiguity_type VARCHAR(50) NOT NULL,
    user_response TEXT NOT NULL,
    frequency INTEGER DEFAULT 1,
    last_used TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    success_rate FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_patterns_user_intent ON clarification_patterns(user_id, intent_type);
CREATE INDEX idx_patterns_frequency ON clarification_patterns(frequency DESC);
```

### Auto-Resolution Logic

```python
# app/services/pattern_matcher.py

async def check_learned_patterns(
    user_id: str,
    query: str,
    intent_type: str
) -> Optional[Dict[str, Any]]:
    """
    Check if we've seen this pattern before and can auto-resolve.

    Example: User always means "today" when saying "recent trades"
    """

    # Find matching patterns
    patterns = await db.fetch("""
        SELECT * FROM clarification_patterns
        WHERE user_id = $1
          AND intent_type = $2
          AND frequency >= 3
          AND success_rate > 0.8
        ORDER BY frequency DESC, last_used DESC
        LIMIT 5
    """, user_id, intent_type)

    # Fuzzy match query
    for pattern in patterns:
        similarity = calculate_similarity(query, pattern["query_pattern"])
        if similarity > 0.7:
            return {
                "auto_resolved": True,
                "response": pattern["user_response"],
                "confidence": similarity,
                "frequency": pattern["frequency"]
            }

    return None


# In ambiguity_detector.py, add before LLM call:
learned = await check_learned_patterns(state.user_id, state.query, state.intent_type)
if learned and learned["confidence"] > 0.8:
    # Auto-resolve!
    state.clarification_responses = json.loads(learned["response"])
    state.thinking.append(f"🧠 Auto-resolved using learned pattern (confidence: {learned['confidence']:.2f})")
    return state
```

---

## Frontend Integration Example

```typescript
// React/TypeScript example for all three options

interface QueryResponse {
  status: 'success' | 'needs_clarification' | 'error';
  thread_id?: string;
  questions?: Array<{
    question: string;
    type: 'multiple_choice' | 'text' | 'number';
    options?: string[];
    context?: string;
  }>;
  generated_query?: string;
}

async function submitQuery(query: string): Promise<QueryResponse> {
  const response = await fetch('/api/v1/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      user_id: currentUser.id,
      enable_clarification: true  // Can be toggled by user
    })
  });

  return await response.json();
}

async function submitClarification(
  threadId: string,
  responses: Record<string, any>
): Promise<QueryResponse> {
  const response = await fetch('/api/v1/query/clarify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      thread_id: threadId,
      responses
    })
  });

  return await response.json();
}

// UI Component
function QueryInterface() {
  const [clarificationState, setClarificationState] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleQuerySubmit = async (query: string) => {
    setLoading(true);
    const result = await submitQuery(query);
    setLoading(false);

    if (result.status === 'needs_clarification') {
      // Show clarification form
      setClarificationState({
        threadId: result.thread_id,
        questions: result.questions
      });
    } else if (result.status === 'success') {
      displayQueryResult(result.generated_query);
    }
  };

  const handleClarificationSubmit = async (responses: Record<string, any>) => {
    setLoading(true);
    const result = await submitClarification(
      clarificationState.threadId,
      responses
    );
    setLoading(false);

    if (result.status === 'needs_clarification') {
      // Another round needed
      setClarificationState({
        threadId: result.thread_id,
        questions: result.questions
      });
    } else if (result.status === 'success') {
      displayQueryResult(result.generated_query);
      setClarificationState(null);
    }
  };

  return (
    <div>
      {!clarificationState ? (
        <QueryInput onSubmit={handleQuerySubmit} loading={loading} />
      ) : (
        <ClarificationForm
          questions={clarificationState.questions}
          onSubmit={handleClarificationSubmit}
          loading={loading}
        />
      )}
    </div>
  );
}

function ClarificationForm({ questions, onSubmit, loading }) {
  const [responses, setResponses] = useState({});

  return (
    <div className="clarification-form">
      <h3>I need some clarification:</h3>
      {questions.map((q, idx) => (
        <div key={idx} className="question">
          <label>{q.question}</label>
          {q.context && <p className="context">{q.context}</p>}

          {q.type === 'multiple_choice' ? (
            <select onChange={(e) => setResponses({...responses, [`q${idx}`]: e.target.value})}>
              <option value="">Select...</option>
              {q.options.map(opt => <option key={opt} value={opt}>{opt}</option>)}
            </select>
          ) : q.type === 'number' ? (
            <input
              type="number"
              onChange={(e) => setResponses({...responses, [`q${idx}`]: e.target.value})}
            />
          ) : (
            <input
              type="text"
              onChange={(e) => setResponses({...responses, [`q${idx}`]: e.target.value})}
            />
          )}
        </div>
      ))}

      <button
        onClick={() => onSubmit(responses)}
        disabled={loading || Object.keys(responses).length < questions.length}
      >
        {loading ? 'Processing...' : 'Submit Clarification'}
      </button>
    </div>
  );
}
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_hitl_ambiguity.py

import pytest
from app.services.query_generation.nodes.ambiguity_detector import check_ambiguity_node
from app.services.query_generator_state import QueryGeneratorState

@pytest.mark.asyncio
async def test_ambiguity_detected_missing_date():
    """Test that missing date range triggers clarification."""

    state = QueryGeneratorState(
        query="Show me all trades for AAPL",
        intent_type="analytical",
        query_schema={"tables": [...]},
        enable_clarification=True,
        llm=get_test_llm()
    )

    result = await check_ambiguity_node(state)

    assert result.requires_clarification == True
    assert len(result.clarification_questions) > 0
    assert any("date" in q.question.lower() for q in result.clarification_questions)


@pytest.mark.asyncio
async def test_no_ambiguity_when_disabled():
    """Test that clarification is skipped when disabled."""

    state = QueryGeneratorState(
        query="Show me trades",
        enable_clarification=False,  # Disabled
        llm=get_test_llm()
    )

    result = await check_ambiguity_node(state)

    assert result.requires_clarification == False


@pytest.mark.asyncio
async def test_clear_query_no_clarification():
    """Test that clear queries don't trigger clarification."""

    state = QueryGeneratorState(
        query="Select all trades from trades table where date=2024-01-01 and sym=`AAPL",
        intent_type="analytical",
        query_schema={"tables": [...]},
        enable_clarification=True,
        llm=get_test_llm()
    )

    result = await check_ambiguity_node(state)

    assert result.requires_clarification == False
```

### Integration Tests

```python
# tests/test_hitl_integration.py

@pytest.mark.asyncio
async def test_full_hitl_flow():
    """Test complete HITL flow from ambiguous query to clarification to result."""

    # Step 1: Submit ambiguous query
    response1 = await client.post("/api/v1/query", json={
        "query": "show me high volume trades",
        "user_id": "test_user",
        "enable_clarification": True
    })

    assert response1.status_code == 202
    data1 = response1.json()
    assert data1["status"] == "needs_clarification"
    assert "thread_id" in data1
    assert len(data1["questions"]) > 0

    # Step 2: Submit clarification
    thread_id = data1["thread_id"]
    response2 = await client.post("/api/v1/query/clarify", json={
        "thread_id": thread_id,
        "responses": {
            "q0": "Today",
            "q1": "AAPL",
            "q2": "100000"
        }
    })

    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["status"] == "success"
    assert "generated_query" in data2
    assert "AAPL" in data2["generated_query"]
    assert "100000" in data2["generated_query"]


@pytest.mark.asyncio
async def test_multi_round_clarification():
    """Test multiple rounds of clarification."""

    # Simulate scenario where first clarification isn't enough
    # (Implementation would depend on specific logic)
    pass
```

---

## Metrics & Monitoring

### Key Metrics to Track

```python
# app/core/metrics.py

class HITLMetrics:
    """Track HITL performance metrics."""

    @staticmethod
    async def track_clarification_requested(
        user_id: str,
        query: str,
        ambiguity_types: List[str],
        num_questions: int
    ):
        """Track when clarification is requested."""
        await metrics_db.insert("clarification_requests", {
            "user_id": user_id,
            "query": query,
            "ambiguity_types": ambiguity_types,
            "num_questions": num_questions,
            "timestamp": datetime.utcnow()
        })

    @staticmethod
    async def track_clarification_provided(
        user_id: str,
        thread_id: str,
        response_time_seconds: float,
        resulted_in_success: bool
    ):
        """Track when user provides clarification."""
        await metrics_db.insert("clarification_responses", {
            "user_id": user_id,
            "thread_id": thread_id,
            "response_time": response_time_seconds,
            "success": resulted_in_success,
            "timestamp": datetime.utcnow()
        })
```

### Dashboard Queries

```sql
-- Clarification request rate
SELECT
    DATE(timestamp) as date,
    COUNT(*) as total_queries,
    SUM(CASE WHEN clarification_requested THEN 1 ELSE 0 END) as requests,
    AVG(CASE WHEN clarification_requested THEN 1.0 ELSE 0.0 END) * 100 as request_rate
FROM query_logs
WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- Most common ambiguities
SELECT
    ambiguity_type,
    COUNT(*) as count,
    AVG(num_questions) as avg_questions
FROM clarification_requests
GROUP BY ambiguity_type
ORDER BY count DESC;

-- User response time
SELECT
    AVG(response_time) as avg_seconds,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY response_time) as median_seconds,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time) as p95_seconds
FROM clarification_responses;

-- Success rate after clarification
SELECT
    COUNT(*) as total,
    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) * 100 as success_rate
FROM clarification_responses;
```

---

## Implementation Timeline

### Option 1 Timeline (Recommended)

**Day 1:**
- Morning: Update state models and schemas
- Afternoon: Implement ambiguity_detector node
- Evening: Update workflow and routing

**Day 2:**
- Morning: Update API endpoints
- Afternoon: Testing and bug fixes
- Evening: Documentation and PR

**Total: 2 days**

### Option 2 Timeline

**Week 1:**
- Days 1-2: Planning layer implementation
- Days 3-4: Sub-agent architecture
- Day 5: Backend strategy and storage

**Total: 1 week**

### Option 3 Timeline

**Weeks 1-2:** Architecture and tool conversion
**Week 3:** DeepAgents integration
**Week 4:** Testing and refinement

**Total: 4 weeks**

---

## Next Steps

When you return to implement:

1. **Read this document fully** to refresh context
2. **Review DeepAgents repo** (link at top) for latest updates
3. **Decide which option** based on timeline and risk tolerance
4. **Create implementation branch** from current feature branch
5. **Follow implementation checklist** for chosen option
6. **Test incrementally** - don't implement everything at once
7. **Gather user feedback** before adding more complexity

---

## Open Questions & Future Considerations

- [ ] Should clarification timeout? (e.g., thread_id expires after 15 minutes)
- [ ] How to handle partial clarification responses?
- [ ] Should we show "estimated accuracy" before/after clarification?
- [ ] Integration with existing retry mechanism?
- [ ] How to display clarification history in conversation UI?
- [ ] Rate limiting on clarification requests per user?
- [ ] A/B testing framework for measuring HITL effectiveness?
- [ ] Support for "skip this question" option?

---

## References & Resources

1. **DeepAgents Framework**
   - GitHub: https://github.com/langchain-ai/deepagents
   - Blog: https://blog.langchain.com (search for DeepAgents)

2. **LangGraph Documentation**
   - Checkpointing: https://langchain-ai.github.io/langgraph/how-tos/persistence/
   - Conditional Edges: https://langchain-ai.github.io/langgraph/how-tos/branching/

3. **Your Current Codebase**
   - `app/services/query_generator.py` - Main pipeline
   - `app/services/query_generation/nodes/` - Node implementations
   - `app/routes/query.py` - API endpoints

---

**END OF DOCUMENT**

When you come back with this document, I will have full context on:
- The problem and motivation
- Your current architecture
- All three implementation options
- Detailed code examples
- Recommended path forward
- Timeline and effort estimates

Good luck with implementation! 🚀

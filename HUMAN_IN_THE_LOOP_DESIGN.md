# Human-in-the-Loop (HITL) Query Clarification System

## Overview

This document outlines the design for a Human-in-the-Loop clarification system that improves query accuracy by proactively asking users for clarification when queries are ambiguous or lack necessary context.

## Problem Statement

**Current Flow:**
```
User Query → Process All Nodes → Generate Query → [FAILURE: "I didn't understand"]
         ↓
User guesses what went wrong → Retry → Maybe success
```

**Issues:**
- System processes through all nodes before failing
- User wastes time guessing what information is missing
- Multiple failed attempts consume LLM tokens and time
- Poor user experience with reactive failure handling

**Goal:**
Detect ambiguity early and ask clarifying questions **before** generating incorrect queries.

## Design Decisions

Based on requirements discussion:

### 1. **Where to Ask for Clarification**

**Two-Stage Approach:**

**Stage 1: Intent Level (Early Detection)**
- Check if intent itself is ambiguous
- Fast failure for completely unclear queries
- Example: "Show me data" (What data? From where?)

**Stage 2: Schema Level (Context-Aware) ⭐ PRIMARY FOCUS**
- After schema retrieval, with full context
- Detect ambiguities in:
  - Column references (which "price"? which "volume"?)
  - Table selection (multiple matching tables)
  - Missing required filters (date range, symbols)
  - Vague qualifiers ("high", "low", "recent")

**Why Schema Level is Primary:**
- Better context for generating meaningful questions
- Can reference specific schema elements
- Higher quality clarifications
- Reduced false positives

### 2. **Latency Tolerance**

✅ **1 extra round trip is acceptable**

Trade-off Analysis:
```
Without HITL:
Query → 3s → Wrong result → Retry → 3s → Maybe correct = 6-10+ seconds

With HITL:
Query → 2s → Clarification (user responds) → 3s → Correct result = 5-8 seconds
+ Better accuracy + Better UX
```

### 3. **Types of Ambiguities**

**Open-ended approach** - System should detect ANY type of ambiguity:
- Date/time ambiguities ("recent", "today", missing ranges)
- Symbol ambiguities (which ticker?)
- Column ambiguities (multiple columns match description)
- Threshold ambiguities ("high volume" - how high?)
- Aggregation ambiguities ("average" - over what period?)
- Table ambiguities (multiple tables could match)
- Conditional ambiguities (unclear filtering logic)

**Strategy:** Use LLM to intelligently detect ambiguities rather than hard-coded rules.

### 4. **Automatic vs Opt-in**

**Hybrid Approach:**

```python
# Default: Automatic clarification enabled
{
  "query": "show me high volume trades",
  "enable_clarification": true  # default
}

# Power users can skip
{
  "query": "show me high volume trades",
  "enable_clarification": false,
  "skip_clarification": true
}
```

**Benefits:**
- Default users get accuracy improvement
- Power users can opt-out for speed
- A/B testing capability
- Can adjust based on user history

### 5. **Learning & Pattern Storage**

✅ **Yes - Store and learn from clarification patterns**

**What to Learn:**
- User-specific preferences (e.g., user always means "today" when saying "recent")
- Common ambiguities per query type
- Schema-specific patterns (AAPL vs TSLA most queried)
- Successful clarification → query mappings

**Storage Strategy:**
```sql
CREATE TABLE clarification_patterns (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100),
    query_pattern TEXT,
    ambiguity_type VARCHAR(50),
    clarification_question TEXT,
    user_response TEXT,
    resulted_in_success BOOLEAN,
    frequency INT DEFAULT 1,
    last_used TIMESTAMP,
    metadata JSONB
);

CREATE INDEX idx_clarification_user ON clarification_patterns(user_id);
CREATE INDEX idx_clarification_pattern ON clarification_patterns(query_pattern);
```

## System Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  Intent Classifier     │
            └────────┬───────────────┘
                     │
                     ▼
            ┌────────────────────────┐
            │  Confidence < 0.6?     │──Yes──┐
            └────────┬───────────────┘       │
                     │ No                    │
                     ▼                       ▼
            ┌────────────────────────┐  ┌──────────────────────┐
            │  Schema Retrieval      │  │ Ask Intent           │
            └────────┬───────────────┘  │ Clarification        │
                     │                  └──────────────────────┘
                     ▼
            ┌────────────────────────┐
            │ Ambiguity Detector     │  ⭐ PRIMARY CHECKPOINT
            │ (LLM-powered)          │
            └────────┬───────────────┘
                     │
                     ▼
            ┌────────────────────────┐
            │ Check Learning DB      │
            │ (Auto-resolve if known)│
            └────────┬───────────────┘
                     │
                ┌────┴────┐
                │         │
         Ambiguous?    Clear?
                │         │
                ▼         ▼
    ┌──────────────┐  ┌──────────────┐
    │ Generate     │  │ Continue to  │
    │ Clarifying   │  │ Query        │
    │ Questions    │  │ Generation   │
    └──────┬───────┘  └──────────────┘
           │
           ▼
    ┌──────────────────┐
    │ Return to User   │
    │ (Status: 202)    │
    └──────┬───────────┘
           │
           ▼
    ┌──────────────────┐
    │ User Responds    │
    └──────┬───────────┘
           │
           ▼
    ┌──────────────────┐
    │ Store Pattern    │
    │ Continue Query   │
    │ Generation       │
    └──────────────────┘
```

## Implementation Components

### 1. Ambiguity Detector Node

```python
# app/services/query_generation/nodes/ambiguity_detector.py

from typing import Dict, Any, List, Optional
from langchain.prompts import ChatPromptTemplate
from app.services.query_generator_state import QueryGeneratorState

AMBIGUITY_DETECTION_PROMPT = """
You are an expert at detecting ambiguities in natural language database queries.

User Query: {query}
Intent Type: {intent_type}
Available Schema:
{schema}

Conversation Context:
{conversation_context}

Analyze if this query has any ambiguities or missing critical information that would prevent generating an accurate database query.

Consider:
1. **Column Ambiguities**: Multiple columns could match the description (e.g., "price" → bid_price, ask_price, last_price)
2. **Missing Filters**: Required information not specified (date range, symbols, thresholds)
3. **Vague Terms**: Unclear qualifiers ("high", "recent", "best", "top" without specifics)
4. **Table Ambiguities**: Multiple tables could satisfy the query
5. **Aggregation Ambiguities**: Unclear time periods or grouping
6. **Threshold Ambiguities**: Comparative terms without values ("greater than", "more than")

Return JSON:
{{
  "needs_clarification": true/false,
  "confidence": 0.0-1.0,
  "ambiguities": [
    {{
      "type": "column_ambiguity" | "missing_filter" | "vague_term" | "threshold",
      "description": "Brief description of the ambiguity",
      "affected_part": "The part of query that's ambiguous",
      "possible_values": ["option1", "option2", ...],  // if applicable
      "severity": "critical" | "important" | "minor"
    }}
  ],
  "reasoning": "Why clarification is or isn't needed"
}}

Be conservative - only ask for clarification if truly necessary to generate accurate query.
"""

async def detect_ambiguities(state: QueryGeneratorState) -> QueryGeneratorState:
    """
    Detect ambiguities in the query after schema retrieval.
    This is the primary checkpoint for HITL clarification.
    """

    # Skip if clarification disabled
    if not state.enable_clarification:
        state.thinking.append("⏭️  Clarification disabled, skipping ambiguity detection")
        return state

    # Skip if already processing clarification response
    if state.clarification_responses:
        state.thinking.append("✓ Processing clarification responses, skipping detection")
        return state

    # Check learned patterns first
    learned_resolution = await check_learned_patterns(state)
    if learned_resolution:
        state.thinking.append(f"🧠 Auto-resolved using learned pattern: {learned_resolution}")
        apply_learned_resolution(state, learned_resolution)
        return state

    try:
        llm = state.llm

        # Format schema for prompt
        schema_text = format_schema_for_ambiguity_detection(state.query_schema)

        # Get conversation context
        conversation_context = format_conversation_context(state.conversation_history)

        # Create prompt
        prompt = ChatPromptTemplate.from_template(AMBIGUITY_DETECTION_PROMPT)
        chain = prompt | llm

        # Invoke LLM
        response = await chain.ainvoke({
            "query": state.query,
            "intent_type": state.intent_type,
            "schema": schema_text,
            "conversation_context": conversation_context
        })

        # Parse response
        import json
        result = json.loads(response.content)

        if result["needs_clarification"]:
            # Generate user-friendly questions from ambiguities
            questions = generate_clarifying_questions(result["ambiguities"], state)

            state.requires_clarification = True
            state.clarification_questions = questions
            state.ambiguity_analysis = result

            state.thinking.append(f"❓ Detected {len(result['ambiguities'])} ambiguities requiring clarification")

            # Short-circuit the pipeline - return to user for clarification
            state.should_halt = True
        else:
            state.thinking.append("✓ No significant ambiguities detected, proceeding")

        return state

    except Exception as e:
        logger.error(f"Error in ambiguity detection: {str(e)}", exc_info=True)
        state.thinking.append(f"⚠️  Ambiguity detection error: {str(e)}")
        # Continue without clarification on error
        return state


def generate_clarifying_questions(
    ambiguities: List[Dict[str, Any]],
    state: QueryGeneratorState
) -> List[Dict[str, Any]]:
    """
    Convert detected ambiguities into user-friendly clarifying questions.
    """
    questions = []

    for idx, ambiguity in enumerate(ambiguities):
        if ambiguity["severity"] == "minor":
            continue  # Skip minor ambiguities

        question = {
            "id": f"q{idx + 1}",
            "type": determine_question_type(ambiguity),
            "ambiguity_type": ambiguity["type"],
            "severity": ambiguity["severity"]
        }

        # Generate question text based on ambiguity type
        if ambiguity["type"] == "column_ambiguity":
            question["question"] = f"You mentioned '{ambiguity['affected_part']}'. Which specific column do you mean?"
            question["options"] = ambiguity.get("possible_values", [])
            question["type"] = "multiple_choice"

        elif ambiguity["type"] == "missing_filter":
            question["question"] = f"What {ambiguity['affected_part']} should I use?"
            question["type"] = "text"
            question["placeholder"] = get_placeholder_for_filter(ambiguity)

        elif ambiguity["type"] == "vague_term":
            question["question"] = f"You used '{ambiguity['affected_part']}'. Can you specify what you mean?"
            question["type"] = "text"
            question["context"] = ambiguity["description"]

        elif ambiguity["type"] == "threshold":
            question["question"] = f"What threshold for '{ambiguity['affected_part']}'?"
            question["type"] = "number"
            question["unit"] = extract_unit_from_context(ambiguity, state)

        questions.append(question)

    return questions


async def check_learned_patterns(state: QueryGeneratorState) -> Optional[Dict[str, Any]]:
    """
    Check if we've seen this pattern before and can auto-resolve.
    """
    from app.services.clarification_pattern_manager import ClarificationPatternManager

    pattern_mgr = ClarificationPatternManager()

    # Look for matching patterns for this user
    pattern = await pattern_mgr.find_matching_pattern(
        user_id=state.user_id,
        query=state.query,
        intent_type=state.intent_type,
        schema=state.query_schema
    )

    if pattern and pattern["confidence"] > 0.8:
        return pattern["resolution"]

    return None
```

### 2. Updated State Model

```python
# app/services/query_generator_state.py

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ClarificationQuestion(BaseModel):
    """Model for a clarification question."""
    id: str
    type: str  # "multiple_choice", "text", "number", "date_range"
    question: str
    ambiguity_type: str
    severity: str
    options: Optional[List[str]] = None
    placeholder: Optional[str] = None
    unit: Optional[str] = None
    context: Optional[str] = None

class QueryGeneratorState(BaseModel):
    """Extended state with HITL clarification fields."""

    # ... existing fields ...

    # Clarification fields
    enable_clarification: bool = True
    requires_clarification: bool = False
    clarification_questions: List[ClarificationQuestion] = Field(default_factory=list)
    clarification_responses: Optional[Dict[str, Any]] = None
    clarification_round: int = 0
    max_clarification_rounds: int = 2  # Prevent infinite loops
    ambiguity_analysis: Optional[Dict[str, Any]] = None
    should_halt: bool = False  # Halt pipeline for clarification
    clarification_id: Optional[str] = None  # Track clarification session
```

### 3. API Endpoints

```python
# app/routes/query.py

from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.schemas.query import QueryRequest, QueryResponse, ClarificationRequest, ClarificationResponse
from uuid import uuid4

@router.post("/query", response_model=QueryResponse, status_code=200)
async def generate_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Generate a database query from natural language.

    Returns:
    - 200 with generated query if successful
    - 202 with clarification questions if clarification needed
    """
    try:
        # Initialize state
        state = await initialize_query_state(request)

        # Run query generation pipeline
        result = await query_generator.generate(state)

        # Check if clarification is needed
        if result.requires_clarification:
            # Generate unique clarification ID
            clarification_id = f"clf_{uuid4().hex[:12]}"

            # Store state for later retrieval
            await store_clarification_state(clarification_id, result)

            return JSONResponse(
                status_code=202,  # Accepted - needs clarification
                content={
                    "status": "needs_clarification",
                    "clarification_id": clarification_id,
                    "questions": [q.dict() for q in result.clarification_questions],
                    "ambiguity_analysis": result.ambiguity_analysis,
                    "thinking": result.thinking[-5:]  # Last 5 thinking steps
                }
            )

        # Normal success response
        return {
            "status": "success",
            "generated_query": result.generated_query,
            "execution_id": result.execution_id,
            "thinking": result.thinking,
            # ... other fields
        }

    except Exception as e:
        logger.error(f"Error in query generation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/clarify", response_model=QueryResponse)
async def submit_clarification(request: ClarificationRequest):
    """
    Submit clarification responses and continue query generation.

    Request:
    {
        "clarification_id": "clf_abc123",
        "responses": {
            "q1": "Today",
            "q2": "AAPL"
        }
    }
    """
    try:
        # Retrieve stored state
        state = await retrieve_clarification_state(request.clarification_id)
        if not state:
            raise HTTPException(status_code=404, detail="Clarification session not found or expired")

        # Apply clarification responses
        state.clarification_responses = request.responses
        state.clarification_round += 1
        state.requires_clarification = False
        state.should_halt = False

        # Store pattern for learning
        background_tasks.add_task(
            store_clarification_pattern,
            state=state,
            responses=request.responses
        )

        # Continue query generation from where we left off
        result = await query_generator.continue_from_clarification(state)

        # Check if another round of clarification is needed
        if result.requires_clarification and result.clarification_round < result.max_clarification_rounds:
            # Generate new clarification ID
            new_clarification_id = f"clf_{uuid4().hex[:12]}"
            await store_clarification_state(new_clarification_id, result)

            return JSONResponse(
                status_code=202,
                content={
                    "status": "needs_clarification",
                    "clarification_id": new_clarification_id,
                    "questions": [q.dict() for q in result.clarification_questions],
                    "round": result.clarification_round
                }
            )

        # Success
        return {
            "status": "success",
            "generated_query": result.generated_query,
            "execution_id": result.execution_id,
            "thinking": result.thinking,
            # ... other fields
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing clarification: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
```

### 4. Pattern Learning Service

```python
# app/services/clarification_pattern_manager.py

from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib
import json

class ClarificationPatternManager:
    """
    Manages learning and retrieval of clarification patterns.
    """

    async def store_pattern(
        self,
        user_id: str,
        query: str,
        intent_type: str,
        ambiguity_type: str,
        clarification_question: str,
        user_response: str,
        resulted_in_success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a clarification pattern for learning."""

        # Generate pattern hash for deduplication
        pattern_key = self._generate_pattern_key(user_id, query, ambiguity_type)

        conn = await db_pool.get_connection()
        try:
            # Check if pattern exists
            existing = await conn.fetchrow(
                """
                SELECT id, frequency FROM clarification_patterns
                WHERE user_id = $1 AND pattern_key = $2
                """,
                user_id, pattern_key
            )

            if existing:
                # Update frequency and metadata
                await conn.execute(
                    """
                    UPDATE clarification_patterns
                    SET frequency = frequency + 1,
                        last_used = $1,
                        resulted_in_success = $2,
                        user_response = $3,
                        metadata = $4
                    WHERE id = $5
                    """,
                    datetime.utcnow(),
                    resulted_in_success,
                    user_response,
                    json.dumps(metadata or {}),
                    existing["id"]
                )
            else:
                # Insert new pattern
                await conn.execute(
                    """
                    INSERT INTO clarification_patterns
                    (user_id, pattern_key, query_pattern, intent_type, ambiguity_type,
                     clarification_question, user_response, resulted_in_success,
                     frequency, last_used, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 1, $9, $10)
                    """,
                    user_id,
                    pattern_key,
                    query,
                    intent_type,
                    ambiguity_type,
                    clarification_question,
                    user_response,
                    resulted_in_success,
                    datetime.utcnow(),
                    json.dumps(metadata or {})
                )

            logger.info(f"Stored clarification pattern for user {user_id}")

        finally:
            await db_pool.release_connection(conn)

    async def find_matching_pattern(
        self,
        user_id: str,
        query: str,
        intent_type: str,
        schema: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Find a matching learned pattern for auto-resolution.

        Returns pattern with confidence score if found.
        """
        conn = await db_pool.get_connection()
        try:
            # Find patterns with high frequency and recent usage
            patterns = await conn.fetch(
                """
                SELECT *
                FROM clarification_patterns
                WHERE user_id = $1
                  AND intent_type = $2
                  AND resulted_in_success = true
                  AND frequency >= 3
                ORDER BY frequency DESC, last_used DESC
                LIMIT 10
                """,
                user_id,
                intent_type
            )

            if not patterns:
                return None

            # Use fuzzy matching to find best pattern
            best_match = None
            best_score = 0.0

            for pattern in patterns:
                score = self._calculate_similarity(query, pattern["query_pattern"])
                if score > best_score and score > 0.7:  # Threshold
                    best_score = score
                    best_match = pattern

            if best_match:
                return {
                    "resolution": {
                        "ambiguity_type": best_match["ambiguity_type"],
                        "response": best_match["user_response"]
                    },
                    "confidence": best_score,
                    "frequency": best_match["frequency"]
                }

            return None

        finally:
            await db_pool.release_connection(conn)

    def _generate_pattern_key(self, user_id: str, query: str, ambiguity_type: str) -> str:
        """Generate a unique key for pattern deduplication."""
        # Normalize query
        normalized = query.lower().strip()
        key_string = f"{user_id}:{normalized}:{ambiguity_type}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries."""
        # Simple token-based similarity (can be improved with embeddings)
        tokens1 = set(query1.lower().split())
        tokens2 = set(query2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union)
```

### 5. Database Migration

```sql
-- scripts/db_scripts/03_add_clarification_patterns.sql

-- Table for storing learned clarification patterns
CREATE TABLE IF NOT EXISTS clarification_patterns (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    pattern_key VARCHAR(32) NOT NULL,  -- MD5 hash for deduplication
    query_pattern TEXT NOT NULL,
    intent_type VARCHAR(50),
    ambiguity_type VARCHAR(50) NOT NULL,
    clarification_question TEXT NOT NULL,
    user_response TEXT NOT NULL,
    resulted_in_success BOOLEAN DEFAULT false,
    frequency INTEGER DEFAULT 1,
    last_used TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Indexes for fast lookups
CREATE INDEX idx_clarification_user_intent ON clarification_patterns(user_id, intent_type);
CREATE INDEX idx_clarification_pattern_key ON clarification_patterns(pattern_key);
CREATE INDEX idx_clarification_frequency ON clarification_patterns(frequency DESC, last_used DESC);
CREATE INDEX idx_clarification_success ON clarification_patterns(resulted_in_success) WHERE resulted_in_success = true;

-- Table for storing temporary clarification sessions
CREATE TABLE IF NOT EXISTS clarification_sessions (
    clarification_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    conversation_id VARCHAR(100),
    state_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    is_resolved BOOLEAN DEFAULT false
);

-- Index for session cleanup
CREATE INDEX idx_clarification_sessions_expiry ON clarification_sessions(expires_at);

-- Cleanup expired sessions (can be run as a cron job)
CREATE OR REPLACE FUNCTION cleanup_expired_clarification_sessions()
RETURNS void AS $$
BEGIN
    DELETE FROM clarification_sessions
    WHERE expires_at < CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;
```

## Integration with Existing Pipeline

### Modified Query Generator Flow

```python
# app/services/query_generator.py

async def generate(self, state: QueryGeneratorState) -> QueryGeneratorState:
    """Modified to support HITL clarification."""

    # Step 1: Intent Classification
    state = await classify_intent(state)

    # Early clarification check (if intent confidence too low)
    if state.intent_confidence < 0.6 and state.enable_clarification:
        state = await request_intent_clarification(state)
        if state.requires_clarification:
            return state  # Halt and return to user

    # Step 2: Schema Retrieval
    state = await retrieve_relevant_schemas(state)

    # Step 3: Ambiguity Detection ⭐ PRIMARY CHECKPOINT
    state = await detect_ambiguities(state)
    if state.requires_clarification:
        return state  # Halt and return to user

    # Step 4: Continue with normal flow
    state = await analyze_query(state)
    state = await generate_query_node(state)
    state = await validate_query(state)

    return state

async def continue_from_clarification(self, state: QueryGeneratorState) -> QueryGeneratorState:
    """
    Continue pipeline execution after receiving clarification responses.
    """

    # Apply clarification responses to state
    state = apply_clarifications_to_state(state)

    # Continue from query generation (skip earlier nodes)
    state = await generate_query_node(state)
    state = await validate_query(state)

    return state
```

## Frontend Integration Example

```typescript
// Example React/TypeScript implementation

interface ClarificationQuestion {
  id: string;
  type: 'multiple_choice' | 'text' | 'number' | 'date_range';
  question: string;
  options?: string[];
  placeholder?: string;
}

interface QueryResponse {
  status: 'success' | 'needs_clarification' | 'error';
  clarification_id?: string;
  questions?: ClarificationQuestion[];
  generated_query?: string;
}

async function submitQuery(query: string): Promise<QueryResponse> {
  const response = await fetch('/api/v1/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      enable_clarification: true
    })
  });

  return await response.json();
}

async function submitClarification(
  clarificationId: string,
  responses: Record<string, any>
): Promise<QueryResponse> {
  const response = await fetch('/api/v1/query/clarify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      clarification_id: clarificationId,
      responses
    })
  });

  return await response.json();
}

// UI Component
function QueryInterface() {
  const [clarificationState, setClarificationState] = useState<any>(null);

  const handleQuerySubmit = async (query: string) => {
    const result = await submitQuery(query);

    if (result.status === 'needs_clarification') {
      // Show clarification UI
      setClarificationState({
        clarificationId: result.clarification_id,
        questions: result.questions
      });
    } else if (result.status === 'success') {
      // Show query result
      displayQueryResult(result.generated_query);
    }
  };

  const handleClarificationSubmit = async (responses: Record<string, any>) => {
    const result = await submitClarification(
      clarificationState.clarificationId,
      responses
    );

    if (result.status === 'success') {
      displayQueryResult(result.generated_query);
      setClarificationState(null);
    }
  };

  return (
    <div>
      {!clarificationState ? (
        <QueryInput onSubmit={handleQuerySubmit} />
      ) : (
        <ClarificationForm
          questions={clarificationState.questions}
          onSubmit={handleClarificationSubmit}
        />
      )}
    </div>
  );
}
```

## Testing Strategy

### Unit Tests

```python
# tests/test_ambiguity_detection.py

import pytest
from app.services.query_generation.nodes.ambiguity_detector import detect_ambiguities

@pytest.mark.asyncio
async def test_column_ambiguity_detection():
    """Test detection of ambiguous column references."""
    state = create_test_state(
        query="Show me the price for AAPL",
        schema={
            "tables": [{
                "name": "trades",
                "columns": [
                    {"name": "bid_price", "type": "float"},
                    {"name": "ask_price", "type": "float"},
                    {"name": "last_price", "type": "float"}
                ]
            }]
        }
    )

    result = await detect_ambiguities(state)

    assert result.requires_clarification == True
    assert len(result.clarification_questions) == 1
    assert result.clarification_questions[0].ambiguity_type == "column_ambiguity"
    assert "price" in result.clarification_questions[0].question.lower()

@pytest.mark.asyncio
async def test_missing_date_filter():
    """Test detection of missing date range."""
    state = create_test_state(
        query="Show me all trades for AAPL",
        intent_type="analytical"
    )

    result = await detect_ambiguities(state)

    assert result.requires_clarification == True
    questions = [q for q in result.clarification_questions if q.ambiguity_type == "missing_filter"]
    assert len(questions) > 0

@pytest.mark.asyncio
async def test_learned_pattern_auto_resolution():
    """Test that learned patterns auto-resolve ambiguities."""
    # Store a pattern
    await pattern_manager.store_pattern(
        user_id="user123",
        query="show me recent trades",
        intent_type="analytical",
        ambiguity_type="vague_term",
        clarification_question="What time range?",
        user_response="Today",
        resulted_in_success=True
    )

    # Simulate same query
    state = create_test_state(
        query="show me recent trades",
        user_id="user123"
    )

    result = await detect_ambiguities(state)

    # Should auto-resolve without requiring clarification
    assert result.requires_clarification == False
    assert "Auto-resolved" in " ".join(result.thinking)
```

### Integration Tests

```python
# tests/test_hitl_integration.py

@pytest.mark.asyncio
async def test_full_clarification_flow():
    """Test complete HITL flow from query to clarification to result."""

    # Step 1: Submit ambiguous query
    response1 = await client.post("/api/v1/query", json={
        "query": "show me high volume trades",
        "enable_clarification": True
    })

    assert response1.status_code == 202
    data1 = response1.json()
    assert data1["status"] == "needs_clarification"
    assert "clarification_id" in data1
    assert len(data1["questions"]) > 0

    # Step 2: Submit clarification responses
    clarification_id = data1["clarification_id"]
    responses = {
        "q1": "Today",
        "q2": "AAPL",
        "q3": "100000"
    }

    response2 = await client.post("/api/v1/query/clarify", json={
        "clarification_id": clarification_id,
        "responses": responses
    })

    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["status"] == "success"
    assert "generated_query" in data2
    assert "AAPL" in data2["generated_query"]
```

## Performance Considerations

### Latency Budget

| Stage | Estimated Time | Notes |
|-------|---------------|-------|
| Intent Classification | 500-800ms | Existing |
| Schema Retrieval | 200-400ms | Existing |
| **Ambiguity Detection** | **600-900ms** | **New - using fast LLM** |
| User Response Time | 5-30s | Human in loop |
| Query Generation | 2-4s | Existing |
| **Total (with clarification)** | **~8-35s** | **Including user response** |
| **Total (without clarification)** | **~3-5s** | **No ambiguity detected** |

### Optimization Strategies

1. **Use Fast LLM for Ambiguity Detection**
   - Gemini Flash or Claude Haiku
   - Lower cost and latency
   - Sufficient for classification task

2. **Parallel Processing**
   - Run ambiguity detection in parallel with other analysis
   - Pre-compute common ambiguities

3. **Caching**
   - Cache ambiguity detection results for similar queries
   - Cache learned patterns in memory

4. **Progressive Clarification**
   - Ask most critical questions first
   - Skip minor ambiguities

## Rollout Plan

### Phase 1: Foundation (Week 1-2)
- [ ] Database schema for patterns and sessions
- [ ] State model extensions
- [ ] Basic ambiguity detector (rule-based)
- [ ] API endpoints (query, clarify)

### Phase 2: LLM-Powered Detection (Week 2-3)
- [ ] LLM-based ambiguity detection
- [ ] Question generation logic
- [ ] Integration with existing pipeline
- [ ] Unit tests

### Phase 3: Learning & Optimization (Week 3-4)
- [ ] Pattern storage and retrieval
- [ ] Auto-resolution logic
- [ ] Performance optimization
- [ ] Integration tests

### Phase 4: Production Rollout (Week 4-5)
- [ ] Frontend integration
- [ ] A/B testing framework
- [ ] Monitoring and metrics
- [ ] Documentation

## Metrics & Monitoring

### Success Metrics

1. **Accuracy Improvement**
   - Successful query generation rate (before vs after)
   - Retry rate reduction

2. **User Experience**
   - Average clarification response time
   - Clarification abandonment rate
   - User satisfaction scores

3. **Efficiency**
   - Auto-resolution rate (learned patterns)
   - Average clarifications per query
   - Token usage (with vs without HITL)

4. **Learning Effectiveness**
   - Pattern database growth rate
   - Pattern reuse frequency
   - Pattern success rate

### Monitoring Dashboard

```sql
-- Key metrics queries

-- Clarification request rate
SELECT
    DATE(created_at) as date,
    COUNT(*) as total_queries,
    SUM(CASE WHEN is_clarification_requested THEN 1 ELSE 0 END) as clarification_requests,
    AVG(CASE WHEN is_clarification_requested THEN 1.0 ELSE 0.0 END) * 100 as clarification_rate
FROM query_logs
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Auto-resolution effectiveness
SELECT
    ambiguity_type,
    COUNT(*) as total_patterns,
    AVG(frequency) as avg_reuse,
    SUM(CASE WHEN resulted_in_success THEN 1 ELSE 0 END) as successful_resolutions
FROM clarification_patterns
GROUP BY ambiguity_type;

-- Most common ambiguities
SELECT
    ambiguity_type,
    COUNT(*) as occurrence_count,
    AVG(CASE WHEN resulted_in_success THEN 1.0 ELSE 0.0 END) * 100 as success_rate
FROM clarification_sessions
GROUP BY ambiguity_type
ORDER BY occurrence_count DESC;
```

## Next Steps

1. **Review this design** and provide feedback
2. **Prioritize implementation phases** based on business needs
3. **Set up development environment** on this branch
4. **Begin Phase 1 implementation** when ready

## Open Questions

- [ ] Should we support multi-turn clarification (clarification → response → more clarification)?
- [ ] How long should clarification sessions be cached? (suggested: 15 minutes)
- [ ] Should we allow users to provide feedback on clarification questions themselves?
- [ ] Integration with existing conversation history - how to display clarifications?
- [ ] Should there be user settings to control clarification aggressiveness?

---

**Document Version:** 1.0
**Last Updated:** 2025-11-23
**Status:** Design Phase - Ready for Review
**Branch:** `feature-human-in-the-loop-clarification`

# Query Generator Server - Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Patterns](#architecture-patterns)
3. [Directory Structure](#directory-structure)
4. [Core Query Generation Pipeline](#core-query-generation-pipeline)
5. [Key Services](#key-services)
6. [Data Flow](#data-flow)
7. [Database Schema](#database-schema)
8. [External Integrations](#external-integrations)
9. [Configuration Management](#configuration-management)
10. [API Structure](#api-structure)
11. [State Management](#state-management)
12. [Caching Strategy](#caching-strategy)
13. [Error Handling](#error-handling)

---

## System Overview

### What is Query Generator Server?

Query Generator Server is a **FastAPI-based REST API server** that converts natural language questions into database-specific queries (primarily KDB/Q syntax). It leverages Large Language Models (LLMs) and vector search to understand user intent and generate accurate, context-aware database queries.

### Key Capabilities

- **Natural Language to Query**: Converts user questions into executable database queries
- **Multi-LLM Support**: Supports Google Gemini and Anthropic Claude models
- **Intelligent Pipeline**: Uses LangGraph for sophisticated multi-step query generation
- **Vector-Powered Schema Search**: Semantic search to find relevant tables and schemas
- **Conversation Context**: Maintains multi-turn conversations for follow-up questions
- **Query Validation & Refinement**: Automatic validation and iterative refinement
- **Real-time Processing**: WebSocket support for streaming responses
- **Business Glossary**: Integration with domain-specific terminology and directives

### Technology Stack

**Core Framework**
- **FastAPI** (0.115.11) - Modern async web framework
- **Uvicorn** (0.34.0) - ASGI server
- **Python 3.11+** - Programming language

**LLM & AI Orchestration**
- **LangChain** (0.3.21) - LLM framework
- **LangGraph** (0.3.18) - Graph-based workflow engine
- **LangChain-Google-VertexAI** - Google Gemini integration
- **LangChain-Anthropic** - Claude API integration

**Data & Database**
- **SQLAlchemy** (2.0.39) - ORM
- **asyncpg** (0.30.0) - Async PostgreSQL driver
- **PostgreSQL** - Primary database with pgvector extension

**Vector Search & Embeddings**
- **Google Cloud VertexAI** - Text embeddings (text-embedding-005)
- **pgvector** - Vector similarity search in PostgreSQL

**Observability**
- **Langfuse** (2.60.7) - LLM tracing and monitoring
- **Custom logging** - Structured logging system

**Real-time Communication**
- **Python-SocketIO** (5.12.1) - WebSocket support

**Data Processing**
- **Pydantic** (2.10.6) - Data validation and serialization
- **Pandas** - Data manipulation

---

## Architecture Patterns

### Overall Pattern: Service-Oriented Architecture with Pipeline Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    Client Layer                                  │
│              (REST API / WebSocket)                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    API Routes Layer                              │
│                     (FastAPI)                                    │
│  /query  /conversations  /schemas  /feedback  /directives       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Service Layer                                 │
│                  (Business Logic)                                │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Query Generation Service (LangGraph Pipeline)          │   │
│  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │ Intent → Schema → Analyze → Generate → Validate │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                  │
│  │  Schema    │ │Conversation│ │  Feedback  │                  │
│  │  Manager   │ │  Manager   │ │  Manager   │                  │
│  └────────────┘ └────────────┘ └────────────┘                  │
│                                                                   │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                  │
│  │    LLM     │ │ Embedding  │ │   Cache    │                  │
│  │  Provider  │ │  Provider  │ │  Manager   │                  │
│  └────────────┘ └────────────┘ └────────────┘                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│              External Integrations Layer                         │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Gemini  │  │  Claude  │  │PostgreSQL│  │  Langfuse│        │
│  │   API    │  │   API    │  │ +pgvector│  │  (Trace) │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Design Patterns Used

#### 1. Singleton Pattern
- **LLMProvider**: Single instance manages all LLM interactions
- **DatabasePool**: Single connection pool for database access
- **EmbeddingProvider**: Single embeddings service instance
- **CacheManager**: Single cache instance (local or Redis)

#### 2. Factory Pattern
- **LLMProvider.get_model()**: Creates appropriate LLM instances (Gemini vs Claude)
- **CacheManager**: Creates appropriate cache implementation (Local vs Redis)

#### 3. Pipeline/DAG Pattern (LangGraph)
- **QueryGenerator**: Directed acyclic graph of processing nodes
- Nodes execute sequentially with state passing
- Conditional routing based on validation results
- Bounded loops for refinement (max 3 iterations)

#### 4. State Machine Pattern
- **QueryGenerationState**: Immutable state object passed through pipeline
- State transitions tracked across nodes
- Escalation and refinement counters prevent infinite loops

#### 5. Repository Pattern
- **SchemaManager**: Abstracts schema data access
- **ConversationManager**: Abstracts conversation storage
- **FeedbackManager**: Abstracts feedback storage

#### 6. Strategy Pattern
- Multiple analysis strategies (intent, intelligent, validation)
- Multiple caching strategies (local vs Redis)
- Multiple LLM provider strategies (Gemini vs Claude)

---

## Directory Structure

```
/Users/naveenramaka/naveen/query-generator-server/
│
├── app/                                    # Main application package
│   │
│   ├── main.py                            # FastAPI app entry point & startup
│   ├── run.py                             # Uvicorn server launcher
│   │
│   ├── core/                              # Core infrastructure
│   │   ├── config.py                      # Settings & environment config (Pydantic)
│   │   ├── db.py                          # Database connection pool (asyncpg)
│   │   ├── logging.py                     # Logging configuration
│   │   ├── langfuse_client.py             # Langfuse observability client
│   │   └── profiling.py                   # Performance profiling utilities
│   │
│   ├── routes/                            # API endpoint handlers
│   │   ├── query.py                       # Query generation endpoints (main)
│   │   ├── conversation.py                # Conversation management endpoints
│   │   ├── directives.py                  # Schema directives endpoints
│   │   ├── feedback.py                    # User feedback collection
│   │   ├── schema_management.py           # Schema upload/management
│   │   ├── schema_manager.py              # Advanced schema management
│   │   ├── schema_api.py                  # Schema API endpoints
│   │   ├── debug_schema.py                # Debug/diagnostic endpoints
│   │   └── websocket.py                   # WebSocket event handlers
│   │
│   ├── services/                          # Business logic services
│   │   │
│   │   ├── query_generator.py             # Main LangGraph orchestrator
│   │   ├── query_generator_state.py       # State model for pipeline
│   │   ├── llm_provider.py                # LLM factory (Gemini/Claude)
│   │   ├── embedding_provider.py          # Vector embedding service
│   │   │
│   │   ├── schema_management.py           # Schema storage & retrieval
│   │   ├── enhanced_schema_service.py     # Advanced schema operations
│   │   ├── conversation_manager.py        # Conversation state management
│   │   ├── conversation_summarizer.py     # Conversation summarization
│   │   ├── feedback_manager.py            # User feedback management
│   │   ├── retry_generator.py             # Retry logic with feedback
│   │   ├── ai_description_service.py      # AI-driven descriptions
│   │   ├── schema_editor.py               # Schema editing operations
│   │   ├── database_connector.py          # Database query execution
│   │   │
│   │   ├── query_generation/              # Query generation pipeline
│   │   │   │
│   │   │   ├── nodes/                     # LangGraph pipeline nodes
│   │   │   │   ├── intent_classifier.py            # Classify query intent
│   │   │   │   ├── schema_retriever.py             # Retrieve relevant schemas
│   │   │   │   ├── enhanced_schema_retriever.py    # Enhanced retrieval
│   │   │   │   ├── intelligent_analyzer.py         # Analyze query complexity
│   │   │   │   ├── query_generator_node.py         # Generate KDB/Q query
│   │   │   │   ├── query_validator.py              # Validate generated query
│   │   │   │   ├── query_refiner.py                # Refine failed queries
│   │   │   │   ├── schema_description_node.py      # Generate schema descriptions
│   │   │   │   └── simplified_query_generator_node.py
│   │   │   │
│   │   │   ├── prompts/                   # LLM prompt templates
│   │   │   │   ├── generator_prompts.py
│   │   │   │   ├── intelligent_analyzer_prompts.py
│   │   │   │   ├── intent_classifier_prompts.py
│   │   │   │   ├── validation_prompts.py
│   │   │   │   ├── refiner_prompts.py
│   │   │   │   ├── retry_prompts.py
│   │   │   │   ├── schema_description_prompts.py
│   │   │   │   ├── ai_description_prompts.py
│   │   │   │   ├── analyzer_prompts.py
│   │   │   │   └── shared_constants.py    # Shared constants across prompts
│   │   │   │
│   │   │   └── tools/                     # Tool utilities
│   │   │
│   │   └── caching/                       # Caching layer
│   │       ├── cache_manager.py           # Cache abstraction manager
│   │       ├── caching_interface.py       # Cache interface definition
│   │       ├── local_cache.py             # In-memory cache implementation
│   │       └── redis_cache.py             # Redis cache implementation
│   │
│   ├── schemas/                           # Pydantic models
│   │   ├── query.py                       # QueryRequest/QueryResponse models
│   │   ├── conversation.py                # Conversation models
│   │   ├── feedback.py                    # Feedback models
│   │   ├── spot.json                      # Example schema file
│   │   └── default.json                   # Default schema file
│   │
│   └── utils/                             # Utility functions
│
├── scripts/                               # Utility scripts
│   ├── init_database.py                   # Database schema initialization
│   ├── import_schema.py                   # Schema file import
│   ├── batch_import_schemas.py            # Batch schema import
│   └── test_vector_search.py              # Vector search testing
│
├── tests/                                 # Test suite
│   ├── conftest.py                        # pytest configuration
│   └── test_glossary_integration.py       # Glossary tests
│
├── docs/                                  # Documentation
│   ├── ARCHITECTURE.md                    # This file
│   ├── Api.md                             # API documentation
│   ├── Configuration.md                   # Configuration guide
│   ├── Development.md                     # Development guide
│   ├── SchemaFormat.md                    # Schema format specification
│   └── Troubleshooting.md                 # Troubleshooting guide
│
├── run.py                                 # Application startup script
├── requirements.txt                       # Python dependencies
├── README.md                              # Project README
└── .env                                   # Environment configuration
```

### Key Module Responsibilities

| Module | Purpose | Key Components |
|--------|---------|----------------|
| `app/main.py` | Application entry point | FastAPI app, CORS, router registration |
| `app/core/` | Infrastructure & configuration | Config, DB pool, logging, profiling |
| `app/routes/` | API endpoint handlers | Route handlers for all endpoints |
| `app/services/` | Business logic | All service layer logic |
| `app/services/query_generation/nodes/` | Pipeline nodes | LangGraph node implementations |
| `app/services/query_generation/prompts/` | Prompt engineering | LLM prompt templates |
| `app/services/caching/` | Caching abstraction | Cache implementations |
| `app/schemas/` | Data models | Pydantic models for validation |
| `scripts/` | Admin utilities | DB init, schema import |

---

## Core Query Generation Pipeline

The query generation pipeline is the **heart of the system**, implemented using LangGraph as a directed acyclic graph (DAG) with conditional routing.

### Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     QUERY GENERATION PIPELINE                     │
│                        (LangGraph DAG)                            │
└──────────────────────────────────────────────────────────────────┘

    USER QUERY
        │
        ▼
┌──────────────────┐
│ INTENT CLASSIFIER│  ← Determines query type, detects follow-ups
└────────┬─────────┘
         │ intent_type, confidence
         ▼
┌──────────────────┐
│ SCHEMA RETRIEVER │  ← Vector search for relevant tables/schemas
└────────┬─────────┘
         │ query_schema (tables, examples)
         ▼
┌──────────────────┐
│ INTELLIGENT      │  ← Analyzes complexity, creates execution plan
│ ANALYZER         │
└────────┬─────────┘
         │ query_complexity, execution_plan, reasoning
         ▼
┌──────────────────┐
│ QUERY GENERATOR  │  ← LLM generates KDB/Q query
└────────┬─────────┘
         │ generated_query
         ▼
┌──────────────────┐
│ QUERY VALIDATOR  │  ← Validates syntax, security, logic
└────────┬─────────┘
         │
         ├─── VALID? ──────────────────┐
         │                              │
         ▼                              ▼
    ┌─────────┐                    ┌────────┐
    │ Invalid │                    │  END   │
    └────┬────┘                    │ Return │
         │                         │ Query  │
         ├── Escalation needed?    └────────┘
         │   (complexity issue)
         │   YES → Back to INTELLIGENT ANALYZER
         │
         ▼
    ┌──────────────┐
    │ QUERY        │  ← Fixes query based on validation errors
    │ REFINER      │
    └──────┬───────┘
           │ refinement_guidance
           │
           └──→ Back to QUERY GENERATOR (max 3 loops)
```

### Pipeline Nodes Detail

#### 1. Intent Classifier Node
**File**: `app/services/query_generation/nodes/intent_classifier.py`

**Responsibilities**:
- Analyze user query to determine intent type
- Detect if query is a follow-up question
- Extract confidence level of classification

**Inputs**:
- `user_query`: Natural language question
- `conversation_history`: Previous messages (if any)

**Outputs**:
- `intent_type`: One of:
  - `query_generation`: User wants a database query
  - `schema_description`: User wants schema information
  - `clarification`: User needs help/clarification
  - `greeting`: Conversational greeting
- `is_follow_up`: Boolean indicating if query references previous context
- `confidence`: Classification confidence (0-1)
- `classification_reasoning`: Explanation of classification

**Example**:
```
Query: "Show me all trades from yesterday"
→ intent_type: query_generation
→ is_follow_up: False
→ confidence: 0.95
```

#### 2. Schema Retriever Node
**File**: `app/services/query_generation/nodes/schema_retriever.py`

**Responsibilities**:
- Use vector search to find relevant database tables
- Retrieve table schemas and relationships
- Fetch few-shot example queries
- Include business glossary directives if applicable

**Inputs**:
- `user_query`: Natural language question
- `schema_group_id`: Optional schema group filter

**Outputs**:
- `query_schema`: Complete schema context including:
  - Relevant table definitions
  - Column descriptions
  - Table relationships
  - Business glossary terms
  - Example queries (few-shot learning)

**Vector Search Process**:
1. Embed user query using Google text-embedding-005
2. Perform cosine similarity search against table embeddings
3. Retrieve top K tables (configurable, default 5)
4. Enrich with relationships and examples
5. Include glossary directives if schema has them

**Example**:
```
Query: "Show trades for AAPL"
→ Retrieves: Trade table, Symbol table
→ Includes: Column definitions, join keys
→ Adds: Example queries for similar questions
```

#### 3. Intelligent Analyzer Node
**File**: `app/services/query_generation/nodes/intelligent_analyzer.py`

**Responsibilities**:
- Analyze query complexity (SINGLE_LINE vs MULTI_LINE)
- Determine query type (select, aggregation, join, etc.)
- Create execution plan
- Evaluate schema constraints
- Provide reasoning for decisions

**Inputs**:
- `user_query`: Natural language question
- `query_schema`: Retrieved schema context
- `conversation_context`: Previous conversation (if any)

**Outputs**:
- `query_complexity`: `SINGLE_LINE` or `MULTI_LINE`
- `query_type`: One of:
  - `select_basic`: Simple SELECT query
  - `select_with_filter`: SELECT with WHERE conditions
  - `aggregation`: COUNT, SUM, AVG, etc.
  - `join`: Multi-table joins
  - `temporal`: Time-based queries
  - `complex`: Complex multi-step queries
- `execution_plan`: Step-by-step query plan
- `schema_constraints`: Relevant constraints to consider
- `analysis_reasoning`: Detailed reasoning

**Complexity Criteria**:
- **SINGLE_LINE**: Simple queries, single table, basic filters
- **MULTI_LINE**: Joins, aggregations, subqueries, complex logic

**Example**:
```
Query: "Average trade volume by symbol in the last week"
→ query_complexity: MULTI_LINE
→ query_type: aggregation
→ execution_plan:
  1. Filter trades by time window (last 7 days)
  2. Group by symbol
  3. Calculate average volume
  4. Order results
```

#### 4. Query Generator Node
**File**: `app/services/query_generation/nodes/query_generator_node.py`

**Responsibilities**:
- Generate database-specific query (KDB/Q)
- Use schema context and execution plan
- Apply few-shot examples
- Include refinement guidance if available

**Inputs**:
- `user_query`: Natural language question
- `query_schema`: Schema context
- `execution_plan`: Analysis from intelligent analyzer
- `refinement_guidance`: Feedback from previous iterations (if any)
- `conversation_context`: Previous conversation

**Outputs**:
- `generated_query`: KDB/Q query string
- `thinking_process`: Explanation of generation logic

**Prompt Engineering**:
- Database-specific syntax guidelines (KDB/Q)
- Schema context with table/column definitions
- Few-shot examples for similar queries
- Execution plan from analyzer
- Refinement guidance (if retry)
- Conversation context (if follow-up)

**Example**:
```
Query: "Show top 10 symbols by trade count today"
Generated Query:
```q
select top 10 count i by sym
from trade
where date=.z.d
```
```

#### 5. Query Validator Node
**File**: `app/services/query_generation/nodes/query_validator.py`

**Responsibilities**:
- Validate query syntax
- Check for security issues (SQL injection patterns)
- Validate against schema constraints
- Use LLM for semantic validation
- Provide detailed feedback on errors

**Inputs**:
- `generated_query`: Query to validate
- `query_schema`: Schema context
- `user_query`: Original question

**Outputs**:
- `validation_result`: `VALID`, `INVALID_REFINABLE`, `INVALID_ESCALATE`
- `validation_errors`: List of specific errors
- `detailed_feedback`: LLM-generated analysis of issues
- `confidence`: Validation confidence level

**Validation Checks**:
1. **Syntax Validation**: Basic syntax correctness
2. **Security Validation**: Check for injection patterns
3. **Schema Validation**: Verify tables/columns exist
4. **Semantic Validation**: LLM analyzes if query answers the question
5. **Business Logic Validation**: Check constraints and rules

**Routing Logic**:
- **VALID**: Query passes all checks → END (return query)
- **INVALID_REFINABLE**: Minor issues, can be refined → QUERY REFINER
- **INVALID_ESCALATE**: Major issues, need re-analysis → INTELLIGENT ANALYZER

**Example**:
```
Generated Query: select from trades where sym=`AAPL
Validation: INVALID_REFINABLE
Error: Table name should be 'trade' not 'trades'
→ Routes to QUERY REFINER
```

#### 6. Query Refiner Node
**File**: `app/services/query_generation/nodes/query_refiner.py`

**Responsibilities**:
- Analyze validation errors
- Provide specific refinement guidance
- Track refinement iterations
- Prevent infinite loops (max 3 refinements)

**Inputs**:
- `generated_query`: Failed query
- `validation_errors`: Specific errors from validator
- `detailed_feedback`: LLM validation feedback
- `refinement_count`: Current refinement iteration

**Outputs**:
- `refinement_guidance`: Specific instructions for query generator
- `refinement_count`: Incremented counter

**Refinement Strategy**:
- Provide targeted fixes (e.g., "Change table name from X to Y")
- Include context from validation errors
- Escalate if max refinements reached (3 iterations)

**Loop Prevention**:
```python
if refinement_count >= 3:
    # Escalate to intelligent analyzer for re-analysis
    return "escalate"
else:
    # Continue refinement
    return "refine"
```

### State Object: QueryGenerationState

**File**: `app/services/query_generator_state.py`

The state object is passed through the entire pipeline, accumulating context and results.

**Key State Fields**:

```python
@dataclass
class QueryGenerationState:
    # Input
    user_query: str
    conversation_id: Optional[str]
    schema_group_id: Optional[str]

    # Intent Classification
    intent_type: Optional[str]
    is_follow_up: bool
    confidence: float
    classification_reasoning: Optional[str]

    # Schema Context
    query_schema: Optional[Dict]
    retrieved_tables: List[str]
    glossary_directives: List[Dict]

    # Analysis
    query_complexity: Optional[str]  # SINGLE_LINE | MULTI_LINE
    query_type: Optional[str]
    execution_plan: Optional[str]
    schema_constraints: List[str]
    analysis_reasoning: Optional[str]

    # Generation
    generated_query: Optional[str]
    thinking_process: Optional[str]

    # Validation
    validation_result: Optional[str]  # VALID | INVALID_REFINABLE | INVALID_ESCALATE
    validation_errors: List[str]
    detailed_feedback: Optional[str]

    # Refinement
    refinement_guidance: Optional[str]
    refinement_count: int = 0
    escalation_count: int = 0

    # Conversation Context
    conversation_context: Optional[str]
    previous_query: Optional[str]

    # Tracing
    trace_id: Optional[str]
    langfuse_observation_id: Optional[str]
```

### Pipeline Execution Flow

**File**: `app/services/query_generator.py`

The `QueryGenerator` class orchestrates the LangGraph pipeline:

```python
class QueryGenerator:
    def __init__(self):
        self.llm_provider = LLMProvider()
        self.schema_manager = SchemaManager()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build LangGraph DAG"""
        graph = StateGraph(QueryGenerationState)

        # Add nodes
        graph.add_node("intent_classifier", intent_classifier_node)
        graph.add_node("schema_retriever", schema_retriever_node)
        graph.add_node("intelligent_analyzer", intelligent_analyzer_node)
        graph.add_node("query_generator", query_generator_node)
        graph.add_node("query_validator", query_validator_node)
        graph.add_node("query_refiner", query_refiner_node)

        # Add edges
        graph.set_entry_point("intent_classifier")
        graph.add_edge("intent_classifier", "schema_retriever")
        graph.add_edge("schema_retriever", "intelligent_analyzer")
        graph.add_edge("intelligent_analyzer", "query_generator")
        graph.add_edge("query_generator", "query_validator")

        # Conditional routing from validator
        graph.add_conditional_edges(
            "query_validator",
            self._route_after_validation,
            {
                "end": END,
                "refine": "query_refiner",
                "escalate": "intelligent_analyzer"
            }
        )

        # Refiner routes back to generator
        graph.add_edge("query_refiner", "query_generator")

        return graph.compile()

    async def generate_query(self, request: QueryRequest) -> QueryResponse:
        """Execute the pipeline"""
        initial_state = QueryGenerationState(
            user_query=request.query,
            conversation_id=request.conversation_id,
            schema_group_id=request.schema_group_id
        )

        # Run LangGraph
        final_state = await self.graph.ainvoke(initial_state)

        return QueryResponse(
            query=final_state.generated_query,
            thinking_process=final_state.thinking_process,
            execution_plan=final_state.execution_plan
        )
```

### Escalation & Refinement Logic

**Escalation** (back to Intelligent Analyzer):
- Occurs when validation indicates fundamental misunderstanding
- Query complexity was misjudged
- Schema constraints were missed
- Max escalations: 2

**Refinement** (via Query Refiner):
- Occurs for minor syntax or logic errors
- Correctable with targeted feedback
- Max refinements: 3

**Loop Prevention**:
```
Total iterations = escalation_count + refinement_count
Max total iterations = 5

If max reached → Return error with explanation
```

---

## Key Services

### 1. LLMProvider (Factory Pattern)
**File**: `app/services/llm_provider.py`

**Purpose**: Centralized LLM management and model selection

**Capabilities**:
- Lazy initialization of LLM clients
- Model selection (Gemini vs Claude)
- Temperature and parameter configuration
- Streaming support
- Token usage tracking

**Usage**:
```python
llm_provider = LLMProvider()

# Get model for generation
model = llm_provider.get_model(
    provider="gemini",  # or "claude"
    temperature=0.2,
    streaming=False
)

# Get streaming model for real-time responses
streaming_model = llm_provider.get_streaming_model(provider="gemini")
```

**Configuration**:
- Model names configurable via environment
- Temperature settings per use case
- Max tokens configurable
- API keys managed via environment variables

### 2. EmbeddingProvider
**File**: `app/services/embedding_provider.py`

**Purpose**: Generate vector embeddings for semantic search

**Model**: Google text-embedding-005

**Capabilities**:
- Embed user queries for table search
- Embed table descriptions for indexing
- Batch embedding for efficiency
- Caching of embeddings

**Usage**:
```python
embedding_provider = EmbeddingProvider()

# Embed user query
query_embedding = await embedding_provider.embed_query(
    "Show me trades for AAPL"
)

# Embed multiple texts (batch)
embeddings = await embedding_provider.embed_documents([
    "Table description 1",
    "Table description 2"
])
```

### 3. SchemaManager
**File**: `app/services/schema_management.py`

**Purpose**: Manage database schemas and vector search

**Capabilities**:
- Store schema definitions in PostgreSQL
- Generate and store embeddings for tables
- Vector similarity search for relevant tables
- Retrieve table relationships
- Fetch example queries for few-shot learning
- Manage business glossary directives

**Key Methods**:

```python
class SchemaManager:
    async def upload_schema(
        self,
        schema_group_name: str,
        schema_data: dict
    ) -> str:
        """Upload and embed schema"""

    async def search_relevant_tables(
        self,
        query: str,
        schema_group_id: str,
        limit: int = 5
    ) -> List[Dict]:
        """Vector search for relevant tables"""

    async def get_table_relationships(
        self,
        table_name: str,
        schema_group_id: str
    ) -> List[Dict]:
        """Get foreign key relationships"""

    async def get_example_queries(
        self,
        query_type: str,
        limit: int = 3
    ) -> List[str]:
        """Fetch few-shot examples"""

    async def get_glossary_directives(
        self,
        schema_group_id: str
    ) -> List[Dict]:
        """Get business glossary terms"""
```

### 4. ConversationManager
**File**: `app/services/conversation_manager.py`

**Purpose**: Manage multi-turn conversations

**Capabilities**:
- Create and store conversations
- Add messages to conversations
- Retrieve conversation history
- Summarize long conversations
- Track conversation context

**Database Schema**:
```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    metadata JSONB
);

CREATE TABLE messages (
    id UUID PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id),
    role VARCHAR(20),  -- 'user' or 'assistant'
    content TEXT,
    query TEXT,
    response JSONB,
    created_at TIMESTAMP
);
```

**Usage**:
```python
conv_manager = ConversationManager()

# Create conversation
conv_id = await conv_manager.create_conversation()

# Add user message
await conv_manager.add_message(
    conversation_id=conv_id,
    role="user",
    content="Show me trades",
    query="SELECT * FROM trade"
)

# Get conversation context for next query
context = await conv_manager.get_conversation_context(conv_id)
```

### 5. FeedbackManager
**File**: `app/services/feedback_manager.py`

**Purpose**: Collect and manage user feedback

**Capabilities**:
- Store user feedback on generated queries
- Track feedback types (thumbs up/down, corrections)
- Associate feedback with queries
- Retrieve feedback for analysis

**Feedback Types**:
- **Rating**: Thumbs up/down
- **Correction**: User provides corrected query
- **Comment**: Free-text feedback

**Usage**:
```python
feedback_manager = FeedbackManager()

await feedback_manager.store_feedback(
    query_id=query_id,
    feedback_type="correction",
    content="Query should filter by date",
    corrected_query="SELECT * FROM trade WHERE date=.z.d"
)
```

### 6. CacheManager
**File**: `app/services/caching/cache_manager.py`

**Purpose**: Caching abstraction layer

**Implementations**:
- **LocalCache**: In-memory LRU cache (development)
- **RedisCache**: Redis-backed cache (production)

**Capabilities**:
- Cache LLM responses
- Cache embeddings
- Cache schema lookups
- TTL-based expiration

**Usage**:
```python
cache_manager = CacheManager()  # Auto-selects implementation

# Cache query result
await cache_manager.set(
    key="query:abc123",
    value={"query": "SELECT ...", "result": [...]}
    ttl=3600  # 1 hour
)

# Retrieve from cache
cached_result = await cache_manager.get("query:abc123")
```

### 7. DatabaseConnector
**File**: `app/services/database_connector.py`

**Purpose**: Execute queries against target database (KDB)

**Capabilities**:
- Execute KDB/Q queries
- Handle pagination
- Parse and format results
- Error handling and retry logic

**Usage**:
```python
db_connector = DatabaseConnector()

result = await db_connector.execute_query(
    query="select from trade where sym=`AAPL",
    limit=100,
    offset=0
)
```

---

## Data Flow

### Complete Query Generation Flow

```
1. Client Request
   ↓
   POST /api/v1/query
   {
       "query": "Show top 10 trades by volume today",
       "schema_group_id": "production",
       "conversation_id": null
   }

2. Route Handler (routes/query.py)
   ↓
   - Validates request (Pydantic)
   - Extracts parameters
   - Calls QueryGenerator service

3. QueryGenerator.generate_query()
   ↓
   - Initializes QueryGenerationState
   - Starts LangGraph pipeline
   - Tracks with Langfuse

4. LangGraph Pipeline Execution
   ↓
   ┌─────────────────────────────────────────┐
   │ Intent Classifier                        │
   │ → LLM call to classify intent           │
   │ → State updated with intent_type        │
   └─────────────────────────────────────────┘
   ↓
   ┌─────────────────────────────────────────┐
   │ Schema Retriever                         │
   │ → Embed query (EmbeddingProvider)       │
   │ → Vector search (SchemaManager)         │
   │ → Retrieve tables, examples             │
   │ → State updated with query_schema       │
   └─────────────────────────────────────────┘
   ↓
   ┌─────────────────────────────────────────┐
   │ Intelligent Analyzer                     │
   │ → LLM call to analyze complexity        │
   │ → Create execution plan                 │
   │ → State updated with analysis           │
   └─────────────────────────────────────────┘
   ↓
   ┌─────────────────────────────────────────┐
   │ Query Generator                          │
   │ → LLM call with schema + plan           │
   │ → Generate KDB/Q query                  │
   │ → State updated with generated_query    │
   └─────────────────────────────────────────┘
   ↓
   ┌─────────────────────────────────────────┐
   │ Query Validator                          │
   │ → Syntax validation                     │
   │ → Security checks                       │
   │ → LLM semantic validation               │
   │ → State updated with validation_result  │
   └─────────────────────────────────────────┘
   ↓
   [Conditional Routing]
   ├─ VALID → Return query
   ├─ INVALID_REFINABLE → Query Refiner → Back to Generator
   └─ INVALID_ESCALATE → Back to Intelligent Analyzer

5. Response Construction
   ↓
   QueryResponse {
       "query": "select top 10 from trade where date=.z.d",
       "thinking_process": "...",
       "execution_plan": "...",
       "query_type": "select_with_filter",
       "complexity": "SINGLE_LINE"
   }

6. Return to Client
   ↓
   HTTP 200 OK
   {
       "status": "success",
       "data": { ... }
   }
```

### Follow-up Query Flow

```
1. Client Request with Conversation ID
   ↓
   POST /api/v1/query
   {
       "query": "Now filter for AAPL only",
       "conversation_id": "conv-123"
   }

2. Route Handler
   ↓
   - Retrieves conversation context
   - Passes to QueryGenerator

3. QueryGenerator
   ↓
   - ConversationManager.get_context(conv_id)
   - Retrieves previous messages
   - Includes context in state

4. Intent Classifier
   ↓
   - Detects is_follow_up = True
   - Understands reference to previous query

5. Pipeline Execution
   ↓
   - Schema retrieval uses same tables
   - Generator includes conversation context
   - Query references previous query

6. Response & Storage
   ↓
   - Query generated: "select from trade where date=.z.d, sym=`AAPL"
   - ConversationManager.add_message(...)
   - Response returned to client
```

### Retry with Feedback Flow

```
1. User Provides Feedback
   ↓
   POST /api/v1/feedback
   {
       "query_id": "q-456",
       "feedback": "Query should only show trades after 10am"
   }

2. FeedbackManager Stores Feedback
   ↓
   - Feedback stored in database
   - Associated with original query

3. User Requests Retry
   ↓
   POST /api/v1/retry
   {
       "query_id": "q-456"
   }

4. RetryGenerator
   ↓
   - Retrieves original query and feedback
   - Analyzes feedback with LLM
   - Creates refinement guidance

5. Pipeline Re-execution
   ↓
   - QueryGenerator runs with feedback context
   - Refinement guidance included in prompts
   - New query generated

6. Improved Query Returned
   ↓
   "select from trade where date=.z.d, time>10:00:00"
```

---

## Database Schema

### Entity-Relationship Diagram

```
┌─────────────────────┐
│  schema_groups      │
│  ─────────────────  │
│  id (PK)            │
│  name               │
│  description        │
│  created_at         │
└──────────┬──────────┘
           │ 1
           │
           │ N
┌──────────▼──────────┐         ┌─────────────────────┐
│ schema_definitions  │    N    │  glossary_directives│
│  ─────────────────  │◄────────┤  ─────────────────  │
│  id (PK)            │         │  id (PK)            │
│  schema_group_id(FK)│         │  schema_group_id(FK)│
│  name               │         │  directive_key      │
│  description        │         │  directive_value    │
│  content (JSONB)    │         │  context            │
│  created_at         │         └─────────────────────┘
└──────────┬──────────┘
           │ 1
           │
           │ N
┌──────────▼──────────┐         ┌─────────────────────┐
│  table_definitions  │    N    │ table_relationships │
│  ─────────────────  │◄────────┤  ─────────────────  │
│  id (PK)            │         │  id (PK)            │
│  schema_id (FK)     │    1    │  source_table_id(FK)│
│  table_name         │────────►│  target_table_id(FK)│
│  description        │         │  relationship_type  │
│  columns (JSONB)    │         │  join_keys          │
│  embedding (vector) │◄───┐    └─────────────────────┘
│  created_at         │    │
└─────────────────────┘    │    ┌─────────────────────┐
                           │    │  schema_examples    │
                           └────┤  ─────────────────  │
                                │  id (PK)            │
                                │  table_id (FK)      │
                                │  question           │
                                │  query              │
                                │  explanation        │
                                └─────────────────────┘

┌─────────────────────┐         ┌─────────────────────┐
│  conversations      │    1    │  messages           │
│  ─────────────────  │────────►│  ─────────────────  │
│  id (PK)            │    N    │  id (PK)            │
│  created_at         │         │  conversation_id(FK)│
│  updated_at         │         │  role               │
│  metadata (JSONB)   │         │  content            │
└─────────────────────┘         │  query              │
                                │  response (JSONB)   │
                                │  created_at         │
                                └─────────────────────┘

┌─────────────────────┐
│  feedback           │
│  ─────────────────  │
│  id (PK)            │
│  query_id           │
│  feedback_type      │
│  content            │
│  corrected_query    │
│  created_at         │
└─────────────────────┘
```

### Key Tables

#### schema_groups
Logical grouping of related schemas (e.g., "production", "staging")

```sql
CREATE TABLE schema_groups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### schema_definitions
Individual database schemas within a group

```sql
CREATE TABLE schema_definitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_group_id UUID REFERENCES schema_groups(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    content JSONB NOT NULL,  -- Full schema JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### table_definitions
Individual table schemas with vector embeddings

```sql
CREATE TABLE table_definitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_id UUID REFERENCES schema_definitions(id),
    table_name VARCHAR(255) NOT NULL,
    description TEXT,
    columns JSONB NOT NULL,  -- Column definitions
    embedding vector(768),   -- Google text-embedding-005
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_table_embedding ON table_definitions
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

**Embedding Storage**: Uses pgvector extension with IVFFlat index for fast similarity search

#### glossary_directives
Business glossary terms and domain-specific language

```sql
CREATE TABLE glossary_directives (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_group_id UUID REFERENCES schema_groups(id),
    directive_key VARCHAR(255) NOT NULL,
    directive_value TEXT NOT NULL,
    context TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Example**:
```json
{
    "directive_key": "vwap",
    "directive_value": "Volume Weighted Average Price",
    "context": "When user asks for VWAP, calculate sum(price*volume)/sum(volume)"
}
```

#### table_relationships
Foreign key relationships between tables

```sql
CREATE TABLE table_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_table_id UUID REFERENCES table_definitions(id),
    target_table_id UUID REFERENCES table_definitions(id),
    relationship_type VARCHAR(50),  -- 'one_to_many', 'many_to_one', etc.
    join_keys JSONB,  -- {"source_key": "id", "target_key": "user_id"}
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### schema_examples
Few-shot example queries for prompt engineering

```sql
CREATE TABLE schema_examples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_id UUID REFERENCES table_definitions(id),
    question TEXT NOT NULL,
    query TEXT NOT NULL,
    explanation TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Example**:
```json
{
    "question": "Show top 10 trades by volume",
    "query": "select top 10 from trade",
    "explanation": "Uses 'top N' syntax to limit results"
}
```

---

## External Integrations

### 1. Google Cloud Platform

**Services Used**:
- **Vertex AI**: Gemini model access
- **Vertex AI Embeddings**: text-embedding-005 model
- **Cloud Storage**: Optional schema file storage

**Authentication**:
- Service account credentials
- Environment variable: `GOOGLE_APPLICATION_CREDENTIALS`

**Configuration**:
```python
# app/core/config.py
GOOGLE_CLOUD_PROJECT = "your-project-id"
GOOGLE_CLOUD_LOCATION = "us-central1"
GEMINI_MODEL = "gemini-1.5-pro-002"
EMBEDDING_MODEL = "text-embedding-005"
```

### 2. Anthropic Claude

**Models Supported**:
- Claude 3.5 Sonnet
- Claude 3 Opus
- Claude 3 Haiku

**Authentication**:
- API key via environment variable: `ANTHROPIC_API_KEY`

**Configuration**:
```python
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
CLAUDE_TEMPERATURE = 0.2
CLAUDE_MAX_TOKENS = 4096
```

### 3. PostgreSQL with pgvector

**Extensions Required**:
- `pgvector`: Vector similarity search
- `uuid-ossp`: UUID generation

**Connection**:
```python
DATABASE_URL = "postgresql://user:pass@host:5432/dbname"
```

**Vector Operations**:
```sql
-- Find similar tables using cosine distance
SELECT table_name, 1 - (embedding <=> query_embedding) AS similarity
FROM table_definitions
ORDER BY embedding <=> query_embedding
LIMIT 5;
```

### 4. Langfuse (Observability)

**Purpose**: LLM tracing and monitoring

**Tracked Metrics**:
- LLM calls and latency
- Token usage and costs
- Pipeline execution traces
- Error rates

**Configuration**:
```python
LANGFUSE_PUBLIC_KEY = "pk-..."
LANGFUSE_SECRET_KEY = "sk-..."
LANGFUSE_HOST = "https://cloud.langfuse.com"
```

**Integration**:
```python
from app.core.langfuse_client import get_langfuse_client

langfuse = get_langfuse_client()

# Trace pipeline execution
trace = langfuse.trace(
    name="query_generation",
    input={"query": user_query},
    metadata={"conversation_id": conv_id}
)

# Track LLM generation
generation = trace.generation(
    name="intent_classifier",
    model="gemini-1.5-pro",
    input=prompt,
    output=response
)
```

### 5. Redis (Optional Caching)

**Purpose**: Distributed caching for production

**Configuration**:
```python
REDIS_URL = "redis://localhost:6379/0"
CACHE_TTL = 3600  # 1 hour
```

**Fallback**: LocalCache used if Redis unavailable

---

## Configuration Management

### Environment Variables

**File**: `app/core/config.py`

Configuration managed via Pydantic Settings with environment variable support:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Application
    DEBUG: bool = False
    API_VERSION: str = "v1"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str

    # Google Cloud
    GOOGLE_CLOUD_PROJECT: str
    GOOGLE_CLOUD_LOCATION: str = "us-central1"
    GEMINI_MODEL: str = "gemini-1.5-pro-002"
    GEMINI_TEMPERATURE: float = 0.2
    GEMINI_MAX_TOKENS: int = 8192

    # Anthropic
    ANTHROPIC_API_KEY: str
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20241022"
    CLAUDE_TEMPERATURE: float = 0.2
    CLAUDE_MAX_TOKENS: int = 4096

    # Embeddings
    EMBEDDING_MODEL: str = "text-embedding-005"
    EMBEDDING_DIMENSION: int = 768

    # Schema Search
    SCHEMA_SIMILARITY_THRESHOLD: float = 0.7
    MAX_RETRIEVED_TABLES: int = 5

    # Pipeline Limits
    MAX_REFINEMENT_ITERATIONS: int = 3
    MAX_ESCALATION_ITERATIONS: int = 2

    # Caching
    CACHE_TYPE: str = "local"  # "local" or "redis"
    REDIS_URL: Optional[str] = None
    CACHE_TTL: int = 3600

    # Langfuse
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

### Configuration Files

**`.env` file** (not committed to git):
```bash
DEBUG=false
DATABASE_URL=postgresql://user:pass@localhost:5432/querydb

GOOGLE_CLOUD_PROJECT=my-project
GEMINI_MODEL=gemini-1.5-pro-002

ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-3-5-sonnet-20241022

LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...

CACHE_TYPE=redis
REDIS_URL=redis://localhost:6379/0
```

### Runtime Configuration

Models and parameters can be overridden per request:

```python
# Request-level model selection
POST /api/v1/query
{
    "query": "...",
    "model_config": {
        "provider": "claude",  # Override default
        "temperature": 0.5     # Override default
    }
}
```

---

## API Structure

### REST API Endpoints

**Base URL**: `/api/v1`

#### Query Generation

```
POST /api/v1/query
Request:
{
    "query": "Show me trades for AAPL",
    "schema_group_id": "production",
    "conversation_id": null,
    "model_config": {
        "provider": "gemini",
        "temperature": 0.2
    }
}

Response:
{
    "status": "success",
    "data": {
        "query": "select from trade where sym=`AAPL",
        "thinking_process": "...",
        "execution_plan": "...",
        "query_type": "select_with_filter",
        "complexity": "SINGLE_LINE",
        "retrieved_tables": ["trade"],
        "trace_id": "trace-123"
    }
}
```

#### Query Retry with Feedback

```
POST /api/v1/retry
Request:
{
    "query_id": "q-456",
    "feedback": "Query should filter by date"
}

Response:
{
    "status": "success",
    "data": {
        "query": "select from trade where sym=`AAPL, date=.z.d",
        ...
    }
}
```

#### Query Execution

```
POST /api/v1/execute
Request:
{
    "query": "select from trade where sym=`AAPL",
    "limit": 100,
    "offset": 0
}

Response:
{
    "status": "success",
    "data": {
        "results": [...],
        "row_count": 42,
        "execution_time_ms": 125
    }
}
```

#### Conversation Management

```
POST /api/v1/conversations
Response:
{
    "conversation_id": "conv-123"
}

GET /api/v1/conversations/{id}
Response:
{
    "id": "conv-123",
    "messages": [
        {
            "role": "user",
            "content": "Show trades",
            "query": "select from trade"
        },
        {
            "role": "assistant",
            "content": "Here is the query..."
        }
    ],
    "created_at": "2024-01-15T10:30:00Z"
}
```

#### Schema Management

```
POST /api/v1/schemas/upload
Request:
{
    "schema_group_name": "production",
    "schema_data": {
        "tables": [...]
    }
}

GET /api/v1/schemas
Response:
{
    "schemas": [
        {
            "id": "schema-1",
            "name": "production",
            "table_count": 15
        }
    ]
}
```

#### Feedback

```
POST /api/v1/feedback
Request:
{
    "query_id": "q-456",
    "feedback_type": "correction",
    "content": "Should filter by date",
    "corrected_query": "select from trade where date=.z.d"
}
```

### WebSocket API

**Endpoint**: `/ws`

**Protocol**: Socket.IO

**Events**:

```javascript
// Client → Server
socket.emit('generate_query', {
    query: "Show trades for AAPL",
    schema_group_id: "production"
});

// Server → Client (streaming response)
socket.on('query_progress', (data) => {
    console.log(data.stage);  // "intent_classification", "schema_retrieval", etc.
});

socket.on('query_result', (data) => {
    console.log(data.query);
});

socket.on('query_error', (error) => {
    console.error(error.message);
});
```

---

## State Management

### Pipeline State Flow

The `QueryGenerationState` object is the backbone of the pipeline, maintaining all context as it flows through nodes.

**State Initialization**:
```python
initial_state = QueryGenerationState(
    user_query="Show me trades",
    conversation_id=None,
    schema_group_id="production",

    # All other fields initialized to defaults
    intent_type=None,
    query_schema=None,
    generated_query=None,
    refinement_count=0,
    escalation_count=0
)
```

**State After Intent Classification**:
```python
state = QueryGenerationState(
    user_query="Show me trades",

    # Intent classifier populated these:
    intent_type="query_generation",
    is_follow_up=False,
    confidence=0.95,
    classification_reasoning="User wants to retrieve trade data"
)
```

**State After Schema Retrieval**:
```python
state = QueryGenerationState(
    ...previous fields...,

    # Schema retriever populated these:
    query_schema={
        "tables": [
            {
                "name": "trade",
                "columns": [...],
                "relationships": [...]
            }
        ],
        "examples": [...]
    },
    retrieved_tables=["trade"],
    glossary_directives=[]
)
```

**Final State**:
```python
state = QueryGenerationState(
    ...all previous fields...,

    # Generator populated:
    generated_query="select from trade",
    thinking_process="...",

    # Validator populated:
    validation_result="VALID",
    validation_errors=[],

    # Tracing:
    trace_id="trace-123"
)
```

### Conversation State

**Persistence**:
- Conversations stored in PostgreSQL
- Messages appended to conversation
- Context retrieved for follow-up queries

**Context Window Management**:
- Last N messages included (configurable, default 10)
- Summarization for long conversations
- Token limit awareness

**Example Context**:
```python
conversation_context = """
Previous messages:
User: Show me trades for AAPL
Assistant: select from trade where sym=`AAPL
User: Now filter for today only
"""
```

---

## Caching Strategy

### Cache Layers

**1. Embedding Cache**
- Cache query embeddings (frequent queries)
- Cache table embeddings (static, long TTL)
- Key: Hash of text
- TTL: 24 hours

**2. Schema Lookup Cache**
- Cache retrieved schemas for queries
- Key: `schema:{schema_group_id}:{query_hash}`
- TTL: 1 hour

**3. LLM Response Cache**
- Cache LLM responses for identical inputs
- Key: `llm:{model}:{prompt_hash}`
- TTL: 30 minutes
- Note: Only for deterministic queries

**4. Conversation Cache**
- Cache conversation context
- Key: `conversation:{conversation_id}`
- TTL: 1 hour

### Cache Implementation

**LocalCache** (Development):
```python
from functools import lru_cache

class LocalCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size

    async def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)

    async def set(self, key: str, value: Any, ttl: int):
        if len(self.cache) >= self.max_size:
            # Evict oldest
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
```

**RedisCache** (Production):
```python
class RedisCache:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    async def get(self, key: str) -> Optional[Any]:
        value = await self.redis.get(key)
        return json.loads(value) if value else None

    async def set(self, key: str, value: Any, ttl: int):
        await self.redis.setex(
            key,
            ttl,
            json.dumps(value)
        )
```

### Cache Invalidation

**Manual Invalidation**:
- Schema updates invalidate all schema caches
- Conversation updates invalidate conversation cache

**TTL-Based Expiration**:
- All caches have TTL
- Automatic cleanup

---

## Error Handling

### Error Types

**1. Validation Errors** (400 Bad Request)
- Invalid request format
- Missing required fields
- Invalid parameter values

```python
raise HTTPException(
    status_code=400,
    detail="Invalid schema_group_id"
)
```

**2. Not Found Errors** (404)
- Conversation not found
- Schema not found

**3. Generation Errors** (500)
- LLM API failures
- Pipeline execution failures
- Database connection errors

**4. Timeout Errors** (504)
- LLM response timeout
- Database query timeout

### Error Response Format

```json
{
    "status": "error",
    "error": {
        "code": "GENERATION_FAILED",
        "message": "Failed to generate query",
        "details": {
            "stage": "query_validator",
            "reason": "Query validation failed after 3 refinement attempts"
        },
        "trace_id": "trace-123"
    }
}
```

### Error Handling Strategy

**Graceful Degradation**:
- If cache fails, proceed without cache
- If embeddings fail, fall back to keyword search
- If one LLM provider fails, try alternative

**Retry Logic**:
- LLM API calls: 3 retries with exponential backoff
- Database queries: 2 retries
- Embedding generation: 2 retries

**User-Friendly Messages**:
```python
# Internal error
"Failed to connect to database: connection timeout"

# User-facing error
"We're having trouble connecting to the database. Please try again in a moment."
```

### Logging

**Structured Logging**:
```python
import logging

logger = logging.getLogger(__name__)

logger.info(
    "Query generated successfully",
    extra={
        "user_query": query,
        "generated_query": result,
        "trace_id": trace_id,
        "duration_ms": duration
    }
)
```

**Log Levels**:
- **DEBUG**: Detailed pipeline state transitions
- **INFO**: Successful operations
- **WARNING**: Degraded performance, fallbacks
- **ERROR**: Operation failures
- **CRITICAL**: System failures

---

## Appendix

### Key Files Reference

| Component | File Path | Purpose |
|-----------|-----------|---------|
| App Entry | `app/main.py` | FastAPI app initialization |
| Config | `app/core/config.py` | Configuration management |
| DB Pool | `app/core/db.py` | Database connection pool |
| Query Pipeline | `app/services/query_generator.py` | LangGraph orchestrator |
| LLM Factory | `app/services/llm_provider.py` | LLM provider management |
| Schema Manager | `app/services/schema_management.py` | Schema operations |
| Intent Classifier | `app/services/query_generation/nodes/intent_classifier.py` | Intent classification |
| Schema Retriever | `app/services/query_generation/nodes/schema_retriever.py` | Vector search |
| Intelligent Analyzer | `app/services/query_generation/nodes/intelligent_analyzer.py` | Query analysis |
| Query Generator | `app/services/query_generation/nodes/query_generator_node.py` | Query generation |
| Query Validator | `app/services/query_generation/nodes/query_validator.py` | Query validation |
| Query Refiner | `app/services/query_generation/nodes/query_refiner.py` | Query refinement |

### Pipeline Decision Tree

```
START
  ↓
Intent Classification
  ├─ query_generation → Continue to schema retrieval
  ├─ schema_description → Schema Description Node → END
  └─ clarification → Help Response → END
  ↓
Schema Retrieval (Vector Search)
  ├─ Found tables → Continue
  └─ No tables → Error: No relevant schema found
  ↓
Intelligent Analysis
  ├─ SINGLE_LINE → Set complexity
  └─ MULTI_LINE → Set complexity
  ↓
Query Generation (LLM)
  ↓
Query Validation
  ├─ VALID → END (Success)
  ├─ INVALID_REFINABLE
  │   ├─ refinement_count < 3 → Query Refiner → Back to Generator
  │   └─ refinement_count >= 3 → Escalate to Analyzer
  └─ INVALID_ESCALATE
      ├─ escalation_count < 2 → Back to Analyzer
      └─ escalation_count >= 2 → Error: Max attempts exceeded
```

### Performance Considerations

**Bottlenecks**:
1. **LLM API Calls**: 1-3 seconds per call
2. **Vector Search**: 50-200ms for 1000s of tables
3. **Database Queries**: 10-100ms

**Optimization Strategies**:
1. **Caching**: Aggressive caching of embeddings and frequent queries
2. **Connection Pooling**: Reuse database connections
3. **Parallel Processing**: Run independent LLM calls in parallel
4. **Streaming**: Stream LLM responses for better UX
5. **Index Optimization**: Proper indexes on vector columns

**Scalability**:
- Stateless design enables horizontal scaling
- Cache layer enables multi-instance deployment
- Async I/O for high concurrency

---

## Conclusion

Query Generator Server is a sophisticated LLM-powered system that transforms natural language into database queries through:

1. **Intelligent Pipeline**: LangGraph-based multi-stage processing
2. **Vector Search**: Semantic table discovery
3. **Multi-LLM Support**: Flexible model selection
4. **Conversation Context**: Multi-turn query understanding
5. **Validation & Refinement**: Automatic query improvement
6. **Production-Ready**: Caching, observability, error handling

The architecture is designed for **extensibility**, **reliability**, and **performance** at scale.

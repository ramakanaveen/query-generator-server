# Architecture Documentation

## System Overview

The Query Generator is a FastAPI application designed to convert natural language queries into database queries (primarily KDB/Q syntax) using large language models (LLMs). The system uses a modular architecture with clear separation of concerns to maintain flexibility and extensibility.

## High-Level Architecture

```
┌───────────────────────────┐
│         API Layer         │
│  (FastAPI Routes & WebSockets) │
└───────────────┬───────────┘
                │
┌───────────────▼───────────┐
│     Service Layer         │
│ (Business Logic & Workflows) │
└───────────────┬───────────┘
                │
┌───────────────▼───────────┐    ┌─────────────────────┐
│   LLM Provider Layer      │◄───┤   Schema Manager    │
│ (Gemini & Claude Clients) │    │  (Vector Search)    │
└───────────────┬───────────┘    └─────────────────────┘
                │                          ▲
┌───────────────▼───────────┐              │
│    External Services      │    ┌─────────┴─────────┐
│  (LLM APIs & Databases)   │    │ Embedding Service │
└───────────────────────────┘    └───────────────────┘
```

## Core Components

### 1. API Layer

The API layer is built with FastAPI and provides both REST endpoints and WebSocket connections:

- **Routes**: Defined in `app/routes/` with separate files for different domains:
  - `query.py`: Query generation endpoints
  - `conversation.py`: Conversation management
  - `directives.py`: Schema directive endpoints
  - `feedback.py`: User feedback collection
  - `schema_management.py`: Schema management endpoints
  - `websocket.py`: Real-time communication

- **Models**: Pydantic models defined in `app/schemas/` provide request/response validation and OpenAPI documentation.

### 2. Service Layer

The service layer contains the core business logic:

- **Query Generator**: Uses a LangGraph workflow to process natural language queries:
  - `QueryGenerator` class in `app/services/query_generator.py`
  - Nodes in the workflow defined in `app/services/query_generation/nodes/`
  - Pipeline: Analyze query → Retrieve schema → Generate query → Validate → Refine if needed

- **Schema Management**: Provides schema management and vector search capabilities:
  - `SchemaManager` class in `app/services/schema_management.py`
  - Stores schema hierarchies, tables, and relationships
  - Performs vector similarity search to find relevant tables for queries

- **Embedding Provider**: Generates vector embeddings for semantic search:
  - `EmbeddingProvider` class in `app/services/embedding_provider.py`
  - Uses Google's text embedding models

- **Conversation Manager**: Maintains conversation state and history:
  - `ConversationManager` class in `app/services/conversation_manager.py`
  - Stores messages and provides context for follow-up questions

- **Feedback Manager**: Collects and stores user feedback:
  - `FeedbackManager` class in `app/services/feedback_manager.py`
  - Used to improve query generation

- **Retry Generator**: Generates improved queries based on feedback:
  - `RetryGenerator` class in `app/services/retry_generator.py`

### 3. LLM Provider Layer

The LLM Provider manages connections to language models:

- **LLMProvider**: Factory for different LLM clients in `app/services/llm_provider.py`
  - Supports Gemini (via Google Vertex AI) and Claude (via Anthropic API)
  - Configurable via environment variables

### 4. Database Schema

The database schema is designed to store and manage schema information:

- **Schema Hierarchy**: 
  - `schema_groups`: Top-level schema grouping
  - `schema_definitions`: Schema definitions within groups
  - `schema_versions`: Versioned schemas for tracking changes
  - `table_definitions`: Table definitions with vector embeddings for search

- **Relationships**: 
  - `table_relationships`: Relationships between tables
  - `schema_relationships`: Relationships between schemas

- **Examples**:
  - `schema_examples`: Example queries with natural language and generated SQL
  - `example_table_mappings`: Maps examples to multiple tables

## Query Generation Pipeline

The query generation process uses LangGraph to create a directed graph of operations:

1. **Query Analysis**: Extract directives, entities, and intent from the natural language query
2. **Schema Retrieval**: Retrieve relevant schema information using vector similarity search
3. **Query Generation**: Use LLM to generate a database query using the schema and examples
4. **Query Validation**: Check query syntax and security
5. **Query Refinement**: If validation fails, refine the query using another LLM call

```
┌───────────┐    ┌───────────┐    ┌───────────┐
│  Analyze  │───►│ Retrieve  │───►│ Generate  │
│   Query   │    │  Schema   │    │   Query   │
└───────────┘    └───────────┘    └─────┬─────┘
                                         │
                  ┌────────────┐         ▼
                  │            │  ┌───────────┐
                  │    END     │◄─┤ Validate  │
                  │            │  │   Query   │
                  └────────────┘  └─────┬─────┘
                        ▲               │ 
                        │               │ Fail
                        │               │
                        │        ┌──────▼─────┐
                        └────────┤   Refine   │
                        Retry    │    Query   │
                        Limit    └──────┬─────┘
                        Reached         │
                                        │
                                 ┌──────▼─────┐
                                 │  Generate  │
                                 │   Query    │
                                 └────────────┘
```

## Schema Management and Vector Search

The system uses vector embeddings to improve schema retrieval:

1. **Schema Upload**: Schema files are uploaded and parsed
2. **Embedding Generation**: Table definitions are embedded using text embeddings
3. **Storage**: Embeddings are stored in the database with pgvector extension
4. **Query Search**: When a natural language query is received:
   - The query text is embedded
   - Similar table definitions are found using vector similarity search
   - Relevant tables are used to inform query generation

```
┌───────────┐    ┌───────────┐    ┌───────────┐
│ Schema    │───►│ Generate  │───►│  Store    │
│ Upload    │    │ Embeddings│    │  Schema   │
└───────────┘    └───────────┘    └───────────┘
                                         ▲
                                         │
                                         │
┌───────────┐    ┌───────────┐    ┌──────────┐
│ Natural   │───►│  Embed    │───►│ Vector   │
│ Language  │    │  Query    │    │ Search   │
└───────────┘    └───────────┘    └──────────┘
```

## Prompting Strategy

The system uses specialized prompts for each stage of the pipeline:

1. **Analyzer Prompt** (`app/services/query_generation/prompts/analyzer_prompts.py`):
   - Extracts entities and intent from the natural language query

2. **Generator Prompt** (`app/services/query_generation/prompts/generator_prompts.py`):
   - Converts the analyzed query into a database query using schema information
   - Includes database-specific syntax guidelines and examples

3. **Refiner Prompt** (`app/services/query_generation/prompts/refiner_prompts.py`):
   - Provides guidance for improving queries that fail validation

4. **Refined Generator Prompt** (`app/services/query_generation/prompts/generator_prompts.py`):
   - Used to generate improved queries based on refinement guidance

## WebSocket Flow

For real-time applications, the system supports WebSockets:

1. Client connects to `/ws` with optional conversation ID
2. Client sends a query message
3. Server processes the query through the generation pipeline
4. Server sends thinking/status updates during processing
5. Server sends the generated query
6. Client can request query execution
7. Server executes the query and returns results

## Configuration

The system uses a centralized configuration system (`app/core/config.py`) with environment variables and defaults, supporting:

- Debug mode
- API keys for LLM services
- Model selection and parameters
- Embedding models and parameters
- Database connection details
- CORS settings
- Schema directory location
- Vector search parameters

## Data Flow

1. **Schema Management Flow**:
   - Schema file is uploaded via API or script
   - Schema is parsed and stored in hierarchical database structure
   - Table definitions are embedded for vector search
   - Examples are processed and mapped to tables

2. **Query Generation Flow**:
   - User input received via REST API or WebSocket
   - Query is analyzed to extract entities and intent
   - Vector search finds relevant schemas and tables
   - Query is generated using LLM with schema context
   - Query is validated and refined if needed
   - Generated query is returned to client
   - Optionally, query is executed against database

3. **Feedback and Refinement Flow**:
   - User provides feedback on generated query
   - Feedback is stored for future improvements
   - Retry service generates improved query
   - Improved query is returned to client

## Security Considerations

- **Input Validation**: All inputs are validated using Pydantic models
- **Query Validation**: Generated queries are checked for syntax issues and potential security problems
- **API Keys**: Stored in environment variables, not in code
- **CORS**: Configured to restrict access to specified origins
- **Database Security**: Vector operations use parameterized queries to prevent injection

## Extensibility

The system is designed for extensibility:

- **New LLMs**: Add new providers to the `LLMProvider` class
- **Database Types**: Support for additional database types in query generation
- **Schema Formats**: Flexible schema structure with versioning
- **Embedding Models**: Configurable embedding service for vector search

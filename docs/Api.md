# API Documentation

This document provides detailed information about the Query Generator API endpoints, request/response formats, and usage examples.

## Base URL

All API endpoints are relative to the base URL:

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API does not implement authentication. In a production environment, you should add appropriate authentication mechanisms.

## Common Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input parameters |
| 404 | Not Found - Resource not found |
| 500 | Server Error - Something went wrong on the server |

## Endpoints

### Query Generation

#### Generate a Query

Converts a natural language query into a database query.

```
POST /query
```

**Request Body:**

```json
{
  "query": "Show me the top 5 AAPL trades by size today",
  "model": "gemini",  // Optional, default: "gemini", options: "gemini", "claude"
  "database_type": "kdb",  // Optional, default: "kdb"
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",  // Optional
  "conversation_history": [  // Optional
    {
      "role": "user",
      "content": "What was the average price of MSFT trades yesterday?"
    },
    {
      "role": "assistant",
      "content": "select avg price from trades where date=.z.d-1, sym=`MSFT"
    }
  ]
}
```

**Response:**

```json
{
  "generated_query": "xdesc `size select top 5 from trades where date=.z.d, sym=`AAPL",
  "execution_id": "550e8400-e29b-41d4-a716-446655440000",
  "thinking": [
    "Received query: Show me the top 5 AAPL trades by size today",
    "Analyzing query to extract directives, entities, and intent...",
    "Extracted directives: []",
    "Extracted entities: [\"AAPL\", \"trades\", \"size\", \"top 5\"]",
    "Identified intent: get_sorted_limited_data",
    "Retrieving schema information using vector search...",
    "Found relevant table: spot.market_price (similarity: 0.78)",
    "Built schema from 1 tables across 1 schemas",
    "Generating database query...",
    "Generated query: xdesc `size select top 5 from trades where date=.z.d, sym=`AAPL",
    "Validating generated query...",
    "Query validation passed"
  ]
}
```

### Conversations

#### Create a New Conversation

Creates a new conversation session for maintaining context across queries.

```
POST /conversations
```

**Response:**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "messages": [],
  "created_at": "2025-03-28T09:30:00.000Z",
  "updated_at": "2025-03-28T09:30:00.000Z",
  "metadata": {}
}
```

#### List All Conversations

Returns a list of all conversations.

```
GET /conversations
```

**Response:**

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "messages": [...],
    "created_at": "2025-03-28T09:30:00.000Z",
    "updated_at": "2025-03-28T09:30:00.000Z",
    "metadata": {}
  },
  ...
]
```

#### Get a Specific Conversation

Returns details of a specific conversation.

```
GET /conversations/{conversation_id}
```

**Response:**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "messages": [
    {
      "id": "660e8400-e29b-41d4-a716-446655440001",
      "role": "user",
      "content": "Show me the top 5 AAPL trades by size today",
      "timestamp": "2025-03-28T09:32:15.000Z",
      "metadata": null
    },
    {
      "id": "770e8400-e29b-41d4-a716-446655440002",
      "role": "assistant",
      "content": "xdesc `size select top 5 from trades where date=.z.d, sym=`AAPL",
      "timestamp": "2025-03-28T09:32:18.000Z",
      "metadata": {
        "execution_id": "880e8400-e29b-41d4-a716-446655440003"
      }
    }
  ],
  "created_at": "2025-03-28T09:30:00.000Z",
  "updated_at": "2025-03-28T09:32:18.000Z",
  "metadata": {}
}
```

#### Add a Message to a Conversation

Adds a message to an existing conversation.

```
POST /conversations/{conversation_id}/messages
```

**Request Body:**

```json
{
  "id": "660e8400-e29b-41d4-a716-446655440001",  // Optional, generated if omitted
  "role": "user",  // Required: "user" or "assistant"
  "content": "Show me the top 5 AAPL trades by size today",  // Required
  "metadata": {  // Optional
    "client_info": "web_app_v1.0"
  }
}
```

**Response:**

Returns the created message with additional fields like timestamp.

```json
{
  "id": "660e8400-e29b-41d4-a716-446655440001",
  "role": "user",
  "content": "Show me the top 5 AAPL trades by size today",
  "timestamp": "2025-03-28T09:32:15.000Z",
  "metadata": {
    "client_info": "web_app_v1.0"
  }
}
```

### Directives

#### Get Available Directives

Returns a list of available directives based on schema files.

```
GET /directives
```

**Response:**

```json
{
  "directives": [
    {
      "id": 1,
      "name": "SPOT",
      "description": "Spot market trading data",
      "icon": "TrendingUp"
    },
    {
      "id": 2,
      "name": "FX",
      "description": "Foreign exchange market data",
      "icon": "RefreshCw"
    },
    ...
  ]
}
```

### Schema Management

#### Upload a Schema

Upload a schema file to import into the database.

```
POST /schemas/upload
```

**Request Form Data:**

- `file`: The schema file to upload (multipart/form-data)
- `name` (optional): Name for the schema
- `description` (optional): Description of the schema

**Response:**

```json
{
  "success": true,
  "schema_id": 1,
  "version_id": 1,
  "name": "SPOT",
  "table_count": 3,
  "example_count": 4,
  "activated": true
}
```

#### List All Schemas

Returns a list of all available schemas.

```
GET /schemas
```

**Response:**

```json
{
  "schemas": [
    {
      "id": 1,
      "name": "SPOT",
      "description": "Spot market trading data",
      "current_version": 1,
      "version_id": 1,
      "table_count": 3
    },
    {
      "id": 2,
      "name": "FX",
      "description": "Foreign exchange market data",
      "current_version": 1,
      "version_id": 2,
      "table_count": 2
    }
  ]
}
```

### Feedback

#### Save Flexible Feedback

Saves user feedback about generated queries in a flexible format.

```
POST /feedback/flexible
```

**Request Body:**

The endpoint accepts any JSON structure, but here's a recommended format:

```json
{
  "query_id": "550e8400-e29b-41d4-a716-446655440000",
  "feedback_type": "positive",  // or "negative"
  "original_query": "Show me the top 5 AAPL trades by size today",
  "generated_query": "xdesc `size select top 5 from trades where date=.z.d, sym=`AAPL",
  "rating": 5,  // Optional numeric rating
  "comment": "Perfect query, exactly what I needed",  // Optional comment
  "conversation_id": "660e8400-e29b-41d4-a716-446655440001"  // Optional
}
```

**Response:**

```json
{
  "id": "770e8400-e29b-41d4-a716-446655440002",
  "status": "success",
  "message": "Feedback saved successfully"
}
```

### Retry

#### Generate Improved Query

Generates an improved query based on user feedback.

```
POST /retry
```

**Request Body:**

```json
{
  "original_query": "Show me the top 5 AAPL trades by size today",
  "original_generated_query": "select top 5, sym, size from trades where date=.z.d, sym=`AAPL",
  "feedback": "I need the query to sort by size in descending order",
  "model": "gemini",  // Optional, default: "gemini"
  "database_type": "kdb",  // Optional, default: "kdb"
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",  // Optional
  "conversation_history": [  // Optional
    {
      "role": "user",
      "content": "Show me the top 5 AAPL trades by size today"
    },
    {
      "role": "assistant",
      "content": "select top 5, sym, size from trades where date=.z.d, sym=`AAPL"
    }
  ]
}
```

**Response:**

```json
{
  "generated_query": "xdesc `size select top 5 from trades where date=.z.d, sym=`AAPL",
  "execution_id": "880e8400-e29b-41d4-a716-446655440003",
  "thinking": [
    "Received request to improve query based on feedback",
    "Original query: Show me the top 5 AAPL trades by size today",
    "Original generated query: select top 5, sym, size from trades where date=.z.d, sym=`AAPL",
    "User feedback: I need the query to sort by size in descending order",
    "Generating improved query based on feedback...",
    "Generated improved query: xdesc `size select top 5 from trades where date=.z.d, sym=`AAPL"
  ]
}
```

## WebSocket API

The WebSocket API provides real-time query generation and execution.

### Connect

Connect to the WebSocket endpoint:

```
ws://localhost:8000/ws
```

You can provide authentication and conversation ID in the connection request.

### Events

#### Client-to-Server Events

**Query Generation**

```json
{
  "event": "query",
  "data": {
    "content": "Show me the top 5 AAPL trades by size today",
    "model": "gemini"  // Optional
  }
}
```

**Query Execution**

```json
{
  "event": "execute",
  "data": {
    "query": "xdesc `size select top 5 from trades where date=.z.d, sym=`AAPL",
    "params": {}  // Optional parameters
  }
}
```

#### Server-to-Client Events

**Status Updates**

```json
{
  "event": "status",
  "data": {
    "type": "thinking"  // or "executing", "connected", etc.
  }
}
```

**Query Generated**

```json
{
  "event": "query_generated",
  "data": {
    "content": "xdesc `size select top 5 from trades where date=.z.d, sym=`AAPL",
    "execution_id": "550e8400-e29b-41d4-a716-446655440000",
    "thinking": [...]
  }
}
```

**Query Results**

```json
{
  "event": "results",
  "data": {
    "results": [
      {
        "time": "09:30:00",
        "ticker": "AAPL",
        "price": 150.25,
        "quantity": 1000
      },
      ...
    ],
    "metadata": {
      "execution_time": 0.05,
      "row_count": 5
    }
  }
}
```

**Error**

```json
{
  "event": "error",
  "data": {
    "message": "Error executing query: Invalid syntax"
  }
}
```

## Error Handling

The API returns HTTP status codes and error details in the response body:

```json
{
  "detail": "Failed to generate query: Invalid input parameter"
}
```

For WebSocket errors, the error message is sent via the "error" event.

## Vector Search Parameters

When using schema-based query generation, you can control how relevant schemas are found:

- `SCHEMA_SIMILARITY_THRESHOLD`: Minimum similarity threshold for matching tables (default: 0.65)
- `SCHEMA_MAX_TABLES`: Maximum number of tables to include in query context (default: 5)

These can be configured in your environment variables.

## Database Schema

The API uses a PostgreSQL database with pgvector extension to store schemas and perform vector searches. The database schema includes:

- `schema_groups`: Top-level schema grouping
- `schema_definitions`: Schema definitions within groups
- `schema_versions`: Versioned schemas for tracking changes
- `table_definitions`: Table definitions with vector embeddings for search
- `schema_examples`: Example queries with natural language and generated SQL

You can initialize the database using the provided scripts:

```bash
python scripts/init_database.py
```

## Rate Limiting

Currently, the API does not implement rate limiting. In a production environment, you should add appropriate rate limiting controls.

# Query Generator API

A FastAPI application that generates database queries from natural language using LLMs (Large Language Models). The system converts plain English questions into valid database queries with a focus on KDB/Q syntax.

## 🌟 Key Features

- **Natural Language to Database Query**: Convert English questions to KDB/Q queries
- **Multi-LLM Support**: Use either Google's Gemini or Anthropic's Claude models
- **Vector Search**: Find relevant schemas and tables using semantic similarity
- **Schema Management**: Upload and manage database schemas via API
- **Conversation Context**: Maintain context for follow-up questions
- **Shared Conversations**: Share conversations via secure links with read-only access ⭐ NEW
- **Real-time Processing**: WebSocket support for real-time applications
- **Feedback & Refinement**: Collect feedback and generate improved queries
- **Database Integration**: Ready for PostgreSQL with pgvector extension

## 🏗️ Architecture

This application follows a service-oriented architecture with the following components:

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  API Layer  │ ──── │  Services   │ ──── │ LLM Modules │
└─────────────┘      └─────────────┘      └─────────────┘
       │                    │                    │
       │                    │                    │
       ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Schemas   │      │  Database   │      │  Embedding  │
│  & Models   │      │ Connectors  │      │   Services  │
└─────────────┘      └─────────────┘      └─────────────┘
```

## 📋 Prerequisites

- Python 3.10 or higher
- PostgreSQL with pgvector extension (for vector search capabilities)
- Google Cloud project with Vertex AI API enabled (for Gemini)
- Anthropic API key (for Claude)
- KDB+ database (optional, for executing generated queries)

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone <URL>
   cd query-generator
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root based on the settings in `app/core/config.py`:
   ```
   DEBUG=True
   API_KEY_ANTHROPIC=your_anthropic_api_key
   GOOGLE_CREDENTIALS_PATH=./google-credentials.json
   GOOGLE_PROJECT_ID=your_google_project_id
   GOOGLE_LOCATION=us-central1
   GOOGLE_EMBEDDING_MODEL_NAME=text-embedding-005
   GOOGLE_EMBEDDING_ENDPOINT=
   DATABASE_URL=postgresql://username:password@localhost:5432/query_generator
   ```

5. Set up your Google Cloud credentials:
   - Download your Google Cloud service account key as JSON
   - Save it as `google-credentials.json` in the project root or update the path in your `.env` file

6. Initialize the database:
   ```bash
   python scripts/init_database.py
   ```

7. Import schema files:
   ```bash
   python scripts/import_schema.py --file app/schemas/spot.json
   ```

## 🏃‍♂️ Running the Application

Start the application with:

```bash
python run.py
```

The API will be available at http://localhost:8000

API documentation (Swagger UI) will be available at http://localhost:8000/docs

## 🔬 Testing Connections

### Test LLM Connections

To verify your LLM connections are working:

```bash
python test_llm_connection.py
```

### Test Vector Search

To test the vector search functionality:

```bash
python scripts/test_vector_search.py "Show me AAPL trades with highest volume"
```

## 📝 Example Usage

### Convert Natural Language to a KDB/Q Query:

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me the top 5 AAPL trades by size today",
    "model": "gemini",
    "database_type": "kdb"
  }'
```

Response:
```json
{
  "generated_query": "xdesc `size select top 5 from trades where date=.z.d, sym=`AAPL",
  "execution_id": "550e8400-e29b-41d4-a716-446655440000",
  "thinking": ["Received query: Show me the top 5 AAPL trades by size today", "..."]
}
```

### Upload a Schema:

```bash
curl -X POST "http://localhost:8000/api/v1/schemas/upload" \
  -F "file=@./my_schema.json" \
  -F "name=BONDS" \
  -F "description=Bond market schema"
```

### Share a Conversation ⭐ NEW:

```bash
# Create a shareable link
curl -X POST "http://localhost:8000/api/v1/conversations/conv-123/share" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-A",
    "access_level": "view"
  }'
```

Response:
```json
{
  "success": true,
  "share_token": "Y5NPUVK82jTseK7fT1uGce7NFqRqO3UjV412G8hoI_8",
  "share_url": "http://localhost:8000/api/v1/shared/Y5NPUVK82..."
}
```

Now anyone with the link can view the conversation (read-only):
```bash
curl "http://localhost:8000/api/v1/shared/Y5NPUVK82jTseK7fT1uGce7NFqRqO3UjV412G8hoI_8"
```

See [Shared Conversations Guide](./docs/SHARED_CONVERSATIONS.md) for complete documentation.

## 📚 API Endpoints

### Query Generation
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/query` | POST | Generate a database query from natural language |
| `/api/v1/retry` | POST | Generate an improved query based on feedback |
| `/api/v1/directives` | GET | Get available directives based on schema files |
| `/api/v1/feedback/flexible` | POST | Save user feedback |

### Conversations
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/conversations` | POST | Create a new conversation |
| `/api/v1/conversations/{id}` | GET | Get a specific conversation |
| `/api/v1/conversations/{id}/messages` | POST | Add a message to a conversation |
| `/api/v1/user/{user_id}/conversations/all` | GET | Get all conversations (owned + shared) |

### Shared Conversations ⭐ NEW
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/conversations/{id}/share` | POST | Create a shareable link |
| `/api/v1/shared/{token}` | GET | Access a shared conversation |
| `/api/v1/conversations/{id}/shares` | GET | List all shares for a conversation |
| `/api/v1/conversations/{id}/shares/{share_id}` | DELETE | Revoke a share |

### Schema Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/schemas/upload` | POST | Upload a schema file |
| `/api/v1/schemas` | GET | List all available schemas |

### WebSocket
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ws` | WebSocket | Real-time query generation and execution |

## 📂 Project Structure

```
app/
├── core/                  # Core configuration and utilities
├── routes/                # API route handlers
├── schemas/               # Pydantic models and JSON schema files
├── services/              # Business logic services
│   ├── embedding_provider.py  # Vector embedding service
│   ├── schema_management.py   # Schema management service
│   ├── query_generator.py     # Query generation orchestrator
│   └── query_generation/      # Query generation pipeline
│       ├── nodes/             # Pipeline nodes
│       └── prompts/           # LLM prompts
└── main.py                # Application entry point
scripts/                   # Utility scripts
├── db_scripts/            # Database initialization scripts
├── import_schema.py       # Schema import script
├── init_database.py       # Database initialization script
└── test_vector_search.py  # Vector search testing script
```

## 📖 Documentation

For more detailed documentation, see the `/docs` directory:

- [Architecture](./docs/Architecture.md)
- [API Reference](./docs/API.md)
- [Development Guide](./docs/Development.md)
- [Configuration Options](./docs/Configuration.md)
- [Schema Format](./docs/SchemaFormat.md)
- [Troubleshooting](./docs/Troubleshooting.md)
- **[Shared Conversations](./docs/SHARED_CONVERSATIONS.md)** ⭐ NEW - Complete guide to sharing conversations

## 🔒 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `False` |
| `API_KEY_ANTHROPIC` | Anthropic API key | `""` |
| `GOOGLE_CREDENTIALS_PATH` | Path to Google credentials JSON | `"./google-credentials.json"` |
| `GOOGLE_PROJECT_ID` | Google Cloud project ID | `""` |
| `GOOGLE_LOCATION` | Google Cloud location | `"us-central1"` |
| `GOOGLE_EMBEDDING_MODEL_NAME` | Google embedding model | `"text-embedding-005"` |
| `GEMINI_MODEL_NAME` | Gemini model to use | `"gemini-1.5-pro-002"` |
| `CLAUDE_MODEL_NAME` | Claude model to use | `"claude-3-sonnet-20240229"` |
| `DATABASE_URL` | Database connection string | `"sqlite:///./test.db"` |
| `SCHEMAS_DIRECTORY` | Path to schema files | `"app/schemas"` |
| `SCHEMA_SIMILARITY_THRESHOLD` | Minimum similarity for schema matching | `0.65` |
| `SCHEMA_MAX_TABLES` | Maximum tables to include in query context | `5` |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

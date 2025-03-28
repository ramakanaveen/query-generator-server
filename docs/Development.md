# Development Guide

This guide provides information for developers who want to contribute to or extend the Query Generator project.

## Development Environment Setup

### Prerequisites

- Python 3.10 or higher
- PostgreSQL with pgvector extension installed
- Git
- A code editor (VSCode, PyCharm, etc.)
- Access to Google Cloud Platform (for Gemini and Embeddings)
- Access to Anthropic API (for Claude)

### Setup Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/query-generator.git
   cd query-generator
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the project root:

   ```
   DEBUG=True
   API_KEY_ANTHROPIC=your_anthropic_api_key
   GOOGLE_CREDENTIALS_PATH=./google-credentials.json
   GOOGLE_PROJECT_ID=your_google_project_id
   GOOGLE_LOCATION=us-central1
   GOOGLE_EMBEDDING_MODEL_NAME=text-embedding-005
   DATABASE_URL=postgresql://username:password@localhost:5432/query_generator
   ```

5. **Set up Google Cloud credentials:**

   - Create a service account in Google Cloud Console
   - Grant it access to Vertex AI
   - Download the JSON key
   - Save it as `google-credentials.json` in the project root

6. **Initialize the database:**

   ```bash
   python scripts/init_database.py
   ```

7. **Import schemas (optional):**

   ```bash
   python scripts/import_schema.py --file app/schemas/spot.json
   ```

8. **Verify LLM connections:**

   ```bash
   python test_llm_connection.py
   ```

9. **Test vector search (if schemas imported):**

   ```bash
   python scripts/test_vector_search.py "Show me AAPL trades"
   ```

10. **Run the development server:**

    ```bash
    python run.py
    ```

    The server will start with auto-reload enabled.

## Project Structure

```
app/
├── core/                            # Core configuration and utilities
│   ├── config.py                    # Configuration settings
│   └── logging.py                   # Logging setup
├── routes/                          # API route handlers
│   ├── conversation.py              # Conversation endpoints
│   ├── directives.py                # Schema directives endpoints
│   ├── feedback.py                  # Feedback collection endpoints
│   ├── query.py                     # Query generation endpoints
│   ├── schema_management.py         # Schema management endpoints
│   └── websocket.py                 # WebSocket handlers
├── schemas/                         # Data models & database schemas
│   ├── conversation.py              # Conversation models
│   ├── feedback.py                  # Feedback models
│   ├── query.py                     # Query models
│   ├── default.json                 # Default database schema
│   └── spot.json                    # Spot market schema
├── services/                        # Business logic
│   ├── conversation_manager.py      # Conversation handling
│   ├── database_connector.py        # Database connection
│   ├── embedding_provider.py        # Embedding generation service
│   ├── feedback_manager.py          # Feedback management
│   ├── llm_provider.py              # LLM provider factory
│   ├── query_generator.py           # Query generation orchestrator
│   ├── retry_generator.py           # Query improvement service
│   ├── schema_management.py         # Schema management service
│   └── query_generation/            # Query generation pipeline
│       ├── nodes/                   # Pipeline nodes
│       │   ├── query_analyzer.py    # Query analysis
│       │   ├── schema_retriever.py  # Schema retrieval
│       │   ├── query_generator_node.py # Query generation
│       │   ├── query_validator.py   # Query validation
│       │   └── query_refiner.py     # Query refinement
│       └── prompts/                 # LLM prompts
│           ├── analyzer_prompts.py  # Analysis prompts
│           ├── generator_prompts.py # Generation prompts
│           └── refiner_prompts.py   # Refinement prompts
└── main.py                          # Application entry point
scripts/                             # Utility scripts
├── db_scripts/                      # Database initialization scripts
│   ├── 01_schema_tables.sql         # Schema tables creation
│   ├── 02_auth_tables.sql           # Authentication tables
│   └── migration_script.sql         # Migration script
├── batch_import_schemas.py          # Batch schema import
├── import_schema.py                 # Schema import script
├── init_database.py                 # Database initialization script
└── test_vector_search.py            # Vector search testing script
```

## Key Components

### 1. FastAPI Application

The main FastAPI application is defined in `app/main.py`. It sets up API routes, middleware, and the WebSocket server.

### 2. Configuration

The application configuration is defined in `app/core/config.py` using Pydantic's `BaseSettings` for environment variable validation and type conversion.

### 3. Routes

API routes are organized by domain in the `app/routes/` directory:

- `query.py`: Natural language to query generation
- `conversation.py`: Conversation management
- `directives.py`: Schema directive management
- `feedback.py`: User feedback collection
- `schema_management.py`: Schema upload and management
- `websocket.py`: Real-time WebSocket communication

### 4. Services

Business logic is encapsulated in service classes in the `app/services/` directory:

- `llm_provider.py`: Factory for LLM clients (Gemini, Claude)
- `embedding_provider.py`: Service for generating embeddings
- `schema_management.py`: Schema storage and vector search
- `query_generator.py`: Main query generation workflow
- `conversation_manager.py`: Conversation context management
- `database_connector.py`: Database connection and query execution
- `feedback_manager.py`: User feedback collection and storage
- `retry_generator.py`: Query improvement based on feedback

### 5. Query Generation Pipeline

The query generation process uses LangGraph to create a pipeline:

- `query_analyzer.py`: Extract entities and intent
- `schema_retriever.py`: Find relevant schemas using vector search
- `query_generator_node.py`: Generate database query using LLM
- `query_validator.py`: Validate query syntax and security
- `query_refiner.py`: Refine query if validation fails

### 6. Database Schema

The database schema is defined in SQL scripts in `scripts/db_scripts/`:

- `01_schema_tables.sql`: Schema management tables
- `02_auth_tables.sql`: Authentication tables
- `migration_script.sql`: Migration script template

## Adding New Features

### Adding a New API Endpoint

1. Identify the appropriate router file in `app/routes/`
2. Add your endpoint function with appropriate decorators:

   ```python
   @router.post("/your-endpoint", response_model=YourResponseModel)
   async def your_endpoint(request: YourRequestModel):
       # Your implementation
       return {"result": "success"}
   ```

3. Add any required models to `app/schemas/`

### Adding a New Schema Format

The system supports JSON schema files that follow the expected structure. To add a new schema:

1. Create a new JSON file with schema information
2. Include tables with columns and their descriptions
3. Add example queries to help the LLM
4. Import using the import script or API:

   ```bash
   python scripts/import_schema.py --file your_schema.json
   ```

### Adding a New LLM Provider

1. Update the `LLMProvider` class in `app/services/llm_provider.py`:

   ```python
   def __init__(self):
       self.models = {
           "gemini": self._init_gemini,
           "claude": self._init_claude,
           "new_model": self._init_new_model  # Add your new model
       }
   
   def _init_new_model(self):
       """Initialize and return a new model instance."""
       # Add your implementation here
       return NewModelClass(
           api_key=settings.NEW_MODEL_API_KEY,
           # Other parameters
       )
   ```

2. Update the configuration in `app/core/config.py` to include settings for your new model

### Adding a New Embedding Provider

1. Update the `EmbeddingProvider` class in `app/services/embedding_provider.py`:

   ```python
   # Add a new method for the new embedding model
   async def get_embedding_from_new_provider(self, text):
       # Implementation for the new provider
       return embeddings
   ```

2. Update the configuration in `app/core/config.py` to include settings for your new embedding model

### Modifying the Query Generation Pipeline

1. To add a new node to the pipeline:
   - Create a new node function in `app/services/query_generation/nodes/`
   - Add the node to the pipeline in `app/services/query_generator.py`

2. To modify an existing node:
   - Update the node function in `app/services/query_generation/nodes/`
   - Ensure it maintains the same interface (accepts and returns a state object)

3. To modify prompts:
   - Update the prompt templates in `app/services/query_generation/prompts/`

## Working with LangGraph

The query generation pipeline uses LangGraph to create a directed graph workflow:

```python
# Initialize the workflow
workflow = StateGraph(QueryGenerationState)

# Add nodes
workflow.add_node("query_analyzer", query_analyzer.analyze_query)
workflow.add_node("schema_retriever", schema_retriever.retrieve_schema)
# ...

# Set the entry point
workflow.set_entry_point("query_analyzer")

# Add edges
workflow.add_edge("query_analyzer", "schema_retriever")
workflow.add_edge("schema_retriever", "query_generator")
# ...

# Add conditional edges
workflow.add_conditional_edges(
    "query_validator",
    lambda state: END if state.validation_result else (
        END if state.refinement_count >= state.max_refinements else "query_refiner"
    )
)

# Add conditional edge to skip query generation if no schema found
workflow.add_conditional_edges(
    "schema_retriever",
    lambda state: END if state.no_schema_found else "query_generator"
)

# Compile the workflow
return workflow.compile()
```

To modify the workflow:
1. Add/modify nodes as needed
2. Update the edges to define the flow between nodes
3. Use conditional edges to create branching logic

## Vector Search and Embeddings

The schema retrieval uses vector embeddings for semantic search:

1. **Table Embedding**: When a schema is uploaded, table definitions are embedded
2. **Query Embedding**: When a query is received, it is embedded
3. **Similarity Search**: Vector similarity is used to find relevant tables
4. **Schema Building**: The most relevant tables are combined into a schema for the LLM

To modify the vector search parameters:
1. Update the `SCHEMA_SIMILARITY_THRESHOLD` and `SCHEMA_MAX_TABLES` settings
2. Change the embedding model in `GOOGLE_EMBEDDING_MODEL_NAME`

## LLM Prompting

The system uses different prompts for each stage of the pipeline:

1. **Analyzer Prompt**: Extracts entities and intent from natural language
2. **Generator Prompt**: Generates database queries
3. **Refiner Prompt**: Provides guidance for queries that fail validation
4. **Refined Generator Prompt**: Generates improved queries based on guidance

When updating prompts:
- Test the prompt with different inputs
- Include examples in the prompt
- Be specific about the expected output format
- Include database-specific syntax guidelines

## Database Management

### Initializing the Database

```bash
python scripts/init_database.py --connection postgresql://user:pass@localhost/db
```

### Importing Schemas

```bash
# Import a single schema file
python scripts/import_schema.py --file path/to/schema.json

# Import all schemas in a directory
python scripts/import_schema.py --directory path/to/schemas
```

### Testing Vector Search

```bash
python scripts/test_vector_search.py "Your query here" --threshold 0.5 --max 10
```

## Testing

### Running Tests

Run tests using pytest:

```bash
python -m pytest
```

### Adding Tests

1. Create test files in a `tests/` directory
2. Use pytest fixtures to set up test data
3. Write tests for each component:

```python
def test_query_generator():
    # Setup
    llm = MockLLM()
    query_generator = QueryGenerator(llm)
    
    # Execute
    result, thinking = await query_generator.generate(
        "Show me AAPL trades",
        database_type="kdb"
    )
    
    # Assert
    assert "AAPL" in result
    assert "trades" in result
```

## Debugging

1. Enable debug mode in `.env`:
   ```
   DEBUG=True
   ```

2. Use the logger for debugging:
   ```python
   from app.core.logging import logger
   
   logger.debug("Variable value: %s", variable)
   ```

3. Monitor the logs in the console when running the application

4. Test vector search with the test script:
   ```bash
   python scripts/test_vector_search.py "Your query here" --threshold 0.5
   ```

## Performance Considerations

1. **Database Connection Pool**: Use connection pooling for database operations
2. **Embedding Caching**: Consider caching embeddings for frequently used queries
3. **LLM Calls**: LLM API calls are expensive and slow. Minimize the number of calls by:
   - Caching responses where appropriate
   - Combining steps where possible
   - Using more efficient prompts
4. **Vector Search**: Fine-tune similarity thresholds for optimal performance
5. **Async Operations**: Use async where possible for I/O-bound operations:
   ```python
   async def your_function():
       result = await async_operation()
       return result
   ```

## Deployment

### Docker Deployment

1. Create a `Dockerfile`:
   ```dockerfile
   FROM python:3.10-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. Build and run the Docker image:
   ```bash
   docker build -t query-generator .
   docker run -p 8000:8000 --env-file .env query-generator
   ```

### Cloud Deployment

For deploying to cloud platforms:

1. **Google Cloud Run**:
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/query-generator
   gcloud run deploy query-generator --image gcr.io/PROJECT_ID/query-generator --platform managed
   ```

2. **Azure App Service**:
   ```bash
   az webapp up --runtime PYTHON:3.10 --sku B1 --logs
   ```

3. **AWS Elastic Beanstalk**:
   ```bash
   eb init -p python-3.10 query-generator
   eb create query-generator-env
   ```

## Contributing Guidelines

1. **Code Style**: Follow PEP 8 guidelines
2. **Documentation**: Document all functions, classes, and modules
3. **Testing**: Add tests for new features
4. **Pull Requests**: Create a feature branch and submit a PR with a clear description
5. **Commits**: Use meaningful commit messages

## Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   - Verify PostgreSQL is running
   - Check the connection string in `.env`
   - Make sure pgvector extension is installed

2. **LLM Connection Failures**:
   - Check API keys in `.env`
   - Verify Google credentials file exists and has correct permissions
   - Run `test_llm_connection.py` to diagnose

3. **Vector Search Not Working**:
   - Ensure schemas have been imported successfully
   - Check the embedding model is accessible
   - Adjust the similarity threshold if results are too restrictive
   - Run `test_vector_search.py` to debug

4. **Schema Import Errors**:
   - Validate your schema JSON format
   - Check database connection and permissions
   - Verify the schema file path is correct

5. **Query Generation Errors**:
   - Check the logs for specific error messages
   - Verify the prompt templates are correct
   - Test with simpler queries first
   - Ensure the LLM services are responding

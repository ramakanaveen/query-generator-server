# Configuration Guide

This guide explains how to configure the Query Generator application for different environments and use cases.

## Environment Variables

The application is configured using environment variables, which can be set in a `.env` file in the project root directory. The following environment variables are available:

### General Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `DEBUG` | Enable debug mode | `False` | `True` |
| `DATABASE_URL` | Database connection string | `sqlite:///./test.db` | `postgresql://user:pass@localhost/db` |
| `SCHEMAS_DIRECTORY` | Path to schema files | `app/schemas` | `/data/schemas` |

### LLM Settings

#### Google Vertex AI (Gemini)

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `GOOGLE_CREDENTIALS_PATH` | Path to Google credentials JSON | `./google-credentials.json` | `/secrets/google-creds.json` |
| `GOOGLE_PROJECT_ID` | Google Cloud project ID | `""` | `my-project-123` |
| `GOOGLE_LOCATION` | Google Cloud location | `us-central1` | `europe-west1` |
| `GEMINI_MODEL_NAME` | Gemini model to use | `gemini-1.5-pro-002` | `gemini-1.5-flash-001` |
| `GEMINI_TEMPERATURE` | Temperature for generation | `0.2` | `0.5` |
| `GEMINI_TOP_P` | Top-p sampling parameter | `0.95` | `0.8` |

#### Google Embedding

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `GOOGLE_EMBEDDING_MODEL_NAME` | Google embedding model | `text-embedding-005` | `text-embedding-004` |
| `GOOGLE_EMBEDDING_ENDPOINT` | Custom embedding endpoint | `""` | `https://custom-endpoint.com` |

#### Anthropic (Claude)

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `API_KEY_ANTHROPIC` | Anthropic API key | `""` | `sk-ant-api...` |
| `CLAUDE_MODEL_NAME` | Claude model to use | `claude-3-sonnet-20240229` | `claude-3-opus-20240229` |
| `CLAUDE_TEMPERATURE` | Temperature for generation | `0.2` | `0.7` |
| `CLAUDE_MAX_TOKENS` | Maximum tokens to generate | `4000` | `8000` |

### Schema Search Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `SCHEMA_SIMILARITY_THRESHOLD` | Minimum similarity for schema matching | `0.65` | `0.5` |
| `SCHEMA_MAX_TABLES` | Maximum tables to include in query context | `5` | `10` |

### Network Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `KDB_HOST` | KDB host address | `localhost` | `kdb.example.com` |
| `KDB_PORT` | KDB port number | `5000` | `5432` |

## Configuration File

The application's configuration is managed in `app/core/config.py` using Pydantic's `BaseSettings`. This provides type validation and default values for all settings:

```python
class Settings(BaseSettings):
    # Debug mode
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    
    # API Keys
    API_KEY_ANTHROPIC: str = os.getenv("API_KEY_ANTHROPIC", "")
    
    # Google settings
    GOOGLE_CREDENTIALS_PATH: str = os.getenv("GOOGLE_CREDENTIALS_PATH", "./google-credentials.json")
    GOOGLE_PROJECT_ID: str = os.getenv("GOOGLE_PROJECT_ID", "")
    GOOGLE_LOCATION: str = os.getenv("GOOGLE_LOCATION", "us-central1")
 
    # Google embedding settings
    GOOGLE_EMBEDDING_MODEL_NAME: str = os.getenv("GOOGLE_EMBEDDING_MODEL_NAME", "text-embedding-005")
    GOOGLE_EMBEDDING_ENDPOINT: str = os.getenv("GOOGLE_EMBEDDING_ENDPOINT", "")

    # Schema settings
    SCHEMA_SIMILARITY_THRESHOLD: float = float(os.getenv("SCHEMA_SIMILARITY_THRESHOLD", "0.65"))
    SCHEMA_MAX_TABLES: int = int(os.getenv("SCHEMA_MAX_TABLES", "5"))
    
    # Gemini model settings
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro-002")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
    GEMINI_TOP_P: float = float(os.getenv("GEMINI_TOP_P", "0.95"))
    
    # Claude model settings
    CLAUDE_MODEL_NAME: str = os.getenv("CLAUDE_MODEL_NAME", "claude-3-sonnet-20240229")
    CLAUDE_TEMPERATURE: float = float(os.getenv("CLAUDE_TEMPERATURE", "0.2"))
    CLAUDE_MAX_TOKENS: int = int(os.getenv("CLAUDE_MAX_TOKENS", "4000"))
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    
    # KDB settings
    KDB_HOST: str = os.getenv("KDB_HOST", "localhost")
    KDB_PORT: int = int(os.getenv("KDB_PORT", "5000"))
    
    # CORS settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # React development server
        "http://localhost:8000",  # For local testing
        "https://your-production-domain.com",
    ]
    
    # Schema directory
    SCHEMAS_DIRECTORY: str = os.getenv("SCHEMAS_DIRECTORY", str(_BASE_DIR / "app" / "schemas"))
```

## Environment-Specific Configuration

### Development Environment

For development, create a `.env` file with:

```
DEBUG=True
API_KEY_ANTHROPIC=your_anthropic_api_key
GOOGLE_CREDENTIALS_PATH=./google-credentials.json
GOOGLE_PROJECT_ID=your_google_project_id
DATABASE_URL=postgresql://user:pass@localhost/query_generator
SCHEMA_SIMILARITY_THRESHOLD=0.5
```

### Production Environment

For production, set environment variables securely in your deployment platform:

```
DEBUG=False
API_KEY_ANTHROPIC=your_anthropic_api_key
GOOGLE_CREDENTIALS_PATH=/secrets/google-credentials.json
GOOGLE_PROJECT_ID=your_google_project_id
DATABASE_URL=postgresql://user:pass@production-db/query_generator
CORS_ORIGINS=https://your-production-domain.com
SCHEMA_SIMILARITY_THRESHOLD=0.7
```

### Testing Environment

For testing, create a `.env.test` file:

```
DEBUG=True
DATABASE_URL=postgresql://user:pass@localhost/query_generator_test
SCHEMA_SIMILARITY_THRESHOLD=0.5
SCHEMA_MAX_TABLES=3
```

## Database Configuration

### PostgreSQL with pgvector

The application uses PostgreSQL with the pgvector extension for vector similarity search. To set up:

1. Install PostgreSQL
2. Install pgvector extension:
   ```sql
   CREATE EXTENSION vector;
   ```
3. Create a database:
   ```sql
   CREATE DATABASE query_generator;
   ```
4. Run the initialization script:
   ```bash
   python scripts/init_database.py --connection postgresql://user:pass@localhost/query_generator
   ```

### Database Schema Initialization

The database schema is defined in SQL scripts in the `scripts/db_scripts/` directory:

1. `01_schema_tables.sql`: Creates the main schema tables
2. `02_auth_tables.sql`: Creates authentication tables

Initialize the database with:

```bash
python scripts/init_database.py
```

## Schema Management

### JSON Schema Format

Schema files should follow this structure:

```json
{
  "group": "MarketData",
  "schemas": [
    {
      "schema": [
        {
          "name": "spot",
          "description": "Spot market trading data",
          "tables": [
            {
              "kdb_table_name": "market_price",
              "description": "Spot market price",
              "columns": [
                {
                  "name": "date",
                  "kdb_type": "d",
                  "type": "Date",
                  "column_desc": "Trade date",
                  "references": null
                },
                {
                  "name": "time",
                  "kdb_type": "t",
                  "type": "Time",
                  "column_desc": "Trade time",
                  "references": null
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

### Uploading Schemas

Upload schemas via API:

```bash
curl -X POST "http://localhost:8000/api/v1/schemas/upload" \
  -F "file=@./my_schema.json" \
  -F "name=SPOT" \
  -F "description=Spot market schema"
```

Or using the import script:

```bash
python scripts/import_schema.py --file path/to/schema.json
```

### Vector Search Configuration

Fine-tune vector search with these settings:

- `SCHEMA_SIMILARITY_THRESHOLD`: Lower for more results (e.g., 0.5), higher for stricter matching (e.g., 0.8)
- `SCHEMA_MAX_TABLES`: Maximum number of tables to include in the query context

## Prompt Configuration

LLM prompts are defined in template files in the `app/services/query_generation/prompts/` directory:

- `analyzer_prompts.py`: Templates for query analysis
- `generator_prompts.py`: Templates for query generation
- `refiner_prompts.py`: Templates for query refinement

### Modifying Prompts

To modify a prompt:

1. Edit the template in the appropriate file
2. Focus on specific instructions and examples
3. Test the modified prompt with different inputs
4. Adjust based on results

## LLM Configuration

### Gemini Configuration

Configure Gemini with these settings:

- `GEMINI_MODEL_NAME`: The model version to use
- `GEMINI_TEMPERATURE`: Control randomness (0.0-1.0)
- `GEMINI_TOP_P`: Control diversity (0.0-1.0)

### Claude Configuration

Configure Claude with these settings:

- `CLAUDE_MODEL_NAME`: The model version to use
- `CLAUDE_TEMPERATURE`: Control randomness (0.0-1.0)
- `CLAUDE_MAX_TOKENS`: Maximum tokens to generate

## CORS Configuration

Cross-Origin Resource Sharing (CORS) is configured in `app/core/config.py` and applied in `app/main.py`:

```python
# In config.py
CORS_ORIGINS: List[str] = [
    "http://localhost:3000",  # React development server
    "http://localhost:8000",  # For local testing
    "https://your-production-domain.com",
]

# In main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

To configure CORS:

1. Add your frontend domain(s) to the `CORS_ORIGINS` list
2. For production, limit to only necessary domains
3. For more restrictive settings, modify `allow_methods` and `allow_headers`

## Logging Configuration

Logging is configured in `app/core/logging.py`:

```python
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("query-generator")
```

To configure logging:

1. Set `DEBUG=True` in your environment for detailed logs
2. Set `DEBUG=False` in production for only important logs
3. Add additional handlers as needed (file handlers, etc.)

## Advanced Configuration

### Adding a New LLM Provider

1. Update `app/core/config.py` with settings for the new provider:

   ```python
   # New LLM settings
   NEW_LLM_API_KEY: str = os.getenv("NEW_LLM_API_KEY", "")
   NEW_LLM_MODEL_NAME: str = os.getenv("NEW_LLM_MODEL_NAME", "default-model")
   NEW_LLM_TEMPERATURE: float = float(os.getenv("NEW_LLM_TEMPERATURE", "0.2"))
   ```

2. Update `app/services/llm_provider.py` to support the new provider:

   ```python
   def __init__(self):
       self.models = {
           "gemini": self._init_gemini,
           "claude": self._init_claude,
           "new_llm": self._init_new_llm
       }
   
   def _init_new_llm(self):
       """Initialize and return a New LLM model instance."""
       from new_llm_library import NewLLMClient
       
       return NewLLMClient(
           api_key=settings.NEW_LLM_API_KEY,
           model=settings.NEW_LLM_MODEL_NAME,
           temperature=settings.NEW_LLM_TEMPERATURE
       )
   ```

### Changing the Embedding Model

1. Update `app/core/config.py` with new embedding settings:

   ```python
   GOOGLE_EMBEDDING_MODEL_NAME: str = os.getenv("GOOGLE_EMBEDDING_MODEL_NAME", "text-embedding-005")
   ```

2. Update `app/services/embedding_provider.py` if needed:

   ```python
   self.embeddings = VertexAIEmbeddings(
       model_name=settings.GOOGLE_EMBEDDING_MODEL_NAME,
       project=settings.GOOGLE_PROJECT_ID,
       location=settings.GOOGLE_LOCATION,
   )
   ```

### WebSocket Configuration

1. Update `app/core/config.py` with WebSocket settings:

   ```python
   # WebSocket settings
   WS_PING_INTERVAL: int = int(os.getenv("WS_PING_INTERVAL", "25"))
   WS_PING_TIMEOUT: int = int(os.getenv("WS_PING_TIMEOUT", "60"))
   WS_MAX_SIZE: int = int(os.getenv("WS_MAX_SIZE", "1048576"))  # 1MB
   ```

2. Update `app/main.py` to use these settings:

   ```python
   sio = socketio.AsyncServer(
       async_mode="asgi",
       cors_allowed_origins=settings.CORS_ORIGINS,
       ping_interval=settings.WS_PING_INTERVAL,
       ping_timeout=settings.WS_PING_TIMEOUT,
       max_http_buffer_size=settings.WS_MAX_SIZE
   )
   ```

### LangGraph Pipeline Configuration

The query generation workflow in `app/services/query_generator.py` can be configured:

```python
# Configure maximum number of refinement attempts
max_refinements: int = Field(default=2, description="Maximum number of refinement attempts")
```

Increase this value to allow more refinement attempts for complex queries.

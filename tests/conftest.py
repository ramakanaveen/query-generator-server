import asyncio
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

@pytest.fixture
def test_client():
    """Return a TestClient instance for FastAPI route testing."""
    return TestClient(app)

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    mock_response = MagicMock()
    mock_response.content = "SELECT FROM table_name WHERE column = `value`"
    return mock_response

@pytest.fixture
def mock_chain():
    """Mock LangChain chain for testing."""
    mock = AsyncMock()
    mock.ainvoke.return_value.content = "SELECT FROM table_name WHERE column = `value`"
    return mock

@pytest.fixture
def mock_llm():
    """Mock LLM provider for testing."""
    mock = MagicMock()
    return mock

@pytest.fixture
def mock_embedding():
    """Mock embedding for testing."""
    return [0.1] * 768  # 768-dimensional embedding vector with all values as 0.1

@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider for testing."""
    mock = AsyncMock()
    mock.get_embedding.return_value = [0.1] * 768
    return mock

@pytest.fixture
def mock_conversation():
    """Sample conversation for testing."""
    return {
        "id": "conv-123",
        "messages": [
            {
                "id": "msg-1",
                "role": "user",
                "content": "Show me AAPL trades",
                "timestamp": "2023-01-01T12:00:00"
            },
            {
                "id": "msg-2",
                "role": "assistant",
                "content": "select from trades where sym=`AAPL",
                "timestamp": "2023-01-01T12:00:10"
            }
        ],
        "created_at": "2023-01-01T12:00:00",
        "updated_at": "2023-01-01T12:00:10",
        "metadata": {}
    }

@pytest.fixture
def mock_schema():
    """Sample schema for testing."""
    return {
        "description": "Schema for testing",
        "tables": {
            "market_price": {
                "description": "Market price data",
                "columns": [
                    {"name": "date", "type": "Date", "kdb_type": "d", "column_desc": "Trading date"},
                    {"name": "sym", "type": "Symbol", "kdb_type": "s", "column_desc": "Ticker symbol"},
                    {"name": "price", "type": "Float", "kdb_type": "f", "column_desc": "Price"}
                ]
            }
        },
        "examples": [
            {
                "natural_language": "Show me AAPL prices",
                "query": "select from market_price where sym=`AAPL"
            }
        ]
    }

@pytest.fixture
def sample_query_request():
    """Sample query request for testing."""
    return {
        "query": "Show me AAPL trades for today",
        "model": "claude",
        "database_type": "kdb",
        "conversation_id": "conv-123",
        "conversation_history": []
    }

@pytest.fixture
def mock_db_connection():
    """Mock database connection for testing."""
    conn = AsyncMock()
    
    # Set up fetch to return a list of dictionaries
    async def mock_fetch(query, *args):
        return [
            {"id": 1, "name": "test_table", "description": "Test table", "content": json.dumps({"columns": []}), 
             "schema_id": 1, "schema_name": "test_schema", "group_id": 1, "group_name": "test_group", 
             "schema_version_id": 1, "schema_version": 1, "similarity": 0.8}
        ]
    
    # Set up fetchval to return a single value
    async def mock_fetchval(query, *args):
        return 1
    
    conn.fetch = mock_fetch
    conn.fetchval = mock_fetchval
    conn.close = AsyncMock()
    conn.transaction = AsyncMock().__aenter__ = AsyncMock().__aexit__ = AsyncMock()
    
    return conn

@pytest.fixture
def mock_asyncpg_connect(mock_db_connection):
    """Mock asyncpg.connect for testing."""
    async def _mock_connect(*args, **kwargs):
        return mock_db_connection
    
    with patch('asyncpg.connect', _mock_connect):
        yield

# Event loop fixture for asyncio tests
@pytest.fixture
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
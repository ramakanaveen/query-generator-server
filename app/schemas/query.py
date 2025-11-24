from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class MessageHistory(BaseModel):
    role: str = Field(..., description="Message role (user or assistant)")
    content: str = Field(..., description="Message content")
    id: Optional[str] = Field(None, description="Message ID")
    timestamp: Optional[str] = Field(None, description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class QueryRequest(BaseModel):
    query: str = Field(..., description="The natural language query")
    model: str = Field(default="gemini", description="The LLM to use (gemini or claude)")
    database_type: str = Field(default="kdb", description="The type of database to query")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for context")
    conversation_history: Optional[List[Dict[str, Any]]] = Field(default=None, description="Recent conversation messages")
    
    class Config:
        # Allow extra fields to be more flexible with client data
        extra = "allow"

class QueryResponse(BaseModel):
    generated_query: Optional[str] = Field(default=None, description="The generated database query")
    generated_content: Optional[str] = Field(default=None, description="Generated content for non-query intents")
    response_type: str = Field(default="query", description="Type of response (query, schema_description, help)")
    execution_id: str = Field(..., description="ID for tracking execution")
    thinking: Optional[List[str]] = Field(default=None, description="LLM thinking steps")
    query_complexity: str = Field(default="SINGLE_LINE", description="Query complexity level (SINGLE_LINE or MULTI_LINE)")

class PaginationParams(BaseModel):
    page: int = Field(default=0, description="Page number (0-indexed)")
    page_size: int = Field(default=100, description="Number of records per page")

class ExecutionRequest(BaseModel):
    query: str = Field(..., description="The database query to execute")
    execution_id: str = Field(..., description="ID from the query generation")
    database_type: str = Field(default="kdb", description="Database type: kdb, starburst, trino, postgres, mysql")
    params: Optional[Dict[str, Any]] = Field(default={}, description="Query parameters")
    pagination: Optional[PaginationParams] = Field(default=None, description="Pagination parameters")
    query_complexity: Optional[str] = Field(default="MULTI_LINE", description="Query complexity (SINGLE_LINE or MULTI_LINE) - KDB only, defaults to MULTI_LINE for safety")
    connection_params: Optional[Dict[str, Any]] = Field(default=None, description="Optional connection parameters override (host, port, etc.). If not provided, uses settings from config.")

class ExecutionResponse(BaseModel):
    """
    Response model for query execution with enhanced pagination support
    """
    results: List[Dict[str, Any]] = Field(..., description="Query results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata")
    pagination: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "currentPage": 0,
            "totalPages": 1,
            "totalRows": 0
        }, 
        description="Pagination information"
    )
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., description="The natural language query")
    model: str = Field(default="gemini", description="The LLM to use (gemini or claude)")
    database_type: str = Field(default="kdb", description="The type of database to query")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for context")

class QueryResponse(BaseModel):
    generated_query: str = Field(..., description="The generated database query")
    execution_id: str = Field(..., description="ID for tracking execution")
    thinking: Optional[List[str]] = Field(default=None, description="LLM thinking steps")

class ExecutionRequest(BaseModel):
    query: str = Field(..., description="The database query to execute")
    execution_id: str = Field(..., description="ID from the query generation")
    params: Optional[Dict[str, Any]] = Field(default={}, description="Query parameters")

class ExecutionResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Query results")
    metadata: Dict[str, Any] = Field(..., description="Execution metadata")
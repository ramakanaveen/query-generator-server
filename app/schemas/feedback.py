from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from app.schemas.query import MessageHistory

class FeedbackRequest(BaseModel):
    query_id: str = Field(..., description="ID of the query being rated")
    feedback_type: str = Field(..., description="Type of feedback (positive/negative)")
    original_query: Optional[str] = Field(None, description="The original natural language query")
    generated_query: Optional[str] = Field(None, description="The generated database query")
    conversation_id: Optional[str] = Field(None, description="Associated conversation ID")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="When the feedback was given")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    # Make the schema more flexible for incoming data
    class Config:
        extra = "allow"  # Allow extra fields in the request

class FeedbackBatchRequest(BaseModel):
    feedback: List[FeedbackRequest] = Field(..., description="List of feedback items")

class FeedbackResponse(BaseModel):
    id: str = Field(..., description="ID of the saved feedback")
    status: str = Field(..., description="Status of the operation")
    message: str = Field(..., description="Status message")

class RetryRequest(BaseModel):
    original_query: str = Field(..., description="Original natural language query")
    original_generated_query: str = Field(..., description="Original generated query")
    feedback: str = Field(..., description="User feedback about what was wrong")
    model: str = Field(default="gemini", description="The LLM to use (gemini or claude)")
    database_type: str = Field(default="kdb", description="The type of database to query")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for context")
    conversation_history: Optional[List[MessageHistory]] = Field(default=None, description="Recent conversation messages")

class RetryResponse(BaseModel):
    generated_query: str = Field(..., description="The improved database query")
    execution_id: str = Field(..., description="ID for tracking execution")
    thinking: Optional[List[str]] = Field(default=None, description="LLM thinking steps")
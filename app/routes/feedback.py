from fastapi import APIRouter, HTTPException, Depends, Body, Request
import uuid
import json
from datetime import datetime
import traceback

from app.schemas.feedback import (
    FeedbackRequest, 
    FeedbackBatchRequest, 
    FeedbackResponse,
    RetryRequest,
    RetryResponse
)
from app.services.feedback_manager import FeedbackManager
from app.services.llm_provider import LLMProvider
from app.services.retry_generator import RetryGenerator
from app.core.logging import logger

router = APIRouter()

# Simplest possible feedback endpoint that accepts any JSON
@router.post("/feedback/flexible")
async def save_flexible_feedback(request: Request):
    """
    Super flexible endpoint to save feedback in any format.
    """
    try:
        # Get raw request body
        body = await request.json()
        logger.info(f"Received feedback: {body}")
        
        feedback_manager = FeedbackManager()
        
        # Add required fields if missing
        feedback_data = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "received_data": body
        }
        
        # Add the feedback data as-is
        if isinstance(body, dict):
            feedback_data.update(body)
        
        # Save it
        saved_feedback = await feedback_manager.save_feedback(feedback_data)
        
        return {
            "id": saved_feedback.get("id", str(uuid.uuid4())),
            "status": "success",
            "message": "Feedback saved successfully"
        }
    
    except Exception as e:
        logger.error(f"Error saving flexible feedback: {str(e)}")
        logger.error(traceback.format_exc())
        # Return success anyway to stop the client from retrying
        return {
            "id": str(uuid.uuid4()),
            "status": "success",
            "message": "Feedback acknowledged (with server error)"
        }

# Retry endpoint with proper response model
@router.post("/retry", response_model=RetryResponse)
async def retry_query(request: RetryRequest):
    """
    Generate an improved query based on user feedback.
    """
    try:
        # Initialize services
        llm_provider = LLMProvider()
        llm = llm_provider.get_model(request.model)
        retry_generator = RetryGenerator(llm)
        
        # Get conversation history if conversation_id is provided
        conversation_history = []
        if request.conversation_id:
            conversation_manager = ConversationManager()
            conversation_history = await conversation_manager.get_conversation_context(
                request.conversation_id,
                limit=5  # Get last 5 messages
            )
        
        # Use conversation history provided in request if available
        if request.conversation_history:
            conversation_history = [msg.model_dump() for msg in request.conversation_history]
        
        # Generate improved query
        execution_id = str(uuid.uuid4())
        improved_query, thinking = await retry_generator.generate_improved_query(
            request.original_query,
            request.original_generated_query,
            request.feedback,
            request.database_type,
            request.conversation_id,
            conversation_history
        )
        
        return RetryResponse(
            generated_query=improved_query,
            execution_id=execution_id,
            thinking=thinking
        )
    
    except ValueError as e:
        logger.error(f"ValueError in retry_query endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Exception in retry_query endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query retry failed: {str(e)}")
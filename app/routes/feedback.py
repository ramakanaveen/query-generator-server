# app/routes/feedback.py
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
from app.services.conversation_manager import ConversationManager
from app.services.feedback_manager import FeedbackManager
from app.services.llm_provider import LLMProvider
from app.services.query_generator import QueryGenerator
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
    Generate an improved query based on user feedback using the full enhanced workflow.
    Now includes schema context, conversation essence, and validation.
    """
    try:
        logger.info(f"🔄 Retry request received for conversation: {request.conversation_id}")
        logger.info(f"📝 Original query: {request.original_query}")
        logger.info(f"💬 User feedback: {request.feedback}")

        # Initialize services
        llm_provider = LLMProvider()
        llm = llm_provider.get_model(request.model)

        # UPDATED: Use enhanced QueryGenerator instead of RetryGenerator
        query_generator = QueryGenerator(llm, use_unified_analyzer=True)

        # Prepare conversation history
        conversation_history = []
        if request.conversation_history:
            conversation_history = [msg.model_dump() for msg in request.conversation_history]
        elif request.conversation_id:
            # If no history provided but conversation_id exists, fetch from database
            try:
                conversation_manager = ConversationManager()
                conversation = await conversation_manager.get_conversation(request.conversation_id)
                if conversation:
                    conversation_history = conversation.get("messages", [])
                    logger.info(f"📚 Loaded {len(conversation_history)} messages from conversation history")
            except Exception as e:
                logger.warning(f"Could not load conversation history: {str(e)}")

        # UPDATED: Generate using enhanced workflow with retry parameters
        execution_id = str(uuid.uuid4())
        result, thinking = await query_generator.generate(
            query=request.original_query,
            database_type=request.database_type,
            conversation_id=request.conversation_id,
            conversation_history=conversation_history,
            user_id=getattr(request, 'user_id', None),  # If user_id is added to schema
            # NEW: Retry-specific parameters
            is_retry=True,
            original_generated_query=request.original_generated_query,
            user_feedback=request.feedback
        )

        # Extract the generated query from result
        generated_query = None
        if isinstance(result, dict):
            generated_query = result.get("generated_query")
        else:
            generated_query = result

        logger.info(f"✅ Retry query generated successfully")
        logger.info(f"🔍 Thinking steps: {len(thinking)}")

        return RetryResponse(
            generated_query=generated_query or "// No query generated",
            execution_id=execution_id,
            thinking=thinking
        )

    except ValueError as e:
        logger.error(f"ValueError in retry_query endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Exception in retry_query endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query retry failed: {str(e)}")

# OPTIONAL: Add endpoint to get conversation essence for debugging
@router.get("/conversation/{conversation_id}/essence")
async def get_conversation_essence(conversation_id: str):
    """
    Get conversation essence for debugging purposes.
    """
    try:
        conversation_manager = ConversationManager()
        essence = await conversation_manager.get_conversation_essence(conversation_id)

        return {
            "conversation_id": conversation_id,
            "essence": essence,
            "has_essence": bool(essence)
        }
    except Exception as e:
        logger.error(f"Error getting conversation essence: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving essence: {str(e)}")

# OPTIONAL: Add endpoint to manually update conversation essence
@router.post("/conversation/{conversation_id}/essence")
async def update_conversation_essence_manual(conversation_id: str, essence_data: dict = Body(...)):
    """
    Manually update conversation essence (for testing/debugging).
    """
    try:
        conversation_manager = ConversationManager()
        success = await conversation_manager.update_conversation_essence(conversation_id, essence_data)

        return {
            "conversation_id": conversation_id,
            "success": success,
            "essence_updated": essence_data
        }
    except Exception as e:
        logger.error(f"Error updating conversation essence: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating essence: {str(e)}")

@router.post("/feedback/positive")
async def save_positive_feedback(request: Request):
    """
    Save positive feedback and store the query as a verified example.
    """
    try:
        body = await request.json()
        
        feedback_manager = FeedbackManager()
        feedback = await feedback_manager.save_positive_feedback(
            query_id=body.get("query_id"),
            user_id=body.get("user_id"),
            original_query=body.get("original_query", ""),
            generated_query=body.get("generated_query", ""),
            conversation_id=body.get("conversation_id"),
            metadata=body.get("metadata", {})
        )
        
        return {
            "id": feedback.get("id", str(uuid.uuid4())),
            "status": "success",
            "message": "Positive feedback saved successfully"
        }
    
    except Exception as e:
        logger.error(f"Error saving positive feedback: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "id": str(uuid.uuid4()),
            "status": "success",
            "message": "Feedback acknowledged (with server error)"
        }

@router.post("/feedback/negative")
async def save_negative_feedback(request: Request):
    """
    Save negative feedback with details about what was wrong.
    """
    try:
        body = await request.json()
        
        feedback_manager = FeedbackManager()
        feedback = await feedback_manager.save_negative_feedback(
            query_id=body.get("query_id"),
            user_id=body.get("user_id"),
            original_query=body.get("original_query", ""),
            generated_query=body.get("generated_query", ""),
            feedback_text=body.get("feedback_text", ""),
            conversation_id=body.get("conversation_id"),
            metadata=body.get("metadata", {})
        )
        
        return {
            "id": feedback.get("id", str(uuid.uuid4())),
            "status": "success",
            "message": "Negative feedback saved successfully"
        }
    
    except Exception as e:
        logger.error(f"Error saving negative feedback: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "id": str(uuid.uuid4()),
            "status": "success",
            "message": "Feedback acknowledged (with server error)"
        }

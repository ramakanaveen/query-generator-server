from fastapi import APIRouter, HTTPException, Depends, Request
import uuid
import json
from typing import Dict, Any, List

from app.schemas.query import QueryRequest, QueryResponse, ExecutionRequest, ExecutionResponse
from app.services.llm_provider import LLMProvider
from app.services.conversation_manager import ConversationManager
from app.services.query_generator import QueryGenerator
from app.services.database_connector import DatabaseConnector
from app.core.logging import logger

router = APIRouter()

# app/routes/query.py
@router.post("/query", response_model=QueryResponse)
async def generate_query(request: QueryRequest):
    """
    Generate a database query or other content from natural language.
    """
    try:
        # Extract values from validated request model
        query = request.query
        model = request.model
        database_type = request.database_type
        conversation_id = request.conversation_id
        conversation_history = request.conversation_history
        
        # Initialize services
        llm_provider = LLMProvider()
        llm = llm_provider.get_model(model)
        
        # Initialize query generator
        query_generator = QueryGenerator(llm)
        
        # Generate result
        execution_id = str(uuid.uuid4())
        result, thinking = await query_generator.generate(
            query,
            database_type,
            conversation_id,
            conversation_history
        )
        
        # Determine response type and content
        response_type = "query"
        generated_query = None
        generated_content = None
        
        if isinstance(result, dict):
            # New format with intent type
            response_type = result.get("intent_type", "query")
            generated_query = result.get("generated_query")
            generated_content = result.get("generated_content")
        else:
            # Legacy format (string query)
            generated_query = result
            response_type = "query"
        
        return QueryResponse(
            generated_query=generated_query,
            generated_content=generated_content,
            response_type=response_type,
            execution_id=execution_id,
            thinking=thinking
        )
    
    except ValueError as e:
        logger.error(f"ValueError in generate_query endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Exception in generate_query endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query generation failed: {str(e)}")
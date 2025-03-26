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

@router.post("/query", response_model=QueryResponse)
async def generate_query(request: Request):
    """
    Generate a database query from natural language.
    """
    try:
        # Get raw request data
        try:
            raw_data = await request.json()
            logger.info(f"Received query request: {json.dumps(raw_data)}")
        except Exception as e:
            logger.error(f"Error parsing request JSON: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON in request: {str(e)}")
        
        # Extract fields manually to avoid validation errors
        query = raw_data.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
            
        model = raw_data.get("model", "gemini")
        database_type = raw_data.get("database_type", "kdb")
        conversation_id = raw_data.get("conversation_id")
        conversation_history = raw_data.get("conversation_history", [])
        
        # Validate and normalize conversation history
        normalized_history = []
        if conversation_history:
            try:
                for msg in conversation_history:
                    if not isinstance(msg, dict):
                        logger.warning(f"Skipping non-dict message: {msg}")
                        continue
                        
                    if "role" not in msg or "content" not in msg:
                        logger.warning(f"Skipping message missing required fields: {msg}")
                        continue
                        
                    normalized_history.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                
                logger.info(f"Normalized conversation history: {json.dumps(normalized_history)}")
            except Exception as e:
                logger.error(f"Error normalizing conversation history: {str(e)}")
                # Continue with empty history instead of failing
        
        # Initialize services
        llm_provider = LLMProvider()
        llm = llm_provider.get_model(model)
        
        # Initialize query generator
        query_generator = QueryGenerator(llm)
        
        # Generate query
        execution_id = str(uuid.uuid4())
        generated_query, thinking = await query_generator.generate(
            query,
            database_type,
            conversation_id,
            normalized_history
        )
        
        return QueryResponse(
            generated_query=generated_query,
            execution_id=execution_id,
            thinking=thinking
        )
    
    except ValueError as e:
        logger.error(f"ValueError in generate_query endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Exception in generate_query endpoint: {str(e)}", exc_info=True)
        logger.error(f"Exception type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Query generation failed: {str(e)}")
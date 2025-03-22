from fastapi import APIRouter, HTTPException, Depends
import uuid
from typing import Dict, Any

from app.schemas.query import QueryRequest, QueryResponse, ExecutionRequest, ExecutionResponse
from app.services.llm_provider import LLMProvider
from app.services.conversation_manager import ConversationManager
from app.services.query_generator import QueryGenerator
from app.services.database_connector import DatabaseConnector

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def generate_query(request: QueryRequest):
    """
    Generate a database query from natural language.
    """
    try:
        # Initialize services
        llm_provider = LLMProvider()
        llm = llm_provider.get_model(request.model)
        
        # Initialize query generator
        query_generator = QueryGenerator(llm)
        
        # Generate query
        execution_id = str(uuid.uuid4())
        generated_query, thinking = await query_generator.generate(
            request.query,
            request.database_type,
            request.conversation_id
        )
        
        return QueryResponse(
            generated_query=generated_query,
            execution_id=execution_id,
            thinking=thinking
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query generation failed: {str(e)}")

@router.post("/execute", response_model=ExecutionResponse)
async def execute_query(request: ExecutionRequest):
    """
    Execute a database query and return results.
    """
    try:
        # Initialize database connector
        db_connector = DatabaseConnector()
        
        # Execute query
        results, metadata = await db_connector.execute(
            request.query,
            request.params
        )
        
        return ExecutionResponse(
            results=results,
            metadata=metadata
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")
import time
from functools import lru_cache
import math
import asyncio
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

# Global lock that will be created only once when the module is imported
_connection_lock = None

def get_connection_lock():
    """Get the global asyncio lock, initializing it in the correct context if needed."""
    global _connection_lock
    if _connection_lock is None:
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            _connection_lock = asyncio.Lock()
        except RuntimeError:
            # If we're outside an event loop context, create a new one
            # This is just for initialization, not for actual usage
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _connection_lock = asyncio.Lock()
    return _connection_lock

# Singleton instance of DatabaseConnector
_database_connector_instance = None

def get_database_connector():
    """
    Cached database connector factory.
    
    Using a singleton pattern ensures only one instance is created and reused.
    """
    global _database_connector_instance
    if _database_connector_instance is None:
        # Pass the connection lock to the constructor
        _database_connector_instance = DatabaseConnector(connection_lock=get_connection_lock())
    return _database_connector_instance

@router.post("/query", response_model=QueryResponse)
async def generate_query(request: QueryRequest):
    """
    Generate a database query or other content from natural language.
    """
    start_time = time.time()
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
        total_time = time.time() - start_time
        logger.info(f"⏱️ Total query generation time: {total_time:.4f} seconds")
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


@router.post("/execute", response_model=ExecutionResponse)
async def execute_query(
    request: ExecutionRequest,
    db_connector: DatabaseConnector = Depends(get_database_connector)
):
    start_time = time.time()
    try:
        # Extract values from the request
        query = request.query
        execution_id = request.execution_id
        params = request.params or {}
        
        # Default pagination values
        page = 0
        page_size = 100
        
        # Check if pagination attribute exists and extract values safely
        if hasattr(request, 'pagination') and request.pagination is not None:
            page = request.pagination.get('page', 0)
            page_size = request.pagination.get('pageSize', 100)
        
        # Execute the query
        results, metadata = await db_connector.execute(query, params)
        
        # Prepare pagination information
        total_rows = len(results)
        total_pages = math.ceil(total_rows / page_size)
        
        # Slice results based on pagination
        start_index = page * page_size
        end_index = start_index + page_size
        paginated_results = results[start_index:end_index]
        total_time = time.time() - start_time
        logger.info(f"⏱️ Total query generation time: {total_time:.4f} seconds")
        return ExecutionResponse(
            results=paginated_results,
            metadata=metadata,
            pagination={
                "currentPage": page,
                "totalPages": total_pages,
                "totalRows": total_rows
            }
        )
    
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Query execution failed: {str(e)}"
        )
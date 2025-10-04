# app/routes/query.py

import time
from functools import lru_cache
import math
import asyncio
from fastapi import APIRouter, HTTPException, Depends, Request
import uuid
import json
from typing import Dict, Any, List

from app.core.config import settings
from app.schemas.query import QueryRequest, QueryResponse, ExecutionRequest, ExecutionResponse
from app.services.llm_provider import LLMProvider
from app.services.conversation_manager import ConversationManager
from app.services.query_generator import QueryGenerator  # Already enhanced
from app.services.database_connector import DatabaseConnector
from app.core.logging import logger

router = APIRouter()

# Global lock that will be created only once when the module is imported
_connection_lock = None

def handle_execution_error(e: Exception, endpoint_version: str = "") -> HTTPException:
    """
    Centralized error handler for query execution endpoints.
    Returns user-friendly error messages suitable for UI display.

    Args:
        e: The exception that was raised
        endpoint_version: Optional version tag for logging (e.g., "[V2]")

    Returns:
        HTTPException with appropriate status code and user-friendly message
    """
    # Re-raise HTTPException as-is (validation errors)
    if isinstance(e, HTTPException):
        return e

    # Handle specific exception types
    if isinstance(e, ConnectionError):
        logger.error(f"{endpoint_version} Database connection error: {str(e)}", exc_info=True)
        return HTTPException(
            status_code=503,
            detail="Unable to connect to the database. Please check your connection and try again."
        )

    if isinstance(e, TimeoutError):
        logger.error(f"{endpoint_version} Query timeout: {str(e)}", exc_info=True)
        return HTTPException(
            status_code=408,
            detail="The query took too long to execute. Please try simplifying your query or contact support."
        )

    if isinstance(e, ValueError):
        logger.error(f"{endpoint_version} Invalid query or parameters: {str(e)}", exc_info=True)
        return HTTPException(
            status_code=400,
            detail="The query or parameters are invalid. Please check your input and try again."
        )

    if isinstance(e, PermissionError):
        logger.error(f"{endpoint_version} Permission denied: {str(e)}", exc_info=True)
        return HTTPException(
            status_code=403,
            detail="You don't have permission to execute this query. Please contact your administrator."
        )

    # For generic exceptions, check error message for common patterns
    logger.error(f"{endpoint_version} Unexpected error executing query: {str(e)}", exc_info=True)

    error_msg = str(e).lower()
    if "syntax" in error_msg or "parse" in error_msg:
        return HTTPException(
            status_code=400,
            detail="The query contains a syntax error. Please review your query and try again."
        )
    elif "connection" in error_msg or "network" in error_msg:
        return HTTPException(
            status_code=503,
            detail="Database connection issue. Please try again in a moment."
        )
    elif "timeout" in error_msg:
        return HTTPException(
            status_code=408,
            detail="The query timed out. Please try a simpler query or contact support."
        )
    else:
        # Generic user-friendly message
        return HTTPException(
            status_code=500,
            detail="An unexpected error occurred while executing your query. Please try again or contact support if the issue persists."
        )

def handle_generation_error(e: Exception, endpoint_version: str = "") -> HTTPException:
    """
    Centralized error handler for query generation endpoints.
    Returns user-friendly error messages suitable for UI display.

    Args:
        e: The exception that was raised
        endpoint_version: Optional version tag for logging (e.g., "[V2]")

    Returns:
        HTTPException with appropriate status code and user-friendly message
    """
    # Re-raise HTTPException as-is (validation errors)
    if isinstance(e, HTTPException):
        return e

    # Handle specific exception types
    if isinstance(e, ValueError):
        logger.error(f"{endpoint_version} Validation error in query generation: {str(e)}", exc_info=True)
        return HTTPException(
            status_code=400,
            detail="Invalid request. Please check your input and try again."
        )

    if isinstance(e, ConnectionError):
        logger.error(f"{endpoint_version} Connection error during query generation: {str(e)}", exc_info=True)
        return HTTPException(
            status_code=503,
            detail="Unable to connect to the AI service. Please try again in a moment."
        )

    if isinstance(e, TimeoutError):
        logger.error(f"{endpoint_version} Timeout during query generation: {str(e)}", exc_info=True)
        return HTTPException(
            status_code=408,
            detail="The request took too long to process. Please try again."
        )

    # For generic exceptions, check error message for common patterns
    logger.error(f"{endpoint_version} Unexpected error in query generation: {str(e)}", exc_info=True)

    error_msg = str(e).lower()
    if "api" in error_msg and ("key" in error_msg or "auth" in error_msg):
        return HTTPException(
            status_code=503,
            detail="AI service authentication issue. Please contact support."
        )
    elif "rate limit" in error_msg or "quota" in error_msg:
        return HTTPException(
            status_code=429,
            detail="Service is currently busy. Please try again in a few moments."
        )
    elif "model" in error_msg and ("not found" in error_msg or "unavailable" in error_msg):
        return HTTPException(
            status_code=503,
            detail="The AI model is currently unavailable. Please try again later."
        )
    elif "timeout" in error_msg:
        return HTTPException(
            status_code=408,
            detail="The request timed out. Please try again."
        )
    elif "connection" in error_msg or "network" in error_msg:
        return HTTPException(
            status_code=503,
            detail="Network issue occurred. Please try again in a moment."
        )
    else:
        # Generic user-friendly message without exposing internal details
        return HTTPException(
            status_code=500,
            detail="Unable to generate query at this time. Please try again or contact support if the issue persists."
        )

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
    LEGACY ENDPOINT - Original query generation (kept for backward compatibility)

    Returns generated query WITHOUT query_complexity field for old clients.
    Use /query/v2 for the new version with query_complexity metadata.

    TODO: Remove this after all clients migrate to /query/v2
    """
    start_time = time.time()
    try:
        # Extract values from validated request model
        query = request.query
        model = request.model
        database_type = request.database_type
        conversation_id = request.conversation_id
        conversation_history = request.conversation_history

        # Extract user_id if available
        user_id = getattr(request, 'user_id', None)

        logger.info(f"📥 [LEGACY] Query request: {query}")
        logger.info(f"⚠️  Using LEGACY /query endpoint - consider migrating to /query/v2")
        if conversation_id:
            logger.info(f"📚 Conversation ID: {conversation_id}")

        # Initialize services
        llm_provider = LLMProvider()
        llm = llm_provider.get_model(model)

        query_generator = QueryGenerator(
            llm=llm,
            use_unified_analyzer=settings.USE_ENHANCED_ANALYZER
        )

        # Generate result
        execution_id = str(uuid.uuid4())
        result, thinking = await query_generator.generate(
            query=query,
            database_type=database_type,
            conversation_id=conversation_id,
            conversation_history=conversation_history,
            user_id=user_id,
            is_retry=False,
            original_generated_query=None,
            user_feedback=None
        )

        # Determine response type and content
        response_type = "query"
        generated_query = None
        generated_content = None

        if isinstance(result, dict):
            response_type = result.get("intent_type", "query")
            generated_query = result.get("generated_query")
            generated_content = result.get("generated_content")
        else:
            generated_query = result
            response_type = "query"

        # Store in conversation if provided
        if conversation_id and generated_query:
            try:
                conversation_manager = ConversationManager()

                await conversation_manager.add_message(
                    conversation_id,
                    {
                        "id": str(uuid.uuid4()),
                        "role": "user",
                        "content": query,
                        "metadata": {
                            "execution_id": execution_id,
                            "model": model,
                            "database_type": database_type
                        }
                    }
                )

                await conversation_manager.add_message(
                    conversation_id,
                    {
                        "id": str(uuid.uuid4()),
                        "role": "assistant",
                        "content": generated_query,
                        "metadata": {
                            "execution_id": execution_id,
                            "response_type": response_type,
                            "thinking_steps": len(thinking)
                        }
                    }
                )

                logger.info(f"💾 Stored query interaction in conversation {conversation_id}")

            except Exception as e:
                logger.warning(f"Could not store conversation: {str(e)}")

        total_time = time.time() - start_time
        logger.info(f"⏱️ Total query generation time: {total_time:.4f} seconds")
        logger.info(f"✅ Query generated successfully with {len(thinking)} thinking steps")

        # Return WITHOUT query_complexity (old schema)
        return QueryResponse(
            generated_query=generated_query,
            generated_content=generated_content,
            response_type=response_type,
            execution_id=execution_id,
            thinking=thinking,
            query_complexity="SINGLE_LINE"  # Default for backward compatibility
        )

    except Exception as e:
        raise handle_generation_error(e)


@router.post("/query/v2", response_model=QueryResponse)
async def generate_query_v2(request: QueryRequest):
    """
    NEW ENDPOINT - Query generation with query_complexity metadata

    Returns generated query WITH query_complexity field for optimized execution.
    This enables the /execute/v2 endpoint to use hybrid pagination strategies.
    """
    start_time = time.time()
    try:
        # Extract values from validated request model
        query = request.query
        model = request.model
        database_type = request.database_type
        conversation_id = request.conversation_id
        conversation_history = request.conversation_history

        # Extract user_id if available
        user_id = getattr(request, 'user_id', None)

        logger.info(f"📥 [V2] Query request: {query}")
        if conversation_id:
            logger.info(f"📚 Conversation ID: {conversation_id}")

        # Initialize services
        llm_provider = LLMProvider()
        llm = llm_provider.get_model(model)

        query_generator = QueryGenerator(
            llm=llm,
            use_unified_analyzer=settings.USE_ENHANCED_ANALYZER
        )

        # Generate result
        execution_id = str(uuid.uuid4())
        result, thinking = await query_generator.generate(
            query=query,
            database_type=database_type,
            conversation_id=conversation_id,
            conversation_history=conversation_history,
            user_id=user_id,
            is_retry=False,
            original_generated_query=None,
            user_feedback=None
        )

        # Determine response type and content
        response_type = "query"
        generated_query = None
        generated_content = None
        query_complexity = "SINGLE_LINE"  # Default

        if isinstance(result, dict):
            response_type = result.get("intent_type", "query")
            generated_query = result.get("generated_query")
            generated_content = result.get("generated_content")
            query_complexity = result.get("query_complexity", "SINGLE_LINE")
        else:
            generated_query = result
            response_type = "query"

        # Store in conversation if provided
        if conversation_id and generated_query:
            try:
                conversation_manager = ConversationManager()

                await conversation_manager.add_message(
                    conversation_id,
                    {
                        "id": str(uuid.uuid4()),
                        "role": "user",
                        "content": query,
                        "metadata": {
                            "execution_id": execution_id,
                            "model": model,
                            "database_type": database_type
                        }
                    }
                )

                await conversation_manager.add_message(
                    conversation_id,
                    {
                        "id": str(uuid.uuid4()),
                        "role": "assistant",
                        "content": generated_query,
                        "metadata": {
                            "execution_id": execution_id,
                            "response_type": response_type,
                            "thinking_steps": len(thinking),
                            "query_complexity": query_complexity  # NEW: Store complexity
                        }
                    }
                )

                logger.info(f"💾 Stored query interaction in conversation {conversation_id}")

            except Exception as e:
                logger.warning(f"Could not store conversation: {str(e)}")

        total_time = time.time() - start_time
        logger.info(f"⏱️ [V2] Total query generation time: {total_time:.4f} seconds")
        logger.info(f"✅ [V2] Query generated successfully with {len(thinking)} thinking steps")
        logger.info(f"📐 [V2] Query complexity: {query_complexity}")

        return QueryResponse(
            generated_query=generated_query,
            generated_content=generated_content,
            response_type=response_type,
            execution_id=execution_id,
            thinking=thinking,
            query_complexity=query_complexity  # NEW: Return complexity
        )

    except Exception as e:
        raise handle_generation_error(e, "[V2]")

@router.post("/execute", response_model=ExecutionResponse)
async def execute_query(
    request: ExecutionRequest,
    db_connector: DatabaseConnector = Depends(get_database_connector)
):
    """
    LEGACY ENDPOINT - Original execute method (kept for backward compatibility)

    This is the old implementation that fetches all results then slices in Python.
    Use /execute/v2 for the new optimized pagination approach.

    TODO: Remove this after /execute/v2 is stable in production
    """
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
        if request.pagination is not None:
            page = request.pagination.page
            page_size = request.pagination.page_size

        # Validate pagination parameters
        if page < 0:
            logger.warning(f"Invalid page number: {page}")
            raise HTTPException(
                status_code=400,
                detail="Page number must be non-negative. Please check your pagination settings."
            )

        if page_size <= 0 or page_size > 10000:
            logger.warning(f"Invalid page size: {page_size}")
            raise HTTPException(
                status_code=400,
                detail="Page size must be between 1 and 10,000. Please adjust your pagination settings."
            )

        logger.info(f"⚠️  Using LEGACY execute endpoint - consider migrating to /execute/v2")

        # Execute the query (OLD METHOD - fetches all results)
        results, metadata = await db_connector.execute(query, params)

        # Prepare pagination information
        total_rows = len(results)
        total_pages = math.ceil(total_rows / page_size)

        # Slice results based on pagination IN PYTHON (inefficient for large results)
        start_index = page * page_size
        end_index = start_index + page_size
        paginated_results = results[start_index:end_index]

        total_time = time.time() - start_time
        logger.info(f"⏱️ Total query execution time: {total_time:.4f} seconds")

        return ExecutionResponse(
            results=paginated_results,
            metadata=metadata,
            pagination={
                "currentPage": page,
                "totalPages": total_pages,
                "totalRows": total_rows,
                "pageSize": page_size,
                "returnedRows": len(paginated_results)
            }
        )

    except Exception as e:
        raise handle_execution_error(e)


@router.post("/execute/v2", response_model=ExecutionResponse)
async def execute_query_v2(
    request: ExecutionRequest,
    db_connector: DatabaseConnector = Depends(get_database_connector)
):
    """
    NEW OPTIMIZED ENDPOINT - Hybrid pagination with database-level optimization

    Features:
    - SINGLE_LINE queries: Optimized count query (exec count i) + pagination in KDB
    - MULTI_LINE queries: Wrapping approach for multi-statement queries
    - Auto-detection: Falls back to optimization if possible
    - Count caching: Avoids re-counting on subsequent pages
    - Safe default: MULTI_LINE prevents server hangs
    """
    start_time = time.time()
    try:
        # Extract values from the request
        query = request.query
        execution_id = request.execution_id
        params = request.params or {}
        query_complexity = request.query_complexity or "MULTI_LINE"  # Safe default

        # Default pagination values
        page = 0
        page_size = 100

        # Check if pagination is provided
        if request.pagination is not None:
            page = request.pagination.page
            page_size = request.pagination.page_size

        # Validate pagination parameters
        if page < 0:
            logger.warning(f"[V2] Invalid page number: {page}")
            raise HTTPException(
                status_code=400,
                detail="Page number must be non-negative. Please check your pagination settings."
            )

        if page_size <= 0 or page_size > 10000:
            logger.warning(f"[V2] Invalid page size: {page_size}")
            raise HTTPException(
                status_code=400,
                detail="Page size must be between 1 and 10,000. Please adjust your pagination settings."
            )

        logger.info(f"🚀 [V2] Executing query with complexity: {query_complexity}")

        # Execute query with pagination at database level (NEW METHOD)
        results, metadata, total_count = await db_connector.execute_paginated(
            query=query,
            page=page,
            page_size=page_size,
            params=params,
            query_complexity=query_complexity
        )

        # Calculate pagination info
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1

        total_time = time.time() - start_time
        logger.info(f"⏱️ [V2] Query execution time: {total_time:.4f} seconds")
        logger.info(f"📊 [V2] Returned {len(results)} of {total_count} total records (page {page + 1}/{total_pages})")

        return ExecutionResponse(
            results=results,
            metadata=metadata,
            pagination={
                "currentPage": page,
                "totalPages": total_pages,
                "totalRows": total_count,
                "pageSize": page_size,
                "returnedRows": len(results)
            }
        )

    except Exception as e:
        raise handle_execution_error(e, "[V2]")
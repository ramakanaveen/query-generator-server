# app/routes/schema_api.py

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel

from app.services.enhanced_schema_service import EnhancedSchemaService
from app.core.logging import logger

router = APIRouter(prefix="/api/v1/enhanced-schema", tags=["enhanced-schema"])

class SchemaRequest(BaseModel):
    query: str
    directives: Optional[List[str]] = None
    user_id: Optional[str] = None
    include_examples: bool = True
    max_tables: int = 5

class SchemaResponse(BaseModel):
    schema: dict
    examples: List[dict]
    metadata: dict
    performance: dict

@router.post("/retrieve", response_model=SchemaResponse)
async def retrieve_schema_with_examples(request: SchemaRequest):
    """
    Retrieve schema structure with relevant examples.

    This endpoint provides the same functionality as the schema retriever node
    but as a REST API for external applications.
    """
    try:
        schema_service = EnhancedSchemaService()

        result = await schema_service.get_schema_for_api(
            query=request.query,
            directives=request.directives,
            user_id=request.user_id,
            include_examples=request.include_examples,
            max_tables=request.max_tables
        )

        return SchemaResponse(**result)

    except Exception as e:
        logger.error(f"Error in enhanced schema API: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Schema retrieval failed: {str(e)}")

@router.get("/retrieve")
async def retrieve_schema_get(
        query: str = Query(..., description="Natural language query"),
        directives: Optional[str] = Query(None, description="Comma-separated directives"),
        user_id: Optional[str] = Query(None, description="User ID for personalized examples"),
        include_examples: bool = Query(True, description="Include examples in response"),
        max_tables: int = Query(5, description="Maximum tables to retrieve")
):
    """
    GET version of schema retrieval for simple URL-based access.
    """
    try:
        # Parse directives
        directive_list = directives.split(",") if directives else None

        # Create request
        request = SchemaRequest(
            query=query,
            directives=directive_list,
            user_id=user_id,
            include_examples=include_examples,
            max_tables=max_tables
        )

        return await retrieve_schema_with_examples(request)

    except Exception as e:
        logger.error(f"Error in enhanced schema GET API: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Schema retrieval failed: {str(e)}")


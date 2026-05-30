"""
Admin endpoints for schema metadata management.
  POST /api/admin/schema-groups/generate-descriptions
      Generates Haiku-based descriptions for all groups/schemas missing one.
      Pass ?overwrite=true to regenerate existing descriptions too.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Query

from app.services.schema_description_generator import generate_all_descriptions

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/schema-groups/generate-descriptions")
async def generate_descriptions(overwrite: bool = Query(default=False)):
    """
    Generate 1-sentence descriptions for schema groups and schemas that have none.
    Uses Claude Haiku for low-cost generation.
    Returns counts of updated groups/schemas and any errors.
    """
    logger.info(f"Generating schema descriptions (overwrite={overwrite})")
    result = await generate_all_descriptions(overwrite=overwrite)
    return {
        "status": "ok",
        "groups_updated": result["groups_updated"],
        "schemas_updated": result["schemas_updated"],
        "errors": result["errors"],
    }

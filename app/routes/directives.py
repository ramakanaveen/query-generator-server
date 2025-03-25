# app/routes/directives.py
from fastapi import APIRouter, HTTPException
import os
import json
from pathlib import Path
from app.core.config import settings
from app.core.logging import logger

router = APIRouter()

# app/routes/directives.py - Update the get_directives function

@router.get("/directives")
async def get_directives():
    """
    Get available directives based on schema files.
    """
    try:
        directives = []
        schemas_dir = Path(settings.SCHEMAS_DIRECTORY)
        
        # If directory doesn't exist, return empty list
        if not schemas_dir.exists() or not schemas_dir.is_dir():
            logger.warning(f"Schemas directory not found: {schemas_dir}")
            return {"directives": []}
        
        # Map directive names to appropriate icons
        icon_map = {
            "SPOT": "TrendingUp",
            "STIRT": "BarChart2",
            "TITAN": "Activity",
            "FX": "RefreshCw", 
            "BONDS": "Landmark",
            "DSP": "LineChart"
        }
        
        # Scan for schema files
        for schema_file in schemas_dir.glob("*.json"):
            try:
                directive_name = schema_file.stem.upper()
                
                # Try to read the schema file to get description
                with open(schema_file, 'r') as f:
                    schema_data = json.load(f)
                    
                # Extract description from schema data
                description = "Schema file"
                
                # Try to find description in different possible locations
                if "description" in schema_data:
                    description = schema_data["description"]
                elif "group" in schema_data and "schemas" in schema_data:
                    for schema in schema_data.get("schemas", []):
                        if "description" in schema:
                            description = schema["description"]
                            break
                
                # Assign appropriate icon or default to Hash
                icon = icon_map.get(directive_name, "Hash")
                
                directives.append({
                    "id": len(directives) + 1,
                    "name": directive_name,
                    "description": description,
                    "icon": icon
                })
            except Exception as e:
                logger.error(f"Error processing schema file {schema_file}: {str(e)}")
                # Continue to next file even if this one fails
        
        return {"directives": directives}
    
    except Exception as e:
        logger.error(f"Error getting directives: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get directives: {str(e)}")
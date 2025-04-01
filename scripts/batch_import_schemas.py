# scripts/batch_import_schemas.py
import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from import_schema import import_schema_file
from app.core.logging import logger

async def batch_import(directory, user_id=None):
    """Import all schema files from a directory."""
    try:
        directory_path = Path(directory)
        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory not found: {directory}")
            return False
        
        # Get all JSON files in the directory
        schema_files = list(directory_path.glob("*.json"))
        if not schema_files:
            logger.warning(f"No JSON files found in {directory}")
            return False
        
        logger.info(f"Found {len(schema_files)} schema files in {directory}")
        
        # Process each file
        success_count = 0
        for schema_file in schema_files:
            logger.info(f"Processing {schema_file.name}...")
            
            # Import the schema
            result = await import_schema_file(
                schema_file=str(schema_file),
                user_id=user_id
            )
            
            if result:
                success_count += 1
                logger.info(f"Successfully imported {schema_file.name}")
            else:
                logger.error(f"Failed to import {schema_file.name}")
        
        logger.info(f"Imported {success_count} out of {len(schema_files)} schema files")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error in batch import: {str(e)}", exc_info=True)
        return False

async def main():
    parser = argparse.ArgumentParser(description="Batch import schema files from a directory")
    parser.add_argument("directory", help="Directory containing schema JSON files")
    parser.add_argument("--user", type=int, help="User ID of the importer")
    
    args = parser.parse_args()
    
    success = await batch_import(
        directory=args.directory,
        user_id=args.user
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
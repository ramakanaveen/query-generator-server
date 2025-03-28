#!/usr/bin/env python
# scripts/import_schema.py

import asyncio
import argparse
import json
import sys
import os
from pathlib import Path

# Add the project root to the Python path to import project modules
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from app.services.schema_management import SchemaManager
from app.core.logging import logger

async def import_schema_file(file_path, user_id=None, verbose=False):
    """
    Import a single schema file into the database.
    
    Args:
        file_path: Path to the schema JSON file
        user_id: User ID of the importer
        verbose: Whether to print detailed logs
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate file path
        schema_path = Path(file_path)
        if not schema_path.exists():
            print(f"Error: Schema file not found: {file_path}")
            return False
            
        if not schema_path.is_file() or schema_path.suffix.lower() != '.json':
            print(f"Error: Not a JSON file: {file_path}")
            return False
        
        # Load file to validate JSON
        try:
            with open(schema_path, 'r') as f:
                schema_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {file_path}: {str(e)}")
            return False
        
        # Validate schema structure
        if "group" not in schema_data:
            print(f"Error: Missing 'group' in schema file {file_path}")
            return False
            
        if "schemas" not in schema_data or not isinstance(schema_data["schemas"], list):
            print(f"Error: Missing or invalid 'schemas' array in {file_path}")
            return False
        
        # Validate the nested schema structure
        for schema_container in schema_data.get("schemas", []):
            if "schema" not in schema_container or not isinstance(schema_container["schema"], list):
                print(f"Error: Missing or invalid 'schema' array in a schemas item in {file_path}")
                return False
                
            # Check at least one schema item in the array
            if len(schema_container["schema"]) == 0:
                print(f"Warning: Empty schema array in {file_path}")
                # Not a critical error, so continue
        
        # Initialize schema manager
        schema_manager = SchemaManager()
        
        # Import schema
        print(f"Importing schema file: {schema_path.name}")
        result = await schema_manager.import_schema(
            file_path=str(schema_path),
            user_id=user_id
        )
        
        if result.get("success", False):
            group_name = result.get("group_name", "unknown")
            schema_count = result.get("schema_count", 0)
            schemas = result.get("schemas", [])
            
            print(f"Successfully imported group '{group_name}' with {schema_count} schemas")
            
            if verbose:
                for schema in schemas:
                    name = schema.get("name", "unknown")
                    table_count = schema.get("table_count", 0)
                    example_count = schema.get("example_count", 0)
                    
                    print(f"  - Schema '{name}': {table_count} tables, {example_count} examples")
            
            return True
        else:
            error = result.get("error", "Unknown error")
            print(f"Error importing schema: {error}")
            return False
        
    except Exception as e:
        print(f"Unexpected error importing schema: {str(e)}")
        return False

async def batch_import_schemas(directory, user_id=None, verbose=False):
    """
    Import all schema files from a directory.
    
    Args:
        directory: Directory containing schema JSON files
        user_id: User ID of the importer
        verbose: Whether to print detailed logs
        
    Returns:
        True if at least one schema was successfully imported, False otherwise
    """
    try:
        # Validate directory
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Error: Directory not found: {directory}")
            return False
            
        if not dir_path.is_dir():
            print(f"Error: Not a directory: {directory}")
            return False
        
        # Find JSON files
        json_files = list(dir_path.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {directory}")
            return False
        
        print(f"Found {len(json_files)} JSON files in {directory}")
        
        # Initialize schema manager
        schema_manager = SchemaManager()
        
        # Process each file individually for better error handling
        success_count = 0
        
        for schema_file in json_files:
            print(f"Processing {schema_file.name}...")
            
            # Import the schema
            success = await import_schema_file(
                file_path=str(schema_file),
                user_id=user_id,
                verbose=verbose
            )
            
            if success:
                success_count += 1
        
        print(f"Successfully imported {success_count} out of {len(json_files)} schema files")
        return success_count > 0
        
    except Exception as e:
        print(f"Unexpected error in batch import: {str(e)}")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Import schema files into the database")
    parser.add_argument("--file", help="Path to a single schema JSON file to import")
    parser.add_argument("--directory", help="Directory containing schema JSON files to import")
    parser.add_argument("--user", type=int, help="User ID of the importer")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed information")
    
    args = parser.parse_args()
    args.file = '/Users/naveenramaka/naveen/query-generator-server/app/schemas/spot.json'
    if not args.file and not args.directory:
        parser.print_help()
        return 1
    
    success = False
    
    if args.file:
        success = await import_schema_file(args.file, args.user, args.verbose)
    
    if args.directory:
        success = await batch_import_schemas(args.directory, args.user, args.verbose)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
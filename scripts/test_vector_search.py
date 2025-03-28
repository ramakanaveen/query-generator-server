# scripts/test_vector_search.py
import asyncio
import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.services.schema_management import SchemaManager
from app.core.logging import logger

async def test_search(query_text, threshold=0.65, max_results=5):
    """Test vector search functionality with a query."""
    try:
        print(f"Testing vector search with query: '{query_text}'")
        print(f"Threshold: {threshold}, Max results: {max_results}")
        
        # Initialize schema manager
        schema_manager = SchemaManager()
        
        # Perform vector search
        relevant_tables = await schema_manager.find_tables_by_vector_search(
            query_text=query_text,
            similarity_threshold=threshold,
            max_results=max_results
        )
        
        # Display results
        if relevant_tables:
            print(f"\nFound {len(relevant_tables)} relevant tables:")
            for i, table in enumerate(relevant_tables, 1):
                print(f"\n{i}. {table['schema_name']}.{table['table_name']}")
                print(f"   Similarity: {table['similarity']:.4f}")
                print(f"   Description: {table['description']}")
                print(f"   Columns: {', '.join(get_column_names(table['content']))}")
        else:
            print("\nNo relevant tables found.")
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing vector search: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        return False

def get_column_names(table_content):
    """Extract column names from table content."""
    columns = []
    if "columns" in table_content:
        for col in table_content["columns"]:
            if isinstance(col, dict) and "name" in col:
                columns.append(col["name"])
            elif isinstance(col, str):
                columns.append(col)
    return columns

async def main():
    parser = argparse.ArgumentParser(description="Test vector search functionality")
    parser.add_argument("query", help="Natural language query to test")
    parser.add_argument("--threshold", type=float, default=0.65, help="Similarity threshold")
    parser.add_argument("--max", type=int, default=5, help="Maximum number of results")
    
    # args = parser.parse_args()
    # Default values when debugging
    query_text = "Show me the top 5 AAPL trades by size today"
    threshold = 0.5
    max_results = 10
    success = await test_search(
        query_text=query_text,
        threshold=threshold,
        max_results=max_results
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
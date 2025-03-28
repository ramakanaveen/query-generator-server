# scripts/init_database.py
import asyncio
import argparse
import os
from pathlib import Path
import asyncpg

# Path to database scripts
SCRIPTS_DIR = Path(__file__).parent / "db_scripts"

async def init_database(connection_string):
    """Initialize the database schema."""
    print(f"Connecting to database: {connection_string}")
    
    conn = await asyncpg.connect(connection_string)
    try:
        # Run schema creation scripts in order
        scripts = [
            "01_schema_tables.sql",
            "02_auth_tables.sql"
        ]
        
        for script_name in scripts:
            script_path = SCRIPTS_DIR / script_name
            if not script_path.exists():
                print(f"Script not found: {script_path}")
                continue
                
            print(f"Running script: {script_name}")
            with open(script_path, 'r') as f:
                sql = f.read()
                await conn.execute(sql)
            
            print(f"Completed script: {script_name}")
        
        print("Database initialization completed successfully")
        return True
            
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        return False
    finally:
        await conn.close()

async def main():
    parser = argparse.ArgumentParser(description="Initialize database schema")
    parser.add_argument("--connection", help="Database connection string", 
                        default=os.environ.get("DATABASE_URL"))
    
    args = parser.parse_args()
    
    if not args.connection:
        print("Error: Database connection string not provided")
        print("Set the DATABASE_URL environment variable or use --connection")
        return 1
    
    success = await init_database(args.connection)
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
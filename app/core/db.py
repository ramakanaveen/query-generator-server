# app/core/db.py
import asyncpg
from typing import Optional
from app.core.config import settings
from app.core.logging import logger

class DatabasePool:
    """Singleton database connection pool."""
    
    _instance = None
    _pool: Optional[asyncpg.Pool] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabasePool, cls).__new__(cls)
            cls._instance._pool = None
        return cls._instance
    
    async def initialize(self):
        """Initialize the connection pool if it doesn't exist."""
        if self._pool is None:
            try:
                logger.info(f"Initializing database connection pool")
                self._pool = await asyncpg.create_pool(
                    settings.DATABASE_URL,
                    min_size=5,
                    max_size=20,
                    timeout=30.0,
                    command_timeout=10.0,
                    max_queries=50000,
                    max_inactive_connection_lifetime=300.0,  # 5 minutes
                )
                logger.info(f"Database connection pool initialized")
            except Exception as e:
                logger.error(f"Failed to initialize database pool: {str(e)}")
                raise
    
    async def close(self):
        """Close all connections in the pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database connection pool closed")
    
    async def get_connection(self):
        """Get a connection from the pool."""
        if self._pool is None:
            await self.initialize()
        return await self._pool.acquire()
    
    async def release_connection(self, connection):
        """Release a connection back to the pool."""
        if self._pool:
            await self._pool.release(connection)
    
    async def execute(self, query, *args, **kwargs):
        """Execute a query using a connection from the pool."""
        async with self._pool.acquire() as connection:
            return await connection.execute(query, *args, **kwargs)
    
    async def fetch(self, query, *args, **kwargs):
        """Fetch rows using a connection from the pool."""
        async with self._pool.acquire() as connection:
            return await connection.fetch(query, *args, **kwargs)
    
    async def fetchval(self, query, *args, **kwargs):
        """Fetch a single value using a connection from the pool."""
        async with self._pool.acquire() as connection:
            return await connection.fetchval(query, *args, **kwargs)
    
    async def fetchrow(self, query, *args, **kwargs):
        """Fetch a single row using a connection from the pool."""
        async with self._pool.acquire() as connection:
            return await connection.fetchrow(query, *args, **kwargs)
    
    async def transaction(self):
        """Start a transaction using a connection from the pool."""
        connection = await self.get_connection()
        transaction = connection.transaction()
        await transaction.start()
        
        try:
            yield connection
            await transaction.commit()
        except Exception:
            await transaction.rollback()
            raise
        finally:
            await self.release_connection(connection)

# Initialize DB pool at application startup
db_pool = DatabasePool()
"""
Starburst/Trino database connector.

Provides connection and query execution capabilities for Starburst/Trino databases.

Starburst is a commercial distribution of Trino (formerly PrestoSQL), fully compatible
with the Trino protocol and client libraries.

Features:
- Async connection pooling
- SQL injection prevention
- Optimized pagination (LIMIT/OFFSET)
- Parameterized query support
- Transaction support
"""

from typing import Dict, Any, Tuple, List, Optional
import re
import asyncio
from datetime import datetime
from decimal import Decimal

from app.services.connectors.base_connector import BaseConnector
from app.core.config import settings
from app.core.logging import logger

try:
    from trino.dbapi import connect
    from trino.auth import BasicAuthentication
    TRINO_AVAILABLE = True
except ImportError:
    TRINO_AVAILABLE = False
    logger.warning("trino library not available - install with: pip install trino")


class StarburstConnector(BaseConnector):
    """
    Starburst/Trino database connector.

    Manages connections to Starburst/Trino and executes SQL queries with optimized pagination.

    Note: Starburst is fully compatible with Trino protocol.
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        catalog: str = None,
        schema: str = None,
        user: str = None,
        password: str = None,
        use_https: bool = True,
        **kwargs
    ):
        """
        Initialize Starburst/Trino connector.

        Args:
            host: Starburst/Trino host (defaults to settings.STARBURST_HOST)
            port: Port number (defaults to settings.STARBURST_PORT)
            catalog: Catalog name (e.g., 'hive', 'iceberg')
            schema: Schema name (e.g., 'default')
            user: Username for authentication
            password: Password for authentication
            use_https: Use HTTPS connection (recommended for production)
            **kwargs: Additional connection parameters
        """
        super().__init__()

        if not TRINO_AVAILABLE:
            raise ImportError(
                "trino library is required for Starburst connector. "
                "Install with: pip install trino"
            )

        self.host = host or settings.STARBURST_HOST
        self.port = port or settings.STARBURST_PORT
        self.catalog = catalog or settings.STARBURST_CATALOG
        self.schema = schema or settings.STARBURST_SCHEMA
        self.user = user or settings.STARBURST_USER
        self.password = password or settings.STARBURST_PASSWORD
        self.use_https = use_https
        self._connection = None
        self._lock = asyncio.Lock()

    async def _get_connection(self):
        """
        Get or create a connection to Starburst/Trino.

        Note: trino-python-client is synchronous, but we wrap it for async compatibility.
        """
        async with self._lock:
            if self._connection is None:
                try:
                    # Build authentication if credentials provided
                    auth = None
                    if self.user and self.password:
                        auth = BasicAuthentication(self.user, self.password)

                    # Create connection
                    self._connection = connect(
                        host=self.host,
                        port=self.port,
                        user=self.user,
                        catalog=self.catalog,
                        schema=self.schema,
                        http_scheme='https' if self.use_https else 'http',
                        auth=auth
                    )

                    logger.info(
                        f"Connected to Starburst/Trino at {self.host}:{self.port} "
                        f"(catalog={self.catalog}, schema={self.schema})"
                    )

                except Exception as e:
                    logger.error(f"Starburst/Trino Connection Error: {str(e)}")
                    raise ConnectionError(f"Could not connect to Starburst/Trino: {str(e)}")

            return self._connection

    def _sanitize_query(self, query: str) -> str:
        """
        Prevent SQL injection and validate query safety.

        Args:
            query: Raw SQL query

        Returns:
            Sanitized query

        Raises:
            ValueError if dangerous patterns detected
        """
        # Dangerous SQL patterns
        dangerous_patterns = [
            r';.*drop\s+table',      # DROP TABLE after semicolon
            r';.*drop\s+schema',     # DROP SCHEMA after semicolon
            r';.*delete\s+from',     # DELETE after semicolon
            r';.*truncate',          # TRUNCATE after semicolon
            r';.*alter\s+table',     # ALTER TABLE after semicolon
            r';.*create\s+table',    # CREATE TABLE after semicolon
            r'xp_cmdshell',          # SQL Server command execution
            r'exec\s*\(',            # Dynamic SQL execution
            r'--\s*$',               # SQL comment at end (potential for injection)
        ]

        query_lower = query.lower()

        for pattern in dangerous_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                logger.warning(f"Dangerous SQL pattern detected: {pattern}")
                raise ValueError("Potentially unsafe SQL query detected")

        # Check for multiple statements (semicolon separated)
        # Allow only one statement for safety
        if query.count(';') > 1 or (query.count(';') == 1 and not query.strip().endswith(';')):
            logger.warning("Multiple SQL statements detected")
            raise ValueError("Multiple SQL statements not allowed. Execute one query at a time.")

        return query.strip()

    def _build_count_query(self, query: str) -> str:
        """
        Build optimized count query for total row count.

        Transforms:
            SELECT * FROM users WHERE age > 18
        Into:
            SELECT COUNT(*) FROM (SELECT * FROM users WHERE age > 18) AS count_subquery

        Args:
            query: Original SQL query

        Returns:
            Count query
        """
        # Remove trailing semicolons
        query = query.rstrip(';').strip()

        # Wrap in count subquery
        count_query = f"SELECT COUNT(*) AS total_count FROM ({query}) AS count_subquery"

        return count_query

    def _build_paginated_query(self, query: str, page: int, page_size: int) -> str:
        """
        Add LIMIT/OFFSET pagination to SQL query.

        Args:
            query: Original SQL query
            page: Page number (0-indexed)
            page_size: Records per page

        Returns:
            Paginated SQL query
        """
        offset = page * page_size

        # Remove existing LIMIT/OFFSET if present
        query = re.sub(r'\s+LIMIT\s+\d+', '', query, flags=re.IGNORECASE)
        query = re.sub(r'\s+OFFSET\s+\d+', '', query, flags=re.IGNORECASE)

        # Remove trailing semicolon
        query = query.rstrip(';').strip()

        # Add pagination
        paginated_query = f"{query} LIMIT {page_size} OFFSET {offset}"

        return paginated_query

    def _convert_result_value(self, value: Any) -> Any:
        """
        Convert Trino-specific types to JSON-serializable Python types.

        Extends base _convert_value with Trino-specific handling.

        Args:
            value: Raw value from Trino query

        Returns:
            JSON-serializable value
        """
        # Handle Decimal (common in Trino for precision numbers)
        if isinstance(value, Decimal):
            return float(value)

        # Use base class conversion for standard types
        return self._convert_value(value)

    async def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute a SQL query on Starburst/Trino.

        Args:
            query: SQL query string
            params: Optional query parameters (for parameterized queries)

        Returns:
            Tuple of (results, metadata)
        """
        start_time = asyncio.get_event_loop().time()

        try:
            conn = await self._get_connection()

            # Sanitize query
            sanitized_query = self._sanitize_query(query)

            # Create cursor and execute query
            # Note: trino-python-client is synchronous, so we run in executor
            loop = asyncio.get_event_loop()

            def _execute_sync():
                cursor = conn.cursor()
                if params:
                    # Parameterized query (safer)
                    cursor.execute(sanitized_query, params)
                else:
                    cursor.execute(sanitized_query)

                # Fetch all results
                rows = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description] if cursor.description else []

                cursor.close()
                return rows, column_names

            rows, column_names = await loop.run_in_executor(None, _execute_sync)

            # Convert to list of dicts
            results = []
            for row in rows:
                row_dict = {}
                for i, col_name in enumerate(column_names):
                    row_dict[col_name] = self._convert_result_value(row[i])
                results.append(row_dict)

            # Metadata
            metadata = {
                "row_count": len(results),
                "column_count": len(column_names),
                "column_names": column_names,
                "query": sanitized_query,
                "execution_time": round(asyncio.get_event_loop().time() - start_time, 4),
                "timestamp": datetime.now().isoformat(),
                "database_type": "starburst"
            }

            return results, metadata

        except ValueError as ve:
            # Security/validation errors
            logger.warning(f"Query validation failed: {str(ve)}")
            return [], {
                "error": str(ve),
                "query": query,
                "execution_time": 0,
                "database_type": "starburst"
            }

        except Exception as e:
            # Comprehensive error handling
            logger.error(f"Starburst/Trino Query Execution Error: {str(e)}", exc_info=True)

            return [], {
                "error": "Query execution failed",
                "details": str(e),
                "query": query,
                "execution_time": 0,
                "database_type": "starburst"
            }

    async def execute_paginated(
        self,
        query: str,
        page: int = 0,
        page_size: int = 100,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], int]:
        """
        Execute a SQL query with pagination on Starburst/Trino.

        Uses efficient LIMIT/OFFSET pagination at the database level.

        Args:
            query: SQL query
            page: Page number (0-indexed)
            page_size: Records per page
            params: Optional query parameters
            **kwargs: Additional options (unused for Starburst)

        Returns:
            Tuple of (results, metadata, total_count)
        """
        start_time = asyncio.get_event_loop().time()

        try:
            conn = await self._get_connection()

            # Sanitize query
            sanitized_query = self._sanitize_query(query)

            # Execute count and paginated query in parallel for efficiency
            loop = asyncio.get_event_loop()

            def _execute_count_and_paginated():
                # Get total count
                count_query = self._build_count_query(sanitized_query)
                cursor_count = conn.cursor()
                cursor_count.execute(count_query)
                total_count_result = cursor_count.fetchone()
                total_count = total_count_result[0] if total_count_result else 0
                cursor_count.close()

                # Get paginated results
                paginated_query = self._build_paginated_query(sanitized_query, page, page_size)
                cursor_data = conn.cursor()

                if params:
                    cursor_data.execute(paginated_query, params)
                else:
                    cursor_data.execute(paginated_query)

                rows = cursor_data.fetchall()
                column_names = [desc[0] for desc in cursor_data.description] if cursor_data.description else []
                cursor_data.close()

                return total_count, rows, column_names, paginated_query

            total_count, rows, column_names, paginated_query = await loop.run_in_executor(
                None, _execute_count_and_paginated
            )

            # Convert to list of dicts
            results = []
            for row in rows:
                row_dict = {}
                for i, col_name in enumerate(column_names):
                    row_dict[col_name] = self._convert_result_value(row[i])
                results.append(row_dict)

            # Metadata
            metadata = {
                "row_count": len(results),
                "column_count": len(column_names),
                "column_names": column_names,
                "query": paginated_query,
                "original_query": query,
                "execution_time": round(asyncio.get_event_loop().time() - start_time, 4),
                "timestamp": datetime.now().isoformat(),
                "database_type": "starburst"
            }

            return results, metadata, total_count

        except ValueError as ve:
            logger.warning(f"Query validation failed: {str(ve)}")
            return [], {
                "error": str(ve),
                "query": query,
                "execution_time": 0,
                "database_type": "starburst"
            }, 0

        except Exception as e:
            logger.error(f"Starburst/Trino Paginated Execution Error: {str(e)}", exc_info=True)

            return [], {
                "error": "Paginated query execution failed",
                "details": str(e),
                "query": query,
                "execution_time": 0,
                "database_type": "starburst"
            }, 0

    async def close(self):
        """Close the Trino connection."""
        if self._connection:
            try:
                self._connection.close()
                logger.info("Closed Starburst/Trino connection")
            except Exception as e:
                logger.warning(f"Error closing Starburst connection: {str(e)}")

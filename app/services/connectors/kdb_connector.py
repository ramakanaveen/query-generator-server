"""
KDB+/q database connector.

Provides connection and query execution capabilities for KDB+ databases.

Features:
- Connection management with locking
- Query sanitization for security
- Multi-statement query detection
- Optimized pagination (SINGLE_LINE vs MULTI_LINE)
- Count caching for performance
"""

from typing import Dict, Any, Tuple, List, Optional
import os
import re
import asyncio
import hashlib
import pykx as kx
import pandas as pd
from datetime import datetime

from app.services.connectors.base_connector import BaseConnector
from app.core.config import settings
from app.core.logging import logger


class KDBConnector(BaseConnector):
    """
    KDB+/q database connector.

    Manages connections to KDB+ and executes q queries with optimized pagination.
    """

    def __init__(self, host: str = None, port: int = None, connection_lock=None):
        """
        Initialize KDB connector.

        Args:
            host: KDB host (defaults to settings.KDB_HOST)
            port: KDB port (defaults to settings.KDB_PORT)
            connection_lock: Optional asyncio.Lock for connection management
        """
        super().__init__()
        self.host = host or settings.KDB_HOST
        self.port = port or settings.KDB_PORT
        self._connection = None
        self._connection_lock = connection_lock
        # Cache for query counts (query_hash -> count)
        self._count_cache: Dict[str, int] = {}

    async def _get_connection(self):
        """
        Asynchronously get a connection to the KDB+ database.
        Uses a lock to prevent multiple simultaneous connection attempts.
        """
        # Ensure we have a lock
        if self._connection_lock is None:
            try:
                self._connection_lock = asyncio.Lock()
            except RuntimeError as e:
                logger.error(f"Could not create asyncio.Lock: {str(e)}")
                # Create a dummy lock that doesn't do anything
                class DummyLock:
                    async def __aenter__(self): pass
                    async def __aexit__(self, *args): pass
                self._connection_lock = DummyLock()

        async with self._connection_lock:
            if self._connection is None:
                try:
                    self._connection = kx.QConnection(host=self.host, port=self.port)
                    logger.info(f"Connected to KDB+ at {self.host}:{self.port}")
                except Exception as e:
                    logger.error(f"KDB+ Connection Error: {str(e)}")
                    raise ConnectionError(f"Could not connect to KDB+: {str(e)}")
            return self._connection

    def _sanitize_query(self, query: str) -> str:
        """
        Comprehensive query sanitization to prevent security risks.

        Args:
            query: Raw query string

        Returns:
            Sanitized query

        Raises:
            ValueError if dangerous patterns are detected
        """
        # Dangerous pattern checks
        dangerous_patterns = [
            r'system\s*"',        # system command
            r'hopen\s*":',        # opening handles to ports
            r'read1\s*`:',        # reading files
            r'set\s*`:',          # writing files
            r'\\"',               # potential command injection
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(f"Dangerous pattern detected: {pattern}")
                raise ValueError("Potentially unsafe query detected")

        # Add result limit if not specified
        if "select" in query.lower() and not re.search(r'(select\s+top|select\[[0-9]+\])', query.lower()):
            query = re.sub(r'(select\s+)', r'\1[1000] ', query, flags=re.IGNORECASE)
            logger.info(f"Added default result limit to query")

        return query

    def _get_query_hash(self, query: str) -> str:
        """Generate a hash for caching query counts."""
        return hashlib.md5(query.encode()).hexdigest()

    def _is_multi_statement_query(self, query: str) -> bool:
        """
        Detect if query contains multiple statements with variable assignments.

        Examples of multi-statement:
            a: select from trades; b: select from quotes; c: aj[...]

        Examples of single-statement:
            select from trades where date=2024.01.01

        Args:
            query: KDB query string

        Returns:
            True if multi-statement, False otherwise
        """
        # Pattern to match variable assignments: varname:
        # But exclude time literals like 09:30:00
        assignment_pattern = r'(?<![0-9])\b(\w+)\s*:'
        matches = re.findall(assignment_pattern, query)

        # Multi-statement if there are 2+ assignments
        is_multi = len(matches) >= 2

        if is_multi:
            logger.info(f"Detected multi-statement query with variables: {matches}")

        return is_multi

    def _extract_last_variable(self, query: str) -> Optional[str]:
        """
        Extract the last assigned variable from a multi-statement query.

        Example:
            a: select from trades;
            b: select from quotes;
            c: aj[`sym`time; a; b]
            → Returns 'c'

        Args:
            query: Multi-statement KDB query

        Returns:
            Last variable name or None
        """
        # Pattern to match variable assignments, excluding time literals
        assignment_pattern = r'(?<![0-9])\b(\w+)\s*:'
        matches = re.findall(assignment_pattern, query)

        if matches:
            last_var = matches[-1]
            logger.info(f"Last variable in multi-statement query: {last_var}")
            return last_var

        return None

    def _modify_query_for_pagination(self, query: str, page: int, page_size: int) -> str:
        """
        Modify a KDB select query to apply pagination at the database level.

        For KDB queries, we use sublist to slice results efficiently:
        - Remove any existing [N] limits from select
        - Apply offset and limit using sublist: (offset + limit) sublist offset _ query

        Args:
            query: Original KDB select query
            page: Page number (0-indexed)
            page_size: Number of records per page

        Returns:
            Modified query with pagination
        """
        # Remove existing limit syntax like select[1000]
        query_without_limit = re.sub(r'select\s*\[\d+\]\s*', 'select ', query, flags=re.IGNORECASE)

        # Calculate offset
        offset = page * page_size
        end = offset + page_size

        # Apply sublist: (start_index + count) sublist start_index _ query
        paginated_query = f"{end} sublist {offset} _ ({query_without_limit})"

        return paginated_query

    def _create_count_query_single_line(self, query: str) -> str:
        """
        Create an optimized count query for SINGLE_LINE queries.

        Instead of count(select ...), which evaluates the full select,
        we transform to: exec count i from table where ...

        Args:
            query: Original single-line select query

        Returns:
            Optimized count query
        """
        # Remove existing limit syntax
        query_without_limit = re.sub(r'select\s*\[\d+\]\s*', 'select ', query, flags=re.IGNORECASE)

        # Pattern: select ... from TABLE ...
        select_pattern = r'select\s+(?:.*?\s+)?from\s+(\w+)(.*)'
        match = re.match(select_pattern, query_without_limit, re.IGNORECASE | re.DOTALL)

        if match:
            table_name = match.group(1)
            rest_of_query = match.group(2)  # includes where, by, etc.

            # Use exec count i which is more efficient than select count
            count_query = f"exec count i from {table_name}{rest_of_query}"
            logger.info(f"Optimized count query (SINGLE_LINE): {count_query}")
            return count_query
        else:
            # Fallback: wrap in count (less optimal but works)
            count_query = f"count ({query_without_limit})"
            logger.warning(f"Using fallback count query: {count_query}")
            return count_query

    def _create_count_query_multi_line(self, query: str) -> str:
        """
        Create count query for MULTI_LINE queries by wrapping entire query.

        Example:
            Input: a: select from trades; b: select from quotes; c: aj[...]
            Output: count (a: select from trades; b: select from quotes; c: aj[...]; c)

        Args:
            query: Multi-statement query

        Returns:
            Wrapped count query
        """
        last_var = self._extract_last_variable(query)

        if last_var:
            # Wrap and return the last variable
            count_query = f"count ({query}; {last_var})"
            logger.info(f"Multi-line count query with last var '{last_var}': count (...; {last_var})")
        else:
            # No variable found, just wrap the query
            count_query = f"count ({query})"
            logger.warning("Multi-line query has no detected variables, using simple wrapping")

        return count_query

    def _modify_query_for_pagination_multi_line(self, query: str, page: int, page_size: int) -> str:
        """
        Apply pagination to MULTI_LINE queries by wrapping.

        Example:
            Input: a: select from trades; b: select from quotes; c: aj[...]
            Output: 100 sublist 0 _ (a: select from trades; b: select from quotes; c: aj[...]; c)

        Args:
            query: Multi-statement query
            page: Page number (0-indexed)
            page_size: Number of records per page

        Returns:
            Wrapped paginated query
        """
        last_var = self._extract_last_variable(query)
        offset = page * page_size
        end = offset + page_size

        if last_var:
            # Wrap and return the last variable with pagination
            paginated_query = f"{end} sublist {offset} _ ({query}; {last_var})"
            logger.info(f"Multi-line paginated query: {end} sublist {offset} _ (...; {last_var})")
        else:
            # No variable found, just wrap the query
            paginated_query = f"{end} sublist {offset} _ ({query})"
            logger.warning("Multi-line query has no detected variables, using simple wrapping")

        return paginated_query

    async def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute a KDB+/q query with comprehensive error handling and result processing.

        Args:
            query: The KDB+/q query to execute
            params: Optional query parameters

        Returns:
            Tuple of (results, metadata)
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Get database connection
            conn = await self._get_connection()

            # Sanitize and prepare query
            sanitized_query = self._sanitize_query(query)

            # Parameter substitution (basic implementation)
            if params:
                import json
                for key, value in params.items():
                    placeholder = f":{key}"
                    sanitized_query = sanitized_query.replace(placeholder, json.dumps(value))

            # Execute query asynchronously
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: conn(sanitized_query))

            # Convert results to list of dictionaries
            if hasattr(result, 'to_pandas'):
                # Convert KDB table to pandas DataFrame
                df = result.to_pandas()
                results = df.apply(self._convert_value).to_dict('records')
            elif isinstance(result, pd.DataFrame):
                results = result.apply(self._convert_value).to_dict('records')
            elif isinstance(result, list):
                results = [{'value': self._convert_value(item)} for item in result]
            elif result is not None:
                # Convert the result value
                converted_result = self._convert_value(result)

                # Check if the result is a dict with array values (columnar format from KDB+)
                if isinstance(converted_result, dict) and any(isinstance(v, list) for v in converted_result.values()):
                    # Convert from columnar to row-based format
                    results = self._convert_columnar_to_rows(converted_result)
                    logger.info("Converted columnar data to row-based format")
                else:
                    # Use standard format
                    results = [{'result': converted_result}]
            else:
                results = []

            # Prepare metadata
            metadata = self._extract_metadata(result)
            metadata.update({
                "query": sanitized_query,
                "execution_time": round(asyncio.get_event_loop().time() - start_time, 4),
                "timestamp": datetime.now().isoformat(),
                "database_type": "kdb"
            })

            return results, metadata

        except ValueError as security_error:
            # Handle security-related errors
            logger.warning(f"Query sanitization failed: {str(security_error)}")
            return [], {
                "error": str(security_error),
                "query": query,
                "execution_time": 0,
                "database_type": "kdb"
            }

        except Exception as e:
            # Comprehensive error handling
            logger.error(f"KDB+ Query Execution Error: {str(e)}", exc_info=True)

            # In production, return minimal error information
            return [], {
                "error": "Query execution failed",
                "details": str(e),
                "query": query,
                "execution_time": 0,
                "database_type": "kdb"
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
        Execute a KDB query with pagination applied at the database level.

        Uses hybrid approach:
        - SINGLE_LINE: Optimized count query + modified select
        - MULTI_LINE: Wrapping approach for multi-statement queries
        - Auto-detection: If complexity is MULTI_LINE but query is actually single-line, optimize

        Args:
            query: Original KDB query
            page: Page number (0-indexed)
            page_size: Number of records per page
            params: Optional query parameters
            **kwargs: Accepts query_complexity="SINGLE_LINE" or "MULTI_LINE"

        Returns:
            Tuple of (results, metadata, total_count)
        """
        start_time = asyncio.get_event_loop().time()
        query_complexity = kwargs.get('query_complexity', 'MULTI_LINE')

        try:
            # Get database connection
            conn = await self._get_connection()

            # Auto-detect: If marked as MULTI_LINE but actually single-line, optimize
            if query_complexity == "MULTI_LINE" and not self._is_multi_statement_query(query):
                logger.info("Auto-detected SINGLE_LINE query, using optimized pagination")
                query_complexity = "SINGLE_LINE"

            # Generate query hash for caching
            query_hash = self._get_query_hash(query + query_complexity)

            # Get or compute total count
            if query_hash in self._count_cache:
                total_count = self._count_cache[query_hash]
                logger.info(f"✅ Using cached count: {total_count}")
            else:
                # Create count query based on complexity
                if query_complexity == "SINGLE_LINE":
                    count_query = self._create_count_query_single_line(query)
                else:
                    count_query = self._create_count_query_multi_line(query)

                logger.info(f"🔢 Executing count query ({query_complexity})...")

                loop = asyncio.get_event_loop()
                count_result = await loop.run_in_executor(None, lambda: conn(count_query))

                # Extract count value
                if hasattr(count_result, 'py'):
                    total_count = int(count_result.py())
                else:
                    total_count = int(count_result)

                # Cache the count
                self._count_cache[query_hash] = total_count
                logger.info(f"💾 Cached count: {total_count} records")

            # Create paginated query based on complexity
            if query_complexity == "SINGLE_LINE":
                paginated_query = self._modify_query_for_pagination(query, page, page_size)
            else:
                paginated_query = self._modify_query_for_pagination_multi_line(query, page, page_size)

            # Parameter substitution if needed
            if params:
                import json
                for key, value in params.items():
                    placeholder = f":{key}"
                    paginated_query = paginated_query.replace(placeholder, json.dumps(value))

            # Execute paginated query
            logger.info(f"🚀 Executing paginated query (page {page}, size {page_size})...")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: conn(paginated_query))

            # Convert results to list of dictionaries
            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
                results = df.apply(self._convert_value).to_dict('records')
            elif isinstance(result, pd.DataFrame):
                results = result.apply(self._convert_value).to_dict('records')
            elif isinstance(result, list):
                results = [{'value': self._convert_value(item)} for item in result]
            elif result is not None:
                converted_result = self._convert_value(result)
                if isinstance(converted_result, dict) and any(isinstance(v, list) for v in converted_result.values()):
                    results = self._convert_columnar_to_rows(converted_result)
                else:
                    results = [{'result': converted_result}]
            else:
                results = []

            # Prepare metadata
            metadata = self._extract_metadata(result)
            metadata.update({
                "query": paginated_query,
                "original_query": query,
                "query_complexity": query_complexity,
                "execution_time": round(asyncio.get_event_loop().time() - start_time, 4),
                "timestamp": datetime.now().isoformat(),
                "database_type": "kdb"
            })

            return results, metadata, total_count

        except Exception as e:
            logger.error(f"KDB Paginated query execution error: {str(e)}", exc_info=True)

            # Return empty results with error metadata
            return [], {
                "error": "Paginated query execution failed",
                "details": str(e),
                "query": query,
                "query_complexity": query_complexity,
                "execution_time": 0,
                "database_type": "kdb"
            }, 0

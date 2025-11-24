"""
Base database connector with shared logic.

All database connectors inherit from BaseConnector and implement:
- execute(): Basic query execution
- execute_paginated(): Query execution with pagination

Shared functionality:
- Value conversion for JSON serialization
- Metadata extraction
- Columnar to row-based data conversion
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
import json
import numpy as np
import pandas as pd
from datetime import datetime, date

from app.core.logging import logger


class BaseConnector(ABC):
    """
    Abstract base class for all database connectors.

    Provides shared utility methods for:
    - Type conversion (datetime, numpy, pandas → JSON)
    - Data format conversion (columnar → rows)
    - Metadata extraction
    """

    def __init__(self, **connection_params):
        """
        Initialize connector with connection parameters.

        Args:
            **connection_params: Database-specific connection parameters
        """
        self.connection_params = connection_params
        self._connection = None
        self._pool = None

    def _convert_value(self, value: Any) -> Any:
        """
        Convert complex types to JSON-serializable Python types.

        Handles:
        - datetime/date → ISO format strings
        - numpy types → Python native types
        - PyKX types → Python types
        - Other types → string fallback

        Args:
            value: Raw value from database

        Returns:
            JSON-serializable Python representation
        """
        # Datetime conversions
        if isinstance(value, (datetime, date)):
            return value.isoformat()

        # Handle numpy types
        if isinstance(value, (np.integer, np.floating)):
            return value.item()

        if isinstance(value, np.ndarray):
            return value.tolist()

        # PyKx specific conversion (for KDB)
        if hasattr(value, 'py'):
            try:
                return value.py()
            except Exception:
                return str(value)

        # Default: try JSON serialization
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)

    def _convert_columnar_to_rows(self, data: Dict[str, List]) -> List[Dict[str, Any]]:
        """
        Convert columnar data (dict with array values) to row-based format.

        Example:
            Input:  {"name": ["Alice", "Bob"], "age": [30, 25]}
            Output: [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]

        Args:
            data: Dictionary with array values (columnar data)

        Returns:
            List of dictionaries (row-based data)
        """
        # If not a dictionary or no array values, return as is
        if not isinstance(data, dict):
            return data

        # Check if any value is an array - indicating columnar data
        has_arrays = any(isinstance(v, list) for v in data.values())
        if not has_arrays:
            return [data]  # Wrap in list to maintain expected format

        # Determine the length of arrays
        array_length = 0
        for val in data.values():
            if isinstance(val, list):
                array_length = len(val)
                break

        # Convert columnar data to row-based format
        row_data = []
        for i in range(array_length):
            row = {}
            for key, val in data.items():
                if isinstance(val, list) and i < len(val):
                    row[key] = val[i]
                else:
                    row[key] = val
            row_data.append(row)

        return row_data

    def _extract_metadata(self, result: Any) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from query results.

        Supports:
        - KDB table-like objects (dtype, names)
        - Pandas DataFrames
        - Other structured data

        Args:
            result: Raw result from database query

        Returns:
            Metadata dictionary with row_count, column_count, column_names, data_types
        """
        metadata = {
            "row_count": 0,
            "column_count": 0,
            "column_names": [],
            "data_types": {}
        }

        try:
            # Handle KDB table-like objects
            if hasattr(result, 'dtype') and hasattr(result, 'names'):
                metadata['column_names'] = list(result.dtype.names)
                metadata['column_count'] = len(result.dtype.names)
                metadata['row_count'] = len(result)
                metadata['data_types'] = {
                    col: str(result[col].dtype) for col in result.dtype.names
                }

            # Handle Pandas DataFrame
            elif isinstance(result, pd.DataFrame):
                metadata['column_names'] = list(result.columns)
                metadata['column_count'] = len(result.columns)
                metadata['row_count'] = len(result)
                metadata['data_types'] = result.dtypes.apply(str).to_dict()

            # Handle list of dicts
            elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                metadata['column_names'] = list(result[0].keys())
                metadata['column_count'] = len(result[0].keys())
                metadata['row_count'] = len(result)

        except Exception as e:
            logger.warning(f"Metadata extraction error: {str(e)}")

        return metadata

    @abstractmethod
    async def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute a database query.

        Must be implemented by each database connector.

        Args:
            query: Database query string
            params: Optional query parameters

        Returns:
            Tuple of (results, metadata)
            - results: List of dictionaries representing rows
            - metadata: Dict with execution info (time, row count, etc.)
        """
        pass

    @abstractmethod
    async def execute_paginated(
        self,
        query: str,
        page: int = 0,
        page_size: int = 100,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], int]:
        """
        Execute a database query with pagination.

        Must be implemented by each database connector.

        Args:
            query: Database query string
            page: Page number (0-indexed)
            page_size: Number of records per page
            params: Optional query parameters
            **kwargs: Database-specific options (e.g., query_complexity for KDB)

        Returns:
            Tuple of (results, metadata, total_count)
            - results: List of dictionaries for the current page
            - metadata: Dict with execution info
            - total_count: Total number of rows across all pages
        """
        pass

    async def close(self):
        """
        Close database connection/pool.

        Can be overridden by specific connectors if needed.
        """
        if self._connection:
            try:
                await self._connection.close()
                logger.info(f"Closed connection for {self.__class__.__name__}")
            except Exception as e:
                logger.warning(f"Error closing connection: {str(e)}")

        if self._pool:
            try:
                await self._pool.close()
                logger.info(f"Closed connection pool for {self.__class__.__name__}")
            except Exception as e:
                logger.warning(f"Error closing pool: {str(e)}")

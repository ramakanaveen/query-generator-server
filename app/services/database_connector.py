from typing import Dict, Any, Tuple, List, Optional, Union
import os
import re
import asyncio
import json
import pykx as kx
import numpy as np
import pandas as pd
from datetime import datetime, date

from app.core.config import settings
from app.core.logging import logger

class DatabaseConnector:
    """
    Robust service for connecting to and executing queries on KDB+ databases.
    Provides generic result handling and comprehensive error management.
    """
    
    def __init__(self, host: str = None, port: int = None, connection_lock=None):
        """
        Initialize database connector with configurable connection parameters.
        
        Args:
            host: Optional override for KDB host
            port: Optional override for KDB port
            connection_lock: Optional external asyncio.Lock to use
        """
        self.host = host or settings.KDB_HOST
        self.port = port or settings.KDB_PORT
        self._connection = None
        # Use provided lock or create a new one if we're in an async context
        self._connection_lock = connection_lock
    
    async def _get_connection(self):
        """
        Asynchronously get a connection to the KDB+ database.
        Uses a lock to prevent multiple simultaneous connection attempts.
        """
        # Ensure we have a lock
        if self._connection_lock is None:
            try:
                # We should be in an async context here, so this should work
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
    
    def _convert_value(self, value):
        """
        Convert complex KDB/q types to JSON-serializable Python types.
        
        Args:
            value: Raw value from KDB query
        
        Returns:
            Serializable Python representation
        """
        # Datetime conversions
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        
        # Handle numpy types
        if isinstance(value, (np.integer, np.floating, np.ndarray)):
            return value.tolist() if hasattr(value, 'tolist') else value.item()
        
        # PyKx specific conversion
        if hasattr(value, 'py'):
            try:
                return value.py()
            except:
                return str(value)
        
        # Default conversion
        try:
            json.dumps(value)
            return value
        except:
            return str(value)
    
    def _convert_columnar_to_rows(self, data):
        """
        Convert columnar data (dict with array values) to row-based format.
        
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
    
    def _extract_metadata(self, result):
        """
        Extract comprehensive metadata from query results.
        
        Args:
            result: Raw result from KDB query
        
        Returns:
            Metadata dictionary
        """
        metadata = {
            "row_count": 0,
            "column_count": 0,
            "column_names": [],
            "data_types": {}
        }
        
        try:
            # Handle different result types
            if hasattr(result, 'dtype') and hasattr(result, 'names'):
                # KDB table-like object
                metadata['column_names'] = list(result.dtype.names)
                metadata['column_count'] = len(result.dtype.names)
                metadata['row_count'] = len(result)
                
                # Capture data types
                metadata['data_types'] = {
                    col: str(result[col].dtype) for col in result.dtype.names
                }
            
            elif isinstance(result, pd.DataFrame):
                # Pandas DataFrame
                metadata['column_names'] = list(result.columns)
                metadata['column_count'] = len(result.columns)
                metadata['row_count'] = len(result)
                metadata['data_types'] = result.dtypes.apply(str).to_dict()
            
            # Add more type handling as needed
        except Exception as e:
            logger.warning(f"Metadata extraction error: {str(e)}")
        
        return metadata
    
    async def execute(self, query: str, params: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
                "timestamp": datetime.now().isoformat()
            })
            
            return results, metadata
        
        except ValueError as security_error:
            # Handle security-related errors
            logger.warning(f"Query sanitization failed: {str(security_error)}")
            return [], {
                "error": str(security_error),
                "query": query,
                "execution_time": 0
            }
        
        except Exception as e:
            # Comprehensive error handling
            logger.error(f"KDB+ Query Execution Error: {str(e)}", exc_info=True)
            
            # Provide mock data in debug mode
            if settings.DEBUG:
                mock_results = [
                    {"time": "09:30:00", "ticker": "AAPL", "price": 150.25, "quantity": 1000},
                    {"time": "09:32:15", "ticker": "MSFT", "price": 290.45, "quantity": 500}
                ]
                
                return mock_results, {
                    "error": str(e),
                    "query": query,
                    "execution_time": 0.05,
                    "is_mock_data": True
                }
            
            # In production, return minimal error information
            return [], {
                "error": "Query execution failed",
                "details": str(e),
                "query": query,
                "execution_time": 0
            }
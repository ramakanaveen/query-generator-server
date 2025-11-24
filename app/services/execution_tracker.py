"""
Execution Tracker Service

Tracks all query executions for auditing, analytics, and debugging purposes.
Logs executions asynchronously to avoid impacting API performance.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import asyncio

from app.core.db import db_pool
from app.core.logging import logger


class ExecutionTracker:
    """Service for tracking query executions in the database."""

    @staticmethod
    def log_execution_background(
        execution_id: str,
        query: str,
        database_type: str,
        status: str,
        **kwargs
    ) -> asyncio.Task:
        """
        Log execution in the background without blocking the response.

        This creates a background task that runs asynchronously and doesn't
        impact the API response time.

        Args:
            execution_id: Unique execution identifier
            query: The executed query
            database_type: Database type (kdb, starburst, etc.)
            status: Execution status (success, failed, timeout, error)
            **kwargs: Additional parameters passed to log_execution

        Returns:
            Background task (fire-and-forget)
        """
        task = asyncio.create_task(
            ExecutionTracker.log_execution(
                execution_id=execution_id,
                query=query,
                database_type=database_type,
                status=status,
                **kwargs
            )
        )
        # Add callback to log any errors in background task
        task.add_done_callback(ExecutionTracker._handle_task_result)
        return task

    @staticmethod
    def _handle_task_result(task: asyncio.Task):
        """Handle completion of background logging task."""
        try:
            task.result()  # This will raise exception if task failed
        except Exception as e:
            logger.error(f"Background execution logging failed: {str(e)}", exc_info=True)

    @staticmethod
    async def log_execution(
        execution_id: str,
        query: str,
        database_type: str,
        status: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        query_complexity: Optional[str] = None,
        execution_time: Optional[float] = None,
        total_rows: Optional[int] = None,
        returned_rows: Optional[int] = None,
        page: int = 0,
        page_size: int = 100,
        error_message: Optional[str] = None,
        error_type: Optional[str] = None,
        connection_params: Optional[Dict[str, Any]] = None,
        request_metadata: Optional[Dict[str, Any]] = None,
        response_metadata: Optional[Dict[str, Any]] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None
    ) -> Optional[int]:
        """
        Log a query execution to the database.

        Args:
            execution_id: Unique execution identifier
            query: The executed query
            database_type: Database type (kdb, starburst, etc.)
            status: Execution status (success, failed, timeout, error)
            user_id: User who executed the query
            conversation_id: Associated conversation ID
            query_complexity: Query complexity (SINGLE_LINE, MULTI_LINE)
            execution_time: Execution time in seconds
            total_rows: Total rows in result set
            returned_rows: Rows returned in current page
            page: Page number
            page_size: Page size
            error_message: Error message if failed
            error_type: Type of error
            connection_params: Connection parameters used
            request_metadata: Additional request metadata
            response_metadata: Additional response metadata
            started_at: Execution start time
            completed_at: Execution completion time

        Returns:
            Execution record ID or None if logging failed
        """
        try:
            query_sql = """
                INSERT INTO query_executions (
                    execution_id, user_id, conversation_id,
                    query, database_type, query_complexity,
                    status, execution_time, total_rows, returned_rows,
                    page, page_size,
                    error_message, error_type,
                    started_at, completed_at,
                    connection_params, request_metadata, response_metadata
                ) VALUES (
                    $1, $2, $3,
                    $4, $5, $6,
                    $7, $8, $9, $10,
                    $11, $12,
                    $13, $14,
                    $15, $16,
                    $17, $18, $19
                )
                RETURNING id
            """

            # Convert dicts to JSON
            connection_params_json = json.dumps(connection_params or {})
            request_metadata_json = json.dumps(request_metadata or {})
            response_metadata_json = json.dumps(response_metadata or {})

            # Use current time if not provided
            if started_at is None:
                started_at = datetime.utcnow()

            record_id = await db_pool.fetchval(
                query_sql,
                execution_id, user_id, conversation_id,
                query, database_type, query_complexity,
                status, execution_time, total_rows, returned_rows,
                page, page_size,
                error_message, error_type,
                started_at, completed_at,
                connection_params_json, request_metadata_json, response_metadata_json
            )

            logger.info(f"✅ Logged execution {execution_id} with status: {status}")
            return record_id

        except Exception as e:
            # Don't let execution tracking failures crash the main request
            logger.error(f"Failed to log execution {execution_id}: {str(e)}", exc_info=True)
            return None

    @staticmethod
    async def get_user_executions(
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        status_filter: Optional[str] = None,
        database_type_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get execution history for a user.

        Args:
            user_id: User ID
            limit: Maximum number of records
            offset: Offset for pagination
            status_filter: Filter by status
            database_type_filter: Filter by database type

        Returns:
            List of execution records
        """
        try:
            query_parts = ["""
                SELECT
                    id, execution_id, user_id, conversation_id,
                    query, database_type, query_complexity,
                    status, execution_time, total_rows, returned_rows,
                    page, page_size,
                    error_message, error_type,
                    started_at, completed_at
                FROM query_executions
                WHERE user_id = $1
            """]

            params = [user_id]
            param_count = 2

            if status_filter:
                query_parts.append(f"AND status = ${param_count}")
                params.append(status_filter)
                param_count += 1

            if database_type_filter:
                query_parts.append(f"AND database_type = ${param_count}")
                params.append(database_type_filter)
                param_count += 1

            query_parts.append("ORDER BY started_at DESC")
            query_parts.append(f"LIMIT ${param_count} OFFSET ${param_count + 1}")
            params.extend([limit, offset])

            query_sql = " ".join(query_parts)

            rows = await db_pool.fetch(query_sql, *params)

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to fetch user executions: {str(e)}", exc_info=True)
            return []

    @staticmethod
    async def get_execution_stats(
        user_id: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get execution statistics.

        Args:
            user_id: Optional user ID to filter by
            days: Number of days to include

        Returns:
            Dictionary with execution statistics
        """
        try:
            query_sql = """
                SELECT
                    COUNT(*) as total_executions,
                    COUNT(CASE WHEN status = 'success' THEN 1 END) as success_count,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_count,
                    COUNT(CASE WHEN status = 'error' THEN 1 END) as error_count,
                    COUNT(CASE WHEN status = 'timeout' THEN 1 END) as timeout_count,
                    AVG(execution_time) as avg_execution_time,
                    MAX(execution_time) as max_execution_time,
                    SUM(total_rows) as total_rows_processed
                FROM query_executions
                WHERE started_at >= NOW() - INTERVAL '%s days'
            """

            params = []
            if user_id:
                query_sql += " AND user_id = $1"
                params.append(user_id)

            # Format the interval
            query_sql = query_sql % days

            row = await db_pool.fetchrow(query_sql, *params)

            return dict(row) if row else {}

        except Exception as e:
            logger.error(f"Failed to fetch execution stats: {str(e)}", exc_info=True)
            return {}

    @staticmethod
    async def get_database_type_stats(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get execution statistics grouped by database type.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            List of statistics per database type
        """
        try:
            query_sql = """
                SELECT
                    database_type,
                    COUNT(*) as execution_count,
                    COUNT(CASE WHEN status = 'success' THEN 1 END) as success_count,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_count,
                    AVG(execution_time) as avg_execution_time
                FROM query_executions
                WHERE started_at >= NOW() - INTERVAL '30 days'
            """

            params = []
            if user_id:
                query_sql += " AND user_id = $1"
                params.append(user_id)

            query_sql += " GROUP BY database_type ORDER BY execution_count DESC"

            rows = await db_pool.fetch(query_sql, *params)

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to fetch database type stats: {str(e)}", exc_info=True)
            return []

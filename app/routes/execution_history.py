# app/routes/execution_history.py

"""
Execution History API Endpoints

Provides endpoints to query execution history, statistics, and analytics.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List, Dict, Any

from app.services.execution_tracker import ExecutionTracker
from app.core.logging import logger

router = APIRouter()


@router.get("/executions/{user_id}")
async def get_user_executions(
    user_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None, regex="^(success|failed|error|timeout)$"),
    database_type: Optional[str] = Query(default=None)
):
    """
    Get execution history for a specific user.

    Args:
        user_id: User ID to query executions for
        limit: Maximum number of records to return (1-1000)
        offset: Offset for pagination
        status: Filter by execution status (success, failed, error, timeout)
        database_type: Filter by database type (kdb, starburst, etc.)

    Returns:
        List of execution records
    """
    try:
        executions = await ExecutionTracker.get_user_executions(
            user_id=user_id,
            limit=limit,
            offset=offset,
            status_filter=status,
            database_type_filter=database_type
        )

        return {
            "user_id": user_id,
            "executions": executions,
            "count": len(executions),
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"Failed to fetch user executions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve execution history"
        )


@router.get("/executions/{user_id}/stats")
async def get_user_execution_stats(
    user_id: str,
    days: int = Query(default=7, ge=1, le=365)
):
    """
    Get execution statistics for a specific user.

    Args:
        user_id: User ID to get stats for
        days: Number of days to include in statistics (1-365)

    Returns:
        Execution statistics including success rate, performance metrics
    """
    try:
        stats = await ExecutionTracker.get_execution_stats(user_id=user_id, days=days)

        # Calculate success rate
        total = stats.get('total_executions', 0)
        success_rate = 0.0
        if total > 0:
            success_rate = (stats.get('success_count', 0) / total) * 100

        return {
            "user_id": user_id,
            "period_days": days,
            "total_executions": total,
            "success_count": stats.get('success_count', 0),
            "failed_count": stats.get('failed_count', 0),
            "error_count": stats.get('error_count', 0),
            "timeout_count": stats.get('timeout_count', 0),
            "success_rate": round(success_rate, 2),
            "avg_execution_time": round(float(stats.get('avg_execution_time', 0) or 0), 4),
            "max_execution_time": round(float(stats.get('max_execution_time', 0) or 0), 4),
            "total_rows_processed": stats.get('total_rows_processed', 0)
        }

    except Exception as e:
        logger.error(f"Failed to fetch user execution stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve execution statistics"
        )


@router.get("/executions/{user_id}/database-stats")
async def get_user_database_stats(user_id: str):
    """
    Get execution statistics grouped by database type for a user.

    Args:
        user_id: User ID to get stats for

    Returns:
        Statistics per database type
    """
    try:
        stats = await ExecutionTracker.get_database_type_stats(user_id=user_id)

        return {
            "user_id": user_id,
            "database_stats": stats
        }

    except Exception as e:
        logger.error(f"Failed to fetch database type stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve database statistics"
        )


@router.get("/executions/stats/global")
async def get_global_execution_stats(
    days: int = Query(default=7, ge=1, le=365)
):
    """
    Get global execution statistics across all users.

    Args:
        days: Number of days to include in statistics (1-365)

    Returns:
        Global execution statistics
    """
    try:
        stats = await ExecutionTracker.get_execution_stats(user_id=None, days=days)

        # Calculate success rate
        total = stats.get('total_executions', 0)
        success_rate = 0.0
        if total > 0:
            success_rate = (stats.get('success_count', 0) / total) * 100

        return {
            "period_days": days,
            "total_executions": total,
            "success_count": stats.get('success_count', 0),
            "failed_count": stats.get('failed_count', 0),
            "error_count": stats.get('error_count', 0),
            "timeout_count": stats.get('timeout_count', 0),
            "success_rate": round(success_rate, 2),
            "avg_execution_time": round(float(stats.get('avg_execution_time', 0) or 0), 4),
            "max_execution_time": round(float(stats.get('max_execution_time', 0) or 0), 4),
            "total_rows_processed": stats.get('total_rows_processed', 0)
        }

    except Exception as e:
        logger.error(f"Failed to fetch global execution stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve global statistics"
        )


@router.get("/executions/database-stats/global")
async def get_global_database_stats():
    """
    Get execution statistics grouped by database type across all users.

    Returns:
        Statistics per database type globally
    """
    try:
        stats = await ExecutionTracker.get_database_type_stats(user_id=None)

        return {
            "database_stats": stats
        }

    except Exception as e:
        logger.error(f"Failed to fetch global database stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve database statistics"
        )

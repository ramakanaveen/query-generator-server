# Execution Tracking System

## Overview

The execution tracking system provides comprehensive auditing, analytics, and debugging capabilities for all query executions. It tracks query execution history, user activity, success/failure metrics, and performance data **asynchronously** to avoid impacting API performance.

## Features

- **Async Background Logging**: All execution tracking happens in background tasks without blocking API responses
- **Comprehensive Tracking**: Captures query, user, database type, status, timing, and error details
- **Multi-Database Support**: Works with KDB, Starburst, and future database connectors
- **Analytics Views**: Pre-built views for execution analytics and user statistics
- **RESTful API**: Query execution history and statistics via REST endpoints

## Database Schema

### Main Table: `query_executions`

```sql
CREATE TABLE query_executions (
    id SERIAL PRIMARY KEY,
    execution_id VARCHAR(100) UNIQUE NOT NULL,
    user_id VARCHAR(100),
    conversation_id VARCHAR(100),

    -- Query details
    query TEXT NOT NULL,
    database_type VARCHAR(50) NOT NULL,
    query_complexity VARCHAR(20),

    -- Execution details
    status VARCHAR(20) CHECK (status IN ('success', 'failed', 'timeout', 'error')),
    execution_time FLOAT,
    total_rows INTEGER,
    returned_rows INTEGER,
    page INTEGER DEFAULT 0,
    page_size INTEGER DEFAULT 100,

    -- Error tracking
    error_message TEXT,
    error_type VARCHAR(100),

    -- Timestamps
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Metadata
    connection_params JSONB DEFAULT '{}',
    request_metadata JSONB DEFAULT '{}',
    response_metadata JSONB DEFAULT '{}'
);
```

### Analytics Views

**`execution_analytics`**: Daily execution metrics per user, database type, and status

**`user_recent_executions`**: Recent executions with ranking per user

## API Endpoints

### 1. Get User Execution History

```bash
GET /api/v1/executions/{user_id}?limit=100&offset=0&status=success&database_type=kdb
```

**Query Parameters:**
- `limit` (1-1000): Max records to return
- `offset` (>=0): Pagination offset
- `status`: Filter by status (success, failed, error, timeout)
- `database_type`: Filter by database type

**Response:**
```json
{
  "user_id": "user123",
  "executions": [
    {
      "id": 1,
      "execution_id": "exec-abc-123",
      "query": "SELECT * FROM users",
      "database_type": "starburst",
      "status": "success",
      "execution_time": 0.234,
      "total_rows": 1500,
      "started_at": "2025-11-24T10:30:00Z"
    }
  ],
  "count": 1,
  "limit": 100,
  "offset": 0
}
```

### 2. Get User Execution Statistics

```bash
GET /api/v1/executions/{user_id}/stats?days=7
```

**Response:**
```json
{
  "user_id": "user123",
  "period_days": 7,
  "total_executions": 150,
  "success_count": 145,
  "failed_count": 3,
  "error_count": 2,
  "timeout_count": 0,
  "success_rate": 96.67,
  "avg_execution_time": 0.523,
  "max_execution_time": 2.145,
  "total_rows_processed": 45000
}
```

### 3. Get Database Type Statistics (Per User)

```bash
GET /api/v1/executions/{user_id}/database-stats
```

**Response:**
```json
{
  "user_id": "user123",
  "database_stats": [
    {
      "database_type": "kdb",
      "execution_count": 80,
      "success_count": 78,
      "failed_count": 2,
      "avg_execution_time": 0.412
    },
    {
      "database_type": "starburst",
      "execution_count": 70,
      "success_count": 67,
      "failed_count": 3,
      "avg_execution_time": 0.651
    }
  ]
}
```

### 4. Get Global Execution Statistics

```bash
GET /api/v1/executions/stats/global?days=7
```

**Response:**
```json
{
  "period_days": 7,
  "total_executions": 5234,
  "success_count": 5102,
  "failed_count": 98,
  "error_count": 32,
  "timeout_count": 2,
  "success_rate": 97.48,
  "avg_execution_time": 0.456,
  "max_execution_time": 8.234,
  "total_rows_processed": 1250000
}
```

### 5. Get Global Database Statistics

```bash
GET /api/v1/executions/database-stats/global
```

## How It Works

### Background Logging (Non-Blocking)

The `/v3/execute` endpoint uses **asynchronous background tasks** to log executions without impacting response time:

```python
# In /v3/execute endpoint
ExecutionTracker.log_execution_background(
    execution_id=execution_id,
    query=query,
    database_type=database_type,
    status="success",
    user_id=user_id,
    # ... other parameters
)

# Returns immediately to user (no waiting!)
return ExecutionResponse(...)
```

### How Background Tasks Work

1. **`log_execution_background()`**: Creates an `asyncio.Task` that runs in the background
2. **Fire-and-forget**: The API response is sent immediately without waiting for logging
3. **Error handling**: Background task failures are logged but don't crash the API
4. **Performance**: Zero impact on API response time

### Automatic Tracking

The system automatically tracks:

- ✅ **Success executions**: Full execution details, timing, row counts
- ❌ **Failed executions**: Error messages, error types, status codes
- ⏱️ **Timeout executions**: Detected from error messages
- 🔍 **All metadata**: Connection params, request/response metadata

## Migration

Run the migration script to create the tables and views:

```bash
psql $DATABASE_URL -f scripts/db_scripts/03_execution_tracking.sql
```

## Usage in Code

### Direct Logging (with await)

```python
from app.services.execution_tracker import ExecutionTracker

await ExecutionTracker.log_execution(
    execution_id="exec-123",
    query="SELECT * FROM table",
    database_type="kdb",
    status="success",
    user_id="user123",
    execution_time=0.543,
    total_rows=1000
)
```

### Background Logging (no await, non-blocking)

```python
# Recommended for API endpoints
ExecutionTracker.log_execution_background(
    execution_id="exec-123",
    query="SELECT * FROM table",
    database_type="kdb",
    status="success",
    user_id="user123",
    execution_time=0.543,
    total_rows=1000
)
# Continues immediately without waiting
```

## Use Cases

### 1. Auditing & Compliance
- Track who executed what query and when
- Monitor database access patterns
- Compliance reporting

### 2. Performance Monitoring
- Identify slow queries
- Track execution time trends
- Optimize query patterns

### 3. Error Analysis
- Debug failed executions
- Identify common error patterns
- Track error rates by database type

### 4. User Analytics
- Understand user behavior
- Track feature adoption (KDB vs Starburst usage)
- Identify power users

### 5. Capacity Planning
- Monitor query volume trends
- Predict resource needs
- Optimize database allocation

## Performance Considerations

- **Background Logging**: All logging happens asynchronously - **zero impact** on API latency
- **Indexed Columns**: Fast queries on user_id, status, database_type, timestamps
- **Composite Indexes**: Optimized for common query patterns
- **Views**: Pre-aggregated analytics for fast reporting

## Security

- User IDs are stored but not validated (soft reference)
- Queries are stored in full (consider encryption for sensitive data)
- Connection params are stored (ensure no passwords in plain text)
- RBAC can be added to execution history endpoints

## Future Enhancements

- [ ] Query result caching based on execution history
- [ ] Anomaly detection for unusual query patterns
- [ ] Automatic query optimization suggestions
- [ ] Real-time execution monitoring dashboard
- [ ] Scheduled reports and alerts
- [ ] Data retention policies (auto-delete old records)

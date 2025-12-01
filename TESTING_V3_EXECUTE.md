# Testing /v3/execute with KDB and Execution Tracking

## Quick Start

### 1. Prerequisites

**Start KDB Server:**
```bash
# If you have KDB+ installed
q -p 5001
```

**Run Database Migration (for execution tracking):**
```bash
# Apply the execution tracking schema
psql $DATABASE_URL -f scripts/db_scripts/03_execution_tracking.sql
```

**Start FastAPI Server:**
```bash
python -m uvicorn app.main:app --reload
```

### 2. Test with cURL

**Simple KDB Query:**
```bash
curl -X POST http://localhost:8000/api/v1/execute/v3 \
  -H "Content-Type: application/json" \
  -d '{
    "query": "10#til 100",
    "execution_id": "test-001",
    "database_type": "kdb",
    "query_complexity": "SINGLE_LINE",
    "pagination": {"page": 0, "page_size": 10}
  }'
```

**Create and Query Table:**
```bash
curl -X POST http://localhost:8000/api/v1/execute/v3 \
  -H "Content-Type: application/json" \
  -d '{
    "query": "([] name:`Alice`Bob`Charlie; age:25 30 35)",
    "execution_id": "test-002",
    "database_type": "kdb",
    "pagination": {"page": 0, "page_size": 10}
  }'
```

### 3. Test with Python Script

```bash
python scripts/test_kdb_v3_execution.py
```

This comprehensive test script will:
- ✅ Execute various KDB queries
- ✅ Test pagination
- ✅ Verify execution tracking
- ✅ Query execution history
- ✅ Show execution statistics

### 4. Check Execution History

**Get User's Execution History:**
```bash
curl http://localhost:8000/api/v1/executions/test_user_123?limit=10
```

**Get User Statistics:**
```bash
curl http://localhost:8000/api/v1/executions/test_user_123/stats?days=7
```

**Get Global Statistics:**
```bash
curl http://localhost:8000/api/v1/executions/stats/global?days=7
```

## What Happens Behind the Scenes

### 1. Query Execution (/v3/execute)
```
User Request → /v3/execute endpoint
              ↓
         Get Connector (KDB/Starburst)
              ↓
    Execute Query with Pagination
              ↓
    Return Results to User ✅ (FAST!)
              ↓
    Background Task → Log to database (async, no blocking)
```

### 2. Execution Tracking (Async)
```python
# In the endpoint (non-blocking):
ExecutionTracker.log_execution_background(
    execution_id=execution_id,
    query=query,
    database_type="kdb",
    status="success",
    execution_time=0.543,
    total_rows=1000,
    # ... more fields
)

# Returns immediately to user!
# Logging happens in background
```

### 3. What Gets Tracked

For each execution, the system logs:
- **Execution ID** - Unique identifier
- **User ID** - Who executed it
- **Query** - The actual query text
- **Database Type** - kdb, starburst, etc.
- **Status** - success, failed, error, timeout
- **Performance** - Execution time, row counts
- **Pagination** - Page number, page size
- **Errors** - Error message and type (if failed)
- **Timestamps** - Started/completed times
- **Metadata** - Connection params, request/response data

## Example Outputs

### Successful Query Execution
```json
{
  "results": [
    {"x": 0},
    {"x": 1},
    {"x": 2}
  ],
  "metadata": {
    "row_count": 3,
    "execution_time": 0.0234,
    "database_type": "kdb"
  },
  "pagination": {
    "currentPage": 0,
    "totalPages": 1,
    "totalRows": 10,
    "pageSize": 10,
    "returnedRows": 3
  }
}
```

### Execution History
```json
{
  "user_id": "test_user_123",
  "executions": [
    {
      "execution_id": "test-001",
      "query": "10#til 100",
      "database_type": "kdb",
      "status": "success",
      "execution_time": 0.0234,
      "total_rows": 10,
      "started_at": "2025-11-24T10:30:00Z"
    }
  ],
  "count": 1
}
```

### Execution Statistics
```json
{
  "user_id": "test_user_123",
  "period_days": 7,
  "total_executions": 150,
  "success_count": 145,
  "failed_count": 5,
  "success_rate": 96.67,
  "avg_execution_time": 0.543,
  "max_execution_time": 2.145,
  "total_rows_processed": 45000
}
```

## Troubleshooting

### KDB Connection Error
```
❌ Error: Could not connect to KDB+: Connection error
```
**Solution**: Start KDB server on localhost:5001
```bash
q -p 5001
```

### Execution Tracking Error
```
❌ Error: 'NoneType' object has no attribute 'acquire'
```
**Solution**: Run the database migration
```bash
psql $DATABASE_URL -f scripts/db_scripts/03_execution_tracking.sql
```

### No Execution History
If execution tracking shows no data:
1. Check PostgreSQL is running
2. Verify migration was applied
3. Ensure `DATABASE_URL` environment variable is set
4. Check application logs for background task errors

## Performance Notes

### Response Time
- **With Execution Tracking**: Same as without! (~10-100ms typical)
- **Tracking Impact**: 0ms (runs in background)
- **Database Write**: Happens asynchronously after response

### Example Timing
```
Request Received:     0.000s
Query Executed:       0.050s  (50ms)
Response Sent:        0.052s  (52ms) ✅ User gets response
------------------------
Background Log Start: 0.052s
Background Log Done:  0.075s  (23ms, user already has response)
```

## Next Steps

1. ✅ **Test Basic Queries** - Use the test script
2. ✅ **Check Execution History** - Verify tracking works
3. ✅ **Monitor Statistics** - Track success rates
4. 📊 **Build Dashboard** - Visualize execution metrics
5. 🔐 **Add Authentication** - Track real user IDs
6. 🚨 **Add Alerting** - Monitor error rates

## Additional Resources

- Full Documentation: `docs/EXECUTION_TRACKING.md`
- API Tests: `tests/test_execution_tracking.py`
- Integration Tests: `tests/test_v3_execute.py`
- Schema: `scripts/db_scripts/03_execution_tracking.sql`

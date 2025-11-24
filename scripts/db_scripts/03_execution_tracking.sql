-- execution_tracking.sql
-- Track all query executions for auditing, analytics, and debugging

CREATE TABLE query_executions (
    id SERIAL PRIMARY KEY,
    execution_id VARCHAR(100) UNIQUE NOT NULL,
    user_id VARCHAR(100),
    conversation_id VARCHAR(100) REFERENCES conversations(id) ON DELETE SET NULL,

    -- Query details
    query TEXT NOT NULL,
    database_type VARCHAR(50) NOT NULL,
    query_complexity VARCHAR(20), -- KDB specific: SINGLE_LINE, MULTI_LINE

    -- Execution details
    status VARCHAR(20) CHECK (status IN ('success', 'failed', 'timeout', 'error')) NOT NULL,
    execution_time FLOAT, -- in seconds
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

    -- Additional metadata
    connection_params JSONB DEFAULT '{}',
    request_metadata JSONB DEFAULT '{}',
    response_metadata JSONB DEFAULT '{}'
);

-- Indexes for efficient querying
CREATE INDEX idx_query_executions_user_id ON query_executions(user_id);
CREATE INDEX idx_query_executions_status ON query_executions(status);
CREATE INDEX idx_query_executions_database_type ON query_executions(database_type);
CREATE INDEX idx_query_executions_started_at ON query_executions(started_at DESC);
CREATE INDEX idx_query_executions_execution_id ON query_executions(execution_id);
CREATE INDEX idx_query_executions_conversation_id ON query_executions(conversation_id);

-- Composite index for common queries
CREATE INDEX idx_query_executions_user_status ON query_executions(user_id, status, started_at DESC);

-- Add foreign key to users table if it exists
-- This will be a soft reference for now, can be enforced later
-- ALTER TABLE query_executions ADD CONSTRAINT fk_query_executions_users
--     FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL;

-- Create a view for execution analytics
CREATE OR REPLACE VIEW execution_analytics AS
SELECT
    user_id,
    database_type,
    status,
    DATE(started_at) as execution_date,
    COUNT(*) as execution_count,
    AVG(execution_time) as avg_execution_time,
    MAX(execution_time) as max_execution_time,
    MIN(execution_time) as min_execution_time,
    SUM(total_rows) as total_rows_processed,
    COUNT(CASE WHEN status = 'success' THEN 1 END) as success_count,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_count,
    COUNT(CASE WHEN status = 'error' THEN 1 END) as error_count,
    COUNT(CASE WHEN status = 'timeout' THEN 1 END) as timeout_count
FROM query_executions
GROUP BY user_id, database_type, status, DATE(started_at);

-- Create a view for recent executions per user
CREATE OR REPLACE VIEW user_recent_executions AS
SELECT
    user_id,
    execution_id,
    query,
    database_type,
    status,
    execution_time,
    started_at,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY started_at DESC) as execution_rank
FROM query_executions
ORDER BY user_id, started_at DESC;

COMMENT ON TABLE query_executions IS 'Tracks all query executions for auditing, analytics, and debugging';
COMMENT ON COLUMN query_executions.execution_id IS 'Unique identifier from the execution request';
COMMENT ON COLUMN query_executions.status IS 'Execution status: success, failed, timeout, error';
COMMENT ON COLUMN query_executions.execution_time IS 'Query execution time in seconds';
COMMENT ON COLUMN query_executions.query_complexity IS 'KDB-specific query complexity classification';

-- ============================================================================
-- MEMORY MODULE - Database Schema
-- ============================================================================
-- This schema stores long-term learnings from user interactions
-- Can be used independently by any AI system
-- ============================================================================

-- Create dedicated schema for memory isolation
CREATE SCHEMA IF NOT EXISTS memory;

-- ============================================================================
-- Core Memory Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS memory.entries (
    -- Identity
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Categorization
    memory_type VARCHAR(50) NOT NULL,
    -- Types: 'syntax_correction', 'user_definition', 'approach_recommendation',
    --        'query_pattern', 'error_correction'

    -- Scope (who does this apply to?)
    user_id VARCHAR(255),              -- NULL = global memory
    schema_group_id UUID,              -- NULL = applies to all schemas

    -- Content
    original_context TEXT NOT NULL,    -- What triggered this memory
    learning_description TEXT NOT NULL, -- What was learned
    corrected_version TEXT,            -- The correction/improvement

    -- Rich metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    -- Examples:
    -- {"table": "trade", "column": "px", "error_type": "syntax"}
    -- {"original_term": "VWAP", "definition": "Volume Weighted Average Price"}

    -- Vector embedding for semantic search
    embedding vector(768),

    -- Provenance (where did this come from?)
    source_type VARCHAR(50),           -- 'feedback', 'correction', 'manual'
    source_conversation_id UUID,
    source_feedback_id UUID,

    -- Quality metrics
    confidence_score FLOAT DEFAULT 0.5 CHECK (confidence_score >= 0 AND confidence_score <= 1),
    success_count INT DEFAULT 0,       -- How many times this helped
    failure_count INT DEFAULT 0,       -- How many times this was wrong

    -- Temporal tracking
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at TIMESTAMP,
    access_count INT DEFAULT 0,

    -- Lifecycle management
    is_active BOOLEAN DEFAULT true,
    is_validated BOOLEAN DEFAULT false, -- Admin validation flag

    -- Constraints
    CONSTRAINT check_memory_type CHECK (
        memory_type IN (
            'syntax_correction',
            'user_definition',
            'approach_recommendation',
            'query_pattern',
            'error_correction',
            'schema_clarification'
        )
    ),
    CONSTRAINT check_source_type CHECK (
        source_type IN ('feedback', 'correction', 'manual', 'auto_detected')
    )
);

-- ============================================================================
-- Indexes for Performance
-- ============================================================================

-- Primary lookup indexes
CREATE INDEX idx_memory_type ON memory.entries(memory_type) WHERE is_active = true;
CREATE INDEX idx_user_id ON memory.entries(user_id) WHERE is_active = true;
CREATE INDEX idx_schema_group ON memory.entries(schema_group_id) WHERE is_active = true;

-- Composite indexes for common queries
CREATE INDEX idx_user_type ON memory.entries(user_id, memory_type) WHERE is_active = true;
CREATE INDEX idx_schema_type ON memory.entries(schema_group_id, memory_type) WHERE is_active = true;

-- Quality-based retrieval
CREATE INDEX idx_quality_score ON memory.entries(
    (confidence_score * (success_count + 1.0) / (success_count + failure_count + 2.0)) DESC
) WHERE is_active = true;

-- Temporal indexes
CREATE INDEX idx_recency ON memory.entries(created_at DESC) WHERE is_active = true;
CREATE INDEX idx_last_accessed ON memory.entries(last_accessed_at DESC NULLS LAST) WHERE is_active = true;

-- Vector similarity search (using pgvector)
CREATE INDEX idx_memory_embedding ON memory.entries
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100)
    WHERE is_active = true;

-- JSONB metadata search
CREATE INDEX idx_memory_metadata ON memory.entries USING gin(metadata);

-- ============================================================================
-- Memory Tags (for flexible categorization)
-- ============================================================================
CREATE TABLE IF NOT EXISTS memory.tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL REFERENCES memory.entries(id) ON DELETE CASCADE,
    tag VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(memory_id, tag)
);

CREATE INDEX idx_tags_memory ON memory.tags(memory_id);
CREATE INDEX idx_tags_tag ON memory.tags(tag);
CREATE INDEX idx_tags_lookup ON memory.tags(tag, memory_id);

-- ============================================================================
-- Memory Usage Tracking (analytics)
-- ============================================================================
CREATE TABLE IF NOT EXISTS memory.usage_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL REFERENCES memory.entries(id) ON DELETE CASCADE,

    -- Context
    query_id UUID,
    conversation_id UUID,
    user_id VARCHAR(255),

    -- Usage details
    was_helpful BOOLEAN,               -- Did this memory help?
    applied_to_query BOOLEAN,          -- Was it actually used?
    similarity_score FLOAT,            -- How relevant was it?

    -- Timestamps
    used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_usage_memory ON memory.usage_log(memory_id, used_at DESC);
CREATE INDEX idx_usage_helpful ON memory.usage_log(was_helpful) WHERE was_helpful = true;

-- ============================================================================
-- Functions for Automatic Updates
-- ============================================================================

-- Update 'updated_at' timestamp on modification
CREATE OR REPLACE FUNCTION memory.update_modified_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_memory_timestamp
    BEFORE UPDATE ON memory.entries
    FOR EACH ROW
    EXECUTE FUNCTION memory.update_modified_timestamp();

-- ============================================================================
-- Helper Views
-- ============================================================================

-- Active, high-quality memories
CREATE OR REPLACE VIEW memory.high_quality_memories AS
SELECT
    id,
    memory_type,
    user_id,
    schema_group_id,
    original_context,
    learning_description,
    corrected_version,
    confidence_score,
    -- Quality score: combines confidence, success rate, and recency
    (
        confidence_score * 0.4 +
        (success_count::float / NULLIF(success_count + failure_count, 0)) * 0.4 +
        (1.0 - EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at)) / (86400.0 * 90)) * 0.2
    ) AS quality_score,
    success_count,
    failure_count,
    access_count,
    created_at,
    last_accessed_at
FROM memory.entries
WHERE is_active = true
ORDER BY quality_score DESC;

-- User-specific memory summary
CREATE OR REPLACE VIEW memory.user_memory_stats AS
SELECT
    user_id,
    memory_type,
    COUNT(*) as memory_count,
    AVG(confidence_score) as avg_confidence,
    SUM(success_count) as total_successes,
    SUM(failure_count) as total_failures,
    MAX(created_at) as last_memory_created
FROM memory.entries
WHERE is_active = true AND user_id IS NOT NULL
GROUP BY user_id, memory_type;

-- ============================================================================
-- Sample Data / Examples (for testing)
-- ============================================================================

-- Example: Global syntax correction
INSERT INTO memory.entries (
    memory_type,
    original_context,
    learning_description,
    corrected_version,
    metadata,
    source_type,
    confidence_score,
    is_validated
) VALUES (
    'syntax_correction',
    'select from trades where sym=`AAPL',
    'Table name should be singular "trade" not plural "trades" in KDB/Q schema',
    'select from trade where sym=`AAPL',
    '{"table": "trade", "common_mistake": "using plural form"}',
    'manual',
    0.95,
    true
) ON CONFLICT DO NOTHING;

-- Example: User-specific definition
INSERT INTO memory.entries (
    memory_type,
    user_id,
    original_context,
    learning_description,
    corrected_version,
    metadata,
    source_type,
    confidence_score
) VALUES (
    'user_definition',
    'user_demo@example.com',
    'Show me VWAP for AAPL',
    'User defines VWAP as: sum(price * volume) % sum(volume)',
    'select sym, sum(price*size)%sum size from trade where sym=`AAPL',
    '{"term": "VWAP", "formula": "sum(price*volume)%sum(volume)"}',
    'feedback',
    0.8
) ON CONFLICT DO NOTHING;

-- Example: Approach recommendation
INSERT INTO memory.entries (
    memory_type,
    original_context,
    learning_description,
    metadata,
    source_type,
    confidence_score
) VALUES (
    'approach_recommendation',
    'Queries with large time ranges and aggregations',
    'For date-range queries with aggregations, always filter by date first before calculating aggregates to improve performance',
    '{"optimization": "filter_before_aggregate", "applies_to": "time_series"}',
    'manual',
    0.9
) ON CONFLICT DO NOTHING;

-- ============================================================================
-- Maintenance Functions
-- ============================================================================

-- Function to apply temporal decay to confidence scores
CREATE OR REPLACE FUNCTION memory.apply_temporal_decay(decay_factor FLOAT DEFAULT 0.01)
RETURNS TABLE(updated_count BIGINT) AS $$
DECLARE
    count_updated BIGINT;
BEGIN
    UPDATE memory.entries
    SET confidence_score = GREATEST(
        confidence_score * (1 - decay_factor),
        0.1  -- Minimum floor
    )
    WHERE is_active = true
      AND confidence_score > 0.1
      AND last_accessed_at < CURRENT_TIMESTAMP - INTERVAL '30 days';

    GET DIAGNOSTICS count_updated = ROW_COUNT;
    RETURN QUERY SELECT count_updated;
END;
$$ LANGUAGE plpgsql;

-- Function to archive low-quality memories
CREATE OR REPLACE FUNCTION memory.archive_low_quality_memories(min_quality FLOAT DEFAULT 0.3)
RETURNS TABLE(archived_count BIGINT) AS $$
DECLARE
    count_archived BIGINT;
BEGIN
    UPDATE memory.entries
    SET is_active = false
    WHERE is_active = true
      AND created_at < CURRENT_TIMESTAMP - INTERVAL '90 days'
      AND (
          confidence_score < min_quality
          OR (failure_count > success_count * 2)
      );

    GET DIAGNOSTICS count_archived = ROW_COUNT;
    RETURN QUERY SELECT count_archived;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Grants (adjust based on your user setup)
-- ============================================================================
-- GRANT USAGE ON SCHEMA memory TO your_app_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA memory TO your_app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA memory TO your_app_user;

-- ============================================================================
-- Comments for Documentation
-- ============================================================================
COMMENT ON SCHEMA memory IS 'Long-term memory system for AI learning from user interactions';
COMMENT ON TABLE memory.entries IS 'Core memory storage: learnings extracted from user feedback and corrections';
COMMENT ON TABLE memory.tags IS 'Flexible tagging system for memory categorization';
COMMENT ON TABLE memory.usage_log IS 'Tracks when and how memories are used to measure effectiveness';
COMMENT ON COLUMN memory.entries.embedding IS 'Vector embedding for semantic similarity search';
COMMENT ON COLUMN memory.entries.confidence_score IS 'How confident we are in this memory (0-1)';
COMMENT ON COLUMN memory.entries.metadata IS 'Flexible JSONB field for memory-specific context';
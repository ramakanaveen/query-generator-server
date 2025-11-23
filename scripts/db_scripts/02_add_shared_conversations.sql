-- Migration: Add Shared Conversations Support
-- Description: Create table and indexes for conversation sharing feature
-- Date: 2025-01-15

-- Create shared_conversations table
CREATE TABLE IF NOT EXISTS shared_conversations (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(100) NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    share_token VARCHAR(100) UNIQUE NOT NULL,
    shared_by VARCHAR(100) NOT NULL,  -- User ID who created the share

    -- Access Control
    access_level VARCHAR(20) DEFAULT 'view' CHECK (access_level IN ('view', 'edit')),
    shared_with VARCHAR(100),  -- NULL = anyone with link, or specific user ID
    is_active BOOLEAN DEFAULT TRUE,

    -- Expiry
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,

    -- Analytics/Tracking
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP WITH TIME ZONE,

    -- Extensibility
    metadata JSONB DEFAULT '{}'
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS shared_conversations_token_idx ON shared_conversations(share_token);
CREATE INDEX IF NOT EXISTS shared_conversations_conversation_id_idx ON shared_conversations(conversation_id);
CREATE INDEX IF NOT EXISTS shared_conversations_shared_by_idx ON shared_conversations(shared_by);
CREATE INDEX IF NOT EXISTS shared_conversations_shared_with_idx ON shared_conversations(shared_with);
CREATE INDEX IF NOT EXISTS shared_conversations_is_active_idx ON shared_conversations(is_active);

-- Add comment for documentation
COMMENT ON TABLE shared_conversations IS 'Stores shareable links for conversations with access control';
COMMENT ON COLUMN shared_conversations.share_token IS 'Unique cryptographically secure token for accessing shared conversation';
COMMENT ON COLUMN shared_conversations.shared_with IS 'NULL for public links, user_id for user-specific shares';
COMMENT ON COLUMN shared_conversations.access_level IS 'view = read-only, edit = can add messages (future)';

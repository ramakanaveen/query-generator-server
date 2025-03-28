-- auth_tables.sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    role VARCHAR(20) CHECK (role IN ('user', 'sme', 'admin')) NOT NULL DEFAULT 'user',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    preferences JSONB DEFAULT '{}'
);

-- Add foreign key constraints for user references
ALTER TABLE schema_versions ADD CONSTRAINT fk_schema_versions_users
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL;

ALTER TABLE verified_queries ADD CONSTRAINT fk_verified_queries_users
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL;
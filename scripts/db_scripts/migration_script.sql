-- migration_script.sql
-- Script to migrate data from existing file-based schemas to new database structure

-- Create temporary table to store schemas during migration
CREATE TEMPORARY TABLE temp_schema_imports (
    name VARCHAR(100),
    content JSONB,
    imported BOOLEAN DEFAULT FALSE
);

-- Add any additional migration steps here
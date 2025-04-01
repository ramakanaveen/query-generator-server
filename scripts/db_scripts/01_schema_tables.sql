CREATE TABLE schema_groups (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- schema_definitions.sql

-- Base schema definition
CREATE TABLE schema_definitions (
    id SERIAL PRIMARY KEY,
    group_id INTEGER REFERENCES schema_groups(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(group_id, name)
);

-- Schema versioning
CREATE TABLE schema_versions (
    id SERIAL PRIMARY KEY,
    schema_id INTEGER REFERENCES schema_definitions(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    status VARCHAR(20) CHECK (status IN ('draft', 'active', 'deprecated')) NOT NULL DEFAULT 'draft',
    created_by INTEGER, -- Will reference users(id) when user table is created
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    activated_at TIMESTAMP WITH TIME ZONE,
    deprecated_at TIMESTAMP WITH TIME ZONE,
    notes TEXT,
    metadata JSONB DEFAULT '{}',
    UNIQUE (schema_id, version)
);

-- Table definitions with vector embeddings
CREATE TABLE table_definitions (
    id SERIAL PRIMARY KEY,
    schema_version_id INTEGER REFERENCES schema_versions(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    content JSONB NOT NULL,
    embedding vector(768), -- For text-embeddings-gecko
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (schema_version_id, name)
);

-- Index for vector similarity search
CREATE INDEX table_definitions_embedding_idx ON table_definitions USING ivfflat (embedding vector_cosine_ops);

-- Track active schema versions
CREATE TABLE active_schemas (
    schema_id INTEGER REFERENCES schema_definitions(id) ON DELETE CASCADE PRIMARY KEY,
    current_version_id INTEGER REFERENCES schema_versions(id) ON DELETE CASCADE NOT NULL,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Store example queries
CREATE TABLE schema_examples (
    id SERIAL PRIMARY KEY,
    schema_version_id INTEGER REFERENCES schema_versions(id) ON DELETE CASCADE,
    table_id INTEGER REFERENCES table_definitions(id) ON DELETE SET NULL, -- can be null for multi-table examples
    natural_language_query TEXT NOT NULL,
    generated_query TEXT NOT NULL,
    description TEXT,
    is_cross_schema BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'
);

-- For examples that span multiple tables
CREATE TABLE example_table_mappings (
    id SERIAL PRIMARY KEY,
    example_id INTEGER REFERENCES schema_examples(id) ON DELETE CASCADE,
    table_id INTEGER REFERENCES table_definitions(id) ON DELETE CASCADE,
    relevance_score FLOAT DEFAULT 1.0,
    UNIQUE(example_id, table_id)
);

-- Track relationships between tables
CREATE TABLE table_relationships (
    id SERIAL PRIMARY KEY,
    schema_version_id INTEGER REFERENCES schema_versions(id) ON DELETE CASCADE,
    source_table_id INTEGER REFERENCES table_definitions(id) ON DELETE CASCADE,
    target_table_id INTEGER REFERENCES table_definitions(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL,
    join_column VARCHAR(100),
    description TEXT,
    metadata JSONB DEFAULT '{}'
);

-- Track relationships between schemas
CREATE TABLE schema_relationships (
    id SERIAL PRIMARY KEY,
    source_schema_id INTEGER REFERENCES schema_definitions(id) ON DELETE CASCADE,
    target_schema_id INTEGER REFERENCES schema_definitions(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL,
    description TEXT,
    metadata JSONB DEFAULT '{}'
);

-- Store verified queries for few-shot learning
CREATE TABLE verified_queries (
    id SERIAL PRIMARY KEY,
    original_query_text TEXT NOT NULL,
    generated_query TEXT NOT NULL,
    tables_used JSONB NOT NULL, -- Array of table IDs used
    user_id INTEGER, -- Will reference users(id) when user table is created
    verification_source VARCHAR(50) CHECK (verification_source IN ('user_feedback', 'sme_verification', 'system')),
    verification_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    embedding vector(768),
    score FLOAT DEFAULT 1.0,
    usage_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'
);

-- Index for verified queries
CREATE INDEX verified_queries_embedding_idx ON verified_queries USING ivfflat (embedding vector_cosine_ops);
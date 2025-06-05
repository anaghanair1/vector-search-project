-- pgvector Documentation: https://github.com/pgvector/pgvector#querying
-- PostgreSQL Vector Operations: https://github.com/pgvector/pgvector#distance
-- Supabase Vector Guide: https://supabase.com/docs/guides/database/extensions/pgvector
-- PostgreSQL Function Creation: https://www.postgresql.org/docs/current/sql-createfunction.html
-- Cosine Distance Operator: https://github.com/pgvector/pgvector#l2-distance
-- RPC Function Patterns: https://supabase.com/docs/guides/database/functions
-- PostgreSQL Full-Text Search: https://www.postgresql.org/docs/current/textsearch.html

-- The main table
CREATE TABLE review_chunks (
    id BIGSERIAL PRIMARY KEY,
    review_id TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding vector(384),
    stars INTEGER NOT NULL,
    text_search_vector tsvector,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Basic search function
CREATE OR REPLACE FUNCTION search_similar_reviews(
    query_embedding vector(384),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id bigint,
    review_id text,
    chunk_text text,
    chunk_index int,
    stars int,
    similarity float
)
LANGUAGE sql
AS $$
    SELECT
        id,
        review_id,
        chunk_text,
        chunk_index,
        stars,
        1 - (embedding <=> query_embedding) AS similarity
    FROM review_chunks
    WHERE 1 - (embedding <=> query_embedding) > match_threshold
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;
-- Add text search index
CREATE INDEX idx_review_chunks_text_search 
ON review_chunks USING gin(text_search_vector);

-- Update existing data
UPDATE review_chunks 
SET text_search_vector = to_tsvector('english', chunk_text);
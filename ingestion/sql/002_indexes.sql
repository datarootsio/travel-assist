-- Indexes for hybrid retrieval.
--
-- Build these AFTER loading, not before: an HNSW index built incrementally over
-- 100k inserts is far slower than one built once over a populated table.

-- Sparse side: GIN over the generated tsvector.
CREATE INDEX IF NOT EXISTS chunks_tsv_idx
    ON chunks USING GIN (tsv);

-- Dense side: HNSW with cosine distance.
--
-- The operator class MUST match the operator used at query time. This index
-- serves `<=>` (cosine). Query with `<->` (L2) instead and Postgres silently
-- ignores the index and sequential-scans — correct results, catastrophic
-- latency, no error. That pairing is worth demonstrating live.
--
-- HNSW rather than DiskANN because it is portable: the same DDL works on
-- whatever Postgres the test suite runs against, with no extension beyond
-- `vector`. On Azure at larger scale, pg_diskann is the native option and is
-- worth a mention in the instructor notes.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_document_idx
    ON chunks (document_id);

-- Schema for the Wikivoyage index.
--
-- Applied by an instructor once against the shared Azure Postgres. Students
-- read this file; they do not run it.
--
-- Requires the psql variable :embedding_dim, e.g.
--   psql -v embedding_dim=3072 -f 001_schema.sql
--
-- On Azure Database for PostgreSQL Flexible Server, `vector` must first be
-- allow-listed via the `azure.extensions` server parameter.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per Wikivoyage article. Every chunk carries its source URL through to
-- the answer, because the corpus is CC BY-SA and attribution is part of the
-- product, not an afterthought.
CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    page_id       INTEGER NOT NULL UNIQUE,
    title         TEXT NOT NULL,
    url           TEXT NOT NULL,
    country       TEXT,
    continent     TEXT,
    article_type  TEXT,
    status        TEXT,
    lat           DOUBLE PRECISION,
    lon           DOUBLE PRECISION,
    revision_id   BIGINT,
    retrieved_at  TIMESTAMPTZ NOT NULL
);

-- One row per retrievable passage.
--
-- `tsv` is a GENERATED column rather than something ingestion maintains: keyword
-- search and dense search then provably see the same text, which is what makes
-- the day-1 bake-off a fair comparison.
CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL PRIMARY KEY,
    document_id   BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    section_path  TEXT,
    content       TEXT NOT NULL,
    token_count   INTEGER,
    embedding     VECTOR(:embedding_dim) NOT NULL,
    tsv           TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- The frozen-model contract. src/travel_assist/db.py reads this at startup and
-- refuses to run if the embedding model or width disagrees with its settings.
-- Expected keys: embedding_model, embedding_dim, dump_date, chunk_strategy,
-- built_at, n_docs, n_chunks.
CREATE TABLE IF NOT EXISTS index_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

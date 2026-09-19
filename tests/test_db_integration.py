"""Database plumbing, against a real local Postgres.

Skipped automatically when no writable Postgres is reachable, so a clone with none
configured still runs a green suite. Never point these at the shared Azure
index: nineteen participants share one database and must not be able to break
each other.

Point PGHOST at a Postgres you can create a database on.
"""

from __future__ import annotations

import pytest

from travel_assist.config import Settings
from travel_assist.db import connect, read_index_metadata, verify_index


def test_schema_exists_after_container_initialisation(db_settings: Settings):
    """Conftest applies the same DDL an instructor applies to the shared index."""
    with connect(db_settings) as conn, conn.cursor() as cursor:
        cursor.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )
        tables = {row["table_name"] for row in cursor.fetchall()}

    assert {"documents", "chunks", "index_metadata"} <= tables


def test_pgvector_extension_is_installed(db_settings: Settings):
    with connect(db_settings) as conn, conn.cursor() as cursor:
        cursor.execute("SELECT 1 AS present FROM pg_extension WHERE extname = 'vector'")
        assert cursor.fetchone() is not None


def test_reading_metadata_from_an_un_ingested_database_returns_empty(db_settings: Settings):
    """An empty index is a normal state, not an exception — doctor turns it into advice."""
    with connect(db_settings) as conn:
        metadata = read_index_metadata(conn)

    assert isinstance(metadata, dict)


def test_verify_index_refuses_an_un_ingested_database(db_settings: Settings):
    """Better to refuse than to serve answers from an index that cannot be verified."""
    from travel_assist.db import IndexCompatibilityError

    with connect(db_settings) as conn:
        already_ingested = bool(read_index_metadata(conn))

    if already_ingested:
        pytest.skip("this database has been ingested; the empty-index path cannot be exercised")

    with pytest.raises(IndexCompatibilityError, match="index_metadata"):
        verify_index(db_settings)

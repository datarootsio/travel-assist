"""Postgres access, and the contract that keeps retrieval honest.

Scaffolding — ships implemented. Students read this file; they do not write it.

Two responsibilities:

1. **Connections.** A small pool over `pgvector`-enabled Postgres, pointed at
   whatever `PGHOST` names — the shared Azure index.
2. **The frozen-embedding guard.** An index is built once, with one embedding
   model, at one width. Query it with anything else and you do not get an
   error — you get results, drawn from a different vector space, that look
   plausible and are wrong. This module refuses that at startup instead.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from travel_assist.config import Settings, get_settings

logger = logging.getLogger(__name__)

INDEX_METADATA_TABLE = "index_metadata"


class IndexCompatibilityError(RuntimeError):
    """The configured embedding setup cannot safely query the configured index.

    Raised at startup rather than allowing a run that would return confident,
    subtly wrong results.
    """


def check_embedding_contract(settings: Settings, metadata: Mapping[str, str]) -> None:
    """Assert the configured embedding model matches the one that built the index.

    Args:
        settings: The configuration this process is running with.
        metadata: Rows from the index's `index_metadata` table, as key/value
            strings. Must contain `embedding_model` and `embedding_dim`.

    Raises:
        IndexCompatibilityError: If the metadata is missing, or if either the
            model name or the vector width disagrees with `settings`.
    """
    if not metadata:
        raise IndexCompatibilityError(
            f"The index has no {INDEX_METADATA_TABLE} rows, so its embedding model cannot\n"
            "be verified. An unverifiable index is not trusted: it may have been built with\n"
            "a different model, which degrades retrieval silently.\n"
            "Re-run ingestion, or point PGHOST at an index built by this project."
        )

    expected_model = settings.embedding_model_name
    index_model = metadata.get("embedding_model", "<absent>")
    index_dim = metadata.get("embedding_dim", "<absent>")

    if index_model != expected_model:
        raise IndexCompatibilityError(
            f"Embedding model mismatch: this process would query with {expected_model!r}, "
            f"but the index was built with {index_model!r}.\n"
            "Queries embedded by a different model land in a different vector space, so "
            "retrieval degrades silently rather than failing.\n"
            "Set AZURE_EMBEDDING_DEPLOYMENT to match the index, or rebuild the index."
        )

    if index_dim != str(settings.embedding_dim):
        raise IndexCompatibilityError(
            f"Embedding dimension mismatch: EMBEDDING_DIM={settings.embedding_dim}, "
            f"but the index stores {index_dim}-dimensional vectors.\n"
            "This one usually surfaces as an operator error from pgvector, but a width that "
            "happens to line up would degrade retrieval silently — so it is checked here.\n"
            "Set EMBEDDING_DIM to match the index."
        )


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------


def dsn(settings: Settings | None = None) -> str:
    """Build the libpq connection string for the configured index backend.

    `connect_timeout` is set deliberately low. In a classroom, a command that
    hangs is worse than one that fails: nobody can tell a slow database from a
    wrong hostname, and the instinct is to wait rather than to read the error.
    """
    settings = settings or get_settings()
    return (
        f"host={settings.pghost} port={settings.pgport} dbname={settings.pgdatabase} "
        f"user={settings.pguser} password={settings.pgpassword} sslmode={settings.pgsslmode} "
        f"connect_timeout={settings.pgconnect_timeout}"
    )


def connect(settings: Settings | None = None) -> Connection[Any]:
    """Open a single short-lived connection, bypassing the pool.

    Used by `doctor`, which must diagnose an unreachable database quickly rather
    than join the pool's background reconnect loop and appear to hang.
    """
    settings = settings or get_settings()
    return Connection.connect(dsn(settings), row_factory=dict_row)


_pool: ConnectionPool[Connection[Any]] | None = None


def get_pool(settings: Settings | None = None) -> ConnectionPool[Connection[Any]]:
    """Return the process-wide connection pool, opening it on first use.

    Kept small on purpose: nineteen participants share one read-only role on the
    Azure index, and a generous per-process pool multiplied by nineteen is how
    one room exhausts a database.
    """
    global _pool
    if _pool is None:
        settings = settings or get_settings()
        _pool = ConnectionPool(
            dsn(settings),
            min_size=1,
            max_size=4,
            open=True,
            timeout=settings.pgconnect_timeout,
            kwargs={"row_factory": dict_row},
        )
    return _pool


def close_pool() -> None:
    """Close the pool. For tests and clean shutdown."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def connection(settings: Settings | None = None) -> Iterator[Connection[Any]]:
    """Yield a pooled connection.

    Yields:
        An open psycopg connection with `dict_row` results.
    """
    with get_pool(settings).connection() as conn:
        yield conn


def list_corpus_destinations(settings: Settings | None = None) -> list[str]:
    """Every document title currently in the index, alphabetically.

    Queried fresh rather than hardcoded, so a router built from this list can
    never drift from what the index actually holds — the same reasoning
    `check_embedding_contract` applies to the embedding model applies here to
    the corpus's contents.

    Returns:
        Every distinct `documents.title`. Empty on an un-ingested index.
    """
    settings = settings or get_settings()
    with connection(settings) as conn, conn.cursor() as cursor:
        cursor.execute("SELECT DISTINCT title FROM documents ORDER BY title")
        return [row["title"] for row in cursor.fetchall()]


def read_index_metadata(conn: Connection[Any]) -> dict[str, str]:
    """Read `index_metadata` as a plain key/value mapping.

    Returns:
        Every row as `{key: value}`, or an empty dict if the table does not
        exist yet — an un-ingested database is a normal state, and the guard
        turns it into a readable error rather than a psycopg exception.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT to_regclass(%s) AS present",
            (f"public.{INDEX_METADATA_TABLE}",),
        )
        row = cursor.fetchone()
        if row is None or row["present"] is None:
            return {}

        cursor.execute(f"SELECT key, value FROM {INDEX_METADATA_TABLE}")  # noqa: S608
        return {record["key"]: record["value"] for record in cursor.fetchall()}


def verify_index(settings: Settings | None = None) -> dict[str, str]:
    """Run every startup check against the configured index.

    Call this once at application startup. It is deliberately fail-fast: a
    process that cannot prove its embeddings match the index it is about to
    query should not serve answers.

    Returns:
        The index metadata, so callers can log the dump date and corpus size.

    Raises:
        IndexCompatibilityError: On any incompatibility.
    """
    settings = settings or get_settings()

    with connection(settings) as conn:
        metadata = read_index_metadata(conn)

    check_embedding_contract(settings, metadata)

    logger.info(
        "index verified: %s chunks, embedding model %s, dump %s",
        metadata.get("n_chunks", "?"),
        metadata.get("embedding_model", "?"),
        metadata.get("dump_date", "?"),
    )
    return metadata

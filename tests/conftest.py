"""Shared test fixtures.

The important one is `db_settings`: database-backed tests create and use their
own `travel_assist_test` database, never the one you have been working in.

That is not fastidiousness. These tests truncate the tables they use, so pointing
them at a working index silently destroys the corpus you just spent quota
building. **Never point them at the shared Azure index** — nineteen participants
read from it, it is meant to be read-only, and `TRUNCATE` does not care.

There is no local database shipped with this repo and no Docker. So these tests
need *a* Postgres you can create a database on — any one will do.

**Every test that depends on `db_settings`/`clean_db` (directly or through a
fixture like `corpus`/`wired`) is auto-marked `@pytest.mark.db`** by the hook
below — no per-test decorator needed. `db`, like `live`, is excluded from the
default `make test` run: recreating the test database and re-applying its
schema on every single invocation was real, avoidable friction during ordinary
exercise iteration, for a check that only needs to happen once per
environment. Run it deliberately with `make test-db` — that is the "does my
Postgres connection work" check, run once, not on every test loop.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest
from psycopg.rows import dict_row

from travel_assist.config import Settings
from travel_assist.db import close_pool, dsn

TEST_DATABASE = "travel_assist_test"
SCHEMA_FILES = (
    Path("ingestion/sql/001_schema.sql"),
    Path("ingestion/sql/002_indexes.sql"),
)
# Must match the width the schema is created with.
TEST_EMBEDDING_DIM = 1536
# Named so `index_metadata` tells the truth about what built the test index.
# Recording a real deployment name over vectors from a test double is precisely
# the mismatch the frozen-embedding guard exists to catch.
TEST_EMBEDDING_MODEL = "test-embeddings"

_DB_FIXTURES = frozenset({"db_settings", "clean_db"})


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-apply `@pytest.mark.db` to every test that touches the local test Postgres.

    `item.fixturenames` is the full transitive fixture closure, so this also
    catches a test requesting `corpus` or `wired` (which themselves request
    `clean_db`) without those fixtures needing to know about the marker.
    """
    for item in items:
        if _DB_FIXTURES.intersection(item.fixturenames):
            item.add_marker(pytest.mark.db)


def _test_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "embedding_dim": TEST_EMBEDDING_DIM,
        "azure_embedding_deployment": TEST_EMBEDDING_MODEL,
        "pgdatabase": TEST_DATABASE,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def _create_test_database(admin: Settings) -> None:
    """Create `travel_assist_test` if it is missing, then apply the schema."""
    with psycopg.connect(dsn(admin), autocommit=True, row_factory=dict_row) as conn:
        exists = conn.execute(
            "SELECT 1 AS present FROM pg_database WHERE datname = %s", (TEST_DATABASE,)
        ).fetchone()
        if exists is None:
            conn.execute(f'CREATE DATABASE "{TEST_DATABASE}"')

    with psycopg.connect(dsn(_test_settings()), autocommit=True) as conn:
        for path in SCHEMA_FILES:
            sql = path.read_text(encoding="utf-8").replace(
                ":embedding_dim", str(TEST_EMBEDDING_DIM)
            )
            conn.execute(sql)  # type: ignore[arg-type]


@pytest.fixture(scope="session")
def db_settings() -> Settings:
    """Settings pointing at a dedicated test database, or skip if none is reachable."""
    settings = _test_settings()

    try:
        _create_test_database(_test_settings(pgdatabase="postgres"))
    except psycopg.Error as exc:
        pytest.skip(
            f"no writable Postgres ({exc.__class__.__name__}); point PGHOST/PGUSER/PGPASSWORD "
            "at one you can create a database on — never the shared index"
        )

    return settings


@pytest.fixture(autouse=True)
def _reset_connection_pool() -> None:
    """Undo `db.get_pool`'s process-wide cache between tests.

    `get_pool` opens once per process and ignores `settings` on every later
    call — fine in production, where one process only ever runs with one DSN.
    In a test session it means whichever test opens the pool first decides
    which database every later test silently queries, `settings=` or not.
    Found running the full suite with `-m ""` for real: a live test opening
    the shared Azure pool first made a later `clean_db` test query Azure's
    `index_metadata` instead of its own freshly seeded one, and fail with a
    real-looking `IndexCompatibilityError` that had nothing to do with the
    thing it was testing.
    """
    yield
    close_pool()


@pytest.fixture
def clean_db(db_settings: Settings) -> Settings:
    """An empty index. Safe, because this is never the development database."""
    with psycopg.connect(dsn(db_settings), row_factory=dict_row) as conn:
        conn.execute("TRUNCATE documents, chunks, index_metadata RESTART IDENTITY CASCADE")
        conn.commit()
    return db_settings

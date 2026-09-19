"""`travel-assist search` — the side-by-side demo that carries the lesson.

It is a demo command that prints a table, so most of what it does is checked by
looking at it. Three things are not safe to check by eye: that both columns
render at all, that an empty column is *narrated* rather than left blank, and
that the command refuses an index some other model built.
"""

from __future__ import annotations

import psycopg
import pytest
from typer.testing import CliRunner

from travel_assist.cli.main import app
from travel_assist.config import Settings
from travel_assist.db import dsn
from travel_assist.retrieval.models import RetrievedChunk
from travel_assist.retrieval.search import vector_search

from .conftest import TEST_EMBEDDING_MODEL
from .retrieval_corpus import QUERY_VECTORS, ScriptedEmbeddings, seed_corpus

runner = CliRunner()


def flat(text: str) -> str:
    """Collapse Rich's line wrapping, which breaks sentences at the terminal width."""
    return " ".join(text.split())


@pytest.fixture
def wired(clean_db: Settings, monkeypatch) -> Settings:
    """Point the CLI at the fixture corpus and the scripted embeddings."""
    seed_corpus(clean_db)
    monkeypatch.setattr("travel_assist.cli.search.get_settings", lambda: clean_db)

    def scripted(query: str, k: int = 8, **kwargs: object) -> list[RetrievedChunk]:
        """The real search, with the fixture's embeddings substituted for the provider."""
        if query not in QUERY_VECTORS:
            return []
        return vector_search(query, k=k, settings=clean_db, embeddings=ScriptedEmbeddings())

    monkeypatch.setattr("travel_assist.cli.search.vector_search", scripted)
    return clean_db


def test_both_methods_are_shown_side_by_side(wired: Settings):
    result = runner.invoke(app, ["search", "where can I eat", "-k", "3"])

    assert result.exit_code == 0
    assert "vector (dense)" in flat(result.stdout)
    assert "keyword (sparse)" in flat(result.stdout)


def test_an_empty_keyword_column_is_explained(wired: Settings):
    """Otherwise a blank column reads as a bug in the command, not a finding.

    The narration *is* the demo. Both of this milestone's failures — the empty
    column here, and a vector column full of plausible-but-wrong passages — look
    like ordinary output unless the command says what just happened.
    """
    result = runner.invoke(app, ["search", "somewhere to swim outdoors"])

    assert result.exit_code == 0
    assert "nothing" in flat(result.stdout).lower()
    assert "cannot match a word that is not there" in flat(result.stdout)


def test_querying_an_index_built_by_a_different_model_is_refused(clean_db: Settings, monkeypatch):
    """The failure this whole repo is most careful about, reachable from the CLI.

    Seed an index whose metadata says one embedding model, then query it
    configured for another. Without a check the search *succeeds* and returns
    confidently ranked nonsense: the two sets of vectors occupy different spaces,
    and cosine distance has an opinion about any two vectors you hand it.

    Found by running `travel-assist search` with a model that had not built the
    index and getting a plausible-looking answer back — which is why the check
    lives on the query path and not only in `doctor`.
    """
    seed_corpus(clean_db)

    with psycopg.connect(dsn(clean_db)) as conn:
        conn.execute(
            "UPDATE index_metadata SET value = 'text-embedding-3-small' "
            "WHERE key = 'embedding_model'"
        )
        conn.commit()

    monkeypatch.setattr("travel_assist.cli.search.get_settings", lambda: clean_db)

    result = runner.invoke(app, ["search", "where can I eat"])

    assert result.exit_code != 0
    output = flat(result.stdout)
    assert "text-embedding-3-small" in output
    assert TEST_EMBEDDING_MODEL in output

"""The frozen-embedding contract: crash loudly rather than retrieve quietly wrong.

The index is pre-built once by an instructor and shared read-only by the whole
room. A query embedded by a different model, or at a different width, lands in a
different vector space — so retrieval keeps returning results, they are just
subtly wrong, and *nothing errors*. That failure is far more expensive than a
crash at startup, which is why this guard exists and why its message has to
explain itself rather than just refuse.

Two tests, because the guard has two jobs: refuse, and teach. The rules are pure
functions over a metadata mapping, so no database is needed here; reading that
mapping out of Postgres is plumbing, covered by the integration test, and the
same refusal reached from the CLI is covered by `test_cli_search.py`.
"""

from __future__ import annotations

import pytest

from travel_assist.config import Settings
from travel_assist.db import IndexCompatibilityError, check_embedding_contract

INDEX_BUILT_BY = "text-embedding-3-large"
QUERIED_WITH = "text-embedding-3-small"

METADATA = {
    "embedding_model": INDEX_BUILT_BY,
    "embedding_dim": "3072",
    "dump_date": "2026-08-01",
    "chunk_strategy": "section-packed-400-65",
}


def settings_for(embedding_deployment: str) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        model_backend="azure",
        azure_openai_api_key="sk-not-a-real-key",
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_embedding_deployment=embedding_deployment,
        embedding_dim=3072,
    )


def test_a_matching_model_passes_and_a_different_one_is_refused():
    check_embedding_contract(settings_for(INDEX_BUILT_BY), METADATA)

    with pytest.raises(IndexCompatibilityError) as excinfo:
        check_embedding_contract(settings_for(QUERIED_WITH), METADATA)

    message = str(excinfo.value)
    assert QUERIED_WITH in message, "the error must name the model that asked"
    assert INDEX_BUILT_BY in message, "and the model that built the index"


def test_the_error_explains_the_silent_failure():
    """A participant who hits this must learn something, not just be blocked."""
    with pytest.raises(IndexCompatibilityError) as excinfo:
        check_embedding_contract(settings_for(QUERIED_WITH), METADATA)

    message = str(excinfo.value).lower()

    assert "silent" in message or "silently" in message

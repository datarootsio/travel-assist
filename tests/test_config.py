"""Settings load, and name exactly one place chat and embeddings come from.

There is no fallback backend. `MODEL_BACKEND` names the one provider, and a
missing credential fails where a client is built (`tests/test_models.py`), not
here.

That split is the behaviour worth pinning. **Constructing Settings never
raises**, whatever is absent from the environment: `doctor` builds Settings in
order to *diagnose* a missing key, and most of the suite builds Settings without
going near a model. A validator that refused would make the one command that
explains a broken environment the first thing to break in it.
"""

from __future__ import annotations

import pytest

from travel_assist.config import ModelBackend, Settings


def make_settings(**overrides: object) -> Settings:
    """Build Settings ignoring any developer's local .env, so tests are hermetic."""
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


DOCUMENTED_DEFAULTS = (
    "MODEL_BACKEND",
    "PGHOST",
    "CONTEXT_TOKEN_BUDGET",
    "HISTORY_TOKEN_BUDGET",
    "COMPACTION_STRATEGY",
    "PROMPT_LABEL",
)


def test_defaults_match_the_documented_environment(monkeypatch):
    """`.env.example` and this list must not drift apart.

    Hermetic on purpose. `_env_file=None` ignores the `.env` file but Settings
    still reads the *process* environment, so a developer with PGHOST exported —
    or a test runner setting COMPACTION_STRATEGY — would turn this red for a
    reason that has nothing to do with defaults. That exact fragility had three
    of the six legs of the old CI matrix failing, unnoticed, for weeks.
    """
    for name in DOCUMENTED_DEFAULTS:
        monkeypatch.delenv(name, raising=False)

    settings = make_settings()

    assert settings.model_backend == "azure"
    assert settings.pghost == "localhost"
    assert settings.context_token_budget == 8000
    assert settings.history_token_budget == 2000
    assert settings.compaction_strategy == "summarise"
    assert settings.mlflow_tracking_uri == "sqlite:///mlflow.db"
    assert settings.mlflow_experiment == "travel-assist"


def test_embedding_dim_fits_pgvectors_hnsw_limit():
    """An HNSW index cannot be built above 2000 dimensions in pgvector.

    `text-embedding-3-large` is natively 3072, which is why the default here is
    the truncated 1536 rather than the native width. Raise this above 2000 and
    the schema fails at `CREATE INDEX`, not at query time — so it fails on the
    instructor's ingestion run, which is the only place the index is built.
    """
    assert make_settings().embedding_dim <= 2000


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("CONTEXT_TOKEN_BUDGET", "1234")
    monkeypatch.setenv("COMPACTION_STRATEGY", "drop_oldest")

    settings = make_settings()

    assert settings.context_token_budget == 1234
    assert settings.compaction_strategy == "drop_oldest"


def test_settings_construct_even_with_no_credentials_at_all():
    """`doctor` has to build Settings before it can tell you the key is missing."""
    settings = make_settings(azure_openai_api_key=None, azure_openai_endpoint=None)

    assert settings.model_backend is ModelBackend.AZURE


def test_the_backend_line_names_the_model_and_the_index_that_answered():
    """Printed on every run, so nobody has to guess which combination replied.

    The index half is the host and database, not a category. Two people getting
    different answers to the same question are almost always pointed at
    different Postgres instances, and this line is what tells them so.
    """
    line = make_settings(
        azure_chat_deployment="gpt-5.4-mini", pghost="shared.postgres.database.azure.com"
    ).describe_backends()

    assert "azure" in line
    assert "gpt-5.4-mini" in line
    assert "shared.postgres.database.azure.com/travel_assist" in line


def test_the_embedding_model_name_is_the_azure_deployment():
    """This string is written into `index_metadata` and checked on every startup."""
    settings = make_settings(azure_embedding_deployment="text-embedding-3-large")

    assert settings.embedding_model_name == "text-embedding-3-large"


def test_an_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        make_settings(model_backend="bedrock")


@pytest.mark.parametrize(
    ("pasted", "expected"),
    [
        # What the Foundry portal actually shows you.
        (
            "https://af-rootsacademy-26-genai.services.ai.azure.com/openai/v1",
            "https://af-rootsacademy-26-genai.services.ai.azure.com",
        ),
        # Same, with the trailing slash people copy along with it.
        (
            "https://af-rootsacademy-26-genai.services.ai.azure.com/openai/v1/",
            "https://af-rootsacademy-26-genai.services.ai.azure.com",
        ),
        # The older form some docs still show.
        ("https://example.openai.azure.com/openai", "https://example.openai.azure.com"),
        # Already correct — must survive untouched.
        ("https://example.openai.azure.com", "https://example.openai.azure.com"),
        ("https://example.openai.azure.com/", "https://example.openai.azure.com"),
        # A template copied but never filled in is unset, not a malformed URL.
        ("", None),
        ("   ", None),
    ],
    ids=["v1", "v1-slash", "openai-suffix", "bare", "bare-slash", "empty", "whitespace"],
)
def test_endpoint_is_reduced_to_a_base_url(pasted: str, expected: str | None):
    """The Azure classes append their own `/openai/deployments/...` path.

    Leaving the portal's `/openai/v1` on the end produces a 404 that looks like
    a wrong deployment name — the single most expensive hour available on the
    morning of day 1.
    """
    assert make_settings(azure_openai_endpoint=pasted).azure_openai_endpoint == expected


def test_the_foundry_project_endpoint_is_reduced_to_the_resource():
    """The portal offers three URLs for one resource; only the base one works.

    The project form is the expensive one: the extra path makes the service
    answer 400 "API version not supported", which points at the api-version
    rather than at the endpoint.
    """
    project = "https://af-rootsacademy-26-genai.services.ai.azure.com/api/projects/proj-default"
    assert (
        make_settings(azure_openai_endpoint=project).azure_openai_endpoint
        == "https://af-rootsacademy-26-genai.services.ai.azure.com"
    )

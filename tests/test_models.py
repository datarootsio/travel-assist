"""The model factory — the one place a chat or embedding client is constructed.

There is no mock backend (cleanup D3). What replaces it is not a second
implementation but an *error*: if the credentials for the configured backend are
absent, the factory says so, names the variables, and stops.

That is the behaviour these tests exist to pin. A missing key used to degrade to
canned text; now it has to produce a sentence a participant can act on in the ten
seconds before they put their hand up. `doctor` turns the same error into its
`model` row, so getting the wording right here fixes it in both places.

Constructing an Azure client does not call Azure, so routing and error handling
are both testable with a fake key. Anything that would need a *live* endpoint —
vector width, unit length — is skipped until the credentials land, and named in
the skip reason so the handover has a checklist.
"""

from __future__ import annotations

import math

import pytest
from langchain_core.language_models import BaseChatModel

from travel_assist.config import Settings
from travel_assist.models import MissingCredentialsError, get_chat_model, get_embeddings

NEEDS_AZURE = "needs real Azure credentials — run at the credential handover"


def azure_settings(**overrides: object) -> Settings:
    """Settings with a syntactically valid but fake Azure credential pair."""
    base: dict[str, object] = {
        "model_backend": "azure",
        "azure_openai_api_key": "sk-not-a-real-key",
        "azure_openai_endpoint": "https://example.openai.azure.com",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


def test_the_factory_builds_an_azure_chat_model():
    """We cannot call Azure in CI, but we can prove the factory routes to it."""
    model = get_chat_model(settings=azure_settings())

    assert isinstance(model, BaseChatModel)
    assert "Azure" in type(model).__name__


def test_the_factory_builds_azure_embeddings():
    embeddings = get_embeddings(settings=azure_settings())

    assert "Azure" in type(embeddings).__name__


@pytest.mark.parametrize("overrides", [{"temperature": 0.5}, {"max_tokens": 128}])
def test_overrides_are_accepted(overrides: dict[str, object]):
    """`get_chat_model(**overrides)` keeps 'swap the model' a one-line change."""
    model = get_chat_model(settings=azure_settings(), **overrides)

    assert isinstance(model, BaseChatModel)


# --------------------------------------------------------------------------
# Missing credentials
# --------------------------------------------------------------------------


def test_the_error_names_every_missing_variable_and_ends_in_an_action():
    """A participant reads this at 09:40, so it has to be complete and actionable.

    Silence here becomes a 401 from deep inside httpx twenty minutes later, and
    naming one missing variable at a time is a bad experience when both are
    absent. `doctor` renders the same error as its `model` row, so this wording
    is fixed in two places at once.
    """
    with pytest.raises(MissingCredentialsError) as excinfo:
        get_chat_model(
            settings=azure_settings(azure_openai_api_key=None, azure_openai_endpoint=None)
        )

    message = str(excinfo.value)
    assert "AZURE_OPENAI_API_KEY" in message
    assert "AZURE_OPENAI_ENDPOINT" in message
    assert ".env" in message


def test_embeddings_fail_the_same_way_as_chat():
    """Retrieval embeds the query, so this is the error most people hit first."""
    with pytest.raises(MissingCredentialsError) as excinfo:
        get_embeddings(settings=azure_settings(azure_openai_api_key=None))

    assert "AZURE_OPENAI_API_KEY" in str(excinfo.value)


# --------------------------------------------------------------------------
# Live checks — the credential-handover smoke tests
# --------------------------------------------------------------------------


@pytest.mark.skip(reason=NEEDS_AZURE)
def test_embeddings_have_the_configured_dimension():
    """`EMBEDDING_DIM` is passed to Azure as `dimensions`; verify it is obeyed.

    `text-embedding-3-large` is natively 3072 and truncated to 1536 by that
    parameter. If Azure ignored it the index would build at the wrong width and
    fail at `CREATE INDEX` — see `test_config.py`.
    """
    settings = azure_settings(embedding_dim=1536)

    vector = get_embeddings(settings=settings).embed_query("Lisbon")

    assert len(vector) == 1536


@pytest.mark.skip(reason=NEEDS_AZURE)
def test_embeddings_are_unit_length():
    """Cosine similarity over non-normalised vectors is a subtle, silent bug."""
    vector = get_embeddings(settings=azure_settings()).embed_query("Lisbon")

    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-6)

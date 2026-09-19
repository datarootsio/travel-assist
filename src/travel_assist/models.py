"""The only place a model client is constructed.

Scaffolding — ships implemented. Everything in the application reaches chat
completions and embeddings through `get_chat_model()` and `get_embeddings()`.
Nothing else builds a client. That keeps "swap the model" a one-line change and
means the credential check happens once rather than in every module that
remembers to do it.

**There is no canned backend** (cleanup D3). An earlier version of this file
carried one, and a missing key degraded the room to hash-derived vectors and
templated prose. That was a worse deal than it looks: canned output teaches
plumbing and nothing about answer quality, and anyone still on it an hour into a
teaching day has been quietly wasting their morning.

So a missing credential raises `MissingCredentialsError` here, naming the
variables and what to do about them. `doctor` renders the same error as its
`model` row, so this message is the one a participant actually reads.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from travel_assist.config import Settings, get_settings

logger = logging.getLogger(__name__)


class MissingCredentialsError(RuntimeError):
    """The configured model backend has nothing to authenticate with.

    Raised when a client is built, not when Settings are read: `doctor` has to
    be able to load the configuration in order to report what is absent from it.
    """


def _require_credentials(settings: Settings) -> None:
    """Refuse to build a client that cannot possibly authenticate.

    Args:
        settings: The configuration this process is running with.

    Raises:
        MissingCredentialsError: If any variable the configured backend needs is
            unset. Every missing variable is named in one message — reporting
            them one at a time turns a single fix into three runs.
    """
    required = (
        ("AZURE_OPENAI_API_KEY", settings.azure_openai_api_key),
        ("AZURE_OPENAI_ENDPOINT", settings.azure_openai_endpoint),
    )

    missing = [name for name, value in required if not value]
    if not missing:
        return

    verb = "is" if len(missing) == 1 else "are"
    raise MissingCredentialsError(
        f"MODEL_BACKEND={settings.model_backend.value} needs "
        f"{' and '.join(missing)}, which {verb} not set.\n"
        "Copy .env.example to .env and fill them in from the credentials handed out at "
        "the start of the day, then run `travel-assist doctor` to confirm.\n"
        "There is no offline mode: retrieval embeds your query, so it needs a real endpoint."
    )


def _log_429(response: httpx.Response) -> None:
    """Make rate limiting visible rather than a mysterious pause.

    Nineteen people against one Foundry deployment on two consecutive days is the
    most likely thing to derail the course, so the retry is deliberately not
    silent: the room should be able to see it happening and understand why.
    """
    if response.status_code == 429:
        retry_after = response.headers.get("retry-after", "unspecified")
        logger.warning(
            "rate limited by Azure (429); retrying with backoff, retry-after=%s", retry_after
        )


def _azure_http_client() -> httpx.Client:
    return httpx.Client(event_hooks={"response": [_log_429]})


def get_chat_model(*, settings: Settings | None = None, **overrides: Any) -> BaseChatModel:
    """Return the configured chat model.

    Args:
        settings: Configuration to use. Defaults to the process-wide Settings.
        **overrides: Passed through to the underlying client — `temperature`,
            `max_tokens`, and so on. This is the one-line hook for swapping
            model behaviour without touching call sites.

    Returns:
        A chat model for `MODEL_BACKEND`.

    Raises:
        MissingCredentialsError: If that backend's credentials are not set.
    """
    settings = settings or get_settings()
    _require_credentials(settings)

    # Canonical field names, not the friendlier aliases (`api_key`,
    # `azure_deployment`). The aliases work at runtime but the two Azure classes
    # expose them inconsistently, and `mypy --strict` only accepts these.
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        openai_api_key=settings.azure_openai_api_key,
        openai_api_version=settings.azure_openai_api_version,
        deployment_name=settings.azure_chat_deployment,
        max_retries=settings.model_max_retries,
        http_client=_azure_http_client(),
        **overrides,
    )


def get_embeddings(*, settings: Settings | None = None) -> Embeddings:
    """Return the configured embedding model.

    The embedding model is FROZEN: it must match whatever built the index being
    queried. `EMBEDDING_DIM` and the deployment name are checked against the
    index's own metadata at startup, because a mismatch degrades retrieval
    silently instead of failing.

    Returns:
        An embedding client for `MODEL_BACKEND`, at `EMBEDDING_DIM` width.

    Raises:
        MissingCredentialsError: If that backend's credentials are not set.
    """
    settings = settings or get_settings()
    _require_credentials(settings)

    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_endpoint,
        openai_api_key=settings.azure_openai_api_key,
        openai_api_version=settings.azure_openai_api_version,
        deployment=settings.azure_embedding_deployment,
        dimensions=settings.embedding_dim,
        max_retries=settings.model_max_retries,
        http_client=_azure_http_client(),
    )

"""Every setting the application reads, in one place.

Scaffolding — ships implemented. Students read this file and run against it; they
do not write it.

Two settings deserve attention because they are **independent**, and conflating
them is the single most confusing thing about running this project:

- `MODEL_BACKEND` decides where chat completions and embeddings come from.
- `PGHOST` and friends decide which Postgres — and so which corpus — is searched.

Running a model against an index some *other* model built is not safe: the two
sets of vectors live in different spaces, so retrieval returns plausible-looking
garbage *silently*. The frozen-embedding guard in `db.py` refuses that at startup.

**Constructing Settings never fails on a missing credential.** It is the job of
`models.py` to refuse when a client is actually built, and of `doctor` to explain
it — and `doctor` cannot explain a missing key if reading the configuration is
itself what breaks.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelBackend(StrEnum):
    """Where chat completions and embeddings come from.

    One member, deliberately. There is no canned backend and no second provider:
    a missing key is an error with a remedy, not a quiet downgrade to output that
    teaches nothing. It stays an enum rather than a bare string so that a stale
    `.env` naming a provider we removed fails loudly, at load, naming the value.
    """

    AZURE = "azure"


class Settings(BaseSettings):
    """Application configuration, read from the environment and `.env`.

    Attributes are grouped to match `.env.example`. Anything marked FROZEN must
    agree with the values recorded in the index's `index_metadata` table — the
    index was built with a specific embedding model, and querying it with a
    different one degrades retrieval silently.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_` is a protected namespace in pydantic v2, and `model_backend`
        # is far too good a name to give up over it.
        protected_namespaces=(),
    )

    # ---- Model -----------------------------------------------------------
    model_backend: ModelBackend = ModelBackend.AZURE
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    # A `-preview` version, which is what this deployment requires. Re-check it
    # on the 15 and 22 Sep verification dates with everything else: preview
    # api-versions get retired, and this one is load-bearing for both clients.
    azure_openai_api_version: str = "2024-12-01-preview"
    azure_chat_deployment: str = "gpt-5.4-mini"
    azure_embedding_deployment: str = "text-embedding-3-small"  # FROZEN
    # -small is natively 1536, no truncation needed
    # Raising the dimension above 2000 breaks `CREATE INDEX`, not queries
    # — so it fails during ingestion, not at query time.
    embedding_dim: int = 1536
    model_max_retries: int = 5

    @field_validator("azure_openai_endpoint", mode="after")
    @classmethod
    def _endpoint_base_url_only(cls, value: str | None) -> str | None:
        """Reduce whatever the Foundry portal shows to the base URL.

        The portal hands out the v1 form —
        `https://<resource>.services.ai.azure.com/openai/v1` — but the Azure
        LangChain classes build `/openai/deployments/<name>/...?api-version=...`
        onto whatever they are given. Passed through unchanged, the request goes
        to `/openai/v1/openai/deployments/...`, which 404s, and a 404 on a
        deployment reads exactly like a wrong deployment name. Nineteen people
        will paste the portal value, so normalise it here rather than in a guide
        nobody re-reads.

        Also treats an empty value as unset: a `.env` copied from the template
        has `AZURE_OPENAI_ENDPOINT=` and nothing after it, and an empty string
        produces a baffling URL error instead of a clean missing-credential one.

        Never raises — `Settings` construction stays total so `doctor` can load
        a broken configuration in order to explain it.
        """
        if value is None:
            return None
        trimmed = value.strip().rstrip("/")
        # The portal shows this resource under three different URLs and only the
        # base one works here. Seen in the wild, all three from the same resource:
        #   .../openai/v1                     the v1 (OpenAI-compatible) surface
        #   .../api/projects/proj-default      the Foundry *project* endpoint
        #   https://<res>.cognitiveservices.azure.com   the inference endpoint
        # The project form is the nastiest: the extra path makes the service
        # answer 400 "API version not supported", which sends you off to change
        # the api-version, which is not the problem.
        trimmed = re.sub(r"/api/projects/[^/]+$", "", trimmed)
        for suffix in ("/openai/v1", "/openai"):
            if trimmed.endswith(suffix):
                trimmed = trimmed[: -len(suffix)]
                break
        return trimmed or None

    # ---- Index -----------------------------------------------------------
    # There is no INDEX_BACKEND switch. `PGHOST` already says which Postgres is
    # being searched, and a second setting that only *described* the first could
    # disagree with it — a label that could drift from the host it names is
    # exactly the kind of quiet wrongness this repo spends its effort avoiding.
    pghost: str = "localhost"
    pgport: int = 5432
    pgdatabase: str = "travel_assist"
    pguser: str = "travel_assist"
    pgpassword: str = "travel_assist"
    pgsslmode: str = "prefer"
    pgconnect_timeout: int = 5

    # ---- Observability and prompt management -----------------------------
    # MLflow, local by default: a SQLite file, not a service. `mlflow ui` renders it.
    # Override to a remote tracking server only if one exists; nothing here needs one.
    #
    # SQLite rather than the older `file:./mlruns` store deliberately: MLflow 3.15
    # put the filesystem backend into maintenance mode and *raises* on it unless
    # MLFLOW_ALLOW_FILE_STORE=true. Found by running it, not by reading about it.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "travel-assist"

    # ---- Context management -----------------------------------------------
    context_token_budget: int = 8000
    history_token_budget: int = 2000
    compaction_strategy: Literal["summarise", "drop_oldest", "none"] = "summarise"

    @property
    def chat_model_name(self) -> str:
        """The deployment name this process would send a prompt to."""
        return self.azure_chat_deployment

    @property
    def embedding_model_name(self) -> str:
        """The embedding model this process would query with.

        Written into `index_metadata` at ingestion time and compared against it
        on every startup, so changing embedding deployment forces a re-ingest
        rather than silently corrupting retrieval. See `db.check_embedding_contract`.
        """
        return self.azure_embedding_deployment

    @property
    def mlflow_local_path(self) -> Path | None:
        """The file or directory a local tracking URI points at, else None.

        `doctor` reports the resolved path: "traces in ./mlflow.db" is actionable
        in a way that echoing the raw URI back is not, and a participant who
        cannot find their traces is usually looking in the wrong directory.
        """
        for prefix in ("sqlite:///", "file:"):
            if self.mlflow_tracking_uri.startswith(prefix):
                return Path(self.mlflow_tracking_uri[len(prefix) :]).expanduser().resolve()
        return None

    @property
    def mlflow_is_local(self) -> bool:
        """True when tracing writes to this machine rather than to a server."""
        return self.mlflow_local_path is not None

    def describe_backends(self) -> str:
        """One line, printed every run, naming who answered and what was searched.

        Both halves matter and people conflate them constantly: a good answer
        over the wrong corpus and a bad answer over the right one look the same
        from the outside. The index half names the host and database rather than
        a category, because that is the fact somebody needs when two people get
        different results from the same question.
        """
        return (
            f"model backend: {self.model_backend.value} ({self.chat_model_name}) · "
            f"index: {self.pghost}/{self.pgdatabase}"
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide Settings, constructing it on first use.

    Cached deliberately: constructing Settings reads the environment and logs the
    backend decision, and we want that to happen exactly once per process.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_cache() -> None:
    """Drop the cached Settings. For tests and for `doctor` re-checks only."""
    global _settings
    _settings = None

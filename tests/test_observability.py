"""Tracing, and its degradation path.

The property that matters most here is the one that is easiest to get wrong:
**tracing being unavailable must never stop a run.** Nineteen people, hybrid,
two days — the observability stack failing should cost visibility, nothing else.
Every test below that looks like it is about failure is really about that.

The second property: a trace that cannot name the backend that served it is not
comparable with any other trace, and comparing two runs later is the entire
reason for keeping them.

Tracing is local (`sqlite:///mlflow.db`), so there is no "unconfigured" state to test
any more — the interesting failures are a broken import and a tracking URI that
points somewhere unwritable.
"""

from __future__ import annotations

import logging

import pytest

from travel_assist.config import Settings
from travel_assist.observability.tracing import (
    configure_tracing,
    reset_tracing,
    traced_config,
)


@pytest.fixture(autouse=True)
def _fresh_tracing():
    """Setup is process-cached, so tests must not inherit each other's."""
    reset_tracing()
    yield
    reset_tracing()


def settings_for(**overrides: object) -> Settings:
    base: dict[str, object] = {"model_backend": "azure"}
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def test_tracing_is_local_by_default():
    """No URL, no keys, no service: the default is a directory on this machine.

    This is the whole reason for the switch away from a hosted tracker — it is
    what makes the observability row of the provisioning runbook empty.
    """
    settings = settings_for()

    assert settings.mlflow_tracking_uri == "sqlite:///mlflow.db"
    assert settings.mlflow_is_local
    assert settings.mlflow_local_path is not None
    assert settings.mlflow_local_path.name == "mlflow.db"


def test_a_remote_tracking_uri_is_not_reported_as_a_local_path():
    """`doctor` prints a filesystem path only when there is one to print."""
    settings = settings_for(mlflow_tracking_uri="http://mlflow.internal:5000")

    assert not settings.mlflow_is_local
    assert settings.mlflow_local_path is None


def test_traced_config_still_works_untraced(monkeypatch):
    """The chain must be invokable whether or not anything is watching."""
    monkeypatch.setattr(
        "travel_assist.observability.tracing.configure_tracing", lambda *a, **k: False
    )
    config = traced_config(phase="p1", milestone="context-budget", settings=settings_for())

    assert config["metadata"]["milestone"] == "context-budget"
    assert config["run_name"] == "context-budget"


def test_every_run_records_which_backend_served_it():
    """A trace that cannot name its provider is not comparable with any other."""
    config = traced_config(phase="p1", milestone="vector-search", settings=settings_for())

    assert config["metadata"]["model_backend"] == "azure"
    assert config["metadata"]["phase"] == "p1"
    assert "/" in config["metadata"]["index"], "the index half names host/database"


def test_session_id_is_attached_when_given():
    config = traced_config(
        phase="p2", milestone="chat", session_id="thread-7", settings=settings_for()
    )

    assert config["metadata"]["session_id"] == "thread-7"


def test_setup_failure_warns_once_not_per_call(caplog, monkeypatch):
    """A broken MLflow costs one warning per process, not one per invocation."""

    def explode(_uri: str) -> None:
        raise RuntimeError("tracking store is unwritable")

    monkeypatch.setattr("mlflow.set_tracking_uri", explode)
    settings = settings_for()

    with caplog.at_level(logging.WARNING):
        results = [configure_tracing(settings) for _ in range(5)]

    assert results == [False] * 5, "a failed setup must stay failed, not retry per call"
    assert caplog.text.count("MLflow tracing unavailable") == 1


def test_a_broken_tracker_does_not_stop_the_chain(monkeypatch, tmp_path):
    """The point of the whole module: visibility degrades, the exercise does not."""
    from langchain_core.runnables import RunnableLambda

    monkeypatch.setattr(
        "mlflow.langchain.autolog", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope"))
    )
    settings = settings_for(mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")

    config = traced_config(phase="p1", milestone="rrf", settings=settings)
    result = RunnableLambda(lambda x: x * 2).invoke(3, config=config)

    assert result == 6


def test_a_real_chain_produces_a_real_trace(tmp_path):
    """End to end against a local store, with no model and no network.

    Autologging is global rather than per-call by design, so the thing worth
    proving is that one `traced_config` is enough to make an ordinary chain show
    up as a trace — nobody has to remember to attach anything.
    """
    import mlflow
    from langchain_core.runnables import RunnableLambda

    settings = settings_for(mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    config = traced_config(phase="p1", milestone="hybrid-search", settings=settings)

    chain = RunnableLambda(lambda q: f"retrieved for {q}")
    assert chain.invoke("porto", config=config) == "retrieved for porto"

    from travel_assist.observability.tracing import flush

    flush(settings)

    # `locations`, not the deprecated `experiment_ids`/`experiment_names`.
    experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment)
    assert experiment is not None
    traces = mlflow.search_traces(locations=[experiment.experiment_id], return_type="list")
    assert traces, "one traced_config call must be enough to record a trace"

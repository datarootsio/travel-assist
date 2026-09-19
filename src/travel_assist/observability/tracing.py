"""MLflow tracing: a local directory, one dashboard, no accounts.

Scaffolding — ships implemented.

Traces exist from the very first milestone, deliberately. Day 1 uses them to see
how context was assembled; day 2 uses them to debug an agent loop that nobody
sequenced. If tracing only arrived on day 2 there would be no history to compare
against, and the first trace anyone read would be the complicated one.

**Nothing to provision.** `MLFLOW_TRACKING_URI` defaults to `sqlite:///mlflow.db`,
so tracing is one file on your laptop. `make mlflow-ui` renders it — trace trees,
span timings, token counts. No hosted service, no keys, no signup, and nothing
that can be down on the morning of the course.

SQLite and not the older `file:./mlruns` store because MLflow 3.15 moved the
filesystem backend into maintenance mode and raises on it by default. That is
the repo's standing rule about upstream churn, collected the hard way: check the
installed package, never a blog post.

**Autologging attaches once, globally, and that is the point.** A callback handler
passed to each individual model call produces a flat list of unrelated spans;
attached above the chain it produces a tree, and the tree is what makes an agent's
behaviour legible. `mlflow.langchain.autolog()` makes the correct version the only
version — there is no per-call variant to get wrong.

The trade we accepted: traces are local, so an instructor cannot open a
participant's run from their own machine. Nineteen laptops, a third of them
remote, made a shared instance the more expensive half of that trade.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from travel_assist.config import Settings, get_settings

logger = logging.getLogger(__name__)

_configured = False
_failed = False


def configure_tracing(settings: Settings | None = None) -> bool:
    """Point MLflow at the tracking store and switch on LangChain autologging.

    Idempotent and cached, including the failure: a broken MLflow should cost one
    warning per process, not one per call.

    Returns:
        True when tracing is active, False when it could not be set up. False is
        not an error — the run continues untraced, because observability must
        never stop the room.
    """
    global _configured, _failed
    settings = settings or get_settings()

    if _configured:
        return True
    if _failed:
        return False

    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        # Traces every LCEL chain and agent invocation in the process, as a tree.
        mlflow.langchain.autolog()
    except Exception as error:  # noqa: BLE001 — observability must never break a run
        logger.warning("MLflow tracing unavailable (%s); continuing untraced", error)
        _failed = True
        return False

    _configured = True
    logger.info(
        "tracing to %s, experiment %r", settings.mlflow_tracking_uri, settings.mlflow_experiment
    )
    return True


def reset_tracing() -> None:
    """Drop the cached setup. For tests and for `doctor` re-checks."""
    global _configured, _failed
    _configured = False
    _failed = False


def traced_config(
    *,
    phase: str,
    milestone: str,
    session_id: str | None = None,
    settings: Settings | None = None,
    **extra: str,
) -> RunnableConfig:
    """Build the config to hand to `chain.invoke`, carrying the trace tags.

    Every run is tagged with the same four things so traces can be filtered and
    compared later: which day it belongs to, which milestone produced it, which
    conversation it is part of, and which model backend served it. The last one
    matters most — traces outlive the day they were recorded, and one that cannot
    name its provider cannot be compared with one from another.

    Calling this also switches tracing on, so no milestone has to remember to.

    Args:
        phase: `p1` for the GenAI day, `p2` for the Agentic day.
        milestone: Milestone slug, e.g. `context-budget`.
        session_id: Conversation this run belongs to, so multi-turn threads group.
        settings: Configuration; defaults to the process-wide Settings.
        **extra: Any further metadata worth attaching.

    Returns:
        A `RunnableConfig` whose metadata becomes span attributes and whose
        `run_name` names the trace. Safe when MLflow is unavailable — autologging
        simply never attaches and the chain runs as normal.
    """
    settings = settings or get_settings()
    configure_tracing(settings)

    metadata: dict[str, Any] = {
        "phase": phase,
        "milestone": milestone,
        "model_backend": settings.model_backend.value,
        "index": f"{settings.pghost}/{settings.pgdatabase}",
        **extra,
    }
    if session_id:
        metadata["session_id"] = session_id

    return RunnableConfig(metadata=metadata, run_name=milestone)


def flush(settings: Settings | None = None) -> None:
    """Wait for buffered traces to reach the tracking store.

    MLflow logs traces on a background thread, so a short-lived process — a CLI
    command, an eval run — can exit before its traces are written. Call this
    before shutdown, or the run you just made will not be in the UI when you look.
    """
    if not configure_tracing(settings):
        return

    import mlflow

    # Named `flush_trace_async_logging` in MLflow 3.x. Guarded rather than
    # assumed: a rename upstream should cost a warning, not a crash on exit.
    flusher = getattr(mlflow, "flush_trace_async_logging", None)
    if flusher is None:
        logger.debug("this MLflow has no explicit trace flush; traces are written inline")
        return
    try:
        flusher()
    except Exception as error:  # noqa: BLE001 — observability must never break a run
        logger.warning("could not flush traces (%s)", error)

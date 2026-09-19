"""Diagnose the environment, one named check at a time.

Scaffolding — ships implemented. Run this first, and run it again whenever
anything behaves strangely.

The design rule here is that a check earns its place only if it can say **what to
do next**. "Database: FAIL" wastes a participant's morning; "Database: FAIL —
could not translate host name 'travel-assist.postgres.database.azure.com'; check
PGHOST against the handout and PGSSLMODE=require" does not.

Every check therefore returns a status, a detail describing what was actually
observed, and a remedy. Checks never raise: a broken environment must still
produce a full report, because the second failure is often the one that explains
the first.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import typer
from rich.console import Console
from rich.table import Table

from travel_assist.config import Settings, get_settings
from travel_assist.db import (
    IndexCompatibilityError,
    check_embedding_contract,
    connect,
    read_index_metadata,
)
from travel_assist.models import get_chat_model
from travel_assist.observability.tracing import configure_tracing

logger = logging.getLogger(__name__)
console = Console()


class CheckStatus(StrEnum):
    """How a single check came out.

    `WARN` means degraded but usable — the room keeps moving. `FAIL` means this
    environment cannot do the exercise.
    """

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class CheckResult:
    """One diagnosis: what was checked, what was seen, and what to do."""

    name: str
    status: CheckStatus
    detail: str
    remedy: str = ""


def check_configuration(settings: Settings) -> CheckResult:
    """Report the resolved backends. Always passes; exists to be read aloud."""
    return CheckResult(
        name="configuration",
        status=CheckStatus.OK,
        detail=settings.describe_backends(),
    )


def check_model(
    settings: Settings,
    probe: Callable[[], None] | None = None,
) -> CheckResult:
    """Confirm the configured model actually answers.

    There is no canned backend to fall back to, so this row is a FAIL rather than
    a WARN when credentials are absent — and it is the most likely row to be red
    on the morning of the course. Building the client is inside the probe on
    purpose: a missing key and a rejected key then produce the same shaped
    diagnosis, differing only in what the message says.

    Args:
        settings: Configuration to check.
        probe: Overrides the live call. Injected by tests; in normal use this
            makes one tiny completion against the configured deployment.
    """
    probe = probe or _default_model_probe(settings)

    try:
        probe()
    except Exception as exc:  # noqa: BLE001 — a diagnosis must survive any failure
        return CheckResult(
            name="model",
            status=CheckStatus.FAIL,
            detail=str(exc),
            remedy=(
                "Check AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT and "
                f"AZURE_CHAT_DEPLOYMENT={settings.azure_chat_deployment!r}. Unset means the "
                ".env was never filled in; a 401 means the key is wrong or expired; a 404 "
                "usually means the deployment name does not exist in this resource; a 429 "
                "means the room is saturating the quota."
            ),
        )

    return CheckResult(
        name="model",
        status=CheckStatus.OK,
        # Name the backend that actually answered. Hardcoding "azure" here made
        # doctor report an Azure deployment while running on a different backend —
        # a PASS row that names the wrong provider is worse than no row at all.
        detail=f"{settings.model_backend.value}: {settings.chat_model_name}",
    )


def _default_model_probe(settings: Settings) -> Callable[[], None]:
    def probe() -> None:
        # Not 1. Reasoning models (the GPT-5 series among them) spend part of
        # this budget on hidden `reasoning_tokens` before any visible output —
        # a 1-token budget can be consumed entirely by reasoning, producing a
        # "could not finish the message" error that looks like a real failure
        # but is only this probe being too stingy for the deployed model.
        get_chat_model(settings=settings, max_tokens=16).invoke("ping")

    return probe


def check_database(
    settings: Settings,
    probe: Callable[[], None] | None = None,
) -> CheckResult:
    """Confirm Postgres is reachable with the configured credentials."""
    probe = probe or _default_database_probe(settings)

    try:
        probe()
    except Exception as exc:  # noqa: BLE001 — a diagnosis must survive any failure
        return CheckResult(
            name="database",
            status=CheckStatus.FAIL,
            detail=str(exc),
            remedy=(
                f"Could not reach {settings.pghost}:{settings.pgport} as {settings.pguser!r}. "
                "Check PGHOST, PGUSER and PGPASSWORD against the credentials you were given, "
                "that PGSSLMODE=require, and that your IP is allowed through the Postgres "
                "firewall — that last one is the usual answer on a new network."
            ),
        )

    return CheckResult(
        name="database",
        status=CheckStatus.OK,
        detail=f"connected to {settings.pghost}:{settings.pgport}/{settings.pgdatabase}",
    )


def _default_database_probe(settings: Settings) -> Callable[[], None]:
    def probe() -> None:
        # A direct connection, not the pool: the pool retries in the background,
        # so a wrong hostname would look like a hang rather than a diagnosis.
        with connect(settings) as conn, conn.cursor() as cursor:
            cursor.execute("SELECT 1")

    return probe


def check_index(
    settings: Settings,
    probe: Callable[[], dict[str, str]] | None = None,
) -> CheckResult:
    """Confirm the index exists and was built by the embedding model we query with."""
    probe = probe or _default_index_probe(settings)

    try:
        metadata = probe()
    except IndexCompatibilityError as exc:
        return CheckResult(
            name="index",
            status=CheckStatus.FAIL,
            detail=str(exc),
            remedy="The message above says exactly what to change. Read it before retrying.",
        )
    except Exception as exc:  # noqa: BLE001 — a diagnosis must survive any failure
        return CheckResult(
            name="index",
            status=CheckStatus.FAIL,
            detail=str(exc),
            remedy="The database answered but the index could not be read. Check the schema.",
        )

    if not metadata:
        return CheckResult(
            name="index",
            status=CheckStatus.FAIL,
            detail="the index is empty — no index_metadata rows",
            remedy=(
                "The schema exists but nothing has been ingested. Locally, seed the sample "
                "corpus (see ingestion/README.md). On the teaching day, point PGHOST at the "
                "shared Azure index."
            ),
        )

    return CheckResult(
        name="index",
        status=CheckStatus.OK,
        detail=(
            f"{metadata.get('n_chunks', '?')} chunks, "
            f"embedding model {metadata.get('embedding_model', '?')}, "
            f"dump {metadata.get('dump_date', '?')}"
        ),
    )


def _default_index_probe(settings: Settings) -> Callable[[], dict[str, str]]:
    def probe() -> dict[str, str]:
        with connect(settings) as conn:
            metadata = read_index_metadata(conn)
        if metadata:
            check_embedding_contract(settings, metadata)
        return metadata

    return probe


def check_tracing(
    settings: Settings,
    probe: Callable[[], bool] | None = None,
) -> CheckResult:
    """Confirm runs will be traced.

    Never a FAIL. Observability going down must degrade the experience, not stop
    the exercise — the run continues untraced.

    There is nothing to provision here: the default tracking URI is a directory
    on this machine. So the interesting failure is not "unreachable", it is "you
    are looking in the wrong place for your traces", which is why the OK row
    prints the resolved path.
    """
    probe = probe or (lambda: configure_tracing(settings))

    try:
        active = probe()
    except Exception as exc:  # noqa: BLE001 — a diagnosis must survive any failure
        return CheckResult(
            name="tracing",
            status=CheckStatus.WARN,
            detail=str(exc),
            remedy=(
                "Runs will not be traced. Not fatal — everything else still works. "
                "Check that mlflow imported cleanly: `uv run python -c 'import mlflow'`."
            ),
        )

    if not active:
        return CheckResult(
            name="tracing",
            status=CheckStatus.WARN,
            detail=f"could not start tracing to {settings.mlflow_tracking_uri}",
            remedy=(
                "Runs will not be traced. Not fatal. If MLFLOW_TRACKING_URI points at a "
                "remote server, check it is up; the default `sqlite:///mlflow.db` needs only a "
                "writable working directory."
            ),
        )

    local = settings.mlflow_local_path
    where = f"{local}" if local else settings.mlflow_tracking_uri
    return CheckResult(
        name="tracing",
        status=CheckStatus.OK,
        detail=f"experiment {settings.mlflow_experiment!r} → {where}"
        + (" · view with `make mlflow-ui`" if local else ""),
    )


def run_checks(settings: Settings | None = None) -> list[CheckResult]:
    """Run every check in order, collecting rather than short-circuiting.

    Order matters: configuration explains the rest, and the index check is only
    interpretable once you know whether the database answered at all.
    """
    settings = settings or get_settings()

    database = check_database(settings)

    # The index check has nothing true to say without a database, and reporting
    # it as a second, separate failure sends people off to debug a schema that
    # was never the problem. Skip it and point at the real cause.
    if database.status is CheckStatus.FAIL:
        index = CheckResult(
            name="index",
            status=CheckStatus.SKIP,
            detail="not checked — the database is unreachable (see above)",
            remedy="Fix the database check first, then run `travel-assist doctor` again.",
        )
    else:
        index = check_index(settings)

    return [
        check_configuration(settings),
        check_model(settings),
        database,
        index,
        check_tracing(settings),
    ]


_STATUS_STYLE = {
    CheckStatus.OK: ("[green]PASS[/green]", "green"),
    CheckStatus.WARN: ("[yellow]WARN[/yellow]", "yellow"),
    CheckStatus.FAIL: ("[red]FAIL[/red]", "red"),
    CheckStatus.SKIP: ("[dim]SKIP[/dim]", "dim"),
}


def render(results: list[CheckResult]) -> None:
    """Print the report, then the remedies for anything that is not OK."""
    table = Table(title="travel-assist doctor", show_lines=False)
    table.add_column("check", style="bold")
    table.add_column("status")
    table.add_column("detail", overflow="fold")

    for result in results:
        label, _ = _STATUS_STYLE[result.status]
        table.add_row(result.name, label, result.detail.splitlines()[0])

    console.print(table)

    for result in results:
        if result.status in (CheckStatus.OK, CheckStatus.SKIP) or not result.remedy:
            continue
        _, colour = _STATUS_STYLE[result.status]
        console.print(f"\n[{colour}]{result.name}[/{colour}] — {result.detail}")
        console.print(f"  [dim]{result.remedy}[/dim]")


def doctor() -> None:
    """Check the environment and report exactly what is broken.

    Exits non-zero if any check failed, so the pre-course runbook and CI can
    depend on it. Warnings do not fail the command: a degraded environment still
    completes every milestone.
    """
    results = run_checks()
    render(results)

    if any(result.status is CheckStatus.FAIL for result in results):
        console.print(
            "\n[red]Some checks failed.[/red] Fix them top to bottom — "
            "a later failure is often caused by an earlier one."
        )
        raise typer.Exit(code=1)

    console.print("\n[green]Environment looks good.[/green]")

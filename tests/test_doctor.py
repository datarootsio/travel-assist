"""`travel-assist doctor` — the command that has to be excellent.

Nineteen people, hybrid, mixed Python levels, two consecutive days. Environment
drift is the single biggest time sink available, and a participant who cannot
tell *which* of five things is broken will lose the morning and take an
instructor with them.

Doctor is scaffolding participants run and read, not an exercise they write, so
what is pinned here is its *behaviour*: it names every check, its exit code is
usable from a script, a warning does not stop the room, and a cascading failure
is reported once rather than five times. The exact wording of each remedy is
read by a person and changed freely; tests that spelled it out only made it
expensive to improve.
"""

from __future__ import annotations

from typer.testing import CliRunner

from travel_assist.cli.doctor import (
    CheckResult,
    CheckStatus,
    check_model,
    check_tracing,
    run_checks,
)
from travel_assist.cli.main import app
from travel_assist.config import Settings

runner = CliRunner()


def settings_for(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def healthy_results() -> list[CheckResult]:
    """A full green report, so CLI-level tests do not depend on the real world.

    Invoking the command for real also made a live chat completion on every
    `make test`, which is neither free nor deterministic.
    """
    return [
        CheckResult(name=name, status=CheckStatus.OK, detail=f"{name} is fine")
        for name in ("configuration", "model", "database", "index", "tracing")
    ]


def test_model_check_fails_when_no_credentials_are_configured():
    """There is no fallback backend any more, so this is a FAIL, not a WARN.

    It is also the single most likely row to be red on the morning of the course,
    which is why it goes through the real code path rather than an injected
    probe: the factory raises, and doctor has to turn that into a remedy.
    """
    result = check_model(settings_for(model_backend="azure", azure_openai_api_key=None))

    assert result.status is CheckStatus.FAIL
    assert "AZURE_OPENAI_API_KEY" in result.detail
    assert "AZURE_OPENAI_API_KEY" in result.remedy


def test_tracing_check_warns_rather_than_fails_when_broken():
    """Tracing failing must never block the room — it degrades, it does not stop."""
    result = check_tracing(settings_for(), probe=lambda: False)

    assert result.status is CheckStatus.WARN


def test_index_is_not_checked_when_the_database_is_unreachable():
    """Cascading failures with a wrong remedy are how a morning gets lost.

    With no database, the index check cannot say anything true. Reporting it as
    a second failure — and blaming the schema — sends a participant off to debug
    something that was never the problem.
    """
    results = {result.name: result for result in run_checks(settings_for(pghost="nope.invalid"))}

    assert results["database"].status is CheckStatus.FAIL
    assert results["index"].status is CheckStatus.SKIP
    assert "database" in results["index"].detail.lower()


def test_doctor_runs_and_names_every_check(monkeypatch):
    monkeypatch.setattr("travel_assist.cli.doctor.run_checks", lambda: healthy_results())

    result = runner.invoke(app, ["doctor"])

    for check_name in ("configuration", "model", "database", "index", "tracing"):
        assert check_name in result.stdout.lower()


def test_doctor_exits_zero_when_everything_passes(monkeypatch):
    monkeypatch.setattr("travel_assist.cli.doctor.run_checks", lambda: healthy_results())

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0


def test_doctor_exits_non_zero_when_a_check_fails(monkeypatch):
    """CI and the pre-course runbook both depend on the exit code.

    The failure is injected, not borrowed from the ambient environment. An
    earlier version of this test invoked the real command and asserted it failed
    "because no database is running" — so it passed only while the developer's
    machine was broken, and turned red the moment the environment came good.
    A test that fails when the world gets better is measuring the wrong thing.
    """
    failing = [
        *healthy_results()[:2],
        CheckResult(
            name="database",
            status=CheckStatus.FAIL,
            detail="could not connect",
            remedy="check PGHOST against the credentials you were given",
        ),
    ]
    monkeypatch.setattr("travel_assist.cli.doctor.run_checks", lambda: failing)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code != 0
    assert "check PGHOST against the credentials you were given" in result.stdout


def test_a_warning_alone_does_not_fail_the_command(monkeypatch):
    """A degraded environment still completes every milestone — see `check_tracing`."""
    warned = [
        *healthy_results()[:4],
        CheckResult(
            name="tracing",
            status=CheckStatus.WARN,
            detail="not configured — runs will not be traced",
            remedy="runs will not be traced",
        ),
    ]
    monkeypatch.setattr("travel_assist.cli.doctor.run_checks", lambda: warned)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0

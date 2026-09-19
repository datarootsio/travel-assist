"""`travel-assist chat` — the REPL wired to `pipeline.chain.answer_query`.

`answer_query` and `verify_index` are both scripted here. The CLI's own job is
the loop and the rendering, not the pipeline itself, so nothing in this file
needs a database, an embedding client or a live chat model.
"""

from __future__ import annotations

from typer.testing import CliRunner

from travel_assist.cli.main import app
from travel_assist.db import IndexCompatibilityError
from travel_assist.pipeline.chain import PipelineTurn
from travel_assist.pipeline.citations import Claim, DestinationAnswer, SelfCheckResult

runner = CliRunner()


def flat(text: str) -> str:
    """Collapse Rich's line wrapping, which breaks sentences at the terminal width."""
    return " ".join(text.split())


def test_a_normal_turn_renders_the_answer_and_its_citations(monkeypatch):
    monkeypatch.setattr("travel_assist.cli.chat.verify_index", lambda settings: {})
    answer = DestinationAnswer(
        claims=[
            Claim(text="It has a bookshop.", source_chunk_id=1),
            Claim(text="This part is unsupported.", source_chunk_id=None),
        ]
    )
    turn = PipelineTurn(
        route="destination_question",
        retrieved=True,
        answer=answer,
        self_check=SelfCheckResult(passed=False, uncited=[answer.claims[1]]),
        context=[],
    )
    monkeypatch.setattr("travel_assist.cli.chat.answer_query", lambda query, **kwargs: turn)

    result = runner.invoke(app, ["chat"], input="what's in Porto\nexit\n")

    assert result.exit_code == 0
    output = flat(result.stdout)
    assert "It has a bookshop." in output
    assert "chunk_id=1" in output
    assert "This part is unsupported." in output
    assert "uncited" in output
    assert "1 claim(s) carry no citation" in output


def test_no_citations_hides_chunk_ids_but_keeps_the_answer_and_self_check(monkeypatch):
    monkeypatch.setattr("travel_assist.cli.chat.verify_index", lambda settings: {})
    answer = DestinationAnswer(
        claims=[
            Claim(text="It has a bookshop.", source_chunk_id=1),
            Claim(text="This part is unsupported.", source_chunk_id=None),
        ]
    )
    turn = PipelineTurn(
        route="destination_question",
        retrieved=True,
        answer=answer,
        self_check=SelfCheckResult(passed=False, uncited=[answer.claims[1]]),
        context=[],
    )
    monkeypatch.setattr("travel_assist.cli.chat.answer_query", lambda query, **kwargs: turn)

    result = runner.invoke(app, ["chat", "--no-citations"], input="what's in Porto\nexit\n")

    assert result.exit_code == 0
    output = flat(result.stdout)
    # The claims still appear once, as the answer text itself — `--no-citations`
    # only suppresses the per-claim `[chunk_id=N]`/`[uncited]` source lines.
    assert "It has a bookshop." in output
    assert output.count("This part is unsupported.") == 1
    assert "chunk_id=1" not in output
    assert "uncited" not in output
    assert "1 claim(s) carry no citation" in output  # self-check stays visible


def test_typing_exit_ends_the_loop_cleanly_without_ever_answering(monkeypatch):
    monkeypatch.setattr("travel_assist.cli.chat.verify_index", lambda settings: {})

    def _must_not_be_called(query: str, **kwargs: object) -> PipelineTurn:
        raise AssertionError("answer_query must not be called")

    monkeypatch.setattr("travel_assist.cli.chat.answer_query", _must_not_be_called)

    result = runner.invoke(app, ["chat"], input="exit\n")

    assert result.exit_code == 0


def test_an_incompatible_index_exits_non_zero_and_prints_the_error(monkeypatch):
    def raise_incompatible(settings):
        raise IndexCompatibilityError("embedding model mismatch: found a stale index")

    monkeypatch.setattr("travel_assist.cli.chat.verify_index", raise_incompatible)

    result = runner.invoke(app, ["chat"])

    assert result.exit_code != 0
    assert "embedding model mismatch" in flat(result.stdout)

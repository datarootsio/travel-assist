"""`travel-assist chat` — the day-1 pipeline, live, multi-turn.

Scaffolding — ships implemented, like `cli/search.py` and `cli/doctor.py`. It
wires together what `pipeline/chain.py` builds; there is no new solution logic
here, only the loop and the rendering.
"""

from __future__ import annotations

from typing import Annotated
from uuid import uuid4

import typer
from rich.console import Console

from travel_assist.config import get_settings
from travel_assist.db import IndexCompatibilityError, verify_index
from travel_assist.observability.tracing import flush, traced_config
from travel_assist.pipeline.chain import PipelineTurn, answer_query
from travel_assist.pipeline.memory import CompactingHistory

console = Console()

_EXIT_WORDS = {"exit", "quit"}


def _render(turn: PipelineTurn, *, show_citations: bool) -> None:
    """Print one turn: the answer or message, its citations, and any self-check flag."""
    console.print(f"[bold cyan]assistant[/bold cyan] {turn.display_text}")

    if show_citations and turn.answer is not None:
        for claim in turn.answer.claims:
            label = (
                f"[dim]\\[chunk_id={claim.source_chunk_id}][/dim]"
                if claim.source_chunk_id is not None
                else "[yellow]\\[uncited][/yellow]"
            )
            console.print(f"  {label} {claim.text}")

    if turn.self_check is not None and not turn.self_check.passed:
        n = len(turn.self_check.uncited)
        console.print(f"[yellow]self-check: {n} claim(s) carry no citation[/yellow]")
    console.print()


def chat(
    citations: Annotated[
        bool,
        typer.Option(
            "--citations/--no-citations",
            help="Show each claim's chunk_id source below the answer.",
        ),
    ] = True,
) -> None:
    """Start a multi-turn chat session against the day-1 pipeline."""
    settings = get_settings()
    console.print(f"[dim]{settings.describe_backends()}[/dim]")

    # Same reasoning as `cli/search.py`: fail before spending a call on an
    # index this process cannot honestly query.
    try:
        verify_index(settings)
    except IndexCompatibilityError as exc:
        console.print(f"\n[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print("[dim]Ask about a destination. Type 'exit' or Ctrl-D to quit.[/dim]\n")

    history = CompactingHistory(settings=settings)
    # One session_id for the whole REPL run, so every turn's trace groups together.
    config = traced_config(phase="p1", milestone="chat", session_id=str(uuid4()), settings=settings)

    try:
        while True:
            try:
                query = typer.prompt("you")
            except (EOFError, KeyboardInterrupt):
                console.print()
                break

            stripped = query.strip()
            if stripped.lower() in _EXIT_WORDS:
                break
            if not stripped:
                continue

            turn = answer_query(query, history=history, settings=settings, config=config)
            _render(turn, show_citations=citations)
    finally:
        # MLflow logs traces on a background thread; without this a short REPL
        # session can exit before its traces reach the tracking store.
        flush(settings)

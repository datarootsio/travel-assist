"""`travel-assist search` — run a query through each retrieval method and look.

Scaffolding — ships implemented. It calls the functions participants write; it is
the instrument, not the exercise.

The point of running both methods side by side is that the lesson here is a
*comparison*, and a comparison read from two separate command outputs is not
felt. Put them in one table and the failure is visible in a second:

    travel-assist search "somewhere to cool off on a hot afternoon"  # keyword empty
    travel-assist search "Abdelouahab Derraq" --term Abdelouahab    # vector wrong

The second is the more instructive of the two, and the harder to believe without
seeing: vector search does not return *nothing* for a rare proper noun, it
returns confident, topically adjacent, wrong passages.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from travel_assist.config import get_settings
from travel_assist.db import IndexCompatibilityError, verify_index
from travel_assist.observability.tracing import configure_tracing, flush
from travel_assist.retrieval.models import RetrievedChunk
from travel_assist.retrieval.search import keyword_search, vector_search

console = Console()


def _column(results: list[RetrievedChunk], term: str | None) -> str:
    """One method's ranking, as lines of `rank. label — snippet`."""
    if not results:
        return "[red]— nothing —[/red]"

    lines = []
    for chunk in results:
        snippet = " ".join(chunk.content.split())[:70]
        if term and term.lower() in chunk.content.lower():
            snippet = snippet.replace(term, f"[bold green]{term}[/bold green]")
        lines.append(f"[dim]{chunk.rank}.[/dim] {chunk.citation_label()}\n   {snippet}")
    return "\n".join(lines)


def search(
    query: Annotated[str, typer.Argument(help="What to search for.")],
    k: Annotated[int, typer.Option("-k", help="Results per method.")] = 5,
    term: Annotated[
        str | None,
        typer.Option("--term", help="Highlight this string where it appears in a passage."),
    ] = None,
) -> None:
    """Search the index with vector and keyword retrieval, side by side."""
    settings = get_settings()
    console.print(f"[dim]{settings.describe_backends()}[/dim]")

    # Fail before spending an embedding on a query we cannot answer honestly.
    # Without this, querying an index built by a different embedding model
    # returns a confidently ranked list drawn from the wrong vector space —
    # no error, no warning, and nothing in the output that looks unusual.
    try:
        verify_index(settings)
    except IndexCompatibilityError as exc:
        console.print(f"\n[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    # vector_search/keyword_search are plain functions, not LCEL runnables (see
    # retrieval/search.py) — there's no `.invoke(config=...)` to tag, but this
    # still switches on autologging so their underlying model/embedding calls trace.
    configure_tracing(settings)

    dense = vector_search(query, k=k, settings=settings)
    sparse = keyword_search(query, k=k, settings=settings)

    table = Table(title=f"{query!r}", show_lines=True, title_justify="left")
    table.add_column("vector (dense)", overflow="fold", ratio=1)
    table.add_column("keyword (sparse)", overflow="fold", ratio=1)
    table.add_row(_column(dense, term), _column(sparse, term))
    console.print(table)

    if not sparse:
        console.print(
            "[yellow]Keyword search found nothing.[/yellow] None of the query's terms appear "
            "in the corpus — sparse retrieval cannot match a word that is not there."
        )
    if term and not any(term.lower() in chunk.content.lower() for chunk in dense):
        console.print(
            f"[yellow]Vector search did not surface {term!r} at all[/yellow] — but it did "
            "return passages, and they look reasonable. That is the expensive failure: "
            "plausible, related, and wrong."
        )

    flush(settings)
